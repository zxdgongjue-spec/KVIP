#!/usr/bin/env bash
# NextGen V1: Q4 variable K, independent first-round generic URMA streaming.
# Usage: CONFIRM_NPU6_IDLE=1 bash this-script URMA_MANIFEST PATCH_EVIDENCE_JSON ORIGINAL_FRAMEWORK_ROOT INIT_EVIDENCE_DIR
set -Eeuo pipefail
umask 077
[[ $# == 4 ]] || { echo "Usage: $0 URMA_MANIFEST PATCH_EVIDENCE_JSON ORIGINAL_FRAMEWORK_ROOT INIT_EVIDENCE_DIR" >&2; exit 2; }
MANIFEST=$(realpath "$1")
PATCH_EVIDENCE=$(realpath "$2")
ORIGINAL_ROOT=$(realpath "$3")
INIT_EVIDENCE=$(realpath "$4")
[[ "${CONFIRM_NPU6_IDLE:-0}" == 1 ]] || { echo 'Confirm physical NPU 6 is healthy, idle and allocated to you first.' >&2; exit 2; }
source "$MANIFEST"
WORK=$(dirname "$VENV")
OUT=$(mktemp -d "$WORK/v1-varlen.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
SERVICE_PID=''
COMPLETE=0
trap 'rc=$?; if (( rc != 0 || COMPLETE != 1 )); then echo "V1_FAILED rc=$rc OUT=$OUT SERVICE_PID=${SERVICE_PID:-not_started}"; if (( rc == 0 )); then exit 99; fi; fi' EXIT
echo "OUTPUT_DIR=$OUT"
SCRIPT=$(realpath "$0")
REPO=$(git -C "$(dirname "$SCRIPT")" rev-parse --show-toplevel)
REL=$(realpath --relative-to="$REPO" "$SCRIPT")
COMMIT=$(git -C "$REPO" rev-parse HEAD)
[[ $(git -C "$REPO" hash-object "$SCRIPT") == $(git -C "$REPO" rev-parse "HEAD:$REL") ]]
cp "$SCRIPT" "$OUT/executed.sh"
cp "$MANIFEST" "$OUT/urma_manifest.env"
cp "$PATCH_EVIDENCE" "$OUT/kvaf_new_files.json"
printf '%s\n' "$COMMIT" > "$OUT/test_commit.txt"
sha256sum "$OUT/executed.sh"
echo "TEST_COMMIT=$COMMIT"

echo '===== 1. FROZEN BUILD GATE ====='
[[ "$KVIP_SHA" == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
[[ "$KVAF_SHA" == 87e261ae4d07d4868cfdbb306d15f3f137edfc3d ]]
[[ $(git -C "$NEW_KVIP" rev-parse HEAD) == "$KVIP_SHA" ]]
[[ $(git -C "$NEW_KVAF" rev-parse HEAD) == "$KVAF_SHA" ]]
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet "$KVIP_SHA" "$COMMIT" -- csrc python
[[ $(git -C "$NEW_KVAF" diff --binary HEAD | sha256sum | awk '{print $1}') == 8132c738cb7e7a79fe1dc14f173954cc0110be322e14b25274debed15a8a9052 ]]
[[ "$SOC" == Ascend950DT_9582 && "$TRANSPORT" == urma ]]
for item in \
  KVAF_ENABLE_AIV_BATCH_GET:BOOL=ON KVAF_ENABLE_AIV_BATCH_PUT:BOOL=ON KVAF_BUILD_PYTHON:BOOL=ON \
  KVAF_ENABLE_CPU_OFFLOAD:BOOL=OFF KVAF_ENABLE_NPU_PULL:BOOL=OFF KVAF_ENABLE_AICPU_URMA:BOOL=OFF \
  KVAF_ENABLE_AICPU_CPU_PUSH:BOOL=OFF BUILD_TESTING:BOOL=OFF; do
    grep -Fxq "$item" "$NPU_BUILD/CMakeCache.txt"
done
sha256sum --strict -c <<EOF
85dd545b92d494d6c0c112133f5626781bd086cd80ba31e274ad5999f402663e  $CPU_SERVICE
724b3e2dd86d86e74232566d7469f5ebbdd674037e3cd18638c6ccb7dab26276  $NPU_BUILD/bin/kvaf_execute_batch_get_kernels.o
02e84f41800ed9ffaca5abe3871601946c45be814bd71f6686af11a81b3f8386  $NPU_BUILD/bin/kvaf_execute_batch_put_kernels.o
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
snapshot() {
    for repo in "$ORIGINAL_ROOT" "$ORIGINAL_ROOT/third_party/KVIP" "$ORIGINAL_ROOT/third_party/KVAF"; do
        echo "$repo"
        git -C "$repo" rev-parse HEAD
        git -C "$repo" status --porcelain
        git -C "$repo" diff --binary HEAD | sha256sum
    done
}
snapshot > "$OUT/original_before.txt"
ps -eo pid,args | grep '[k]vaf_cpu_service.*--port 28999' > "$OUT/old_service_before.txt"
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export USER="${USER:-$(id -un)}" ASCEND_RT_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1
export PATH="$VENV/bin:$PATH" PYTHON="$VPY"
export ASCEND_HOME_PATH="$CANN" ASCEND_CANN_PACKAGE_PATH="$CANN" SOC_VERSION="$SOC"
LIDU_EXT="$NEW_KVIP/csrc/operators/lidu/torch_extension"
export PYTHONPATH="$NEW_KVIP/python:$NPU_BUILD/python:$LIDU_EXT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$NEW_KVIP/csrc/operators/lidu/_custom_opp/vendors/customize"
export KVAF_BATCH_GET_KERNEL="$NPU_BUILD/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$NPU_BUILD/bin/kvaf_execute_batch_put_kernels.o"
cd "$NEW_KVIP"
"$VPY" - "$NEW_KVIP" "$NEW_KVAF" "$VENV" "$PATCH_EVIDENCE" <<'PY'
import hashlib, json, sys
from pathlib import Path
k, a, v, evidence = map(lambda x: Path(x).resolve(), sys.argv[1:])
assert Path(sys.prefix).resolve() == v
import kvip, kvaf
import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp as lidu
for module, root in [(kvip, k/'python'), (kvaf, a/'build-npu/python'), (lidu, k/'csrc/operators/lidu/torch_extension')]:
    path = Path(module.__file__).resolve()
    print('MODULE_ORIGIN', module.__name__, path)
    assert path.is_relative_to(root)
records = json.loads(evidence.read_text())
expected = {'access_operator/algorithms/get/direct/urma_aiv/include/kvaf_batch_get_device.h',
            'access_operator/include/ubmem/fused_batch_get_device.h'}
assert len(records) == 2 and {r['path'] for r in records} == expected
for r in records:
    assert hashlib.sha256((a/r['path']).read_bytes()).hexdigest() == r['sha256']
print('FROZEN_MODULE_GATE_PASS')
PY
# Execute the committed validation code, while keeping the built operator tree frozen.
export KVIP_TEST_OPERATOR_ROOT="$NEW_KVIP"
ENTRY="$REPO/tests/operators/lifetch/test_lifetch.py"
echo "TEST_ENTRY=$ENTRY"
echo "OPERATOR_ROOT=$KVIP_TEST_OPERATOR_ROOT"
"$VPY" - "$REPO" "$NEW_KVAF" "$CANN" "$NEW_KVIP" <<'PY'
import sys
from pathlib import Path
repo, kvaf, cann, operator = map(lambda x: Path(x).resolve(), sys.argv[1:])
sys.path.insert(0, str(repo/'tests/operators/lifetch/support'))
import execution
from kvip.runtime.build import kvaf_build_paths
assert Path(execution.__file__).resolve().is_relative_to(repo)
assert execution.ROOT == operator
actual_cann, headers = kvaf_build_paths(kvaf/'build-npu')
assert actual_cann.resolve() == cann
assert execution.find_headers(kvaf) == headers
assert headers.resolve() == kvaf/'access_operator/algorithms/get/direct/urma_aiv/include'
print('TEST_SUPPORT_ORIGIN', execution.__file__)
print('KVAF_CANN', actual_cann, 'KVAF_HEADERS', headers)
print('TEST_HEADER_COMPATIBILITY_PASS')
PY
check_summary() {
    "$VPY" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, kind = Path(sys.argv[1]), sys.argv[2]
files = list(root.glob('*/summary.json'))
assert len(files) == 1, files
run = files[0].parent
rows = json.loads(files[0].read_text())['cases']
assert len(rows) == (14 if kind == 'contract' else 1), rows
assert len({r['case'] for r in rows}) == len(rows)
for r in rows:
    print('CHECK_ROW', r['case'], r['correctness'], r['execution'], r['accepted'])
    assert r['correctness'] == 'PASS' and r['execution'] == 'COMPLETED' and r['accepted'] is True, r
checks = json.loads((run/'cases'/kind/'checks.json').read_text())
assert len(checks) == len(rows) and all(c['correctness'] == 'PASS' for c in checks)
assert {r['case'] for r in rows} == {kind + '/' + c['case'] for c in checks}
assert json.loads((run/'cases'/kind/'worker-result.json').read_text())['exit_code'] == 0
if kind == 'init':
    assert rows[0]['case'] == 'init/init'
    assert 'two independent create/Put/Get/lease/release cycles' in rows[0]['detail']
print('RAW_SUMMARY_PASS', kind, files[0])
PY
}
echo '===== 2. REUSE VERIFIED CONTRACT AND INIT ====='
grep -Fxq M0C_INIT_REPO_PASS "$INIT_EVIDENCE/run.log"
grep -Fxq c994318d74a7b11ba353109bf9fdbbe3ac608cf4 "$INIT_EVIDENCE/test_commit.txt"
cmp "$MANIFEST" "$INIT_EVIDENCE/urma_manifest.env"
check_summary "$INIT_EVIDENCE/contract" contract
check_summary "$INIT_EVIDENCE/init" init
printf '%s\n' "$INIT_EVIDENCE" > "$OUT/init_evidence_path.txt"
echo 'PREVIOUS_INIT_EVIDENCE_PASS'
V0="$WORK/v0-host.1gz9Nq"
grep -Fxq NEXTGEN_V0_REPO_PASS "$V0/run.log"
grep -Fxq 26a0e198ec501d21f4ae20f3328aab8027c2fa43 "$V0/test_commit.txt"
echo 'PREVIOUS_V0_EVIDENCE_PASS'

# Require both absence of reported processes and near-idle HBM.
# Conservative lab threshold, not a hardware limit. Do not reset a device.
idle_gate() {
    local label="$1"
    npu-smi info | tee "$OUT/npu_$label.txt"
    "$VPY" - "$OUT/npu_$label.txt" <<'PY'
import re, sys
from pathlib import Path
data = Path(sys.argv[1]).read_text()
rows = data.splitlines()
matches = []
for i, row in enumerate(rows[:-1]):
    if re.match(r'^\|\s*6\s*\|\s*Ascend950DT\s*\|', row):
        health = row.split('|')[3].strip()
        hbm = re.search(r'(\d+)\s*/\s*(\d+)\s*\|\s*$', rows[i+1])
        if hbm:
            matches.append((health, int(hbm[1]), int(hbm[2])))
if len(matches) != 1:
    raise SystemExit("NPU6_IDLE_GATE_BLOCKED: unrecognized or ambiguous HBM status")
health, used, total = matches[0]
no_process = bool(re.search(r'^\|\s*No running processes found in NPU 6\s*\|', data, re.M))
print(f'NPU6_IDLE_GATE health={health} hbm_used_mb={used} hbm_total_mb={total} no_process={no_process}')
if health != 'OK' or total != 98304 or used > 8192 or not no_process:
    raise SystemExit('NPU6_IDLE_GATE_BLOCKED: device must be healthy, process-free and HBM <= 8192 MB')
print('NPU6_IDLE_GATE_PASS')
PY
}
echo '===== 3. PHYSICAL DEVICE GATE ====='
idle_gate before
"$VPY" - "$CANN" <<'PY'
import ctypes, os, sys, torch, torch_npu, acl
from kvip.runtime.device import query_device_resources
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '6'
assert torch.npu.device_count() == 1
lib = ctypes.CDLL(sys.argv[1] + '/lib64/libascendcl.so')
f = lib.aclrtGetPhyDevIdByLogicDevId
f.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
f.restype = ctypes.c_int
p = ctypes.c_int32(-1)
rc = f(0, ctypes.byref(p))
print('LOGICAL=0 PHYSICAL=', p.value, 'RC=', rc)
assert rc == 0 and p.value == 6
torch.npu.set_device(0)
soc = acl.get_soc_name()
cores = query_device_resources(sys.argv[1], 0)
print('SOC=', soc, 'CORE_BUDGET=', cores)
assert soc == 'Ascend950DT_9582' and cores['effective_aic'] == 32
print('NPU6_MAPPING_PASS')
PY

echo '===== 4. START ISOLATED CPU SERVICE ====='
PORT=29009
"$VPY" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.bind(('127.0.0.1', int(sys.argv[1])))
print('PORT_FREE')
PY
"$CPU_SERVICE" --listen 127.0.0.1 --port "$PORT" --gather-threads 8 --gather-mode active > "$OUT/cpu_service.log" 2>&1 &
SERVICE_PID=$!
printf '%s\n' "$SERVICE_PID" > "$OUT/service.pid"
"$VPY" - "$SERVICE_PID" "$PORT" <<'PY'
import os, sys, time
from pathlib import Path
pid = int(sys.argv[1])
addr = f'0100007F:{int(sys.argv[2]):04X}'
for _ in range(100):
    os.kill(pid, 0)
    sockets = set()
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            link = os.readlink(fd)
        except FileNotFoundError:
            continue
        if link.startswith('socket:['):
            sockets.add(link[8:-1])
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == addr and fields[3] == '0A' and fields[9] in sockets:
            print('OWNED_LISTENER_PASS', pid, addr)
            sys.exit(0)
    time.sleep(.1)
raise SystemExit('service listener not ready')
PY

echo '===== 5. VARIABLE K FIRST-ROUND PROBE (Q4) ====='
for BS in 4 16; do
    CASE="basic_bs$BS"
    idle_gate "before_$CASE"
    echo "V1_CASE_START $CASE"
    "$VPY" -u "$ENTRY" check --transport urma --device 0 --suite basic --cases "$CASE" \
        --check-varlen-probe --kvaf-root "$NEW_KVAF" --kvaf-server "127.0.0.1:$PORT" \
        --output "$OUT/$CASE" --timeout-seconds 900 2>&1 | tee "$OUT/$CASE.log"
    "$VPY" - "$OUT" "$CASE" "$BS" "$LIFETCH_LIBRARY" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
out,case,bs,frozen=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),Path(sys.argv[4])
summaries=list((out/case).glob('*/summary.json'))
assert len(summaries)==1
rows=json.loads(summaries[0].read_text())['cases']
assert len(rows)==1 and rows[0]['correctness']=='PASS' and rows[0]['execution']=='COMPLETED' and rows[0]['accepted'] is True, rows
worker=summaries[0].parent/'cases'/case
assert json.loads((worker/'worker-result.json').read_text())['exit_code']==0
path=Path(rows[0]['report']).resolve()
assert path.is_relative_to(worker.resolve()) and path.name=='report.json'
r=json.loads(path.read_text())
assert r['correctness']=='PASS' and not r['correctness_failures']
assert r['batch']==bs and r['history']==131072 and r['resident']==16384 and r['index_heads']==32
expected=[32768,65536,98304,131072] if bs==4 else [131072]*8+[32768]*8
assert r['actual_k']==expected and r['actual_q']==[4]*bs
assert r['explicit_generic_streaming'] is True and r['parallel_terminal'] is False
assert r['production_lease_type']=='KvafGetLease' and r['first_round_verified']==3 and r['stateful_rounds']==0
assert r['sfa'] is None
with np.load(path.parent/'v1_topk_evidence.npz') as ev:
    assert ev['actual_k'].tolist()==expected
    for name in ('native','fused'):
        ids=ev[name]
        assert ids.shape==(bs,4,2048)
        assert (ids>=0).all() and (ids<np.asarray(expected)[:,None,None]).all()
lib=(worker/'universal-build'/'libv9_bridge.so').resolve()
assert hashlib.sha256(lib.read_bytes()).hexdigest()==hashlib.sha256(frozen.read_bytes()).hexdigest()
(out/(case+'.evidence.json')).write_text(json.dumps(dict(report=str(path),actual_k=expected,
    binary_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),scope=r['scope']),indent=2)+'\n')
print('V1_RAW_PROBE_PASS',case,json.dumps(dict(actual_k=expected,verified=r['first_round_verified'],
    exact_same_topk=r['exact_same_topk'],checks=r['checks'])))
PY
    npu-smi info > "$OUT/npu_after_$CASE.txt"
done
# On success stop only the child service; on failure preserve it for diagnosis.
kill -TERM "$SERVICE_PID"
wait "$SERVICE_PID" || true
SERVICE_PID=''

echo '===== 6. FINAL PROTECTION GATE ====='
snapshot > "$OUT/original_after.txt"
cmp "$OUT/original_before.txt" "$OUT/original_after.txt"
ps -eo pid,args | grep '[k]vaf_cpu_service.*--port 28999' > "$OUT/old_service_after.txt"
cmp "$OUT/old_service_before.txt" "$OUT/old_service_after.txt"
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
npu-smi info | tee "$OUT/npu_after.txt"
COMPLETE=1
echo 'NEXTGEN_V1_GENERIC_FIRST_ROUND_PASS'
echo "OUTPUT_DIR=$OUT"
