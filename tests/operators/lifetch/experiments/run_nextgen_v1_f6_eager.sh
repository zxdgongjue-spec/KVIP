#!/usr/bin/env bash
# NextGen F6: single BS16 eager using exact metadata-guard candidate.
# Requires F3/F4 PASS, exclusive NPU7 and cleared prior quarantine.
set -Eeuo pipefail
umask 077
[[ $# == 9 ]] || { echo "Usage: $0 URMA_MANIFEST PATCH_EVIDENCE_JSON ORIGINAL_FRAMEWORK_ROOT INIT_EVIDENCE_DIR V3B_EVIDENCE_DIR F2_BUILD_DIR F3_EVIDENCE_DIR F4_EVIDENCE_DIR F6_BUILD_DIR" >&2; exit 2; }
MANIFEST=$(realpath "$1")
PATCH_EVIDENCE=$(realpath "$2")
ORIGINAL_ROOT=$(realpath "$3")
INIT_EVIDENCE=$(realpath "$4")
V3B_EVIDENCE=$(realpath "$5")
F2=$(realpath "$6")
F3=$(realpath "$7")
F4=$(realpath "$8")
F6=$(realpath "$9")
[[ "${CONFIRM_NEXTGEN_QUARANTINE_CLEARED:-0}" == 1 ]] || { echo "F3_BLOCKED: require administrator-confirmed transport quiescence and release of previous test service"; exit 2; }
[[ ! -e /proc/123779 && ! -e /proc/123595 && ! -e /proc/126581 && ! -e /proc/126398 ]] || { echo "F3_BLOCKED: prior worker/service PID still visible"; exit 2; }
unset NEXTGEN_F3_LIBRARY NEXTGEN_F4_REPLAYS NEXTGEN_F5_BS16 NEXTGEN_F5_DIAGNOSTIC NEXTGEN_F6_EAGER
[[ ! -e /proc/129363 && ! -e /proc/129546 && ! -e /proc/129544 ]] || { echo "F6_BLOCKED previous diagnostic worker/service PID still visible"; exit 2; }
[[ ! -e /proc/128765 && ! -e /proc/128579 ]] || { echo "F5_DIAG_BLOCKED previous worker/service PID still visible"; exit 2; }
PHYSICAL_DEVICE="${NEXTGEN_PHYSICAL_DEVICE:-}"
[[ "$PHYSICAL_DEVICE" == 7 ]] || { echo "F3 incident replay requires physical NPU7"; exit 2; }
[[ "${CONFIRM_NEXTGEN_NPU_IDLE:-0}" == 1 ]] || { echo 'Confirm selected physical NPU is healthy, idle and allocated to you first.' >&2; exit 2; }
source "$MANIFEST"
WORK=$(dirname "$VENV")
OUT=$(mktemp -d "$WORK/v1-f6-eager.XXXXXX")
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
printf '%s\n' "$PHYSICAL_DEVICE" > "$OUT/physical_device.txt"
echo "SELECTED_PHYSICAL_DEVICE=$PHYSICAL_DEVICE"

echo '===== 1. FROZEN BUILD GATE ====='
[[ $(cat "$F2/test_commit.txt") == af85cc3b1de72b0a4fc3d92a335f12c2a971c80c ]]
grep -Fxq V1_F2_COMPILE_ONLY_PASS "$F2/run.log"
CANDIDATE=$(realpath "$(cat "$F2/candidate_library.txt")")
[[ "$CANDIDATE" == "$F2/build-cache/21042f06167008a5e836/libv9_bridge.so" ]]
sha256sum --strict -c <<EOF
fc56dc00b505a552cf71356188fc93790e9082cbcf32f7d6e64fab5265c52261  $CANDIDATE
EOF
sha256sum --strict -c "$F2/frozen-sha256.txt"
printf '%s\n' "$F2" > "$OUT/f2_evidence_path.txt"
"$VPY" -u "$(dirname "$SCRIPT")/check_nextgen_v1_f3_evidence.py" "$F3" "$CANDIDATE" "$OUT/f3-evidence-sha256.json"
printf '%s\n' "$F3" > "$OUT/f3_evidence_path.txt"
"$VPY" -u "$(dirname "$SCRIPT")/check_nextgen_v1_f4_evidence.py" "$F4" "$CANDIDATE" "$OUT/f4-evidence-sha256.json"
printf '%s\n' "$F4" > "$OUT/f4_evidence_path.txt"
[[ $(cat "$F6/test_commit.txt") == 225f77f03b798827c2b7e7aa5ddd0318c0538427 ]]
grep -Fxq V1_F6_COMPILE_ONLY_PASS "$F6/run.log"
grep -Fxq 'V1_F6_SINGLE_VARIABLE_SOURCE_PASS csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/incremental_methods.h' "$F6/run.log"
CANDIDATE=$(realpath "$(cat "$F6/candidate_library.txt")")
[[ "$CANDIDATE" == "$F6/build-cache/054e5eb4d7609d29d5f0/libv9_bridge.so" ]]
sha256sum --strict -c <<EOF
cea9c320043a074074808bcb7595edebd0fc1ebbaae1766955d858d8c38b13b4  $CANDIDATE
b83173331923462c25e6fa206c2bbea122c35a5d1a52dd31544531c2e0ecf40e  $F6/f6-metadata-writeback.patch
EOF
sha256sum --strict -c "$F6/frozen-sha256.txt"
git -C "$F6/source" diff --binary HEAD | cmp - "$F6/applied-source.diff"
printf '%s\n' "$F6" > "$OUT/f6_evidence_path.txt"
echo F6_CANDIDATE_BUILD_GATE_PASS
"$VPY" - <<'PY'
import socket
with socket.socket() as s:s.bind(("127.0.0.1",29009))
print("F3_PREVIOUS_TEST_PORT_FREE")
PY
[[ "$KVIP_SHA" == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
[[ "$KVAF_SHA" == 87e261ae4d07d4868cfdbb306d15f3f137edfc3d ]]
[[ $(git -C "$NEW_KVIP" rev-parse HEAD) == "$KVIP_SHA" ]]
[[ $(git -C "$NEW_KVAF" rev-parse HEAD) == "$KVAF_SHA" ]]
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
# Independent experimental sources do not participate in the frozen V1 build.
git -C "$REPO" diff --quiet "$KVIP_SHA" "$COMMIT" -- csrc python \
    ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v2_score_queue/**' \
    ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v3_local_topk/**' \
    ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v3_global_topk/**'
# No modifications to validated V3b device sources are admitted.
git -C "$REPO" diff --quiet dd3f154b2d7d44d3f9a86f960452f7f17a61c00a "$COMMIT" -- csrc python
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
export USER="${USER:-$(id -un)}" ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" PYTHONNOUSERSITE=1
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
grep -Fxq V3B_TOPK_MATRIX_PASS "$V3B_EVIDENCE/run.log"
grep -Fxq dd3f154b2d7d44d3f9a86f960452f7f17a61c00a "$V3B_EVIDENCE/test_commit.txt"
"$VPY" -u "$REPO/tests/operators/lifetch/experiments/check_nextgen_v3b_evidence.py" "$V3B_EVIDENCE" "$OUT/v3b_evidence_sha256.json"
printf '%s\n' "$V3B_EVIDENCE" > "$OUT/v3b_evidence_path.txt"

# Reuse V3b's exact, opt-in NPU7 incident exception. Default denies all
# INVALID rows. Between independent workers, poll for actual context release.
GATE_DIR="$REPO/tests/operators/lifetch/experiments"
"$VPY" -u "$GATE_DIR/test_nextgen_v3b_resource_gate.py"
"$VPY" -u "$GATE_DIR/test_nextgen_v1_f3_recorder.py"
exec 9>"$WORK/v3b-npu${PHYSICAL_DEVICE}.lock"
flock -n 9 || { echo "NEXTGEN_VALIDATION_LOCK_BUSY"; exit 2; }
idle_gate() {
    local label="$1"
    "$VPY" -u "$GATE_DIR/nextgen_v3b_resource_gate.py" \
      --physical "$PHYSICAL_DEVICE" --log "$OUT/npu_$label.txt"
}
echo '===== 3. PHYSICAL DEVICE GATE ====='
idle_gate before
"$VPY" - "$CANN" "$PHYSICAL_DEVICE" <<'PY'
import ctypes, os, sys, torch, torch_npu, acl
from kvip.runtime.device import query_device_resources
physical = int(sys.argv[2])
assert physical in (6, 7)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == str(physical)
assert torch.npu.device_count() == 1
lib = ctypes.CDLL(sys.argv[1] + '/lib64/libascendcl.so')
f = lib.aclrtGetPhyDevIdByLogicDevId
f.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
f.restype = ctypes.c_int
p = ctypes.c_int32(-1)
rc = f(0, ctypes.byref(p))
print('LOGICAL=0 PHYSICAL=', p.value, 'RC=', rc)
assert rc == 0 and p.value == physical
torch.npu.set_device(0)
soc = acl.get_soc_name()
cores = query_device_resources(sys.argv[1], 0)
print('SOC=', soc, 'CORE_BUDGET=', cores)
assert soc == 'Ascend950DT_9582' and cores['effective_aic'] == 32
print(f'NPU{physical}_MAPPING_PASS')
PY

idle_gate after_mapping

echo '===== 4. START ISOLATED CPU SERVICE ====='
PORT=29009
"$VPY" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.bind(('127.0.0.1', int(sys.argv[1])))
print('PORT_FREE')
PY
"$CPU_SERVICE" --listen 127.0.0.1 --port "$PORT" --gather-threads 8 --gather-mode active > "$OUT/cpu_service.log" 2>&1 9>&- &
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

# Earlier M0-C evidence was on card 6. Verify fresh real Put/Get and lease
# lifecycle on the selected physical card before the larger worker.
echo '===== 4B. SELECTED DEVICE REAL TRANSPORT INIT ====='
idle_gate before_selected_init
"$VPY" -u "$ENTRY" check --transport urma --device 0 --suite init \
    --kvaf-root "$NEW_KVAF" --kvaf-server "127.0.0.1:$PORT" \
    --output "$OUT/selected_init" --timeout-seconds 300 2>&1 | tee "$OUT/selected_init.log"
check_summary "$OUT/selected_init" init
echo "SELECTED_DEVICE_INIT_PASS physical=$PHYSICAL_DEVICE"

echo '===== 5. F6 BS16 SINGLE EAGER ====='
idle_gate before_basic_bs16
# Preserve worker failure status and the diagnostic artifact locations.
set +e
NEXTGEN_F3_LIBRARY="$CANDIDATE" NEXTGEN_F4_REPLAYS=0 NEXTGEN_F5_BS16=1 NEXTGEN_F5_DIAGNOSTIC=1 NEXTGEN_F6_EAGER=1 "$VPY" -u "$ENTRY" check --transport urma --device 0 --suite basic --cases basic_bs16 \
    --check-varlen-probe --kvaf-root "$NEW_KVAF" --kvaf-server "127.0.0.1:$PORT" \
    --output "$OUT/basic_bs16" --timeout-seconds 900 2>&1 | tee "$OUT/basic_bs16.log"
worker_rc=$?
set -e
"$VPY" - "$OUT" "$worker_rc" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rc=int(sys.argv[2])
paths=list(root.glob("basic_bs16/*/cases/basic_bs16/case0-b16-prototype-graph/f5_topk_diagnostic.json"))
assert len(paths)==1,("snapshot missing; inspect worker logs",paths)
path=paths[0];raw=path.with_suffix(".npz")
d=json.loads(path.read_text())
assert d["batch"]==16 and len(d["queries"])==64
assert hashlib.sha256(raw.read_bytes()).hexdigest()==d["npz_sha256"]
print("V1_F5_DIAGNOSTIC_EVIDENCE_READY",path,raw,"worker_rc",rc)
print("DIAGNOSTIC_COMPLETION",json.dumps(d["completion"]))
print("DIAGNOSTIC_EVENTS",json.dumps(d["events"]))
for row in d["queries"]:
    if row["invalid"] or row["duplicates"] or row["below_cutoff"] or row["mandatory_missing"]:
        print("FAILING_QUERY",json.dumps(row))
print("DIAGNOSTIC_QUERY_COUNTS",d["failing_queries"],"of",64)
PY
if (( worker_rc != 0 )); then
    echo "F5_DIAGNOSTIC_WORKER_FAILED rc=$worker_rc; original correctness gate preserved; service retained"
    exit "$worker_rc"
fi
"$VPY" -u "$GATE_DIR/check_nextgen_v1_f6_eager.py" "$OUT" "$CANDIDATE"
npu-smi info > "$OUT/npu_after_basic_bs16.txt"
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
sha256sum --strict -c "$F2/candidate-sha256.txt"
sha256sum --strict -c "$F6/candidate-sha256.txt"
echo 'NEXTGEN_V1_F6_EAGER_PASS'
echo "OUTPUT_DIR=$OUT"
