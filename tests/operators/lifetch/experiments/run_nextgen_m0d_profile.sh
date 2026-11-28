#!/usr/bin/env bash
# NextGen M0-D2a: fixed-fixture kernel timelines and long-tail evidence.
# Usage: CONFIRM_NPU6_IDLE=1 bash this-script URMA_MANIFEST PATCH_EVIDENCE_JSON ORIGINAL_FRAMEWORK_ROOT INIT_EVIDENCE_DIR BASIC_EVIDENCE_DIR PERF_EVIDENCE_DIR
set -Eeuo pipefail
umask 077
[[ $# == 6 ]] || { echo "Usage: $0 URMA_MANIFEST PATCH_EVIDENCE_JSON ORIGINAL_FRAMEWORK_ROOT INIT_EVIDENCE_DIR BASIC_EVIDENCE_DIR PERF_EVIDENCE_DIR" >&2; exit 2; }
MANIFEST=$(realpath "$1")
PATCH_EVIDENCE=$(realpath "$2")
ORIGINAL_ROOT=$(realpath "$3")
INIT_EVIDENCE=$(realpath "$4")
BASIC_EVIDENCE=$(realpath "$5")
PERF_EVIDENCE=$(realpath "$6")
[[ "${CONFIRM_NPU6_IDLE:-0}" == 1 ]] || { echo 'Confirm physical NPU 6 is healthy, idle and allocated to you first.' >&2; exit 2; }
source "$MANIFEST"
WORK=$(dirname "$VENV")
OUT=$(mktemp -d "$WORK/m0d-profile.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
SERVICE_PID=''
COMPLETE=0
trap 'rc=$?; if (( rc != 0 || COMPLETE != 1 )); then echo "M0C_FAILED rc=$rc OUT=$OUT SERVICE_PID=${SERVICE_PID:-not_started}"; if (( rc == 0 )); then exit 99; fi; fi' EXIT
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
grep -Fxq M0C_BASIC_3_CASES_PASS "$BASIC_EVIDENCE/run.log"
grep -Fxq 95e1aa38556b2a90ba6f413e0fde64a5877fc750 "$BASIC_EVIDENCE/test_commit.txt"
cmp "$MANIFEST" "$BASIC_EVIDENCE/urma_manifest.env"
for BS in 1 4 16; do
    test -s "$BASIC_EVIDENCE/basic_bs$BS.evidence.json"
done
printf '%s\n' "$BASIC_EVIDENCE" > "$OUT/basic_evidence_path.txt"
echo 'PREVIOUS_BASIC_EVIDENCE_PASS'
grep -Fxq M0D_FIXED_PERF_6_CASES_PASS "$PERF_EVIDENCE/run.log"
grep -Fxq adca9d38f153ac196b6eeb9000ff23a5d2b6abd3 "$PERF_EVIDENCE/test_commit.txt"
cmp "$MANIFEST" "$PERF_EVIDENCE/urma_manifest.env"
"$VPY" - "$PERF_EVIDENCE" "$OUT" <<'PY'
import json, sys
from pathlib import Path
old, out = map(Path, sys.argv[1:])
evidence = []
for rnd in (1, 2):
    for bs in (1, 4, 16):
        ev = json.loads((old/f'round{rnd}'/f'basic_bs{bs}.evidence.json').read_text())
        report = json.loads(Path(ev['report']).read_text())
        counts = report['fused_read_counts']
        samples = report['timing']['prototype_with_update']['samples_us']
        assert len(samples) == len(counts) == 50
        for i, us in sorted(enumerate(samples), key=lambda x: x[1], reverse=True)[:5]:
            row = dict(round=rnd, batch=bs, sample=i, event_us=us, reads=counts[i])
            evidence.append(row)
            print('PREVIOUS_TOP5', json.dumps(row))
(out/'previous_top5.json').write_text(json.dumps(evidence, indent=2)+'\n')
PY
echo 'PREVIOUS_PERF_EVIDENCE_PASS'

echo '===== 3. PHYSICAL DEVICE GATE ====='
npu-smi info | tee "$OUT/npu_before.txt"
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

echo '===== 5. FIXED-FIXTURE RAW KERNEL PROFILE ====='
ROUND_OUT="$OUT/cases"
mkdir -p "$ROUND_OUT"
for BS in 4 16; do
    CASE="basic_bs$BS"
    echo "CASE_START $CASE"
    "$VPY" -u "$ENTRY" check --transport urma --device 0 --suite basic --cases "$CASE" \
        --kvaf-root "$NEW_KVAF" --kvaf-server "127.0.0.1:$PORT" \
        --check-warmup 10 --check-iterations 50 --check-profile-rounds 5 \
        --output "$ROUND_OUT/$CASE" --timeout-seconds 1800 2>&1 | tee "$ROUND_OUT/$CASE.log"
    "$VPY" - "$ROUND_OUT" "$CASE" "$BS" "$LIFETCH_LIBRARY" "$BASIC_EVIDENCE" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
out, case, bs, frozen = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
summaries = list((out/case).glob('*/summary.json'))
assert len(summaries) == 1, summaries
run = summaries[0].parent
rows = json.loads(summaries[0].read_text())['cases']
assert len(rows) == 1, rows
row = rows[0]
assert row['case'].startswith(case + '/'), row
assert row['correctness'] == 'PASS' and row['execution'] == 'COMPLETED' and row['accepted'] is True, row
assert row['batch'] == bs and row['transport'] == 'urma' and row['miss_percent'] == 10, row
assert row['history'] == 131072 and row['resident'] == 16384, row
worker = run/'cases'/case
assert json.loads((worker/'worker-result.json').read_text())['exit_code'] == 0
report_path = Path(row['report']).resolve()
assert report_path.is_relative_to(worker.resolve()) and report_path.name == 'report.json'
report = json.loads(report_path.read_text())
assert report['correctness'] == 'PASS' and report['batch'] == bs
assert report['history'] == 131072 and report['resident'] == 16384 and report['index_heads'] == 32
assert report['stateful_rounds'] == 3, report['stateful_rounds']
assert report['production_lease_type'] == 'KvafGetLease', report['production_lease_type']
assert report['parallel_terminal'] is True, report['parallel_terminal']
assert report['read_accounting'].startswith('completed_row_and_cq_totals')
assert report['boundary_coverage']['hot-victim-collision_assignments']['former_selected_hot'] > 0
for phase in ('regular', 'cold-start', 'hot-victim-collision'):
    assignment = report['boundary_coverage'][phase + '_assignments']
    assert assignment['assignment_source'] == 'untimed_terminal_diagnostic_before_retirement'
    assert assignment['read_counts']['total'] == assignment['hot_changes'] + assignment['cold_changes']
    assert (report_path.parent/(phase + '_terminal_assignments.npz')).is_file()
    print('TERMINAL_ASSIGNMENT_EVIDENCE', phase, assignment)
assert report['read_samples']['prototype_with_update']
for sample in report['read_samples']['prototype_with_update']:
    assert sample['total'] == sample['prefetch'] + sample['final']
assert not report['correctness_failures'], report['correctness_failures']
baseline = json.loads((Path(sys.argv[5])/(case+'.evidence.json')).read_text())
old = json.loads(Path(baseline['report']).read_text())
for key in ('batch', 'history', 'resident', 'index_heads', 'seed', 'fixture', 'native_misses_per_request', 'resource_geometry', 'build_options'):
    assert report[key] == old[key], (key, report[key], old[key])
protocol = report['sampling_protocol']
assert protocol['warmup'] == 10 and protocol['iterations'] == 50
assert len(report['fused_read_counts']) == 50
print('SAME_FIXED_FIXTURE_PASS', case)
assert report['resources']['effective_aic'] == 32
for name in ('native_chain', 'prototype_with_update'):
    timing = report['timing'][name]
    assert timing['samples'] == 50 and len(timing['samples_us']) == 50
    assert all(math.isfinite(x) and x > 0 for x in timing['samples_us'])
    print('PROFILE_RUN_UNINSTRUMENTED_TIMING', case, name, json.dumps({k: v for k, v in timing.items() if k != 'samples_us'}))
    replays = json.loads((report_path.parent/(name+'_replay_records.json')).read_text())
    assert len(replays) == 50 and [x['sample'] for x in replays] == list(range(50))
    assert [x['event_us'] for x in replays] == timing['samples_us']
    for x in sorted(replays, key=lambda x: x['event_us'], reverse=True)[:5]:
        print('NEW_TOP5', case, name, json.dumps(x))
    cap = report['fixed_profile'][name]
    assert cap['profile_rounds'] == 5
    for key in ('trace', 'samples', 'timeline'):
        path = Path(cap[key])
        assert path.is_file() and path.resolve().is_relative_to(report_path.parent.resolve())
    timelines = json.loads(Path(cap['timeline']).read_text())
    assert len(timelines) == 5
    print('PROFILE_EVIDENCE_CAPTURED', case, name, json.dumps(cap))
    for x in timelines:
        print('KERNEL_TIMELINE', case, name, x['sample'], json.dumps(x['timeline']))
    if cap['kernel_scopes_found'] != 5:
        print('ATTRIBUTION_INCOMPLETE', case, name, cap['kernel_scopes_found'])
library = (worker/'universal-build'/'libv9_bridge.so').resolve()
assert library.is_file(), library
digest = hashlib.sha256(library.read_bytes()).hexdigest()
frozen_digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
assert digest == frozen_digest, (library, digest, frozen_digest)
record = dict(case=case, batch=bs, summary=str(summaries[0]), report=str(report_path),
              library=str(library), binary_sha256=digest, frozen_sha256=frozen_digest,
              same_binary_as_init=digest == frozen_digest, stateful_rounds=3,
              production_lease_type=report['production_lease_type'], parallel_terminal=report['parallel_terminal'],
              resources=report['resources'], resource_geometry=report['resource_geometry'],
              final_workers=report['final_workers'], build_options=report['build_options'])
for prev in out.glob('basic_bs*.evidence.json'):
    p = json.loads(prev.read_text())
    assert p['binary_sha256'] == digest, (p, record)
(out/(case+'.evidence.json')).write_text(json.dumps(record, indent=2)+'\n')
print('BINARY_SHA256', digest, 'SAME_AS_INIT', record['same_binary_as_init'])
print('RESOURCE_GEOMETRY', record['resource_geometry'], 'FINAL_WORKERS', record['final_workers'])
print('RAW_BASIC_PASS', case, report_path)
PY
    echo "CASE_PASS $CASE"
    npu-smi info > "$ROUND_OUT/npu_after_$CASE.txt"
done
"$VPY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for bs in (4, 16):
    ev = json.loads((out/'cases'/f'basic_bs{bs}.evidence.json').read_text())
    r = json.loads(Path(ev['report']).read_text())
    row = dict(batch=bs, fixed_profile=r['fixed_profile'], report=ev['report'])
    rows.append(row)
    print('FIXED_PROFILE_SUMMARY', json.dumps(row))
(out/'fixed_profile_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
PY
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
echo 'M0D_PROFILE_CAPTURE_2_CASES_PASS'
echo "OUTPUT_DIR=$OUT"
