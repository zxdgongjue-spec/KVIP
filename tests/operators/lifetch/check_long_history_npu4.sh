#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run long-history or same-graph dedup checks against a temporary loopback URMA peer.
set -eo pipefail
work=${1:?usage: bash check_long_history_npu4.sh VALIDATION_WORK [long-history|dedup-replay]}
suite=${2:-long-history}
case "$suite" in long-history|dedup-replay) ;; *) echo "Unknown check suite: $suite" >&2; exit 2 ;; esac
root=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$work"
out=$(mktemp -d "$work/long-history-npu4.XXXXXX")
echo "RESULT_ROOT=$out"
exec > >(tee "$out/run.log") 2>&1
export ASCEND_RT_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="/usr/local/python3.12.13/bin:$PATH"
if [[ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
fi
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
set -u
py=${PYTHON:-$work/peer-baseline/python-da318bdb/bin/python}
if [[ ! -x $py ]]; then py=$(command -v python3); fi
export PYTHON="$py" PATH="${py%/*}:$PATH"
sdk=${KVIP_KVAF_ROOT:-$work/fused-scan-sources/unified-sdk-813f1750-d6cfd392}
mkdir -p "$(dirname "$sdk")"
exec 8>"$sdk.lock"
flock 8
"$py" -B "$root/scripts/build.py" --prepare-only --transport urma --kvaf-root "$sdk"
cache="$work/peer-baseline/KVAF-de4fd37e/build-cpu/CMakeCache.txt"
if [[ -r $cache ]]; then
    headers=$(sed -n 's/^KVAF_URMA_ROOT:[^=]*=//p' "$cache")
    lib=$(sed -n 's/^KVAF_URMA_LIBRARY:[^=]*=//p' "$cache")
    if [[ -f $headers/include/urma_api.h ]]; then export KVAF_UMDK_ROOT="$headers"; fi
    if [[ -f $lib ]]; then export KVAF_URMA_LIBRARY="$lib"; fi
fi
bash "$sdk/build.sh" cpu --jobs "${KVAF_BUILD_JOBS:-2}" >"$out/build-cpu.log" 2>&1 || { tail -30 "$out/build-cpu.log"; exit 1; }
bash "$sdk/build.sh" npu --without-aicpu --jobs "${KVAF_BUILD_JOBS:-2}" >"$out/build-npu.log" 2>&1 || { tail -30 "$out/build-npu.log"; exit 1; }
flock -u 8
export PYTHONPATH="$root/python:$sdk/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$sdk/build-npu/bin/kvaf_execute_batch_put_kernels.o"
cpu="$sdk/build-cpu/runtime/kvaf_cpu_service"
"$py" -B -c 'import torch_npu, kvaf; assert hasattr(kvaf.NpuRuntime, "get_graph_urma_resources")'

ub_device=${KVAF_CPU_UB_DEVICE:-}
if [[ -z $ub_device ]]; then
    # Prefer a measured HCA from a previous run made on this host and physical card.
    ub_device=$("$py" - "$work/peer-baseline" <<'PY_HCA'
import json, socket, sys
from pathlib import Path
area = Path(sys.argv[1])
machine = Path('/etc/machine-id')
roots = (p for prefix in ('unified-main-five', 'main-five', 'li-window-five', 'fused')
         for p in area.glob(prefix + '.*') if p.is_dir())
for root in sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        if (root/'host.txt').read_text().strip() != socket.gethostname() or (root/'physical_npu.txt').read_text().strip() != '4':
            continue
        archived_machine = root/'machine_id.txt'
        if machine.is_file() and archived_machine.is_file() and machine.read_text().strip() != archived_machine.read_text().strip():
            continue
        for path in sorted(root.glob('g*/*/thp.json')):
            info = json.loads(path.read_text())
            if info.get('status') == 'measured' and info.get('ub_device'):
                print(info['ub_device'])
                sys.exit(0)
    except (OSError, ValueError):
        continue
PY_HCA
)
fi
args=()
if [[ -n $ub_device ]]; then args+=(--ub-device "$ub_device"); fi
port=$("$py" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
"$cpu" --listen 127.0.0.1 --port "$port" --memory-page-mode thp "${args[@]}" >"$out/peer.log" 2>&1 &
peer=$!
trap 'kill "$peer" 2>/dev/null || true; wait "$peer" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((i=0; i<30; i++)); do
    if grep -Fq "LISTEN address=127.0.0.1 port=$port" "$out/peer.log"; then break; fi
    kill -0 "$peer" 2>/dev/null || break
    sleep 1
done
grep -Fq "LISTEN address=127.0.0.1 port=$port" "$out/peer.log" || { tail -30 "$out/peer.log"; exit 1; }
echo "KVIP=$(git -C "$root" rev-parse HEAD) KVAF=$(git -C "$sdk" rev-parse HEAD) physical_NPU=4 logical_NPU=0 HCA=${ub_device:-SDK_auto}"
"$py" -B "$root/tests/operators/lifetch/test_lifetch.py" check --transport urma --device 0 \
    --suite "$suite" --kvaf-root "$sdk" --kvaf-server "127.0.0.1:$port" --output "$out/checks" --timeout-seconds 1800
if [[ $suite == dedup-replay ]]; then
    "$py" - "$out/checks" <<'PY_PLAN'
import json, sys
from pathlib import Path
result, = Path(sys.argv[1]).glob('*/cases/dedup_replay/dedup_result.json')
data = json.loads(result.read_text())
if not data['passed']:
    raise SystemExit(f'Failed dedup graph replay: {result}')
for r in data['replays']:
    print(f"PASS max_end={r['max_end']} words={r['words']} mode={r['mode']} TopK/KV/Index/private_CQ=PASS")
PY_PLAN
    exit 0
fi
"$py" - "$out/checks" <<'PY_CHECK'
import json, sys
from pathlib import Path
records = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('*/cases/*/max_history_result.json')]
if sorted(r['history'] for r in records) != [524288, 524544] or not all(r['passed'] for r in records):
    raise SystemExit('Missing successful eager/graph/regression results')
for r in sorted(records, key=lambda r: r['history']):
    print(f"PASS history={r['history']} Q_ends={r['query_ends']} max_token={r['max_token_id'][0]} "
          f"BS={r['batch']} TopK=PASS KV=PASS ordered_Index=PASS private_CQ=PASS "
          f"issued_at_or_above_262144={r['issued_at_or_above_262144']}")
PY_CHECK
