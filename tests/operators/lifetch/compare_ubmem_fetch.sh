#!/usr/bin/env bash
# Same LI/Resolver/Index kernels, inputs and native KVAF MTE pipeline in A and B.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
# Remember explicit connection settings locally; subsequent runs need no arguments.
settings="$repo/tests/results/ubmem-fetch-settings.json"
selection="$(python3 - "$settings" "$repo" "$@" <<'PYSET'
import json, os, sys
from pathlib import Path
path, repo = Path(sys.argv[1]), Path(sys.argv[2])
saved = json.loads(path.read_text()) if path.exists() else {}
args = sys.argv[3:]
if len(args) > 4:
    raise SystemExit('Usage: compare_ubmem_fetch.sh [KVAF_ROOT DEVICE SERVER [BATCHES]]')
values = [args[i] if i < len(args) else saved.get(key, default)
          for i, (key, default) in enumerate((
              ('kvaf_root', os.environ.get('UBMEM_KVAF_ROOT', '')),
              ('device', '4'), ('server', os.environ.get('UBMEM_SERVER', '')),
              ('batches', '1,2,4,8,16')))]
if not values[0] or not values[2]:
    raise SystemExit('First run: bash tests/operators/lifetch/compare_ubmem_fetch.sh KVAF_ROOT DEVICE SERVER; settings are remembered for later runs.')
values[0] = str(Path(values[0]).resolve())
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(dict(zip(('kvaf_root', 'device', 'server', 'batches'), values)), indent=2)+'\n')
print('\n'.join(values))
PYSET
)"
mapfile -t options <<< "$selection"
kvaf_root="${options[0]}"; device="${options[1]}"; server="${options[2]}"; batches="${options[3]}"
set +u
source /usr/local/Ascend/cann/set_env.sh
set -u
run="$repo/tests/results/ubmem-fetch-ab/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
cd "$repo"
git rev-parse HEAD > "$run/commit.txt"
for mode in serial overlap; do
  python tests/operators/lifetch/test_lifetch.py \
    --transport ubmem --optimized-resolve --ubmem-fetch-mode "$mode" \
    --device "$device" --soc Ascend950PR_9599 --kvaf-root "$kvaf_root" \
    --kvaf-server "$server" --batches "$batches" --history 131072 \
    --resident 16384 --index-heads 32 --intermediate-fetches 8 \
    --prefetch-tokens 512 --miss-percent 10 --q-workers 2 \
    --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 1800 \
    --output "$run/$mode" 2>&1 | tee "$run/$mode.log"
done
python - "$run" <<'PY' | tee "$run/summary.log"
import re
import sys
from pathlib import Path
sys.path.insert(0, 'tests/operators/lifetch/support')
from compare_transport_tail import summarize, read_cases
root = Path(sys.argv[1])
data = {}
for mode in ('serial', 'overlap'):
    logs = list((root/mode).rglob('worker.log'))
    if len(logs) != 1:
        raise SystemExit(f'{mode}: expected exactly one worker log, got {len(logs)}')
    rows = {}
    pending = None
    for line in logs[0].read_text().splitlines():
        if line.startswith('UBMEM_GRAPH_EVENT_US'):
            pending = float(re.search(r'p50=([\d.]+)', line)[1])
        if line.startswith('UBMEM_LIFETCH_PASS'):
            batch = int(re.search(r'batch=(\d+)', line)[1])
            if pending is None:
                raise SystemExit(f'{mode}: BS{batch} missing timing')
            rows[batch] = pending
            pending = None
    data[mode] = rows
print('BS | serial full p50 us | overlap full p50 us | reduction')
if data['serial'].keys() != data['overlap'].keys() or not data['serial']:
    raise SystemExit('Incomplete A/B cases')
for batch, before in data['serial'].items():
    after = data['overlap'][batch]
    print(f'{batch} | {before:.3f} | {after:.3f} | {(1-after/before)*100:+.2f}%')
for mode in ('serial', 'overlap'):
    print(f'=== {mode}: separately profiled stages and Q/core timeline ===')
    for batch, (path, samples) in read_cases(root/mode, 'ubmem').items():
        if len(samples) != 3 or any('error' in s['accounting'] or not s.get('streaming_cycles', s.get('streaming')) for s in samples):
            raise SystemExit(f'Incomplete detailed trace: {mode} BS{batch} {path}')
    summarize(root/mode, 'ubmem')
print(f'Full results: {root}')
PY
