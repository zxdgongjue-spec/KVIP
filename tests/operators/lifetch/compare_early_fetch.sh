#!/usr/bin/env bash
# Compare frozen PR11 against the isolated early-fetch experiment.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: bash tests/operators/lifetch/compare_early_fetch.sh KVAF_ROOT DEVICE HOST:PORT}"
device="${2:?Missing device ID}"
server="${3:?Missing KVAF CPU service address}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/early-fetch-ab/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
trap 'status=$?; if (( status != 0 )); then printf "FAIL: exit=%s; logs=%s\n" "$status" "$run" | tee "$run/summary.log"; fi' EXIT
source "$repo/scripts/env.sh"
for variant in baseline early; do
    args=()
    if [[ "$variant" == early ]]; then args+=(--early-fetch); fi
    echo "=== $variant ==="
    python "$repo/tests/operators/lifetch/test_lifetch.py" \
        --transport urma --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
        --batches 16 --history 131072 --resident 16384 --index-heads 32 \
        --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
        --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 300 \
        --output "$run/$variant" ${args[@]+"${args[@]}"} 2>&1 | tee "$run/$variant.log"
done
python - "$run" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = ['BS16 Q4 H131072 R16384 miss10%; independent graph-event p50 and 3 profiled replays.',
         'Early experiment moves all transport completion before Index update; not a stage-only speedup.']
for variant in ('baseline', 'early'):
    text = (root / f'{variant}.log').read_text()
    traces = list((root / variant).rglob('full_attribution.json'))
    match = re.search(r'prototype_with_update:([\d.]+)', text)
    if 'PASS' not in text or not match or len(traces) != 1:
        raise SystemExit(f'{variant}: checked summary or full trace missing: {root}')
    rows = json.loads(traces[0].read_text())['samples']
    if any('error' in row['accounting'] for row in rows):
        raise SystemExit(f'{variant}: trace attribution failed: {traces[0]}')
    rows.sort(key=lambda row: row['us'])
    row = rows[len(rows) // 2]
    account = row['accounting']
    stages = ' '.join(f'{name}={value:.3f}' for name, value in account['stages_us'].items())
    lines.append(f"{variant}: graph_p50={float(match[1]):.3f} trace={row['us']:.3f} {stages} "
                 f"gap={account['inter_kernel_gap_us']:.3f} boundary={account['boundary_residual_us']:.3f} "
                 f"prefetch={row['prefetch']} final={row['final']}")
result = '\n'.join(lines) + '\n'
(root / 'summary.log').write_text(result)
print(result, end='')
print(f"Summary: {root / 'summary.log'}")
PY
