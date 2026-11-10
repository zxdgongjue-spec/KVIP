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
for variant in baseline early early-safe; do
    args=()
    if [[ "$variant" == early* ]]; then args+=(--early-fetch); fi
    if [[ "$variant" == early-safe ]]; then args+=(--avoid-prefetch-slots); fi
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
         'Early experiments submit nonconflicting final reads before Index update; prefetch completion remains after it.']
for variant in ('baseline', 'early', 'early-safe'):
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
    if variant in ('early', 'early-safe'):
        detail = row.get('detail', [])
        if len(detail) != 16:
            raise SystemExit(f'early: per-request device timestamps missing: {traces[0]}')
        lines.append(f"{variant} all requests: safe={sum(x['safe'] for x in detail)} conflict={sum(x['conflict'] for x in detail)} pending_batches={sum(x['pending'] for x in detail)}")
        early_request = max(enumerate(detail), key=lambda x: x[1]['ticks'][2] - x[1]['ticks'][0])
        final_request = max(enumerate(detail), key=lambda x: x[1]['ticks'][7] - x[1]['ticks'][3])
        early_row, final_row = early_request[1], final_request[1]
        et, ft = early_row['ticks'], final_row['ticks']
        early_ticks, final_ticks = et[2] - et[0], ft[7] - ft[3]
        if early_ticks <= 0 or final_ticks <= 0:
            raise SystemExit('early: invalid device timestamps')
        # Clock frequency is device-specific: relative scaling uses the
        # enclosing profiler kernel interval and is explicitly approximate.
        er = account['stages_us']['early_fetch'] / early_ticks
        fr = account['stages_us']['fetch'] / final_ticks
        def metric(label, ticks, factor):
            return f'{label}={ticks}cycles(~{ticks * factor:.2f}us)'
        lines.append(f"early slowest request={early_request[0]} safe={early_row['safe']} "
                     f"conflict={early_row['conflict']} prefetch_pending_batches={early_row['pending']} "
                     + metric('check_and_publish', et[1] - et[0], er) + ' '
                     + metric('safe_get_total', et[2] - et[1], er) + ' '
                     + metric('safe_post', early_row['kvaf_ticks'][0], er) + ' '
                     + metric('safe_get_cq', early_row['kvaf_ticks'][1], er))
        lines.append(f"final slowest request={final_request[0]} safe={final_row['safe']} "
                     f"conflict={final_row['conflict']} prefetch_pending_batches={final_row['pending']} "
                     + metric('update_status', ft[4] - ft[3], fr) + ' '
                     + metric('prefetch_cq', ft[5] - ft[4], fr) + ' '
                     + metric('conflict_get_total', ft[6] - ft[5], fr) + ' '
                     + metric('conflict_post', final_row['kvaf_ticks'][2], fr) + ' '
                     + metric('conflict_get_cq', final_row['kvaf_ticks'][3], fr) + ' '
                     + metric('status_writeback', ft[7] - ft[6], fr))
        lines.append('Microseconds above are estimates calibrated to each enclosing kernel, not a verified device clock; '
                     'KVAF post/CQ values are components of Get total and must not be added twice.')
result = '\n'.join(lines) + '\n'
(root / 'summary.log').write_text(result)
print(result, end='')
print(f"Summary: {root / 'summary.log'}")
PY
