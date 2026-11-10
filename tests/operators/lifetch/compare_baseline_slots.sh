#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_baseline_slots.sh KVAF_ROOT DEVICE HOST:PORT}"
device="${2:?Missing device}"
server="${3:?Missing server}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/baseline-slots/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
trap 'status=$?; if ((status != 0)); then echo "FAIL exit=$status logs=$run" | tee "$run/summary.log"; fi' EXIT
source "$repo/scripts/env.sh"
for variant in baseline baseline-safe; do
    args=()
    if [[ "$variant" == baseline-safe ]]; then args+=(--avoid-prefetch-slots); fi
    echo "=== $variant ==="
    python "$repo/tests/operators/lifetch/test_lifetch.py" \
      --transport urma --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
      --batches 16 --history 131072 --resident 16384 --index-heads 32 \
      --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
      --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 300 \
      --output "$run/$variant" ${args[@]+"${args[@]}"} 2>&1 | tee "$run/$variant.log"
done
python3 - "$run" <<'PY'
import json
import statistics
import sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ['BS16 Q4 H131072 R16384 miss10%; same baseline schedule; independent runs.',
         'Counts are measured after each timed replay; device stages are separate profiler replays.']
for version in ('baseline', 'baseline-safe'):
    reports = list((root / version).rglob('report.json'))
    traces = list((root / version).rglob('full_attribution.json'))
    if len(reports) != 1 or len(traces) != 1:
        raise SystemExit(f'{version}: expected one report and trace: {root}')
    report = json.loads(reports[0].read_text())
    rows = json.loads(traces[0].read_text())['samples']
    if any('error' in row['accounting'] for row in rows):
        raise SystemExit(f'{version}: trace attribution failed')
    samples = report['split_samples']['prototype_with_update']
    counts = {key: [min(x[key] for x in samples), max(x[key] for x in samples)]
              for key in ('final_reads', 'nonconflicting_reads', 'conflicting_reads')}
    stages = {key: statistics.median(row['accounting']['stages_us'][key] for row in rows)
              for key in ('init', 'li', 'resolve', 'publish', 'apply', 'fetch')}
    lines.append(f"{version}: graph_p50_us={report['timing']['prototype_with_update']['p50_us']:.3f} "
                 f"stage_median_us={json.dumps(stages)} count_ranges={json.dumps(counts)}")
result='\n'.join(lines)+'\n'
(root/'summary.log').write_text(result)
print(result, end='')
print(f'Summary: {root / "summary.log"}')
PY
