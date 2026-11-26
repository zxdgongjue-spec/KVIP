#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$here/run_small_batch_a5.sh"
phase=${1:---help}
if (( $# )); then shift; fi

case "$phase" in
    --prepare-deps|--build-deps)
        exec bash "$runner" "$phase" "$@"
        ;;
    --sweep)
        exec bash "$runner" --cold-window-matrix "$@"
        ;;
    --retired)
        if (( $# < 2 )) || [[ $1 != --sweep-summary ]]; then
            printf 'Usage: %s --retired --sweep-summary /path/to/summary.csv [timing or trace options]\n' "$0" >&2
            exit 2
        fi
        summary=$2
        shift 2
        head=$(git -C "$here" rev-parse HEAD)
        candidates=$("${KVIP_PYTHON:-python3}" -B - "$summary" "$head" <<'PY'
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys

summary = Path(sys.argv[1])
manifest = json.loads((summary.parent / 'manifest.json').read_text())
if (manifest.get('status') != 'PASS' or not manifest.get('cold_window_matrix')
        or manifest.get('measurement') != 'graph_event' or manifest.get('commit') != sys.argv[2]):
    sys.exit('Select a completed, unmarked P/W sweep from the current commit.')

timings = {}
with summary.open(newline='') as stream:
    for row in csv.DictReader(stream):
        if row['accepted'].lower() != 'true' or float(row['miss_percent']) != 10:
            continue
        key = int(row['batch']), int(row['cold_score_parts']), int(row['windows'])
        value = float(row['kv_ready_us'])
        if math.isfinite(value) and value > 0:
            timings.setdefault(key, []).append(value)

for batch in (1, 2, 8):
    expected = {1: {(2, 2), (4, 2), (8, 2), (32, 2)},
                2: {(p, w) for p in (2, 4, 16) for w in (2, 4)},
                8: {(p, w) for p in (2, 4) for w in (2, 4)}}[batch]
    measured = {key[1:]: values for key, values in timings.items() if key[0] == batch}
    if not expected.issubset(measured):
        sys.exit(f'P/W sweep lacks BS{batch} cells: {sorted(expected - measured.keys())}')
    ranking = sorted((median(measured[p, w]), p, w) for p, w in expected)
    best, best_parts, best_windows = ranking[0]
    duration, parts, windows = next(entry for entry in ranking if entry[1] > 2)
    print(f'BS{batch} fastest P{best_parts}/W{best_windows} {best:.2f} us, '
          f'retired A/B P{parts}/W{windows} {duration:.2f} us', file=sys.stderr)
    print(f'{batch}:{parts}:{windows}')
PY
        )
        pair_args=()
        while IFS= read -r candidate; do
            pair_args+=(--retired-candidate "$candidate")
        done <<< "$candidates"
        exec bash "$runner" "${pair_args[@]}" "$@"
        ;;
    --service-probes)
        bash "$runner" --window-probe bs1-p8 "$@"
        exec bash "$runner" --window-probe bs2-w3 "$@"
        ;;
    --dry-run)
        bash "$runner" --cold-window-matrix --dry-run "$@"
        bash "$runner" --retired-candidate 1:8:2 --retired-candidate 2:16:2 \
            --retired-candidate 8:4:4 --dry-run "$@"
        bash "$runner" --window-probe bs1-p8 --dry-run "$@"
        exec bash "$runner" --window-probe bs2-w3 --dry-run "$@"
        ;;
    --help|-h)
        cat <<'HELP'
Usage: run_unified_a5.sh PHASE [runner options]
  --prepare-deps, --build-deps
  --sweep
  --retired --sweep-summary /path/to/summary.csv
  --service-probes
  --dry-run
HELP
        ;;
    *)
        printf 'Unknown phase: %s\n' "$phase" >&2
        exit 2
        ;;
esac
