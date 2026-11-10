#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_streaming.sh KVAF_ROOT DEVICE HOST:PORT [BATCHES]}"
device="${2:?Missing device}"
server="${3:?Missing server}"
batches="${4:-1,2,4,8,16}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/lifetch-streaming-ab/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
trap 'status=$?; if ((status != 0)); then echo "FAIL exit=$status logs=$run" | tee "$run/summary.log"; fi' EXIT
source "$repo/scripts/env.sh"
for variant in baseline-safe streaming; do
    args=()
    if [[ "$variant" == baseline-safe ]]; then args+=(--avoid-prefetch-slots); else args+=(--streaming); fi
    echo "=== $variant; batches=$batches ==="
    python "$repo/tests/operators/lifetch/test_lifetch.py" \
      --transport urma --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
      --batches "$batches" --history 131072 --resident 16384 --index-heads 32 \
      --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
      --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 1800 \
      --output "$run/$variant" "${args[@]}" 2>&1 | tee "$run/$variant.log"
done
python3 - "$run" <<'PY'
import json
import statistics
import sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ['URMA graph A/B: Q4 H131072 R16384 heads32 miss10% prefetch512.',
         'Separate runs; full graph includes LI, resolver, update and KVAF fetch. Verification outside timing.',
         'Streaming breakdown is a separate instrumented replay; counters are raw device cycles, include instrumentation and may overlap.',
         'The streaming resolve_update kernel fuses resolver and index update; it has no separate device boundary.',
         'batch | variant | full_p50_us | init_us | li_us | resolve_us | publish_us | apply_us | resolve_update_us | fetch_us']
for variant in ('baseline-safe', 'streaming'):
    reports = list((root / variant).rglob('report.json'))
    if not reports:
        raise SystemExit(f'{variant}: missing reports')
    for report_path in sorted(reports, key=lambda p: int(p.parent.name.split('-')[1].lstrip('b'))):
        report = json.loads(report_path.read_text())
        trace_path = report_path.with_name('full_attribution.json')
        rows = json.loads(trace_path.read_text())['samples']
        if any('error' in row['accounting'] for row in rows):
            raise SystemExit(f'{variant}: failed profiler attribution: {trace_path}')
        stages = {key: statistics.median(row['accounting']['stages_us'][key] for row in rows)
                  for key in rows[0]['accounting']['stages_us']}
        batch = report_path.parent.name.split('-')[1].lstrip('b')
        val = lambda k: f'{stages[k]:.2f}' if k in stages else '--'
        lines.append(f"{batch} | {variant} | {report['timing']['prototype_with_update']['p50_us']:.2f} | "
                     + ' | '.join(val(k) for k in ('init','li','resolve','publish','apply','resolve_update','fetch')))
        if variant == 'streaming':
            details = [s.get('streaming_cycles') for s in rows]
            if any(not d for d in details):
                raise SystemExit(f'streaming BS={batch}: missing per-request device timestamps: {trace_path}')
            def worst_median(fn):
                return round(statistics.median(max(fn(x) for x in replay) for replay in details))
            li_keys = ('mapping_wait','barrier','load_first','classify_first','wait_second','classify_second','publish')
            li = {k: worst_median(lambda x, k=k: max(q[k] for q in x['li'])) for k in li_keys}
            qkeys = ('resolve', 'output', 'pack', 'publish')
            producer = {k: worst_median(lambda x, k=k: sum(q[k] for q in x['producer'])) for k in qkeys}
            producer['input_and_bitmap'] = worst_median(lambda x: x['input_and_bitmap'])
            producer['initialize'] = worst_median(lambda x: x['initialize'])
            updater = {f'phase{i}': worst_median(lambda x, i=i: x['update'][i]) for i in range(5)}
            updater.update({f'wait_before_phase{i}': worst_median(lambda x, i=i: x['update_wait'][i]) for i in range(5)})
            updater['metadata'] = worst_median(lambda x: x['metadata'])
            overlap = sum(int(q['overlap']) for x in details[-1] for q in x['li'])
            topk_only = sum(int(q['topk_only']) for x in details[-1] for q in x['li'])
            lines.append(f"  cycles BS={batch} LI_max_query={json.dumps(li)} metadata_overlap={overlap}/{len(details[-1])*4} topk_only={topk_only}/{len(details[-1])*4}")
            lines.append(f"  cycles BS={batch} producer_max_request={json.dumps(producer)} updater_max_request={json.dumps(updater)}")
text = '\n'.join(lines) + '\n'
(root / 'summary.log').write_text(text)
print(text, end='')
print(f'Summary: {root / "summary.log"}')
PY
