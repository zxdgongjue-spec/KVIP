#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: compare_query_workers.sh KVAF_ROOT DEVICE HOST:PORT [BATCHES]}"
device="${2:?Missing device}"
server="${3:?Missing server}"
batches="${4:-16}"
clock_mhz="${5:-}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/lifetch-query-workers-ab/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
trap 'status=$?; if ((status != 0)); then echo "FAIL exit=$status logs=$run" | tee "$run/summary.log"; fi' EXIT
source "$repo/scripts/env.sh"
for variant in baseline q1 q2 q4; do
    args=()
    args+=(--streaming --streaming-fetch --fused-final)
    case "$variant" in q1) args+=(--q-workers 1);; q2) args+=(--q-workers 2);; q4) args+=(--q-workers 4);; esac
    echo "=== $variant; batches=$batches ==="
    python "$repo/tests/operators/lifetch/test_lifetch.py" \
      --transport urma --device "$device" --kvaf-root "$kvaf" --kvaf-server "$server" \
      --batches "$batches" --history 131072 --resident 16384 --index-heads 32 \
      --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
      --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 300 \
      --output "$run/$variant" "${args[@]}" 2>&1 | tee "$run/$variant.log"
done
python3 - "$run" "$clock_mhz" <<'PY'
import json
import statistics
import sys
from pathlib import Path
root = Path(sys.argv[1])
clock_mhz = float(sys.argv[2]) if sys.argv[2] else None
if clock_mhz is not None and clock_mhz <= 0:
    raise SystemExit('clock MHz must be positive')
lines = ['URMA graph A/B: Q4 H131072 R16384 heads32 miss10% prefetch512.',
         'Separate runs; full graph includes LI, resolver, update and KVAF fetch. Verification outside timing.',
         'Streaming breakdown is a separate instrumented replay; counters are raw device cycles, include instrumentation and may overlap.',
         f"GetSystemCycle frequency: {clock_mhz if clock_mhz else 'independently calibrated per q-worker run'} MHz; approximate.",
         'Q workers use two dedicated Jettys and native KVAF BatchGet; mode=3 counters are shared per worker group.',
         'Native Get-only GB/s uses selected miss bytes / Get-only graph event p50; scheduling included.',
         'Block IDs are logical launch indices; physical AIV IDs are not measured.',
         'batch | variant | full_p50_us | init_us | li_us | resolve_us | publish_us | apply_us | resolve_update_us | fetch_us']
for variant in ('baseline', 'q1', 'q2', 'q4'):
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
        effective_clock = clock_mhz or report.get('counter_clock_mhz')
        if effective_clock:
            lines.append(f'  BS={batch} counter_mhz={effective_clock:.2f} (independent graph-event calibration, approximate)')
        val = lambda k: f'{stages[k]:.2f}' if k in stages else '--'
        native_get = report['timing']['native_get_only']['p50_us']
        native_bytes = sum(report['native_misses_per_request']) * 656
        lines.append(f"  native Get-only BS={batch} bytes={native_bytes} graph_event_p50_us={native_get:.2f} effective_GBps={native_bytes / native_get / 1000:.3f}")
        lines.append(f"{batch} | {variant} | {report['timing']['prototype_with_update']['p50_us']:.2f} | "
                     + ' | '.join(val(k) for k in ('init','li','resolve','publish','apply','resolve_update','fetch')))
        if variant in ('baseline', 'q1', 'q2', 'q4'):
            details = [s.get('streaming_cycles') for s in rows]
            if any(not d for d in details):
                raise SystemExit(f'streaming BS={batch}: missing per-request device timestamps: {trace_path}')
            def worst_median(fn):
                return round(statistics.median(max(fn(x) for x in replay) for replay in details))
            li_keys = ('mapping_wait','barrier','load_first','classify_first','wait_second','classify_second','bitmap_merge','publish')
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
            if variant != 'baseline':
                worker_count = int(variant[1:])
                replay = details[-1]
                scope = replay[0].get('prefetch_scope')
                if scope:
                    cycles = scope['interval_cycles']
                    if cycles:
                        rate = (f"{scope['bytes'] * effective_clock / cycles / 1000:.3f} GB/s" if effective_clock
                                else f"{scope['bytes'] / cycles:.4f} * F_GHz GB/s")
                        lines.append(f"  prefetch {json.dumps(scope)} effective={rate}")
                for q in range(4):
                    for lane in range(min(worker_count, int(batch))):
                        group = [x for x in replay if x['batch'] % worker_count == lane]
                        safe = sum(x['final_counts'][q]['safe'] for x in group)
                        conflict = sum(x['final_counts'][q]['conflict'] for x in group)
                        timing = group[0]['fetch']['queries'][q]
                        def rate(count, cycles):
                            if not count or not cycles: return '--'
                            return f'{count*656*effective_clock/cycles/1000:.3f} GB/s' if effective_clock else f'{count*656/cycles:.4f} * F_GHz GB/s'
                        lines.append(f"  group Q{q+1} lane={lane} jettys=2 safe={safe} bytes={safe*656} call_cycles={timing['safe_submit_or_sync']} effective={rate(safe,timing['safe_submit_or_sync'])} conflict={conflict} bytes={conflict*656} call_cycles={timing['conflict_get']} effective={rate(conflict,timing['conflict_get'])}")
                for request in details[-1]:
                    lines.append(f"  request={request['batch']} resolver_block={2*request['batch']} update_block={2*request['batch']+1} q_blocks={[2*int(batch)+q*int(variant[1:])+request['batch']%int(variant[1:]) for q in range(4)]} prefetch_changes={request['prefetch_changes']} q_new_misses={request['q_new_misses']} fetch_cycles={json.dumps(request['fetch'])} producer_cycles={json.dumps(request['producer'])} update_cycles={json.dumps(request['update'])} update_wait_cycles={json.dumps(request['update_wait'])}")
                    if effective_clock:
                        fetch = request['fetch']
                        def convert(value):
                            if isinstance(value, dict):
                                return {k: (v if k.endswith('_groups') or k.endswith('_mode') or k.endswith('_polls') or k.endswith('_observed') else convert(v)) for k, v in value.items()}
                            if isinstance(value, list):
                                return [convert(v) for v in value]
                            return round(value / effective_clock, 2)
                        lines.append(f"  request={request['batch']} fetch_us={json.dumps(convert(fetch))} producer_us={json.dumps(convert(request['producer']))} update_us={json.dumps(convert(request['update']))} update_wait_us={json.dumps(convert(request['update_wait']))}")
            for query in range(4):
                qstage = {key: worst_median(lambda x, k=key, q=query: x['producer'][q][k]) for key in qkeys}
                qli = {key: worst_median(lambda x, k=key, q=query: x['li'][q][k]) for key in li_keys}
                lines.append(f"  cycles BS={batch} Q{query+1}: LI={json.dumps(qli)} resolver={json.dumps(qstage)}")
text = '\n'.join(lines) + '\n'
(root / 'summary.log').write_text(text)
print(text, end='')
print(f'Summary: {root / "summary.log"}')
PY
