# SPDX-License-Identifier: Apache-2.0
"""Compare source scopes from checked BS8 Cold Score replays."""

from collections import defaultdict
import csv
import math
from statistics import median

from run_mr25_handoff import save
from run_mr25_score import compare_reads


ROUTES = ('p2', 'cold_p4')
SCOPES = {
    218: 'partitioned_half_wait',
    151: 'predecessor_wait',
    21: 'cold_aic_score_scan',
    22: 'cold_aiv_score_scan',
    191: 'cold_score_vf_sample',
    192: 'score_bitmap_progress_mte3_sample',
    193: 'mte3_output_bank_wait_sample',
    155: 'window_topk',
    201: 'window_topk_vf',
    152: 'service_candidates_and_dedup',
    203: 'service_dedup_vf_issue',
    153: 'prefetch_descriptor_prepare',
    154: 'prefetch_wqe_submit',
    205: 'prefetch_sq_mte3',
}
SAMPLED = {191, 192, 193}
REFERENCE = dict(
    cold_service_us={'p2': 186.97, 'cold_p4': 205.55}, delta_us=18.58,
    total_get_per_call={'p2': 1800.0, 'cold_p4': 1874.6},
    timing='historical uninstrumented graph benchmark; local_counts ON; different per-request Get totals',
)


def union_us(intervals):
    end = -math.inf
    duration = 0.0
    for start, stop in sorted(intervals):
        if stop > end:
            duration += stop - max(start, end)
            end = stop
    return duration


def summarize_scopes(document, cold_kernel, windows):
    scope_events = defaultdict(list)
    for event in document['traceEvents']:
        args = event.get('args', {})
        phase = args.get('phase_id')
        if event.get('ph') != 'X' or phase not in SCOPES or args.get('kernel_entry') != cold_kernel:
            continue
        window = args.get('window_epoch')
        start, duration = event['ts'], event['dur']
        if window not in range(windows) or not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f'Invalid Cold source scope {event}')
        record = ((event['pid'], event['tid']), start, start + duration)
        scope_events[(None, phase)].append(record)
        scope_events[(window, phase)].append(record)

    rows = []
    for window in (None, *range(windows)):
        for phase, name in SCOPES.items():
            records = scope_events.get((window, phase), ())
            lanes = defaultdict(list)
            for lane, start, stop in records:
                lanes[lane].append((start, stop))
            durations = [stop - start for _, start, stop in records]
            rows.append(dict(
                window=window, phase_id=phase, scope=name, sampled=phase in SAMPLED,
                events=len(records), lanes=len(lanes),
                event_median_us=median(durations) if durations else None,
                max_event_us=max(durations, default=0.0),
                source_union_us=union_us((start, stop) for _, start, stop in records),
                max_lane_union_us=max((union_us(spans) for spans in lanes.values()), default=0.0),
            ))
    return rows


def matched_gets(base, variant):
    def vector(report):
        return {row['request']: (row['prefetch_by_window'], row['final_get_by_query'])
                for row in report['requests']}
    kind, changed = compare_reads(vector(base), vector(variant))
    return dict(status=kind, changed_requests=changed)


def write_comparison(output, captures, local_counts):
    """Keep native-wall, source-scope and Get-placement comparisons separate."""
    base, variant = (captures[route]['report'] for route in ROUTES)
    same = ('fixture_sha256', 'library_sha256', 'geometry', 'miss_percent', 'soc', 'aics', 'windows',
            'requested_windows', 'prefetch_tokens', 'prefetch_rank_start', 'warmup', 'service_lanes')
    different = [key for key in same if base[key] != variant[key]]
    if different:
        raise ValueError(f'Cold trace pair has different fixture or build fields: {different}')
    route_rows, scopes = {}, {}
    for route, parts in zip(ROUTES, (2, 4), strict=True):
        capture = captures[route]
        report, intervals = capture['report'], capture['intervals']
        if (report['cold_route'] != route or report['cold_parts'] != parts or report['hot_parts'] != 2
                or report['local_counts'] is not local_counts):
            raise ValueError(f'Cold trace route mismatch: {route}')
        phases = {row['name']: row for row in intervals}
        cold = phases['Cold Score + rotating Service']
        route_rows[route] = dict(
            cold_parts=parts, cold_mix_groups=parts * report['geometry']['batch'],
            service_lanes=report['service_lanes'],
            native_kernel_us={row['name']: row['duration_us'] for row in intervals},
            native_fused_kernel_span_us=phases['Terminal']['end_us'] - phases['Hot Score']['begin_us'],
            prefetch_requests=report['total_prefetch'], final_get_requests=report['total_final_get'],
            total_get_requests=report['total_prefetch'] + report['total_final_get'],
            prefetch_by_window=[sum(row['prefetch_by_window'][epoch] for row in report['requests'])
                                for epoch in range(report['windows'])],
        )
        scopes[route] = summarize_scopes(capture['source'], cold['kernel'], report['windows'])

    scope_comparison = []
    for primary, candidate in zip(*(scopes[route] for route in ROUTES), strict=True):
        row = {key: primary[key] for key in ('window', 'phase_id', 'scope', 'sampled')}
        for metric in ('events', 'lanes', 'event_median_us', 'max_event_us', 'source_union_us', 'max_lane_union_us'):
            left, right = primary[metric], candidate[metric]
            row.update({f'p2_{metric}': left, f'cold_p4_{metric}': right,
                        f'delta_{metric}': right - left if left is not None and right is not None else None})
        scope_comparison.append(row)
    cold_delta = (route_rows['cold_p4']['native_kernel_us']['Cold Score + rotating Service']
                  - route_rows['p2']['native_kernel_us']['Cold Score + rotating Service'])
    result = dict(
        historical_reference=REFERENCE,
        local_counts=local_counts,
        routes=route_rows,
        cold_service_native_delta_us=cold_delta,
        get_comparison=matched_gets(base, variant),
        interpretation=('One instrumented eager replay per route. Native kernel clocks and calibrated device-source '
                        'scopes are separate. Scope unions include overlapping/nested work and cannot be summed. '
                        'MTE3/VF scan samples are only the first/last tile on each producer; source timings '
                        'reflect issue/existing fence boundaries, with no PMU busy-time claim.'),
        scopes=scope_comparison,
    )
    save(output / 'cold_trace_comparison.json', result)
    with (output / 'cold_trace_scopes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scope_comparison[0])
        writer.writeheader()
        writer.writerows(scope_comparison)
    return result
