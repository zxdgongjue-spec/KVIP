#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize checked small-batch A5 source scopes and final Get queues offline."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

from lifetch_cold_trace_comparison import union_us


QUERIES = 4
SCOPES = {
    23: 'resident_topk', 155: 'window_topk', 160: 'final_topk',
    164: 'final_wqe', 167: 'final_cq', 168: 'prefetch_cq',
}


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def scope_metrics(records, expected):
    by_lane = defaultdict(list)
    for lane, _, begin, end in records:
        by_lane[lane].append((begin, end))
    return dict(
        events=len(records), lanes=len(by_lane), expected_records=expected,
        observed_records=len({record[1] for record in records}),
        max_lane_union_us=max(map(union_us, by_lane.values()), default=None),
    )


def source_scopes(document, batch, windows):
    """Use calibrated source X events; kernel envelopes and msprof are separate."""
    phases = defaultdict(list)
    cq_spans = defaultdict(list)
    for event in document['traceEvents']:
        args = event.get('args') or {}
        phase = args.get('phase_id')
        if event.get('ph') != 'X' or phase not in SCOPES:
            continue
        begin, duration = event['ts'], event['dur']
        request, query = args['request_row'], args['query']
        window = args.get('window_epoch') if phase == 155 else None
        if (args.get('core_type') != 'AIV' or request not in range(batch) or query not in range(QUERIES)
                or (phase == 155 and window not in range(windows)) or (phase == 168 and query != 1)
                or not math.isfinite(begin) or not math.isfinite(duration) or duration < 0):
            raise ValueError(f'Invalid source scope {phase} for request {request}, Q{query}')
        lane = event['pid'], event['tid']
        end = begin + duration
        phases[phase].append((lane, (request, query, window), begin, end))
        if phase in (167, 168):
            cq_spans[phase, request, query].append((begin, end))
    summary = {name: scope_metrics(phases[phase],
                                   batch * (windows * QUERIES if phase == 155 else 1 if phase == 168 else QUERIES))
               for phase, name in SCOPES.items() if phase != 164}
    # Zero-post queues skip WQE generation. The queue ledger supplies its
    # expected coverage once the matching stage report has been decoded.
    summary['final_wqe'] = scope_metrics(phases[164], None)
    summary['window_topk_by_epoch'] = [
        dict(window=window, **scope_metrics(
            [record for record in phases[155] if record[1][2] == window], batch * QUERIES))
        for window in range(windows)]
    return summary, {key: union_us(spans) for key, spans in cq_spans.items()}


def checked_vector(row, field, length, total):
    vector = row.get(field)
    if vector is not None and (not isinstance(vector, list) or len(vector) != length
                               or any(type(count) is not int or count < 0 for count in vector)
                               or sum(vector) != total):
        raise ValueError(f'Request {row["request"]}: inconsistent {field}')
    return vector


def requests_and_queues(report, batch, windows, cq):
    requests = []
    for row in sorted(report['requests'], key=lambda r: r['request']):
        request = row['request']
        early, final = row['prefetch_requests'], row['final_get_requests']
        if any(type(count) is not int or count < 0 for count in (early, final)):
            raise ValueError(f'Request {request}: invalid completed Get counts')
        per_window = checked_vector(row, 'prefetch_by_window', windows, early)
        by_query = checked_vector(row, 'final_get_by_query', QUERIES, final)
        posted = checked_vector(row, 'final_posted_counts', QUERIES, final)
        owners = checked_vector(row, 'final_original_owner_counts', QUERIES, final)
        if by_query is not None and posted is not None and by_query != posted:
            raise ValueError(f'Request {request}: final post counters disagree')
        if posted is None:
            posted = by_query
        requests.append(dict(
            request=request, prefetch_requests=early, final_get_requests=final, prefetch_by_window=per_window,
            final_posted_by_query=posted, final_original_owner_by_query=owners,
            peak_final_queue_gets=max(posted) if posted is not None else None,
            peak_original_owner_gets=max(owners) if owners is not None else None,
            final_cq_us_by_query=[cq.get((167, request, query)) for query in range(QUERIES)],
            prefetch_cq_q1_us=cq.get((168, request, 1)),
        ))
    if ([r['request'] for r in requests] != list(range(batch))
            or sum(r['prefetch_requests'] for r in requests) != report['total_prefetch']
            or sum(r['final_get_requests'] for r in requests) != report['total_final_get']):
        raise ValueError('Request IDs or completed Get totals do not match the report')
    return requests


def arm_files(arm, manifest, batch):
    # Prefer local files so copied A5 trace bundles need no path rewriting.
    reports = list(arm.glob(f'worker/miss*/case-b{batch}/baseline_stage_report.json'))
    if not reports:
        reports = [Path(manifest['stage_report'])]
    if len(reports) != 1:
        raise ValueError(f'{arm}: expected exactly one baseline_stage_report.json')
    report = reports[0]
    for source in (arm / 'lifetch_business_trace.json', report.with_name('baseline_stage_trace.json')):
        if source.is_file():
            return report, source
    return report, Path(manifest['business_trace'])


def load_arm(arm, manifest, batch):
    report_path, source_path = arm_files(arm, manifest, batch)
    report, source = read_json(report_path), read_json(source_path)
    mode = arm.name
    windows = report['windows']
    if (report['correctness'] != 'PASS' or report['execution'] != 'COMPLETED'
            or report['geometry']['batch'] != batch or report.get('small_batch_mode', mode) != mode
            or not 1 <= windows <= 4 or report.get('traced_invocations', 1) != 1):
        raise ValueError(f'{arm}: expected one checked replay with matching route geometry')
    metrics, cq = source_scopes(source, batch, windows)
    requests = requests_and_queues(report, batch, windows, cq)
    posted_vectors_complete = all(row['final_posted_by_query'] is not None for row in requests)
    if posted_vectors_complete:
        metrics['final_wqe']['expected_records'] = sum(
            count > 0 for row in requests for count in row['final_posted_by_query'])
    active_cq = [duration for row in requests if row['final_posted_by_query'] is not None
                 for count, duration in zip(row['final_posted_by_query'], row['final_cq_us_by_query'])
                 if count and duration is not None]
    balanced = None
    if all(row['peak_final_queue_gets'] is not None and row['peak_original_owner_gets'] is not None for row in requests):
        balanced = sum(row['peak_final_queue_gets'] < row['peak_original_owner_gets'] for row in requests)
    histogram_us = next((interval['duration_us'] for interval in manifest.get('kernel_intervals', [])
                         if interval['kernel'] == 'v9_li_kernel_resident_topk_high'), None)
    return dict(
        batch=batch, variant=mode, windows=windows, service_lanes=report.get('service_lanes'),
        fixture_sha256=report.get('fixture_sha256'), library_sha256=report.get('library_sha256'),
        resident_seed_sha256=report.get('resident_seed_sha256'), source_clock=report.get('clock'),
        source=str(source_path), stage_report=str(report_path), scopes=metrics,
        resident_histogram_msprof_us=histogram_us,
        prefetch_requests=report['total_prefetch'], final_get_requests=report['total_final_get'],
        requests_balanced=balanced, active_final_cq_samples=len(active_cq),
        max_active_final_cq_us=max(active_cq, default=None),
        peak_final_queue_gets=max((r['peak_final_queue_gets'] for r in requests
                                   if r['peak_final_queue_gets'] is not None), default=None),
        peak_original_owner_gets=max((r['peak_original_owner_gets'] for r in requests
                                      if r['peak_original_owner_gets'] is not None), default=None),
        requests=requests,
    )


def compare_p2(base, arm):
    hashes = ('fixture_sha256', 'library_sha256')
    if any(not base[field] or base[field] != arm[field] for field in hashes):
        return dict(paired=False, reason='Fixture or library SHA is missing or differs')
    matched = list(zip(base['requests'], arm['requests']))
    early = all(a['prefetch_requests'] == b['prefetch_requests'] for a, b in matched)
    final = all(a['final_get_requests'] == b['final_get_requests'] for a, b in matched)
    total = all(a['prefetch_requests'] + a['final_get_requests'] ==
                b['prefetch_requests'] + b['final_get_requests'] for a, b in matched)
    seed = base['resident_seed_sha256'], arm['resident_seed_sha256']
    return dict(paired=True, same_per_request_prefetch_counts=early,
                same_per_request_final_get_counts=final, same_per_request_total_get_counts=total,
                resident_seed_match=seed[0] == seed[1] if all(seed) else None)


def analyze(directory):
    rows, skipped = [], []
    for manifest_path in sorted(directory.glob('bs*/*/manifest.json')):
        arm = manifest_path.parent
        batch = int(arm.parent.name[2:])
        manifest = read_json(manifest_path)
        if manifest.get('status') != 'PASS':
            skipped.append(dict(batch=batch, variant=arm.name, status=manifest.get('status')))
            continue
        rows.append(load_arm(arm, manifest, batch))
    if not rows:
        raise ValueError(f'{directory}: no successful small-batch trace arms')
    p2 = {row['batch']: row for row in rows if row['variant'] == 'p2'}
    for row in rows:
        if row['variant'] != 'p2' and row['batch'] in p2:
            row['vs_p2'] = compare_p2(p2[row['batch']], row)
    return dict(rows=rows, skipped=skipped, notes=[
        'Source scope durations use the source trace counter calibration. Max-lane union merges intervals on '
        'each logical lane then takes the largest lane total. Scope metrics overlap and are not additive.',
        'Observed / expected coverage counts unique request,Q,window scope keys. A missing window TopK scope '
        'can reflect the native readiness gate. Missing scope duration is null, not zero.',
        'Per-query queue counts are submitted requests, not the maximum number simultaneously in flight. '
        'Phase 167 is per-Q final CQ completion. Phase 168 is the prefetch CQ join on Q1.',
        'The extra Resident histogram kernel, when present, uses the separately labeled msprof device clock. '
        'Its duration is not included in source phase 23. Each arm contains one instrumented replay.',
    ])


def print_csv(rows):
    prefix = ('batch', 'variant', 'windows', 'service_lanes', 'prefetch_requests', 'final_get_requests',
              'peak_final_queue_gets', 'peak_original_owner_gets', 'requests_balanced',
              'active_final_cq_samples', 'max_active_final_cq_us', 'resident_histogram_msprof_us')
    output = []
    for arm in rows:
        row = {field: arm[field] for field in prefix}
        for name in SCOPES.values():
            scope = arm['scopes'][name]
            row[f'{name}_max_lane_us'] = scope['max_lane_union_us']
            expected = scope['expected_records']
            row[f'{name}_coverage'] = f"{scope['observed_records']}/{expected if expected is not None else '?'}"
        output.append(row)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(output[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path, help='run_small_batch_a5.sh --trace output directory')
    parser.add_argument('--format', choices=('json', 'csv'), default='json')
    args = parser.parse_args()
    try:
        result = analyze(args.run_directory.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    if args.format == 'csv':
        print_csv(result['rows'])
    else:
        json.dump(result, sys.stdout, indent=2, allow_nan=False)
        print()


if __name__ == '__main__':
    main()
