# SPDX-License-Identifier: Apache-2.0
"""Compare checked BS16 fixed/dynamic traces by native intervals and source milestones."""

from collections import defaultdict
import csv

from lifetch_cold_trace_comparison import matched_gets, union_us
from lifetch_msprof_timeline import READY_SERVICE_ROUTES
from run_mr25_handoff import save


MATCH_FIELDS = ('fixture_sha256', 'library_sha256', 'geometry', 'miss_percent', 'soc', 'aics',
                'windows', 'requested_windows', 'prefetch_tokens', 'prefetch_rank_start',
                'warmup', 'service_lanes', 'local_counts')


def write_ready_comparison(output, captures):
    base, candidate = (captures[route]['report'] for route in READY_SERVICE_ROUTES)
    mismatches = [field for field in MATCH_FIELDS if base[field] != candidate[field]]
    if mismatches:
        raise ValueError(f'Dynamic trace pair has different fixture or build fields: {mismatches}')
    rows, timeline, scope_rows = {}, [], []
    for tid, route in enumerate(READY_SERVICE_ROUTES, 1):
        capture = captures[route]
        report, intervals = capture['report'], capture['intervals']
        if (report['cold_route'] != route or report['geometry']['batch'] != 16
                or report['windows'] != 4 or report['aics'] != 32
                or report['service_lanes'] != 16 or report['local_counts'] is not False):
            raise ValueError(f'Dynamic trace replay has unexpected geometry: {route}')
        origin = intervals[0]['begin_us']
        scopes = defaultdict(list)
        for event in capture['source']['traceEvents']:
            args = event.get('args', {})
            if event.get('ph') == 'X' and 'phase_id' in args:
                scopes[args['phase_id'], args.get('window_epoch')].append(event)
        resident = scopes[23, None]
        cold_mmad = [event for epoch in (1, 2, 3) for event in scopes[184, epoch]]
        first_w1_doorbell = min((event['ts'] for event in scopes[223, 1]), default=None)
        rows[route] = dict(
            native_stages=intervals,
            hot_to_terminal_device_us=intervals[-1]['end_us'] - origin,
            resident_cold_service_device_us=intervals[-2]['end_us'] - intervals[1]['begin_us'],
            first_cold_mmad_source_us=min((event['ts'] for event in cold_mmad), default=None),
            last_resident_end_source_us=max((event['ts'] + event['dur'] for event in resident), default=None),
            first_w1_doorbell_source_us=first_w1_doorbell,
            prefetch_requests=report['total_prefetch'],
            final_get_requests=report['total_final_get'],
            prefetch_by_window=[sum(row['prefetch_by_window'][w] for row in report['requests']) for w in range(4)],
        )
        if route == 'dynamic':
            last_resident = rows[route]['last_resident_end_source_us']
            rows[route]['service_jobs_before_last_resident'] = sum(
                event['ts'] < last_resident for epoch in range(4) for event in scopes[249, epoch])
        timeline.append(dict(name='thread_name', ph='M', pid=1, tid=tid, args={'name': route}))
        for stage in intervals:
            timeline.append(dict(name=stage['name'], ph='X', pid=1, tid=tid, cat='KVIP native kernel',
                                 ts=stage['begin_us'] - origin, dur=stage['duration_us'], args=stage))
        for (phase, window), events in scopes.items():
            if not events:
                continue
            spans = [(event['ts'], event['ts'] + event['dur']) for event in events]
            last = max(events, key=lambda event: event['ts'] + event['dur'])
            scope_rows.append(dict(route=route, phase_id=phase, window=window, scope=events[0]['name'],
                                   events=len(events), lanes=len({(e['pid'], e['tid']) for e in events}),
                                   first_us=min(begin for begin, _ in spans),
                                   last_us=last['ts'] + last['dur'],
                                   last_core=last['args']['core_type'], last_block=last['args']['block'],
                                   source_union_us=union_us(spans)))

    status = matched_gets(base, candidate)
    result = dict(
        fixture={field: base[field] for field in MATCH_FIELDS}, routes=rows, get_comparison=status,
        raw_hot_to_terminal_delta_us=(rows['dynamic']['hot_to_terminal_device_us']
                                      - rows['static_ready']['hot_to_terminal_device_us']),
        timing='One marked device replay per arm. Native timestamps are separate from calibrated source scopes '
               'and graph Event performance. A transport-workload-matched ratio requires exact Get counts.',
    )
    save(output / 'dynamic_trace_comparison.json', result)
    save(output / 'dynamic_trace_native.json', dict(traceEvents=timeline, displayTimeUnit='ms'))
    with (output / 'dynamic_trace_scopes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scope_rows[0]))
        writer.writeheader()
        writer.writerows(scope_rows)
    return result
