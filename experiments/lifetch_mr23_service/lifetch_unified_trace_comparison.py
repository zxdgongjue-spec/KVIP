# SPDX-License-Identifier: Apache-2.0
"""Compare three checked LIFetch replays without mixing clock domains."""

from collections import defaultdict
import csv

from lifetch_cold_trace_comparison import matched_gets, union_us
from lifetch_msprof_timeline import ROUTE_PHASES
from run_mr25_handoff import save


ROUTES = ('p2', 'full_p2', 'full_uniform')
ROLES = ('score', 'threshold', 'cold_service', 'terminal')
MATCH_FIELDS = ('fixture_sha256', 'library_sha256', 'geometry', 'miss_percent', 'soc', 'aics',
                'windows', 'requested_windows', 'prefetch_tokens', 'prefetch_rank_start',
                'warmup', 'service_lanes', 'local_counts')


def native_stages(route, intervals):
    expected = [kernel for _, kernel, _ in ROUTE_PHASES[route]]
    if [row['kernel'] for row in intervals] != expected:
        raise ValueError(f'{route} did not capture the four expected kernel tasks')
    origin = intervals[0]['begin_us']
    end = origin
    stages = []
    for role, interval in zip(ROLES, intervals, strict=True):
        begin = interval['begin_us']
        if begin < end - 0.001 or interval['end_us'] <= begin:
            raise ValueError(f'{route} has invalid kernel order at {role}')
        stages.append(dict(role=role, name=interval['name'], kernel=interval['kernel'],
                           begin_us=begin - origin, end_us=interval['end_us'] - origin,
                           duration_us=interval['duration_us'], gap_before_us=max(0.0, begin - end)))
        end = interval['end_us']
    return stages


def source_scopes(source, stages):
    """Return event envelopes and interval unions by role, phase and window."""
    roles = {stage['kernel']: stage['role'] for stage in stages}
    groups = defaultdict(list)
    for event in source['traceEvents']:
        args = event.get('args', {})
        if event.get('ph') != 'X' or 'phase_id' not in args:
            continue
        role = roles[args['kernel_entry']]
        window = args.get('window_epoch')
        groups[(role, args['phase_id'], window)].append(event)
        if window is not None:
            groups[(role, args['phase_id'], None)].append(event)

    rows = {}
    for key, events in groups.items():
        lanes = defaultdict(list)
        for event in events:
            lanes[event['pid'], event['tid']].append((event['ts'], event['ts'] + event['dur']))
        spans = [span for per_lane in lanes.values() for span in per_lane]
        last = max(events, key=lambda event: event['ts'] + event['dur'])
        rows[key] = dict(
            scope=events[0]['name'], events=len(events), lanes=len(lanes),
            sampled=all(event['args'].get('sampled_scan_tile', False) for event in events),
            first_us=min(begin for begin, _ in spans), last_us=max(end for _, end in spans),
            last_core=last['args'].get('core_type'), last_block=last['args'].get('block'),
            source_union_us=union_us(spans), max_lane_union_us=max(map(union_us, lanes.values())),
        )
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_scope_rows(scopes):
    keys = set().union(*(arm.keys() for arm in scopes.values()))
    keys = sorted(keys, key=lambda key: (ROLES.index(key[0]), key[1], -1 if key[2] is None else key[2]))
    fields = ('events', 'lanes', 'sampled', 'first_us', 'last_us', 'last_core', 'last_block',
              'source_union_us', 'max_lane_union_us')
    output = []
    for role, phase, window in keys:
        phase_rows = {route: scopes[route].get((role, phase, window), {}) for route in ROUTES}
        labels = {row['scope'] for row in phase_rows.values() if row}
        row = dict(role=role, phase_id=phase, window=window, scope=' / '.join(sorted(labels)))
        for route, metrics in phase_rows.items():
            row.update({f'{route}_{field}': metrics.get(field) for field in fields})
        for route in ROUTES[1:]:
            for field in ('source_union_us', 'max_lane_union_us'):
                left, right = row[f'p2_{field}'], row[f'{route}_{field}']
                row[f'{route}_delta_{field}'] = right - left if left is not None and right is not None else None
        output.append(row)
    return output


def write_unified_comparison(output, captures, local_counts):
    reference = captures['p2']['report']
    routes, scopes, stage_rows, timeline = {}, {}, [], []
    for tid, (route, parts) in enumerate(zip(ROUTES, (2, 2, 4), strict=True), 1):
        capture = captures[route]
        report = capture['report']
        mismatches = [field for field in MATCH_FIELDS if report[field] != reference[field]]
        if mismatches:
            raise ValueError(f'{route} has different fixture or build fields: {mismatches}')
        if (report['cold_route'] != route or report['hot_parts'] != parts or report['cold_parts'] != 2
                or report['local_counts'] is not local_counts
                or report['geometry']['batch'] != 8 or report['aics'] != 32
                or report['windows'] != 4 or report['service_lanes'] != 8):
            raise ValueError(f'Unified trace replay has unexpected geometry: {route}')
        stages = native_stages(route, capture['intervals'])
        scopes[route] = source_scopes(capture['source'], stages)
        if not scopes[route]:
            raise ValueError(f'{route} has no instrumented source intervals')
        doorbell = scopes[route].get(('cold_service', 223, None))
        terminal = scopes[route].get(('terminal', 1, None))
        if terminal is None or (report['total_prefetch'] > 0 and doorbell is None):
            raise ValueError(f'{route} is missing Terminal or observed prefetch doorbell source scopes')
        if doorbell and doorbell['last_us'] > terminal['first_us']:
            raise ValueError(f'{route} has a prefetch doorbell after Terminal begins')
        routes[route] = dict(
            score_parts=parts, cold_parts=2, service_lanes=report['service_lanes'],
            score_to_kv_ready_device_us=stages[-1]['end_us'],
            source_first_doorbell_us=doorbell['first_us'] if doorbell else None,
            source_last_doorbell_us=doorbell['last_us'] if doorbell else None,
            source_first_doorbell_to_terminal_us=terminal['first_us'] - doorbell['first_us'] if doorbell else None,
            prefetch_requests=report['total_prefetch'], final_get_requests=report['total_final_get'],
            prefetch_by_window=[sum(row['prefetch_by_window'][window] for row in report['requests'])
                                for window in range(report['windows'])],
            get_comparison=matched_gets(reference, report),
            native_stages=stages,
        )
        timeline.append(dict(name='thread_name', ph='M', pid=1, tid=tid, args={'name': route}))
        for stage, baseline in zip(stages, routes['p2']['native_stages'], strict=True):
            stage_rows.append(dict(route=route, **stage,
                                   delta_duration_us=stage['duration_us'] - baseline['duration_us'],
                                   delta_gap_before_us=stage['gap_before_us'] - baseline['gap_before_us']))
            timeline.append(dict(name=stage['name'], ph='X', pid=1, tid=tid, cat='KVIP device kernel',
                                 ts=stage['begin_us'], dur=stage['duration_us'], args=stage))
        routes[route]['delta_score_to_kv_ready_device_us'] = (
            routes[route]['score_to_kv_ready_device_us'] - routes['p2']['score_to_kv_ready_device_us'])

    source_rows = paired_scope_rows(scopes)
    report = dict(
        fixture={field: reference[field] for field in MATCH_FIELDS}, routes=routes,
        timing='One checked, instrumented replay per arm. The device span runs from Score task start '
               'to Terminal task end. Graph Event mean is measured by the separate non-trace benchmark.',
        scope_timing='Source times use each arm’s calibrated device counter and its own first scope. '
                     'Native task timing is independent. Nested scopes and parallel cores must not be summed. '
                     'Blank fields are inapplicable or unobserved. First/last tile scopes are samples.',
    )
    save(output / 'unified_trace_comparison.json', report)
    save(output / 'unified_trace_native.json', dict(traceEvents=timeline, displayTimeUnit='ms'))
    write_csv(output / 'unified_trace_stages.csv', stage_rows)
    write_csv(output / 'unified_trace_scopes.csv', source_rows)
    return report
