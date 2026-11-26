# SPDX-License-Identifier: Apache-2.0
"""Verify and summarize the BS2 P4/W2 vector workers in one source trace."""

from collections import defaultdict


def analyze_window_q(source, workers_per_request):
    phases = defaultdict(list)
    for event in source['traceEvents']:
        args = event.get('args', {})
        if (event.get('ph') == 'X' and args.get('core_type') in ('AIC', 'AIV')
                and args.get('kernel_id') == 22 and args.get('window_epoch') == 1):
            phases[args['phase_id']].append(event)

    def key(event, field):
        return event['args'][field]

    def end_us(event):
        return event['ts'] + event['dur']

    owners = {key(event, 'service_lane'): event for event in phases[150]}
    if len(phases[150]) != 2 or {b: event['tid'] for b, event in owners.items()} != {0: 8, 1: 9}:
        raise ValueError('W1 requires one original Service owner per request')
    for phase, field in ((152, 'request_row'), (153, 'request_row'), (154, 'request_or_service_lane')):
        for event in phases[phase]:
            owner = owners.get(key(event, field))
            if owner is None or event['tid'] != owner['tid']:
                raise ValueError('Only the W1 request owner may select, prepare or submit WQE')

    width = 4 // workers_per_request
    slices = {
        (key(event, 'request_row'), key(event, 'query')): event for event in phases[253]
    }
    expected = {(b, q) for b in range(2) for q in range(0, 4, width)} if workers_per_request > 1 else set()
    if set(slices) != expected or len(phases[253]) != len(expected):
        raise ValueError('Incomplete or duplicate W1 TopK worker slices')
    for (b, q), event in slices.items():
        worker = q // width
        if event['tid'] != (8 + b if worker == 0 else 2 * worker + b):
            raise ValueError('W1 TopK ran on an unexpected Score AIV')
    for phase in (251, 252):
        if workers_per_request > 1:
            if len(phases[phase]) != 2 or {key(e, 'request_row'): e['tid'] for e in phases[phase]} != {0: 8, 1: 9}:
                raise ValueError('Missing owner snapshot or join in split-Q W1')
        elif phases[phase]:
            raise ValueError('Split-Q owner scopes appeared in the serial control')
    if expected:
        score_end = {event['tid']: end_us(event) for event in phases[22]}
        snapshots = {key(event, 'request_row'): end_us(event) for event in phases[251]}
        joins = {key(event, 'request_row'): end_us(event) for event in phases[252]}
        for (b, _), event in slices.items():
            if event['tid'] not in score_end or event['ts'] < score_end[event['tid']]:
                raise ValueError('W1 TopK worker started before finishing its Score scan')
            if event['ts'] < snapshots[b] or end_us(event) > joins[b]:
                raise ValueError('W1 TopK worker ran outside the owner snapshot and join')
        for phase, field in ((152, 'request_row'), (153, 'request_row'), (154, 'request_or_service_lane')):
            for event in phases[phase]:
                if event['ts'] < joins[key(event, field)]:
                    raise ValueError('W1 request owner started WQE work before the TopK join')

    seen = set()
    topk_by_request = defaultdict(list)
    for event in phases[155]:
        b, q = (key(event, field) for field in ('request_row', 'query'))
        worker = slices[b, q // width * width] if expected else owners[b]
        if (b, q) in seen or event['tid'] != worker['tid']:
            raise ValueError('Window TopK query has duplicate or incorrect ownership')
        seen.add((b, q))
        topk_by_request[b].append(event)

    latest_aic = max(map(end_us, phases[21]))
    latest_aiv = max(map(end_us, phases[22]))
    latest_service = max(map(end_us, owners.values()))
    selected_phases = (150, 151, 152, 153, 154, 155, 243, 244, 251, 252, 253)
    scopes = [
        dict(phase=phase, core=event['tid'], source_start_us=event['ts'],
             source_duration_us=event['dur'], details=event['args'])
        for phase in selected_phases for event in phases[phase]
    ]
    return dict(
        w1_topk_worker_aivs=len(slices) if expected else len(owners),
        w1_service_owner_aivs=len(owners),
        w1_after_last_aiv_score_scan_us=max(0, latest_service - latest_aiv),
        w1_after_last_aic_qk_scan_us=max(0, latest_service - latest_aic),
        w1_max_owner_service_us=max(e['dur'] for e in owners.values()),
        w1_max_owner_topk_join_us=max((e['dur'] for e in phases[252]), default=0),
        w1_max_request_topk_elapsed_us=max(
            (max(map(end_us, spans)) - min(e['ts'] for e in spans) for spans in topk_by_request.values()), default=0),
        w1_per_q_topk_spans=len(seen),
    ), scopes
