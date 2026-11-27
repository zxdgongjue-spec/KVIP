# SPDX-License-Identifier: Apache-2.0
"""Verify and summarize the W2 Top2048 vector workers in one source trace."""

from collections import defaultdict


def _p2_w0_aiv(request):
    return 4 * (request // 2) + request % 2


def _owner_aiv(batch, request):
    if batch == 16:
        return 32 + _p2_w0_aiv(request)
    if batch == 1:
        return 4 + request
    return 8 * (batch // 2 + request // 2) + request % 2


def _worker_aiv(batch, request, worker):
    if worker == 0:
        return _owner_aiv(batch, request)
    if batch == 16:
        previous = _p2_w0_aiv(request)
        return (previous + 2, previous + 34, previous)[worker - 1]
    if batch == 1:
        return (2, 3, 6)[worker - 1]
    return 2 * (4 * (request // 2) + worker) + request % 2


def analyze_window_q(source, workers_per_request, batch=2):
    if batch not in (1, 2, 4, 8, 16) or workers_per_request not in (1, 2, 4):
        raise ValueError('Unsupported W2 window TopK geometry')
    phases = defaultdict(list)
    previous_service = []
    for event in source['traceEvents']:
        args = event.get('args', {})
        if (event.get('ph') == 'X' and args.get('core_type') in ('AIC', 'AIV')
                and args.get('kernel_id') == 22):
            if args.get('window_epoch') == 1:
                phases[args['phase_id']].append(event)
            elif batch == 16 and args.get('window_epoch') == 0 and args['phase_id'] == 150:
                previous_service.append(event)

    def key(event, field):
        return event['args'][field]

    def end_us(event):
        return event['ts'] + event['dur']

    service_lanes = {key(event, 'service_lane'): event for event in phases[150]}
    # BS1 has one inactive request lane; its owner still publishes the original Service event.
    expected_lanes = {lane: _owner_aiv(batch, lane) for lane in range(max(2, batch))}
    if (len(phases[150]) != len(expected_lanes)
            or {lane: event['tid'] for lane, event in service_lanes.items()} != expected_lanes):
        raise ValueError('W1 requires one original Service owner per request')
    owners = {row: service_lanes[row] for row in range(batch)}
    previous_owners = {key(event, 'service_lane'): event for event in previous_service}
    if batch == 16 and (
            len(previous_service) != batch
            or {row: event['tid'] for row, event in previous_owners.items()}
            != {row: _p2_w0_aiv(row) for row in range(batch)}):
        raise ValueError('BS16 P2/W2 requires one original W0 Service owner per request')
    posting_phases = {152: 'request_row', 153: 'request_row', 154: 'request_or_service_lane',
                      203: 'request_row', 206: 'request_row'}
    for phase, field in posting_phases.items():
        for event in phases[phase]:
            owner = owners.get(key(event, field))
            if owner is None or event['tid'] != owner['tid']:
                raise ValueError('Only the W1 request owner may select, prepare or submit WQE')

    width = 4 // workers_per_request
    slices = {
        (key(event, 'request_row'), key(event, 'query')): event for event in phases[253]
    }
    expected = {(b, q) for b in range(batch) for q in range(0, 4, width)} if workers_per_request > 1 else set()
    if set(slices) != expected or len(phases[253]) != len(expected):
        raise ValueError('Incomplete or duplicate W1 TopK worker slices')
    for (b, q), event in slices.items():
        worker = q // width
        if event['tid'] != _worker_aiv(batch, b, worker):
            raise ValueError('W1 TopK ran on an unexpected Score AIV')
    for phase in (251, 252):
        if workers_per_request > 1:
            expected_owners = {b: e['tid'] for b, e in owners.items()}
            if (len(phases[phase]) != batch
                    or {key(e, 'request_row'): e['tid'] for e in phases[phase]} != expected_owners):
                raise ValueError('Missing owner snapshot or join in split-Q W1')
        elif phases[phase]:
            raise ValueError('Split-Q owner scopes appeared in the serial control')
    if expected:
        score_end = {event['tid']: end_us(event) for event in phases[22]}
        snapshots = {key(event, 'request_row'): end_us(event) for event in phases[251]}
        joins = {key(event, 'request_row'): end_us(event) for event in phases[252]}
        for (b, q), event in slices.items():
            if event['tid'] not in score_end or event['ts'] < score_end[event['tid']]:
                raise ValueError('W1 TopK worker started before finishing its Score scan')
            if batch == 16 and q // width == 3 and event['ts'] < end_us(previous_owners[b]):
                raise ValueError('W0 Service owner began W1 TopK before finishing W0 Service')
            if event['ts'] < snapshots[b] or end_us(event) > joins[b]:
                raise ValueError('W1 TopK worker ran outside the owner snapshot and join')
        for phase, field in posting_phases.items():
            for event in phases[phase]:
                if event['ts'] < joins[key(event, field)]:
                    raise ValueError('W1 request owner started WQE work before the TopK join')

    seen = set()
    topk_by_request = defaultdict(list)
    for event in phases[155]:
        b, q = (key(event, field) for field in ('request_row', 'query'))
        if b not in owners or not 0 <= q < 4:
            raise ValueError('Window TopK query has an invalid request or query index')
        worker = slices[b, q // width * width] if expected else owners[b]
        if (b, q) in seen or event['tid'] != worker['tid']:
            raise ValueError('Window TopK query has duplicate or incorrect ownership')
        seen.add((b, q))
        topk_by_request[b].append(event)

    latest_aic = max(map(end_us, phases[21]))
    latest_aiv = max(map(end_us, phases[22]))
    latest_service = max(map(end_us, owners.values()))
    selected_phases = (150, 151, 152, 153, 154, 155, 203, 206, 243, 244, 251, 252, 253)
    scopes = [
        dict(phase=phase, core=event['tid'], source_start_us=event['ts'],
             source_duration_us=event['dur'], details=event['args'])
        for phase in selected_phases for event in phases[phase]
    ]
    scopes.extend(
        dict(phase=150, core=event['tid'], source_start_us=event['ts'],
             source_duration_us=event['dur'], details=event['args'])
        for event in previous_service
    )
    def max_owner_total(phase):
        totals = defaultdict(float)
        for event in phases[phase]:
            totals[event['tid']] += event['dur']
        return max(totals.values(), default=0.0)

    return dict(
        w1_topk_worker_aivs=len(slices) if expected else len(owners),
        w1_service_owner_aivs=len(owners),
        w1_reused_w0_owner_aivs=batch if batch == 16 and workers_per_request == 4 else 0,
        w1_empty_service_lanes=len(service_lanes) - len(owners),
        w1_after_last_aiv_score_scan_us=max(0, latest_service - latest_aiv),
        w1_after_last_aic_qk_scan_us=max(0, latest_service - latest_aic),
        w1_max_owner_service_us=max(e['dur'] for e in owners.values()),
        w1_max_owner_candidate_select_us=max_owner_total(152),
        w1_max_owner_dedup_vf_issue_us=max_owner_total(203),
        w1_max_owner_candidate_mte2_us=max_owner_total(206),
        w1_max_owner_prefetch_prepare_us=max((e['dur'] for e in phases[153]), default=0),
        w1_max_owner_wqe_submit_us=max((e['dur'] for e in phases[154]), default=0),
        w1_max_owner_topk_join_us=max((e['dur'] for e in phases[252]), default=0),
        w1_max_request_topk_elapsed_us=max(
            (max(map(end_us, spans)) - min(e['ts'] for e in spans) for spans in topk_by_request.values()), default=0),
        w1_per_q_topk_spans=len(seen),
    ), scopes
