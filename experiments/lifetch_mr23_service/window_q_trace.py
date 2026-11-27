# SPDX-License-Identifier: Apache-2.0
"""Verify and summarize last-window Top2048 workers in one source trace."""

from collections import defaultdict


def _p2_w0_aiv(request):
    return 4 * (request // 2) + request % 2


def _owner_aiv(batch, request, epoch, parts):
    if parts == 2:
        rank = epoch * ((batch + 1) // 2) + request // 2
        group = 2 * (rank % batch) + rank // batch
        return 2 * group + request % 2
    if batch == 1:
        return 4 + request
    return 8 * (batch // 2 + request // 2) + request % 2


def _worker_aiv(batch, request, worker, epoch, parts):
    if worker == 0:
        return _owner_aiv(batch, request, epoch, parts)
    if parts == 2:
        if epoch == 3:
            return _owner_aiv(batch, request, worker - 1, parts)
        if batch == 1:
            return (0, 1, 3)[worker - 1]
        previous = _p2_w0_aiv(request)
        return previous + (2, 2 * batch + 2, 0)[worker - 1]
    if batch == 1:
        return (2, 3, 6)[worker - 1]
    return 2 * (4 * (request // 2) + worker) + request % 2


def window_prefetch_get_counts(requests, windows, *, calls=1):
    totals = [0] * windows
    for request in requests:
        posted = request['prefetch_by_window']
        if (len(posted) != windows or sum(posted) != request['prefetch_requests']
                or any(not 0 <= count <= 128 for count in posted)):
            raise ValueError(f"Request {request['request']}: invalid window prefetch Get counters")
        for epoch, count in enumerate(posted):
            totals[epoch] += count
    return {f'w{epoch}_prefetch_get_requests_per_call': count / calls
            for epoch, count in enumerate(totals)}


def analyze_window_q(source, workers_per_request, batch=2, *, windows=2, parts=None):
    if parts is None:
        parts = 2 if batch == 16 else 4
    if (batch not in (1, 2, 4, 8, 16) or parts not in (2, 4)
            or workers_per_request not in (1, 2, 4) or windows not in (2, 4)
            or (parts == 4 and batch == 16)
            or (windows == 4 and (parts != 2 or batch == 1 or workers_per_request == 2))):
        raise ValueError('Unsupported window TopK geometry')
    last = windows - 1
    window = f'W{last}'
    by_epoch = [defaultdict(list) for _ in range(windows)]
    for event in source['traceEvents']:
        args = event.get('args', {})
        epoch = args.get('window_epoch')
        if (event.get('ph') == 'X' and args.get('core_type') in ('AIC', 'AIV')
                and args.get('kernel_id') == 22 and epoch in range(windows)):
            by_epoch[epoch][args['phase_id']].append(event)
    phases = by_epoch[last]
    if any(by_epoch[epoch][phase] for epoch in range(last) for phase in (251, 252, 253)):
        raise ValueError(f'Split-Q TopK may run only in {window}')

    def key(event, field):
        return event['args'][field]

    def end_us(event):
        return event['ts'] + event['dur']

    owners_by_epoch, inactive_service = {}, {}
    for epoch in range(windows) if parts == 2 else (last,):
        bodies = by_epoch[epoch][150]
        service_lanes = {key(event, 'service_lane'): event for event in bodies}
        # The inactive BS1 Service lane still publishes its completion event.
        expected_lanes = {lane: _owner_aiv(batch, lane, epoch, parts) for lane in range(max(2, batch))}
        if (len(bodies) != len(expected_lanes)
                or {lane: event['tid'] for lane, event in service_lanes.items()} != expected_lanes):
            raise ValueError(f'W{epoch} requires one original Service owner per request')
        owners_by_epoch[epoch] = {row: service_lanes[row] for row in range(batch)}
        if parts == 2 and batch == 1:
            inactive = service_lanes[1]
            inactive_service[inactive['tid']] = inactive
    owners = owners_by_epoch[last]
    posting_phases = {152: 'request_row', 153: 'request_row', 154: 'request_or_service_lane',
                      203: 'request_row', 206: 'request_row'}
    for phase, field in posting_phases.items():
        for event in phases[phase]:
            owner = owners.get(key(event, field))
            if owner is None or event['tid'] != owner['tid']:
                raise ValueError(f'Only the {window} request owner may select, prepare or submit WQE')

    width = 4 // workers_per_request
    slices = {
        (key(event, 'request_row'), key(event, 'query')): event for event in phases[253]
    }
    expected = {(b, q) for b in range(batch) for q in range(0, 4, width)} if workers_per_request > 1 else set()
    if set(slices) != expected or len(phases[253]) != len(expected):
        raise ValueError(f'Incomplete or duplicate {window} TopK worker slices')
    for (b, q), event in slices.items():
        worker = q // width
        if event['tid'] != _worker_aiv(batch, b, worker, last, parts):
            raise ValueError(f'{window} TopK ran on an unexpected Score AIV')
    for phase in (251, 252):
        if workers_per_request > 1:
            expected_owners = {b: e['tid'] for b, e in owners.items()}
            if (len(phases[phase]) != batch
                    or {key(e, 'request_row'): e['tid'] for e in phases[phase]} != expected_owners):
                raise ValueError(f'Missing owner snapshot or join in split-Q {window}')
        elif phases[phase]:
            raise ValueError('Split-Q owner scopes appeared in the serial control')
    if expected:
        score_end = {event['tid']: end_us(event) for event in phases[22]}
        snapshots = {key(event, 'request_row'): end_us(event) for event in phases[251]}
        joins = {key(event, 'request_row'): end_us(event) for event in phases[252]}
        for (b, q), event in slices.items():
            if event['tid'] not in score_end or event['ts'] < score_end[event['tid']]:
                raise ValueError(f'{window} TopK worker started before finishing its Score scan')
            for epoch, earlier_owners in owners_by_epoch.items():
                if epoch >= last:
                    continue
                earlier = earlier_owners[b]
                if event['tid'] == earlier['tid'] and event['ts'] < end_us(earlier):
                    raise ValueError(f'W{epoch} Service owner began {window} TopK before finishing W{epoch} Service')
            empty = inactive_service.get(event['tid'])
            if empty is not None and event['ts'] < end_us(empty):
                raise ValueError(f'{window} TopK helper started before finishing inactive Service')
            if event['ts'] < snapshots[b] or end_us(event) > joins[b]:
                raise ValueError(f'{window} TopK worker ran outside the owner snapshot and join')
        for phase, field in posting_phases.items():
            for event in phases[phase]:
                if event['ts'] < joins[key(event, field)]:
                    raise ValueError(f'{window} request owner started WQE work before the TopK join')

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
    selected_phases = (21, 22, 150, 151, 152, 153, 154, 155, 180, 203, 206, 243, 244, 251, 252, 253)
    scopes = [
        dict(phase=phase, core=event['tid'], source_start_us=event['ts'],
             source_duration_us=event['dur'], details=event['args'])
        for window_phases in by_epoch for phase in selected_phases for event in window_phases[phase]
    ]
    def max_owner_total(phase):
        totals = defaultdict(float)
        for event in phases[phase]:
            totals[event['tid']] += event['dur']
        return max(totals.values(), default=0.0)

    metrics = dict(
        topk_worker_aivs=len(slices) if expected else len(owners),
        service_owner_aivs=len(owners),
        reused_w0_owner_aivs=sum(event['tid'] == owners_by_epoch[0][b]['tid']
                                 for (b, _), event in slices.items()) if 0 in owners_by_epoch else 0,
        empty_service_lanes=len(phases[150]) - len(owners),
        after_last_aiv_score_scan_us=max(0, latest_service - latest_aiv),
        after_last_aic_qk_scan_us=max(0, latest_service - latest_aic),
        max_owner_service_us=max(e['dur'] for e in owners.values()),
        max_owner_candidate_select_us=max_owner_total(152),
        max_owner_dedup_vf_issue_us=max_owner_total(203),
        max_owner_candidate_mte2_us=max_owner_total(206),
        max_owner_prefetch_prepare_us=max((e['dur'] for e in phases[153]), default=0),
        max_owner_wqe_submit_us=max((e['dur'] for e in phases[154]), default=0),
        max_owner_topk_join_us=max((e['dur'] for e in phases[252]), default=0),
        max_request_topk_elapsed_us=max(
            (max(map(end_us, spans)) - min(e['ts'] for e in spans) for spans in topk_by_request.values()), default=0),
        per_q_topk_spans=len(seen),
    )
    metrics = {f'w{last}_{name}': value for name, value in metrics.items()}
    for epoch, window_phases in enumerate(by_epoch):
        owner_waits = window_phases[151]
        if epoch in owners_by_epoch:
            expected_owners = owners_by_epoch[epoch]
            for event in owner_waits:
                row = key(event, 'service_lane')
                if row in expected_owners and event['tid'] != expected_owners[row]['tid']:
                    raise ValueError(f'W{epoch} predecessor wait ran on a non-owner AIV')
        metrics[f'w{epoch}_max_owner_predecessor_wait_us'] = max(
            (event['dur'] for event in owner_waits), default=0.0)
        credit_waits = [event['dur'] for event in window_phases[180] if key(event, 'core_type') == 'AIC']
        metrics[f'w{epoch}_sampled_aic_credit_wait_spans'] = len(credit_waits)
        metrics[f'w{epoch}_max_sampled_aic_credit_wait_us'] = max(credit_waits, default=None)
    metrics.update({
        f'w{last}_aic_score_span_us': latest_aic - min(event['ts'] for event in phases[21]),
        f'w{last}_aiv_score_span_us': latest_aiv - min(event['ts'] for event in phases[22]),
        f'w{last}_reused_earlier_owner_aivs': (
            sum(event['tid'] == prior[b]['tid'] for (b, _), event in slices.items()
                for epoch, prior in owners_by_epoch.items() if epoch < last)
        ),
    })
    return metrics, scopes
