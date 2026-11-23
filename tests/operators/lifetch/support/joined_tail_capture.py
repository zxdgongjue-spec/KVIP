"""Capture the diagnostic owner TopK kernel on the same 20-slot graph as production."""

from math import isclose, isfinite

import numpy as np

from correctness import topk_fingerprint
from native_li_reference import check_positions, fused_li_output, native_ends
from owner_pack_tail import workload
from peer_delta import read
from reporting import atomic_json
from score_equivalence import check_scores


PASSES = ('off_before', 'odd_on', 'even_on', 'off_after')


def archived_g2_serial(root, inputs, identity):
    folder, = (root / 'g2/logical-ties/results').glob('*/cases/diagnose/fused')
    receipt = read(folder / 'score_topk_check.json')
    source = receipt['native_reference']
    if (receipt['group'] != 2 or receipt['calls'] != 20 or receipt['queries'] != 1280
            or source['origin'] not in ('fresh', 'validated-cache')
            or any(source['identity'][key] != identity[key] for key in
                   ('inputs', 'host', 'npu', 'visible', 'operator', 'sparse_mode', 'torch', 'torch_npu', 'cann'))):
        raise ValueError('G2 native reference differs from the measured production capture')
    with np.load(folder / 'score_topk_ids.npz', allow_pickle=False) as raw:
        serial = raw['strong_serial']
    check_positions(serial, native_ends(inputs))
    if topk_fingerprint(serial) != receipt['serial_set_digest']:
        raise ValueError('Saved G2 native TopK differs from the checked production reference')
    return {'strong_serial': serial}, dict(source, origin='completed-archive',
                                          archive_path=str(folder / 'score_topk_ids.npz'), replay_identity=identity)


def trace_enabled(name, call):
    return (name == 'odd_on' and call % 2 == 1) or (name == 'even_on' and call % 2 == 0)


def kernel_duration(accounting, call):
    groups = accounting['intervals_us']
    if (accounting['invocation'] != call - 1 or set(groups) != {'li', 'resolve_update'}
            or len(groups['li']) != 3 or len(groups['resolve_update']) != 1
            or accounting.get('owner_topk_kernel') != 'v9_owner_topk_resolve_update'):
        raise ValueError(f'Call {call} did not execute three LI kernels and the joined owner tail')
    intervals = sorted(groups['li']) + groups['resolve_update']
    if (any(not isfinite(a) or not isfinite(b) or a >= b for a, b in intervals)
            or any(b > c + 1e-5 for (_, b), (c, _) in zip(intervals, intervals[1:]))):
        raise ValueError(f'Invalid kernel ordering in call {call}')
    a, b = intervals[-1]
    if not isclose(b-a, accounting['stages_us']['resolve_update'], abs_tol=.025):
        raise ValueError(f'Inconsistent joined tail span in call {call}')
    return b-a


def capture(states, inputs, config, directory, ticks_per_us, profile_pass, native_reference):
    if len(states) != 20 or not isfinite(ticks_per_us) or ticks_per_us <= 0:
        raise ValueError('Joined capture needs twenty calibrated graph slots')
    fields = list(states[0].forward.query_owner_trace_fields)
    if len(fields) < 64 or len(set(fields)) != len(fields) or fields[0] != 'entry':
        raise ValueError(f'Unexpected owner timestamp schema: {fields}')
    refs, reference = native_reference
    passes = []
    for name in PASSES:
        for call, state in enumerate(states, 1):
            state.reset()
            state.forward.set_query_owner_capture(trace_enabled(name, call))
            state.forward.query_owner_trace.zero_()
        profile = profile_pass(directory / 'joined-tail-profiler' / name)
        timing = profile['timing']
        if (timing['graph_calls'] != len(states) or timing['measured_graphs'] != 1
                or timing['timing_protocol'] != 'one_profiled_graph'
                or not isfinite(timing['graph_event_us']) or timing['graph_event_us'] <= 0):
            raise ValueError(f'{name}: expected one complete twenty-slot graph')
        calls = []
        for call, (state, accounting) in enumerate(zip(states, profile['samples'], strict=True), 1):
            tail = kernel_duration(accounting, call)
            state.validate(inputs)
            score = check_scores(inputs, refs['strong_serial'], [fused_li_output(state)])
            requests = workload(state)
            totals = {k: sum(r[k] for r in requests) for k in ('prefetch', 'final')}
            if (any(accounting['work_totals'][k] != v for k, v in totals.items()) or
                    state.query_owners != dict(owners=64, queues=64, processed=totals['final'])):
                raise ValueError(f'{name} call {call}: profiler slot disagrees with physical Get receipts')
            enabled = trace_enabled(name, call)
            ticks = state.forward.query_owner_trace.cpu().numpy()
            if (ticks.shape != (64, len(fields)) or ticks.dtype != np.int64 or
                    (np.any(ticks[:, 0] <= 0) if enabled else np.any(ticks))):
                raise ValueError(f'{name} call {call}: trace on/off slot ownership mismatch')
            item = dict(call=call, traced=enabled, tail_us=tail, accounting=accounting,
                        score_check=score, requests=requests, layout_receipts=state.layout_receipts,
                        query_owners=state.query_owners, topk_set_digest=state.topk_set_digest)
            if enabled:
                item['ticks'] = ticks.tolist()
            calls.append(item)
        passes.append(dict(name=name, graph_event_us=timing['graph_event_us'], calls=calls))
    checked = sum(len(p['calls']) for p in passes)
    atomic_json(directory / 'joined_tail.json', dict(schema_version=1, group=config.group,
                warmup=config.warmup, independent_states=len(states), source_round=inputs.source_round,
                native_revision=config.production_revision, input_paths=list(map(str, inputs.source_paths)),
                ticks_per_us=ticks_per_us, trace_fields=fields, native_reference=reference,
                checked_calls=checked, passes=passes))
    return checked
