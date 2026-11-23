"""One independently checked NPU graph of final-window availability diagnostics."""
from contextlib import ExitStack
from pathlib import Path
import math

import numpy as np


def decode_call(number, records, details):
    records = np.asarray(records, dtype=np.uint64)
    batch = records.shape[0]
    if records.shape != (batch, 4, 32) or batch < 1:
        raise ValueError(f'Unexpected rank-progress shape {records.shape}')
    head = details['streaming_cycles'][0]
    clock = float(head['terminal_timeline']['clock_mhz'])
    services = head['prefetch_services']
    if not math.isfinite(clock) or clock <= 0 or len(services) != batch * 4:
        raise ValueError('Clock or window service count is invalid')
    by_aiv = {}
    for e in services:
        lane, epoch = e['lane'], e['epoch']
        # In split-Q, both vector subcores of an AIC serve consecutive lanes.
        aiv = 2 * e['core'] + lane % 2
        if (aiv in by_aiv or e['event'] != epoch * batch + lane or
                not 0 <= epoch < 4 or not 0 <= lane < batch or e['end_cycles'] < e['begin_cycles']):
            raise ValueError(f'Invalid split-Q event mapping: {e}')
        by_aiv[aiv] = e
    if set(by_aiv) != set(range(4 * batch)):
        raise ValueError('One or more physical vector workers did not report')
    rows, workers = [], []
    for aiv, record in enumerate(records.reshape(-1, 32)[:, 16:]):
        score_end, safe, freeze_start, freeze_end, *stamps = map(int, record)
        e = by_aiv[aiv]
        if not 0 < score_end <= safe or safe < e['end_cycles']:
            raise ValueError(f'Invalid scoring/drain clocks for call {number} AIV {aiv}')
        workers.append(dict(aiv=aiv, service_epoch=e['epoch'], service_lane=e['lane'],
                            score_end_cycles=score_end, safe_cycles=safe))
        if e['epoch'] != 3:
            if freeze_start or freeze_end or any(stamps):
                raise ValueError(f'Non-owner AIV {aiv} wrote last-window stamps')
            continue
        if not score_end <= freeze_start <= freeze_end <= e['begin_cycles']:
            raise ValueError(f'Invalid frontier snapshot on AIV {aiv}')
        previous = freeze_end
        queries = []
        for q in range(4):
            begin, end, packed = stamps[3*q:3*q+3]
            if not previous <= begin < end <= e['begin_cycles']:
                raise ValueError(f'Invalid per-query clock for call {number} AIV {aiv} query {q}')
            previous = end
            queries.append(dict(q=q, begin_cycles=begin, end_cycles=end,
                                ready0=packed & 0xffffffff, ready1=packed >> 32))
        rows.append(dict(row=e['lane'], owner_aiv=aiv, freeze_begin_cycles=freeze_start,
                         freeze_end_cycles=freeze_end, queries=queries,
                         post_begin_cycles=e['begin_cycles'], post_end_cycles=e['end_cycles']))
    if sorted(r['row'] for r in rows) != list(range(batch)):
        raise ValueError('Duplicate or missing last-window owner')
    return dict(call=number, clock_mhz=clock, workers=workers, rows=sorted(rows, key=lambda r: r['row']))


def measure(torch, states, sequence, config, directory, *, fused):
    import execution as ex
    from diagnostics import algorithm_before, algorithm_after, decode_rank_progress
    from reporting import atomic_json

    if not fused or len(sequence.rounds) != 1 or not states:
        raise ValueError('Expected a single-round fused capture')
    target = Path(directory) / 'diagnose' / 'round-1'
    target.mkdir(parents=True, exist_ok=True)
    inputs = next(iter(sequence))
    stream = torch.npu.current_stream(states[0].device)
    capture, graph = torch.npu.Stream(), torch.npu.NPUGraph()
    checks = ex.CorrectnessRecorder()
    pending = False
    try:
        # The first Get on each distinct output registers it with the transport.
        # This must finish outside capture, before arming diagnostic writes.
        for i, state in enumerate(states):
            state.reset()
            state.forward.set_diagnostics(False)
            pending = True
            state.forward()
            stream.synchronize()
            checks.check(f'pre_capture/input{i}', lambda: state.validate(inputs))
            pending = False
            state.reset()
        stream.synchronize()
        leases = {id(state.lease): state.lease for state in states}.values()
        with ExitStack() as stack:
            for lease in leases:
                stack.enter_context(lease.capture_scope())
            pending = True
            with torch.npu.graph(graph, stream=capture):
                for state in states:
                    state.forward()
            capture.synchronize()
            pending = False
        for lease in leases:
            if hasattr(lease, 'retain_graph'):
                lease.retain_graph(graph)
        clock = states[0].forward.calibrate_counter_mhz()
        before = []
        for state in states:
            state.reset()
            state.forward.counter_clock_mhz = clock
            state.forward.set_diagnostics(True)
            state.prepare(inputs)
            snapshot = algorithm_before(state.rotating)
            snapshot['logical_ids'] = state.input_layout.copy()
            before.append(snapshot)
            state.forward.times.zero_()
            state.rotating.rank_progress.zero_()
        stream.synchronize()
        pending = True
        graph.replay()
        stream.synchronize()
        calls, details, raw_rank = [], [], []
        for i, state in enumerate(states):
            checks.check(f'detail/input{i}/round{inputs.source_round}', lambda: state.validate(inputs))
        pending = False
        for i, state in enumerate(states):
            records = state.rotating.rank_progress.cpu().numpy().astype(np.uint64)
            raw_rank.append(records)
            bounds = state.rotating.query_ends.cpu().tolist()
            detail = dict(round=i, source_round=inputs.source_round, streaming_cycles=state.forward.diagnostic_cycles(),
                          algorithm=algorithm_after(state.rotating, before[i], admissions=state.forward.events_src.cpu().numpy()),
                          rank_progress=[[b, q+1, w, *decode_rank_progress(int(v), bounds[b][q])]
                              for b, qs in enumerate(records) for q, words in enumerate(qs)
                              for w, v in enumerate(words[:4]) if v])
            details.append(detail)
            calls.append(decode_call(i, records, detail))
        np.savez_compressed(target/'raw_rank_progress.npz', records=np.stack(raw_rank))
        atomic_json(target/'detail_samples.json', details)
        atomic_json(target/'owner_availability.json', dict(schema=1, group=config.group, calls=calls,
                    safe_definition='last cold block returned, both UB score banks released, MTE2/MTE3/V joined to scalar',
                    query_definition='original AdvanceIncremental scalar start/return including its existing overlap with writeback of adjacent queries',
                    post_definition='original scan, dedup, and submission until the service completion marker'))
        return dict(correctness='FAIL' if checks.failures else 'PASS', failures=checks.failures,
                    diagnostic_only=True, diagnostic_calls=len(calls))
    except BaseException as error:
        if pending:
            ex.NativeFetchSession.quarantine('availability_diagnostic_completion_uncertain', error)
        raise
    finally:
        if not pending:
            for state in states:
                state.forward.set_diagnostics(False)
            graph.reset()
