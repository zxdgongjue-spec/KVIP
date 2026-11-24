"""One independently checked NPU graph of final-window availability diagnostics."""
from contextlib import ExitStack
from pathlib import Path
import math

import numpy as np


def decode_call(number, records, details, event_words, history, resident):
    """Match each producer DMA interval to its real row, shard, and service."""
    records = np.asarray(records, dtype=np.uint64)
    batch = records.shape[0]
    if records.shape != (batch, 4, 32) or batch < 1 or (history-resident) % 256:
        raise ValueError(f'Unexpected split-Q geometry {records.shape} H={history} R={resident}')
    raw = np.asarray(event_words).reshape(-1, 128).view('<u8')
    head = details['streaming_cycles'][0]
    clock = float(head['terminal_timeline']['clock_mhz'])
    if not math.isfinite(clock) or clock <= 0 or raw.shape != (4 * batch, 16):
        raise ValueError('Invalid clock or service records')
    half_tiles = (((history - resident) // 256 + 1) // 2 + 2) // 3 * 2
    workers = []
    seen = set()
    for e in head['prefetch_services']:
        lane, epoch = e['lane'], e['epoch']
        aiv = 2 * e['core'] + lane % 2
        idx = epoch * batch + lane
        if (not 0 <= epoch < 4 or not 0 <= lane < batch or idx != e['event']
                or not 0 <= aiv < batch * 4 or aiv in seen):
            raise ValueError(f'Invalid or duplicate service on AIV {aiv}')
        seen.add(aiv)
        words = list(map(int, records.reshape(-1, 32)[aiv, 16:]))
        origin, first, freeze0, freeze1 = words[:4]
        wait, acquired = int(raw[idx, 4]), int(raw[idx, 13])
        post, end = e['begin_cycles'], e['end_cycles']
        if not 0 < origin <= wait <= acquired <= freeze0 <= freeze1 <= post <= end:
            raise ValueError(f'Service clocks out of order on call {number} AIV {aiv}')
        first_time, first_ok = first & ((1 << 63) - 1), bool(first >> 63)
        if (epoch and not wait <= first_time <= acquired) or (not epoch and first != 0):
            raise ValueError(f'Invalid first predecessor read on AIV {aiv}')
        shard = (aiv // 2) % 2
        samples = []
        last_done = origin
        for segment in range(3):
            start = resident // 128 + (2 * segment + shard) * half_tiles
            stop = min(start + half_tiles, history // 128)
            group = []
            for q in range(4):
                packed = words[4 + segment * 4 + q]
                issued = origin + (packed & 0xffffffff)
                done = issued + ((packed >> 32) & 65535)
                tile = packed >> 48
                threshold = start + ((stop - start) * (q + 1) + 3) // 4
                sample = [tile, issued, done]
                same = bool(group) and sample == group[-1]
                if (not threshold <= tile <= stop or (q == 3 and tile != stop)
                        or not packed or (not same and issued < last_done)
                        or (segment < epoch and done > wait)
                        or (segment >= epoch and issued < end)):
                    raise ValueError(f'Invalid score frontier on call {number} AIV {aiv} segment {segment+1} mark {q+1}: {sample}')
                group.append(sample)
                last_done = done
            samples.append(group)
        workers.append(dict(aiv=aiv, scoring_row=aiv//4, shard=shard, first_query=2*(aiv%2),
                            service_row=lane, service_epoch=epoch, clock_origin=origin,
                            wait_begin=wait, first_poll=first_time, first_poll_complete=first_ok,
                            acquired=acquired, freeze_begin=freeze0, freeze_end=freeze1,
                            post_begin=post, post_end=end, segments=samples))
    if len(seen) != 4 * batch:
        raise ValueError('Missing physical vector workers')
    return dict(call=number, clock_mhz=clock, workers=sorted(workers, key=lambda x:x['aiv']))


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
            events = state.rotating.events.cpu().numpy().reshape(-1, 128)
            calls.append(decode_call(i, records, detail, events, config.history, config.resident))
        np.savez_compressed(target/'raw_rank_progress.npz', records=np.stack(raw_rank))
        atomic_json(target/'detail_samples.json', details)
        atomic_json(target/'cold_publication.json', dict(schema=1, group=config.group, calls=calls,
                    segment_sample_fields=['exclusive_tile_end_128_tokens', 'before_frontier_dma_cycle',
                                           'after_frontier_dma_join_cycle']))
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
