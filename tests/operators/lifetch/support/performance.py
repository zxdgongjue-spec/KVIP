# SPDX-License-Identifier: Apache-2.0
"""Graph measurement and the existing native LI/LIDU/Get reference chain."""
from types import SimpleNamespace

from reporting import CorrectnessRecorder, timing_summary


def measure_graph(torch, call, reset, validate, *, warmup, iterations, diagnose=None):
    """Restore outside events; preserve completed samples on numerical failures.

    validate must raise RuntimeError/ValueError for unsafe completion or invalid
    addresses. Only numerical AssertionErrors are recoverable.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError('invalid graph iteration counts')
    stream = torch.npu.current_stream()
    capture = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    checks = CorrectnessRecorder()
    samples = []
    try:
        reset()
        call()
        stream.synchronize()
        checks.check('eager', validate)
        reset()
        stream.synchronize()
        with torch.npu.graph(graph, stream=capture):
            call()
        capture.synchronize()
        for step in range(warmup + iterations):
            reset()
            stream.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            if step >= warmup:
                samples.append(start.elapsed_time(end) * 1000)
            checks.check(f'replay{step}', validate)
        if diagnose is not None:
            diagnose(graph, reset, lambda: checks.check('diagnostic', validate))
        return dict(timing=timing_summary(samples), correctness=checks.status,
                    failures=checks.failures)
    finally:
        graph.reset()


def measure_ubmem_serial(torch, torch_npu, runtime, device_table, workspace, storage,
                         source_cpu, *, batch, history, resident, heads, warmup, iterations, trace_rounds=0, trace_folder=None):
    """Use NativeMtp and KVAF BatchGet before acquiring the fused UBmem lease.

    Serial state is independent. Both paths use identical real Q/K/scale tensors,
    the same remote access table, and the same initial hot population.
    """
    import numpy as np
    from native_mtp import NativeMtp
    from native_support import _load_native_operators
    from real_fetch_inputs import native_li
    from kvip.cache.lru import SlotLRU

    _, registration, _ = _load_native_operators('c8_offload')
    registration.require_a5_dsa_registration(('lidu_fused_batch_get_prepare_mtp_out',))
    device = workspace.inputs.query.device
    forward = torch.full((batch, history), -1, dtype=torch.int32, device=device)
    forward[:, :resident] = torch.arange(resident, dtype=torch.int32, device=device)
    session = SimpleNamespace(runtime=runtime, device_table=device_table,
                              cpu_payload=torch.from_numpy(source_cpu.reshape(batch, history, 656)),
                              batch_get_attempted=False)
    owner = SimpleNamespace(batch=batch, resident_budget=resident, cache_map=forward,
                            lease=None, session=session)
    fixture = dict(
        query=workspace.inputs.query.reshape(batch, 4, heads, 128),
        key=storage.keys.view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
        actual_query=torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 4,
        li_query_lengths=workspace.query_lengths, actual_key=workspace.key_lengths,
        block_table=workspace.inputs.block_table,
        native_inputs=dict(weights=workspace.inputs.weights.reshape(batch, 4, heads),
                           query_scale=workspace.inputs.scale.reshape(batch, 4, heads),
                           key_scale=storage.scales.reshape(-1, 128, 1)))
    native = NativeMtp(torch, device, owner, fixture, source_tokens=history, padded_tokens=history)
    native.bitmap = torch.zeros((batch, 4176), dtype=torch.int32, device=device)
    selected = native_li(torch_npu, fixture)
    initial = selected.cpu().numpy().reshape(batch, -1)
    for row in range(batch):
        policy = SlotLRU()
        policy.resize(resident, resident)
        policy.commit(initial[row][initial[row] < resident])
        native.preferred[row].copy_(torch.from_numpy(policy.slots[0]).to(device))
        native.bitmap[row, :len(policy.bitmap[0])].copy_(torch.from_numpy(policy.bitmap[0]).to(device))
    state = {}

    def call():
        state['selected'] = native_li(torch_npu, fixture)
        native.launch(state['selected'])

    def validate():
        counts = native.verify(state['selected'])
        expected = [len(np.unique(row[row >= resident])) for row in initial]
        assert counts[:-1] == expected, f'native miss counts {counts[:-1]} != {expected}'

    def diagnose(graph, reset, verify):
        from device_trace import collect_native
        collect_native(torch, graph, reset, verify, trace_rounds, trace_folder)
    return measure_graph(torch, call, native.reset, validate, warmup=warmup, iterations=iterations,
                         diagnose=diagnose if trace_rounds and trace_folder else None)


def measure_sequence(torch, rounds, graphs, reset, prepare, validate, *, warmup, iterations, save):
    """Measure pre-captured per-round graphs with independent continuous state.

    reset runs once per whole sequence. prepare imports this round's real Q and
    Index values into the path's current layout, outside timing; it must not
    restore captured residency. Graphs retain their fixed buffers for the run.
    save receives each completed measured sample before validation, so a later
    device or numerical failure cannot erase earlier timing evidence.
    """
    if warmup < 0 or iterations < 1 or not graphs:
        raise ValueError('invalid sequence measurement configuration')
    stream = torch.npu.current_stream()
    checks = CorrectnessRecorder()
    samples = {number: [] for number in graphs}
    totals = []
    for repetition in range(warmup + iterations):
        reset()
        sequence_total = 0.0
        seen = []
        for round_input in rounds:
            number = round_input.source_round
            if number not in graphs or number in seen:
                raise ValueError('Replay round does not have exactly one captured graph')
            seen.append(number)
            prepare(round_input)
            stream.synchronize()
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record(stream)
            graphs[number].replay()
            end.record(stream)
            end.synchronize()
            elapsed = begin.elapsed_time(end) * 1000
            if repetition >= warmup:
                # Validate timing before it enters any aggregate.
                timing_summary([elapsed])
                samples[number].append(elapsed)
                sequence_total += elapsed
                save(dict(repeat=repetition - warmup, source_round=number,
                          scheduler_round=round_input.scheduler_round, elapsed_us=elapsed,
                          correctness='NOT_CHECKED'))
            checks.check(f'sequence{repetition}/round{number}', lambda: validate(round_input))
            if repetition >= warmup:
                save(dict(repeat=repetition - warmup, source_round=number,
                          scheduler_round=round_input.scheduler_round, elapsed_us=elapsed,
                          correctness=checks.status, failures=list(checks.failures)))
        if set(seen) != set(graphs):
            raise ValueError('Replay sequence ended before all captured rounds')
        if repetition >= warmup:
            totals.append(sequence_total)
    return dict(rounds={number: timing_summary(values) for number, values in samples.items()},
                sequence_graph_sum=timing_summary(totals),
                all_rounds=timing_summary([value for values in samples.values() for value in values]),
                correctness=checks.status, failures=checks.failures)
