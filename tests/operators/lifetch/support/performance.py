# SPDX-License-Identifier: Apache-2.0
"""Graph timing and paired serial/fused comparisons."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
from reporting import CorrectnessRecorder, timing_summary

FETCH_AB_MAGIC = 0x4B5641464247444F

TOKEN_BYTES = 656

TOPK = 2048


def measure_graph(torch, call, reset, validate, *, warmup, iterations, diagnose=None, validate_each=True):
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
            if validate_each or step == warmup + iterations - 1:
                checks.check(f'replay{step}', validate)
        if diagnose is not None:
            diagnose(graph, reset, lambda: checks.check('diagnostic', validate))
        return dict(timing=timing_summary(samples), correctness=checks.status,
                    failures=checks.failures)
    finally:
        graph.reset()


def measure_ubmem_serial(torch, torch_npu, runtime, device_table, workspace, storage,
                         source_cpu, *, batch, history, resident, heads, warmup, iterations, trace_rounds=0, trace_folder=None, reference=None, bench_only=False):
    """Use NativeMtp and KVAF BatchGet before acquiring the fused UBmem lease.

    Serial state is independent. Both paths use identical real Q/K/scale tensors,
    the same remote access table, and the same initial hot population.
    """
    import numpy as np
    from execution import NativeMtp, _load_native_operators, native_li
    from kvip.cache.lru import SlotLRU

    _, registration, _ = _load_native_operators('c8_offload')
    registration.require_a5_dsa_registration(('lidu_fused_batch_get_prepare_mtp_out',))
    device = workspace.inputs.query.device
    forward = torch.full((batch, history), -1, dtype=torch.int32, device=device)
    forward[:, :resident] = torch.arange(resident, dtype=torch.int32, device=device)
    session = SimpleNamespace(runtime=runtime, device_table=device_table,
                              cpu_payload=torch.from_numpy(source_cpu.reshape(batch, history, 656)),
                              batch_get_attempted=False, reference=reference)
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
        from diagnostics import collect_native
        collect_native(torch, graph, reset, verify, trace_rounds, trace_folder)
    return measure_graph(torch, call, native.reset, validate, warmup=warmup, iterations=iterations,
                         diagnose=diagnose if trace_rounds and trace_folder else None, validate_each=not bench_only)


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


def build_group_plan(posts, blocks, offsets, destinations, batch):
    stride = batch * TOPK
    scratch = np.zeros((8, 3 * stride + ((3 * batch + 31) // 32) * 32), dtype=np.int32)
    counts = np.zeros((8, 2), dtype=np.int32)
    descriptor_indices = []
    for q in range(4):
        query_destinations = []
        for lane in range(2):
            ids = []
            # Same order as FetchQueryGroup: all safe ranges then conflicts.
            for phase in range(2):
                before = len(ids)
                for b in range(lane, batch, 2):
                    post = posts[b, q]
                    count, start, safe = int(post[0]), int(post[4]), int(post[5])
                    if not (0 <= safe <= count <= TOPK and start == b * 8192 + q * TOPK):
                        raise AssertionError(f'invalid descriptor batch={b} q={q}')
                    first, length = (start, safe) if phase == 0 else (start + safe, count - safe)
                    ids.extend(range(first, first + length))
                counts[q * 2 + lane, phase] = len(ids) - before
            group = q * 2 + lane
            if len(ids) > stride:
                raise AssertionError('native group exceeds packed capacity')
            scratch[group, :len(ids)] = blocks[ids]
            scratch[group, stride:stride + len(ids)] = offsets[ids]
            scratch[group, 2 * stride:2 * stride + len(ids)] = destinations[ids]
            query_destinations.extend(int(x) for x in destinations[ids])
        descriptor_indices.append(query_destinations)
    all_dest = sum(descriptor_indices, [])
    if len(all_dest) != len(set(all_dest)):
        raise AssertionError('duplicate final destination in captured fixture')
    return scratch, counts, descriptor_indices, all_dest


def compare_fetch_only(torch, owner, rotating, pipeline, args):
    if pipeline.q_workers != 2:
        raise ValueError('fetch-only A/B requires two workers per Q')
    batch, device = pipeline.batch, rotating.device
    posts = rotating.final_posts.cpu().numpy().copy()
    blocks = rotating.blocks.cpu().numpy().reshape(-1).copy()
    offsets = rotating.offsets.cpu().numpy().reshape(-1).copy()
    destinations = rotating.destinations.cpu().numpy().reshape(-1).copy()
    reference = owner.output.clone()
    original = owner.initial_output
    scratch, counts, descriptor_indices, all_dest = build_group_plan(
        posts, blocks, offsets, destinations, batch)
    capacity = batch * 8192
    native_packed = torch.empty((3, capacity), dtype=torch.int32, device=device)
    native_count = torch.zeros(1, dtype=torch.int32, device=device)
    native_status = torch.zeros((8, owner.abi_sizes['status']), dtype=torch.uint8, device=device)
    native_reference = torch.tensor([native_packed[0].data_ptr(), native_packed[1].data_ptr(),
                                     native_packed[2].data_ptr(), native_count.data_ptr(),
                                     native_status.data_ptr()], dtype=torch.uint64, device=device)
    native = rotating.library.kvip_native_fetch_only
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    native.restype = ctypes.c_uint32
    fused = rotating.library.kvip_query_fetch_only
    fused.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    fused.restype = ctypes.c_uint32
    stream = torch.npu.current_stream(device)
    capture = torch.npu.Stream(device=device)
    rows = []
    for query in range(5):
        selected = all_dest if query == 4 else descriptor_indices[query]
        expected = original.clone().reshape(-1, TOKEN_BYTES)
        if selected:
            ids = torch.tensor(selected, dtype=torch.int64, device=device)
            expected[ids] = reference.reshape(-1, TOKEN_BYTES)[ids]
        expected = expected.reshape_as(owner.output)
        group_ids = list(range(8)) if query == 4 else [query * 2, query * 2 + 1]
        for version, function in [('native', native), ('fused', fused)]:
            stage_lines = []
            if version == 'native':
                # Native KVAF receives one contiguous request list and performs
                # its own standard block_count partition across 2 or 8 AIVs.
                stride = batch * TOPK
                parts = [scratch[g, k * stride:k * stride + int(counts[g].sum())]
                         for k in range(3) for g in group_ids]
                for k in range(3):
                    packed = np.concatenate(parts[k * len(group_ids):(k + 1) * len(group_ids)])
                    if len(packed) != len(selected):
                        raise AssertionError('native descriptor count differs from fused')
                    native_packed[k, :len(packed)].copy_(torch.from_numpy(packed).to(device))
                native_count.fill_(len(selected))
            def prepare():
                owner.output.copy_(original)
                if version != 'native':
                    for q in (range(4) if query == 4 else [query]):
                        pipeline.flags[:, 8 + q, :3].zero_()
                else:
                    native_status.zero_()
            prepare()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            owner.graphs.append(graph)
            owner.mark_launched()
            with torch.npu.graph(graph, stream=capture):
                if version == 'native':
                    result = function(pipeline.workspace.data_ptr(), native_reference.data_ptr(), capture.npu_stream, query)
                else:
                    result = function(pipeline.workspace.data_ptr(), capture.npu_stream, query)
                if result:
                    raise RuntimeError(f'{version} fetch-only launch failed: {result}')
            capture.synchronize()
            graph.replay()
            stream.synchronize()

            def verify():
                if version == 'native':
                    status = native_status.cpu().numpy().view('<u8')
                    cores = 8 if query == 4 else 2
                    for core in range(cores):
                        expected_count = len(selected) * (core + 1) // cores - len(selected) * core // cores
                        if int(status[core, 0]) != FETCH_AB_MAGIC or int(status[core, 2]) or int(status[core, 1]) != expected_count:
                            raise RuntimeError(f'native core={core} status={status[core, :4].tolist()} expected={expected_count}')
                else:
                    flags = pipeline.flags.cpu().numpy()
                    qs = range(4) if query == 4 else [query]
                    for b in range(batch):
                        for q in qs:
                            flag = flags[b, 8 + q]
                            if int(flag[0]) != 1 or int(flag[2]):
                                raise RuntimeError(f'fused batch={b} q={q} flag={flag[:3].tolist()}')
                if not torch.equal(owner.output, expected):
                    raise AssertionError(f'{version} Q={query} full output byte mismatch')
                owner.mark_completed(completion_verified=True)

            verify()
            samples = []
            for iteration in range(args.warmup + args.iterations):
                prepare()
                stream.synchronize()
                owner.mark_launched()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
                if iteration >= args.warmup:
                    samples.append(start.elapsed_time(end) * 1000)
                verify()
            if version == 'native':
                prepare()
                profile_magic = torch.tensor(list((0x4b56414650524f46).to_bytes(8, 'little')),
                                             dtype=torch.uint8, device=device)
                native_status[:, 64:72].copy_(profile_magic)
                stream.synchronize()
                owner.mark_launched()
                graph.replay()
                stream.synchronize()
                verify()
                statuses = native_status.cpu().numpy().view('<u8')
                for core in range(8 if query == 4 else 2):
                    t = statuses[core, 8:]
                    stage_lines.append(f'FETCH STAGES native Q={query} core={core} cycles: '
                          f'get={int(t[7])-int(t[1])} post={int(t[-3])} cq={int(t[-2])}')
            if version != 'native':
                prepare()
                pipeline.times.zero_()
                pipeline.set_diagnostics(True)
                stream.synchronize()
                owner.mark_launched()
                graph.replay()
                stream.synchronize()
                verify()
                ticks = pipeline.times.cpu().numpy()[batch * 128:batch * 256].reshape(batch, 128)
                qs = range(4) if query == 4 else [query]
                for q in qs:
                    for lane in range(min(2, batch)):
                        t = ticks[lane, q * 24:q * 24 + 24]
                        stage_lines.append(f'FETCH STAGES {version} Q{q+1} lane={lane} cycles: '
                              f'prepare={int(t[18])} pack={int(t[19])} '
                              f'get={int(t[4])-int(t[3])} post={int(t[12])} cq={int(t[13])}')
                pipeline.set_diagnostics(False)
                stream.synchronize()
            graph.reset()
            owner.graphs.remove(graph)
            median = float(np.median(samples))
            rows.append(dict(query='Q4_all' if query == 4 else f'Q{query+1}', version=version,
                             aiv=8 if query == 4 else 2, jettys_per_aiv=2,
                             count=len(selected), bytes=len(selected)*TOKEN_BYTES, stages=stage_lines,
                             safe_count=int(counts[group_ids, 0].sum()),
                             conflict_count=int(counts[group_ids, 1].sum()),
                             **timing_summary(samples),
                             effective_GBps=(len(selected)*TOKEN_BYTES/median/1000 if selected else None)))
    return rows
