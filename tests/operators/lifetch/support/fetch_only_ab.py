"""Graph A/B of the actual fused Q worker and the KVAF device entry.

Inputs are captured from the real completed LIFetch resolver. The reference
compacts descriptors outside timing (the native LIDU contract); the fused
worker retains its actual in-kernel packing. All transports use the same lease,
queue pairs, byte payloads, Q partition and AIV count.
"""
import ctypes

import numpy as np


MAGIC = 0x4B5641464247444F
TOKEN_BYTES = 656
TOPK = 2048


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


def run(torch, owner, rotating, pipeline, args):
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
                if version == 'fused':
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
                        if int(status[core, 0]) != MAGIC or int(status[core, 2]) or int(status[core, 1]) != expected_count:
                            raise AssertionError(f'native core={core} status={status[core, :4].tolist()} expected={expected_count}')
                else:
                    flags = pipeline.flags.cpu().numpy()
                    qs = range(4) if query == 4 else [query]
                    for b in range(batch):
                        for q in qs:
                            flag = flags[b, 8 + q]
                            if int(flag[0]) != 1 or int(flag[2]):
                                raise AssertionError(f'fused batch={b} q={q} flag={flag[:3].tolist()}')
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
            graph.reset()
            owner.graphs.remove(graph)
            median = float(np.median(samples))
            rows.append(dict(query='Q4_all' if query == 4 else f'Q{query+1}', version=version,
                             aiv=8 if query == 4 else 2, jettys_per_aiv=2,
                             count=len(selected), bytes=len(selected)*TOKEN_BYTES,
                             safe_count=int(counts[group_ids, 0].sum()),
                             conflict_count=int(counts[group_ids, 1].sum()),
                             p50_us=median, p99_us=float(np.percentile(samples, 99)),
                             effective_GBps=(len(selected)*TOKEN_BYTES/median/1000 if selected else None)))
    return rows
