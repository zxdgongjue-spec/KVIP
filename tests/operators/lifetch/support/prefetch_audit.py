# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixture audit; reconcile measured counters without emulating NPU timing."""

import numpy as np


def actual_reads(mapping, selected, rows, resident):
    """Inspect completed replay outputs outside timing; count unique prefetches."""
    result = dict(
        prefetch_hot=0, prefetch_cold=0, unused_hot=0, unused_cold=0, final=int(rows[:, 247].astype(np.uint64).sum())
    )
    for row, tokens in zip(mapping, selected):
        fetched = np.flatnonzero(row >= 0)
        used = np.isin(fetched, tokens.reshape(-1))
        hot = fetched < resident
        result["prefetch_hot"] += int(hot.sum())
        result["prefetch_cold"] += int((~hot).sum())
        result["unused_hot"] += int((hot & ~used).sum())
        result["unused_cold"] += int((~hot & ~used).sum())
    prefetch = result["prefetch_hot"] + result["prefetch_cold"]
    if prefetch != int(rows[:, 9].astype(np.uint64).sum()):
        raise ValueError("Prefetch mapping and device request counts disagree")
    result["total"] = prefetch + result["final"]
    result["unused"] = result["unused_hot"] + result["unused_cold"]
    return result


def completion_failures(rows, posts, statuses, prefetch, final):
    """Decode existing device status buffers; no device work or state changes."""
    failures = []
    for b in range(len(rows)):
        cq = {
            name: words[2 * b : 2 * b + 2, :8].tolist()
            for name, words in (("prefetch_cq", prefetch), ("final_cq", final))
        }
        cq_bad = any(any(w[0] != 0x4B5641464247444F or w[2] for w in pair) for pair in cq.values())
        if not (rows[b, 11] or rows[b, 211] or rows[b, 248] or np.any(posts[b, :, 1:4]) or statuses[b, 0] or cq_bad):
            continue
        failures.append(
            dict(
                request=b,
                prefetch_error=int(rows[b, 11]),
                prefetch_pending=int(rows[b, 211]),
                final_pending=int(rows[b, 248]),
                final_requests=int(rows[b, 247]),
                resolver_status=statuses[b, :5].tolist(),
                posts_count_pending_error_deferred_descriptor=posts[b, :, :5].tolist(),
                **cq,
            )
        )
    return failures


def audit_wqes(wqes, entity, head, remote, destinations, token_bytes):
    """Compare the submitted SQ with KVAF's scalar READ wire format on CPU."""
    entity = np.asarray(entity, dtype="<u4")
    meta = entity.view("<u8")
    count = len(wqes)
    expected = np.zeros((count, 16), dtype="<u4")
    sequence = np.arange(count, dtype=np.uint64) + head
    depth = int(entity[19])
    if depth < 2 or depth & (depth - 1):
        raise ValueError(f"Invalid SQ depth: {depth}")
    expected[:, 0] = np.where(sequence & depth, 0, 0x80000000).astype(np.uint32) | 0x30150000
    if count:
        expected[-1, 0] = (expected[-1, 0] & 0x80000000) | 0x30360000
    expected[:, 1] = 6 << 8
    expected[:, 2] = (entity[20] & 0xFFFFFF) | 0x1000000
    expected[:, 3] = entity[6] & 0xFFFFF
    expected[:, 4:8] = entity[22:26]
    expected[:, 8] = entity[7]
    remote = np.asarray(remote, dtype=np.uint64)
    local = meta[4] + np.asarray(destinations, dtype=np.uint64) * token_bytes
    for column, addresses in ((10, remote), (14, local)):
        expected[:, column] = addresses & 0xFFFFFFFF
        expected[:, column + 1] = addresses >> 32
    expected[:, 12], expected[:, 13] = token_bytes, entity[12]
    different = np.asarray(wqes, dtype=np.uint32) != expected
    invalid_source = (remote < meta[1]) | (remote + token_bytes > meta[1] + meta[2])
    invalid_destination = (local < meta[4]) | (local + token_bytes > meta[4] + meta[5])
    bad = np.flatnonzero(different.any(axis=1) | invalid_source | invalid_destination)
    return dict(
        checked=count,
        mismatched_wqes=int(different.any(axis=1).sum()),
        mismatched_fields=np.count_nonzero(different, axis=0).tolist(),
        source_out_of_range=int(invalid_source.sum()),
        destination_out_of_range=int(invalid_destination.sum()),
        examples=[dict(index=int(i), actual=wqes[i].tolist(), expected=expected[i].tolist()) for i in bad[:3]],
    )


def snapshot_failed_transport(owner, rotating, posts, failures, folder):
    """Read failed queues after device execution; never consume CQ or retry DMA.

    Used only on test failure, outside graph/timing. Raw SQ/CQ evidence is saved
    once so subsequent analysis does not require another NPU run.
    """
    import ctypes  # Failure-only ACL dependency; CPU audit tests need no CANN.

    torch = rotating.torch
    snapshot = rotating.library.kvip_transport_snapshot
    snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    snapshot.restype = ctypes.c_uint32

    def read(address, size):
        if not address or address % 32 or not 0 < size <= 64 * 1024 * 1024 or size % 32:
            raise ValueError(f"Invalid diagnostic read: address={address:#x} bytes={size}")
        # Driver queue VAs are device-accessible, not ACL-copyable allocations.
        # Copy on device first; only the owned output is transferred to the CPU.
        result = torch.empty(size, dtype=torch.uint8, device=rotating.device)
        stream = torch.npu.current_stream(rotating.device)
        rc = snapshot(address, result.data_ptr(), size, stream.npu_stream)
        if rc:
            raise RuntimeError(f"Device diagnostic launch failed: {rc}")
        return result.cpu().numpy()

    contexts = owner.device_args.cpu().numpy().view("<u8")
    remote = rotating.fallback_addresses.cpu().numpy().reshape(-1)
    destinations = rotating.destinations.cpu().numpy().reshape(-1)
    raw, queues = {}, []
    batch, workers = posts.shape[:2]
    for failure in failures:
        b = failure["request"]
        for worker, post in enumerate(posts[b]):
            count, pending, error, deferred, descriptor = map(int, post[:5])
            if not count or deferred or error:
                continue  # No direct SQ submission to inspect.
            view = b * 2 + 1 if worker == 0 else batch * 2 + b * (workers - 1) + worker - 1
            entity = read(int(contexts[view, 8]), 192).view("<u4")
            meta = entity.view("<u8")
            queue = read(int(contexts[view, 9]), 64).view("<u8")
            depth, cqe_size, cq_depth = map(int, (entity[19], entity[32], entity[31]))
            head = int(post[6]) | (int(post[7]) << 32)
            if not 0 < count < depth <= 1048576 or not 4 <= cqe_size <= 4096 or cq_depth < 2:
                raise ValueError("Invalid transport ABI/depth in failure snapshot")
            start = head % depth
            first = min(count, depth - start)
            pieces = [read(int(meta[7]) + start * 64, first * 64)]
            if first < count:
                pieces.append(read(int(meta[7]), (count - first) * 64))
            sq = np.concatenate(pieces).view("<u4").reshape(count, 16)
            ci = int(queue[2])
            cq = read(int(meta[13]) + (ci % cq_depth) * cqe_size, cqe_size).view("<u4")
            token_bytes = int(contexts[view].view("<u4")[23])
            audit = audit_wqes(
                sq,
                entity,
                head,
                remote[descriptor : descriptor + count],
                destinations[descriptor : descriptor + count],
                token_bytes,
            )
            key = f"b{b}_w{worker}"
            raw.update({key + "_sq": sq, key + "_cq": cq, key + "_entity": entity, key + "_queue": queue})
            queues.append(
                dict(
                    request=b,
                    worker=worker,
                    pending=pending,
                    head=head,
                    producer=int(queue[1]),
                    consumer=ci,
                    cq_ready=bool(((int(cq[0]) >> 2) & 1) != ((ci // cq_depth) & 1)),
                    cq_status=int(cq[0]) >> 16,
                    **audit,
                )
            )
    np.savez(folder / "transport_snapshot.npz", **raw)
    return dict(
        queues=queues,
        checked=sum(q["checked"] for q in queues),
        mismatched_wqes=sum(q["mismatched_wqes"] for q in queues),
        source_out_of_range=sum(q["source_out_of_range"] for q in queues),
        destination_out_of_range=sum(q["destination_out_of_range"] for q in queues),
        ready_cqs=sum(q["cq_ready"] for q in queues),
        queue_count=len(queues),
    )
