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


def destination_overlaps(raw):
    """Check actual WQE destination byte intervals, including partial overlaps."""
    spans = []
    for key, words in raw.items():
        if not key.endswith("_sq"):
            continue
        phase = "prefetch" if "_prefetch_" in key else "final"
        for index, wqe in enumerate(words):
            start = int(wqe[14]) | (int(wqe[15]) << 32)
            size = int(wqe[12])
            if size:
                spans.append((start, start + size, phase, key, index))
    counts = {"final_final": 0, "prefetch_prefetch": 0, "prefetch_final": 0}
    examples, active = [], []
    for span in sorted(spans):
        active = [prior for prior in active if prior[1] > span[0]]
        for prior in active:
            kind = span[2] + "_" + span[2] if prior[2] == span[2] else "prefetch_final"
            counts[kind] += 1
            if len(examples) < 5:
                examples.append(
                    dict(
                        kind=kind,
                        first=[prior[3], prior[4]],
                        second=[span[3], span[4]],
                        start=hex(span[0]),
                        end=hex(min(span[1], prior[1])),
                    )
                )
        active.append(span)
    return dict(
        **counts,
        wqes=len(spans),
        examples=examples,
        prefetch_snapshot_present=any("_prefetch_sq" in key for key in raw),
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
    rows = rotating.rows.cpu().numpy() if hasattr(rotating, "rows") else None
    for failure in failures:
        b = failure["request"]
        if rows is not None:
            # The prefetch queue is exclusive to this request. Its current-round
            # WQEs are the last row.requests entries, even after CQ consumption.
            entity = read(int(contexts[b * 2, 8]), 192).view("<u4")
            meta = entity.view("<u8")
            queue = read(int(contexts[b * 2, 9]), 64).view("<u8")
            count, depth, producer = int(rows[b, 9]), int(entity[19]), int(queue[1])
            if not 0 <= count < depth or producer < count:
                raise ValueError("Prefetch SQ history unavailable or overwritten")
            start = (producer - count) % depth
            first = min(count, depth - start)
            pieces = []
            if first:
                pieces.append(read(int(meta[7]) + start * 64, first * 64))
            if first < count:
                pieces.append(read(int(meta[7]), (count - first) * 64))
            raw[f"b{b}_prefetch_sq"] = (
                np.concatenate(pieces).view("<u4").reshape(count, 16) if pieces else np.empty((0, 16), dtype="<u4")
            )
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
        destination_overlaps=destination_overlaps(raw),
        checked=sum(q["checked"] for q in queues),
        mismatched_wqes=sum(q["mismatched_wqes"] for q in queues),
        source_out_of_range=sum(q["source_out_of_range"] for q in queues),
        destination_out_of_range=sum(q["destination_out_of_range"] for q in queues),
        ready_cqs=sum(q["cq_ready"] for q in queues),
        queue_count=len(queues),
    )


def summarize_saved_transport(folder):
    """Inspect existing snapshots only: no ACL, kernel launch or queue mutation."""
    import json
    from collections import Counter
    from pathlib import Path

    folder = Path(folder)
    failure = json.loads((folder / "completion_failure.json").read_text())
    queues = failure["transport"]["queues"]
    lines = ["SAVED TRANSPORT AUDIT: no device execution", f"phase={failure['phase']}"]
    pending = [q for q in queues if q["pending"]]
    lines.append(f"queues={len(queues)} pending={len(pending)} completed={len(queues) - len(pending)}")
    lines.append("completed=" + str([(q["request"], q["worker"]) for q in queues if not q["pending"]]))
    lines.append("producer_mismatch=" + str(sum(q["producer"] != q["head"] + q["checked"] for q in queues)))
    fields, regions, doorbells = {}, {"SQ": [], "CQ": []}, {"SQ": [], "CQ": []}

    def record(name, value):
        fields.setdefault(name, []).append(int(value))

    with np.load(folder / "transport_snapshot.npz", allow_pickle=False) as raw:
        overlaps = destination_overlaps(raw)
        lines.append("WQE destination overlaps: " + json.dumps(overlaps))
        if not overlaps["prefetch_snapshot_present"]:
            lines.append("prefetch/final intersection UNKNOWN: old snapshot contains no prefetch SQ")
        for q in queues:
            key = f"b{q['request']}_w{q['worker']}"
            entity = raw[key + "_entity"].astype("<u4", copy=False)
            meta = entity.view("<u8")
            state, cq = raw[key + "_queue"], raw[key + "_cq"]
            for name, value in (
                ("SQ_depth", entity[19]),
                ("WQE_bytes", entity[21]),
                ("CQ_depth", entity[31]),
                ("CQE_bytes", entity[32]),
                ("queue_initialized", state.view("<u4")[1]),
                ("queue_error", state[4]),
                ("queue_poisoned", state[5]),
                ("SQ_head", q["head"]),
                ("CQ_consumer", q["consumer"]),
                ("CQ_completed_batches", state[3]),
                ("local_token", entity[12]),
                ("remote_token", entity[6]),
                ("remote_token_value", entity[7]),
                ("CQ_word0", cq[0]),
            ):
                record(name, value)
            for kind, base, depth, size, db in (
                ("SQ", meta[7], entity[19], entity[21], meta[8]),
                ("CQ", meta[13], entity[31], entity[32], meta[14]),
            ):
                regions[kind].append((int(base), int(base) + int(depth) * int(size), key))
                doorbells[kind].append(int(db))
    for name, values in fields.items():
        counts = Counter(values)
        detail = (
            dict(sorted(counts.items()))
            if len(counts) <= 8
            else f"min={min(values)} max={max(values)} unique={len(counts)}"
        )
        lines.append(f"{name}: {detail}")
    for kind, spans in regions.items():
        overlaps = [
            (a[2], b[2]) for i, a in enumerate(spans) for b in spans[i + 1 :] if max(a[0], b[0]) < min(a[1], b[1])
        ]
        lines.append(
            f"{kind}: unique_base={len({a[0] for a in spans})} overlapping_pairs={len(overlaps)} "
            f"unique_doorbell={len(set(doorbells[kind]))} examples={overlaps[:3]}"
        )
    lines.append("CQ_consumer and SQ_head have different units; do not subtract them.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Summarize an existing transport_snapshot.npz without NPU execution")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    summary = summarize_saved_transport(args.folder)
    (args.folder / "transport_brief.log").write_text(summary + "\n")
    print(summary)
