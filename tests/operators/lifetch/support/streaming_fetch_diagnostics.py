"""Failure-only forensic report for the staged URMA path."""
import numpy as np


def summarize_mismatches(row, selected, ids, slots, output, payload, destinations,
                         addresses, posts, lru, mapping, stride, resident, remote_addresses, limit=8):
    selected = np.asarray(selected).reshape(-1)
    ids = np.asarray(ids).reshape(-1)
    slots = np.asarray(slots).reshape(-1)
    out = np.asarray(output)
    reference = np.asarray(payload)
    destinations = np.asarray(destinations).reshape(-1)
    addresses = np.asarray(addresses).reshape(-1)
    remote_addresses = np.asarray(remote_addresses).reshape(-1).view(np.uint64)
    posts = np.asarray(posts)
    mapping = np.asarray(mapping)
    lru = np.asarray(lru)
    bad = np.flatnonzero(np.any(out[slots] != reference[ids], axis=1))
    entries = []
    for p in bad[:limit]:
        slot = int(slots[p]); actual = out[slot]
        candidates = np.flatnonzero(np.all(reference[:, :16] == actual[:16], axis=1))
        actual_ids = [int(i) for i in candidates if np.array_equal(reference[i], actual)][:8]
        writers = []
        for q, post in enumerate(posts):
            count, start = int(post[0]), int(post[4])
            for i in range(start, start + count):
                if 0 <= i < len(destinations) and int(destinations[i]) == row * stride + slot:
                    remote = int(addresses[i]) & ((1 << 64) - 1)
                    source_ids = np.flatnonzero(remote_addresses == remote)
                    writers.append(dict(q=q + 1, descriptor=i, remote_address=hex(remote),
                                        source_tokens=[int(x) for x in source_ids[:4]]))
        virtual = np.flatnonzero(mapping >= 0)
        prefetched = [int(t) for t in virtual
                      if 0 <= int(mapping[t]) - resident < len(lru)
                      and int(lru[int(mapping[t]) - resident]) == slot]
        entries.append(dict(query=int(p // 2048 + 1), position=int(p % 2048),
                            selected_physical=int(selected[p]), expected_token=int(ids[p]), slot=slot,
                            actual_matching_tokens=actual_ids, prefetch_writers=prefetched, actual_prefix=bytes(actual[:16]).hex(),
                            expected_prefix=bytes(reference[ids[p], :16]).hex(), final_writers=writers,
                            lru_ordinal=next((int(i) for i in np.flatnonzero(lru == slot)), None)))
    return dict(batch=int(row), mismatch_positions=int(len(bad)), examples=entries,
                posts=posts[:, :6].astype(np.int64).tolist())


def audit_queue_writes(owner, rows, posts, row, bad_slots):
    """Read already-completed SQ entries on failure; no device work is submitted."""
    import ctypes

    torch = owner.torch
    snapshot = owner.abi_library.kvip_transport_snapshot
    snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    snapshot.restype = ctypes.c_uint32

    def read(address, size):
        if not address or address % 32 or not 0 < size <= 2 * 1024 * 1024 or size % 32:
            raise ValueError(f"invalid queue snapshot: {address:#x}, {size}")
        result = torch.empty(size, dtype=torch.uint8, device=owner.device)
        stream = torch.npu.current_stream(owner.device)
        code = snapshot(address, result.data_ptr(), size, stream.npu_stream)
        if code:
            raise RuntimeError(f"queue snapshot launch returned {code}")
        return result.cpu().numpy()

    contexts = owner.device_args.cpu().numpy().view('<u8')
    workers = posts.shape[1]
    batch = posts.shape[0]
    queue_specs = []
    for owner_row in range(batch):
        initial = int(rows[owner_row, 212]) | (int(rows[owner_row, 213]) << 32)
        queue_specs.append((f'b{owner_row}_prefetch', owner_row * 2, initial, int(rows[owner_row, 9])))
        for q, post in enumerate(posts[owner_row]):
            view = owner_row * 2 + 1 if q == 0 else batch * 2 + owner_row * (workers - 1) + q - 1
            count = int(post[0])
            if not count:
                continue
            queue = read(int(contexts[view, 9]), 64).view('<u8')
            queue_specs.append((f'b{owner_row}_final_q{q+1}', view, int(queue[1]) - count, count))
    result = []
    for name, view, start, count in queue_specs:
        if not count:
            continue
        entity = read(int(contexts[view, 8]), 192).view('<u4')
        meta = entity.view('<u8')
        depth = int(entity[19])
        if start < 0 or not 0 < count < depth <= 1048576:
            raise ValueError(f'invalid SQ range {name}: {start}, {count}, {depth}')
        first = min(count, depth - start % depth)
        chunks = [read(int(meta[7]) + (start % depth) * 64, first * 64)]
        if first < count:
            chunks.append(read(int(meta[7]), (count-first) * 64))
        sq = np.concatenate(chunks).view('<u4').reshape(count, 16)
        local = sq[:, 14].astype(np.uint64) | (sq[:, 15].astype(np.uint64) << np.uint64(32))
        remote = sq[:, 10].astype(np.uint64) | (sq[:, 11].astype(np.uint64) << np.uint64(32))
        for slot in bad_slots:
            expected = int(owner.output.data_ptr()) + (row * owner.output.shape[1] + int(slot)) * 656
            lengths = sq[:, 12].astype(np.uint64)
            for i in np.flatnonzero((local < expected + 656) & (local + lengths > expected)):
                result.append(dict(queue=name, sq_index=int((start + int(i)) % depth), slot=int(slot),
                                   local=hex(int(local[i])), remote=hex(int(remote[i])),
                                   length=int(sq[i, 12])))
    return result
