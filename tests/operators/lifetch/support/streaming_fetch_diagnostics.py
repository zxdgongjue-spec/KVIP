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
