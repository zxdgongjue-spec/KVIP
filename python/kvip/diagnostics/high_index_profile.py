# SPDX-License-Identifier: Apache-2.0
"""Oracle and markers for the full-AIV terminal Index on an eight-window HIGH step."""

import hashlib

import numpy as np


def event_work(flags, sources, destinations, *, history):
    """Fingerprint the exact ordered row/prefetch/Q event stream outside the timed graph."""
    flags, sources, destinations = map(np.asarray, (flags, sources, destinations))
    batch = len(sources)
    if (flags.dtype != np.int32 or flags.ndim != 3 or flags.shape[0] != batch
            or flags.shape[1] < 16 or flags.shape[2] != 32 or sources.dtype != np.int32
            or destinations.dtype != np.int32 or sources.shape != (batch, 10240)
            or destinations.shape != sources.shape):
        raise ValueError("Unexpected HIGH event/flag tensor layout")
    ready = flags[:, [0, 12, 13, 14, 15], :]
    counts = ready[:, :, 1].copy()
    if (np.any(ready[:, :, 0] != 1) or np.any(ready[:, :, 2] != 0)
            or np.any(counts < 0) or np.any(counts > 2048)):
        raise ValueError("Incomplete HIGH event publication")
    digest = hashlib.sha256(counts.astype("<i4").tobytes())
    for row, portions in enumerate(counts):
        final = int(sum(portions[1:]))
        for start, size in ((0, portions[0]), (2048, final)):
            src = sources[row, start:start + size]
            dst = destinations[row, start:start + size]
            if np.any(src < 0) or np.any(src >= history) or np.any(dst < 0) or np.any(dst >= history):
                raise ValueError("HIGH Index event outside history")
            digest.update(src.astype("<i4", copy=False).tobytes())
            digest.update(dst.astype("<i4", copy=False).tobytes())
    return dict(event_sha256=digest.hexdigest(), event_count=int(counts.sum()))


def verify_permutation(flags, sources, destinations, before, after, owners):
    """Replay each simultaneous 512-element assignment plus its displaced-row closure."""
    history = before.shape[1]
    two_cycle_moves = []
    for row, owner in enumerate(owners):
        at = np.arange(history, dtype=np.int32)
        location = at.copy()
        offset = 0
        for phase, count in enumerate(flags[row, [0,12,13,14,15], 1]):
            if phase == 1:
                offset = 2048
            for start in range(offset, offset+int(count), 512):
                stop = min(start+512, offset+int(count))
                src = location[sources[row, start:stop]]
                dst = destinations[row, start:stop]
                if len(np.unique(src)) != len(src) or len(np.unique(dst)) != len(dst):
                    raise ValueError("Conflicting HIGH Index assignments inside a chunk")
                evicted = dst[~np.isin(dst, src)]
                freed = src[~np.isin(src, dst)]
                source = np.concatenate((src, evicted))
                target = np.concatenate((dst, freed))
                original = at[source]
                at[target] = original
                location[original] = target
            offset += int(count)
        if not np.array_equal(before[owner, at], after[owner]):
            raise ValueError(f"HIGH Index permutation differs from ordered prefetch/Q updates: row={row}")
        # Count endpoints, not pairs. Use the composed permutation; the two
        # endpoints of an input event need not be partners after later phases.
        dirty = np.flatnonzero(at != np.arange(history))
        two_cycle_moves.append(int(np.count_nonzero(at[at[dirty]] == dirty)))
    return two_cycle_moves


def describe_row_load(flags, sources, destinations, moved, resident):
    """Describe real work distribution using the already copied diagnostic inputs."""
    counts = flags[:, [0, 12, 13, 14, 15], 1]
    batch = len(counts)
    row_events = counts.sum(axis=1)
    row_moves = np.asarray(moved).reshape(-1, batch).sum(axis=0)
    simple = 0
    for row, portions in enumerate(counts):
        prefetched, final = int(portions[0]), int(portions[1:].sum())
        if not prefetched + final:
            continue
        def used(table):
            return np.concatenate((table[row, :prefetched], table[row, 2048:2048 + final]))
        src, dst = used(sources), used(destinations)
        if (np.all(src >= resident) and np.all(dst < resident)
                and np.unique(src).size == src.size and np.unique(dst).size == dst.size):
            simple += 1
    return dict(row_events=row_events.tolist(), row_moves=row_moves.tolist(),
                max_row_events=int(row_events.max()), max_row_moves=int(row_moves.max()),
                dense_eligible_rows=int(np.count_nonzero(2 * row_events >= 4096)),
                simple_swap_rows=simple)


def validate_move_counts(profile, before, after, owners):
    """Physical Index rows must match the verified, persistent identity permutation."""
    before, after = map(np.asarray, (before, after))
    moved = np.asarray(profile["moved"])
    cores, batch = len(moved), len(owners)
    if before.shape != after.shape or before.ndim != 2 or cores != 64 or batch != 16:
        raise ValueError("Unexpected full-AIV Index metadata shape")
    for row, owner in enumerate(owners):
        count = int(np.count_nonzero(before[owner] != after[owner]))
        lanes = len(moved[row::batch])
        expected = np.diff(np.arange(lanes+1, dtype=np.int64) * count // lanes)
        if not np.array_equal(moved[row::batch], expected):
            raise ValueError("HIGH moved rows differ from the verified pre/post Index identities")
    profile["persistent_moves_checked"] = True


def descriptor_fingerprint(posts, arrays, addresses, *, final_get, arena_base):
    """Hash only submitted layout-relative block, token offset and destination triples."""
    posts = np.asarray(posts)
    if posts.ndim != 3 or posts.shape[1:] != (4, 32) or posts.dtype != np.int32:
        raise ValueError("Unexpected HIGH final-post table")
    arrays = tuple(map(np.asarray, arrays))
    if len(arrays) != 3 or any(v.shape != (len(posts)*8192,) or v.dtype != np.int32 for v in arrays):
        raise ValueError("Unexpected HIGH Get descriptor storage")
    addresses = np.asarray(addresses)
    if addresses.dtype != np.int64 or addresses.size != len(arrays[0]):
        raise ValueError("Invalid fallback address table")
    addresses = addresses.reshape(-1).view(np.uint64)
    counts = posts[:, :, 0]
    if np.any(counts < 0) or np.any(counts > 2048) or int(counts.sum()) != final_get:
        raise ValueError("Final descriptor count differs from completed HIGH Gets")
    digest = hashlib.sha256(counts.astype("<i4").tobytes())
    for record in posts.reshape(-1, 32):
        count, begin = int(record[0]), int(record[4])
        if not count:
            continue
        if not 0 <= begin <= len(arrays[0]) - count:
            raise ValueError("Final Get descriptors outside storage")
        src = arrays[0][begin:begin+count]
        if np.any(src < 0) or np.any(src >= len(addresses)) or np.any(addresses[src] < arena_base):
            raise ValueError("Invalid HIGH Get source or unrelocated DDR address")
        digest.update((addresses[src] - np.uint64(arena_base)).astype("<u8").tobytes())
        for values in arrays:
            digest.update(values[begin:begin+count].astype("<i4", copy=False).tobytes())
    return digest.hexdigest()


def decode_high_index(records, detail, *, wait_first, final_get, work):
    records = np.asarray(records)
    if records.dtype != np.int64 or records.shape != (64, 16):
        raise ValueError("HIGH requires one 128-byte record per AIV")
    rows = records.view(np.uint64)
    flags = rows[:, 2]
    if (np.any(rows[:, 0]) or np.any(rows[:, 3]) or np.any(rows[:, 14] != 2)
            or not np.array_equal(rows[:, 1], rows[:, 15])
            or sum(map(int, rows[:, 1])) != final_get or np.any(flags > 3)
            or np.any(((flags & 2) != 0) != wait_first)):
        raise ValueError("Incomplete HIGH Post/Index order, status, or Get count")
    result = dict(**work, final_get_by_core=rows[:, 1].tolist(), core_count=len(rows),
                  wait_first=wait_first, fallback=bool(np.any(flags & 1)), fallback_cores=int(np.count_nonzero(flags & 1)))
    if detail is None:
        # Inactive markers may still hold a preceding instrumented launch.
        return result
    detail = np.asarray(detail)
    if detail.dtype != np.int64 or detail.shape != (len(rows), 32):
        raise ValueError("Unexpected HIGH timestamp layout")
    marks = detail.view(np.uint64)
    if np.any(marks[:, 2] != 0x50494831):
        raise ValueError("Missing PIH1 markers on one or more AIVs")
    order = [3, 7, 4, 5, 6] if wait_first else [3, 4, 5, 6, 7]
    ticks = marks[:, order]
    if (np.any(ticks == 0) or np.any(ticks[:, 1:] < ticks[:, :-1])
            or np.any(rows[:, 7] > marks[:, 3]) or np.any(rows[:, 7] == 0)):
        raise ValueError("Unordered HIGH Post/Index/CQ markers")
    assigned, moved = marks[:, 9], marks[:, 10]
    if np.any(assigned != moved):
        raise ValueError("Full-AIV move count differs from the ordered Index permutation")
    # All endpoints are from the same launch and device clock. Include skew and joins.
    def span(starts, ends):
        return int(ends.max()) - int(starts.min())

    drain_start = marks[:, 3 if wait_first else 6]
    cq_span = span(drain_start, marks[:, 7])
    # The drain span includes software polling and warp skew; it is not a wire RTT.
    result.update(assigned=assigned.tolist(), moved=moved.tolist(), active_aiv=int(np.count_nonzero(moved)),
                  index_span=span(marks[:, 4], marks[:, 5]), cq_span=cq_span,
                  index_core_p50=float(np.median(marks[:, 5] - marks[:, 4])),
                  cq_core_p50=float(np.median(marks[:, 7] - drain_start)))
    # [7] ends local polling; [6] ends an Index collective. Only [13] follows
    # the last collective in both orders. Their maximum is an asymmetric proxy.
    joined = marks[:, 13]
    if np.any(joined):
        if np.any(joined == 0) or np.any(joined < np.maximum(marks[:, 6], marks[:, 7])):
            raise ValueError("Incomplete or unordered final HIGH collective markers")
        result["joined_span"] = span(marks[:, 3], joined)
    subphases = marks[:, [4, 11, 12, 5]]
    if np.any(subphases[:, 1:3]):
        if np.any(subphases == 0) or np.any(subphases[:, 1:] < subphases[:, :-1]):
            raise ValueError("Incomplete or unordered PIH1 plan/gather/scatter markers")
        # Differences between phase-end envelopes partition the measured Index span.
        result.update(plan_join=int(marks[:, 11].max()) - int(marks[:, 4].min()),
                      gather_join=int(marks[:, 12].max()) - int(marks[:, 11].max()),
                      scatter_end=int(marks[:, 5].max()) - int(marks[:, 12].max()))
        # The first two endpoints follow a collective. Last arrival *after*
        # that collective identifies the observed endpoint, not its cause.
        result["last_end_aiv"] = {name: int(np.argmax(marks[:, col])) for name, col in
                                  (("plan_join", 11), ("gather_join", 12), ("scatter", 5))}
    return result
