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


def verify_permutation(flags, sources, destinations, before, after, owners, *, resident=None, invalidations=None):
    """Replay each simultaneous 512-element assignment plus its displaced-row closure."""
    history = before.shape[1]
    two_cycle_moves = []
    for row, owner in enumerate(owners):
        at = np.arange(history, dtype=np.int32)
        location = at.copy()
        offset = 0
        cleared = 0
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
                if invalidations is not None:
                    cleared += int(np.count_nonzero(freed < resident))
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
        if invalidations is not None:
            invalidations.append(cleared)
    return two_cycle_moves


def describe_row_load(flags, sources, destinations, moved, resident, *, planned_row_moves=None):
    """Describe real work distribution using the already copied diagnostic inputs."""
    counts = flags[:, [0, 12, 13, 14, 15], 1]
    batch = len(counts)
    row_events = counts.sum(axis=1)
    # Legacy lanes own one request each. New workers consume flat ranges across rows.
    row_moves = (np.asarray(moved).reshape(-1, batch).sum(axis=0)
                 if planned_row_moves is None else np.asarray(planned_row_moves))
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


def validate_move_counts(profile, before, after, owners, *, two_cycle_moves=None, invalidations=None):
    """Physical Index rows must match the verified, persistent identity permutation."""
    before, after = map(np.asarray, (before, after))
    moved = np.asarray(profile["moved"])
    cores, batch = len(moved), len(owners)
    if before.shape != after.shape or before.ndim != 2 or cores != 64 or batch != 16:
        raise ValueError("Unexpected full-AIV Index metadata shape")
    counts = [int(np.count_nonzero(before[o] != after[o])) for o in owners]
    if prepared := profile.get("preplan"):
        if counts != prepared["row_moves"] or list(two_cycle_moves) != [2*p for p in prepared["row_pairs"]]:
            raise ValueError("Preplanned moves/pairs disagree with persistent Index permutation")
        # The lane owning the first endpoint of a pair moves both physical rows.
        # Derive its work from the oracle, independently of the device's planner counters.
        cuts = np.arange(cores+1, dtype=np.int64)*sum(counts)//cores
        expected_pairs = np.zeros(cores, dtype=np.int64)
        expected_singles = expected_pairs.copy()
        base = 0
        for count, endpoints in zip(counts, two_cycle_moves):
            local = np.clip(cuts-base, 0, count)
            expected_pairs += np.diff((np.minimum(local, endpoints)+1)//2)
            expected_singles += np.diff(np.maximum(local-endpoints, 0))
            base += count
        invalid = np.asarray(invalidations, dtype=np.int64)
        lanes = cores // batch
        expected_invalidations = np.diff(np.arange(lanes+1)[:, None]*invalid//lanes, axis=0).ravel()
        if (not np.array_equal(expected_pairs, prepared["pairs_by_core"])
                or not np.array_equal(expected_singles, prepared["long_by_core"])
                or not np.array_equal(expected_invalidations, prepared["invalidations_by_core"])):
            raise ValueError("Global HIGH transfer or hot invalidation work differs from its oracle partition")
        profile["persistent_moves_checked"] = True
        return
    for row, count in enumerate(counts):
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


def _preplan_detail(marks, moved, work):
    """Independently time early private planning, including publication waits."""
    cores = len(marks)
    batch = cores // 4
    lane2 = np.zeros(cores, dtype=bool)
    lane2[2*batch:3*batch] = True
    if (np.any(marks[~lane2][:, [14, 15, 20, 21, 22]])
            or np.any(marks[lane2, 14] == 0)
            or np.any(marks[lane2, 15] < marks[lane2, 14])
            or np.any(marks[lane2, 15] > marks[lane2, 3])):
        raise ValueError("Missing, late or misplaced HIGH private preplan marker")
    start, end = marks[lane2, 14], marks[lane2, 15]
    active = marks[lane2, 21]
    if np.any(active > end-start):
        raise ValueError("HIGH planner active duration exceeds its wall duration")
    pairs, singles, spill, invalid = marks[:, 16:20].T
    row_moves, row_pairs = marks[lane2, 20], marks[lane2, 22]
    # Conservative bounded arithmetic before operations on raw unsigned marker values.
    if (any(np.any(v > 32768) for v in (pairs, singles, spill, invalid, row_moves, row_pairs))
            or np.any(2*pairs+singles != moved) or np.any((spill != 0) & (spill != singles))
            or np.any(2*row_pairs > row_moves) or row_pairs.sum() != pairs.sum()
            or row_moves.sum() != moved.sum() or int(invalid.sum()) > work["event_count"]):
        raise ValueError("Invalid per-core HIGH pair, long-cycle, spill or invalidation accounting")
    intervals = [(int(a), int(b)) for a,b in zip(start, end)]
    post = (int(marks[:, 4].min()), int(marks[:, 5].max()))
    if max(b for _,b in intervals) > post[0]:
        raise ValueError("Private planning finished after the post-permission global barrier")
    # The separate preplan intervals on 16 AIVs overlap. Count their time union only once.
    wall = 0
    previous = 0
    for a,b in sorted(intervals):
        wall += max(0, b-max(a, previous))
        previous = max(previous, b)
    return dict(row_moves=row_moves.tolist(), row_pairs=row_pairs.tolist(),
                pairs_by_core=pairs.tolist(), long_by_core=singles.tolist(),
                spill_by_core=spill.tolist(), invalidations_by_core=invalid.tolist(),
                active_by_row=active.tolist(), wall_by_row=(end-start).tolist(),
                early_union=wall, early_active_max=int(active.max()),
                early_wait_overhead_max=int((end-start-active).max()),
                early_to_post_gap=post[0]-int(end.max()),
                index_union_including_early=wall + post[1]-post[0],
                first_preplan_to_last_index=post[1]-int(start.min()))


def decode_high_index(records, detail, *, wait_first, final_get, work, cold_plan_expected=False):
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
    from kvip.diagnostics.high_index_cold_plan import decode_choices

    cold = decode_choices(marks, cold_plan_expected)
    if cold:
        result["cold_plan"] = cold
    if cold or np.any(marks[:, 14:23]):
        result["preplan"] = _preplan_detail(marks, moved, work)
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
