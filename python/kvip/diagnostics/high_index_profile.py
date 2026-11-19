# SPDX-License-Identifier: Apache-2.0
"""Oracle and markers for the full-AIV terminal Index on an eight-window HIGH step."""

import hashlib

import numpy as np

from kvip.diagnostics.index_assistance import survey_index_helpers


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


def _cycle_histogram(at, dirty, paired):
    """Count nontrivial cycles in the composed destination-to-original permutation."""
    pair_moves = int(np.count_nonzero(paired))
    counts = {2: pair_moves // 2} if pair_moves else {}
    visited = set()
    for first in map(int, dirty[~paired]):
        if first in visited:
            continue
        pos, length = first, 0
        while pos not in visited:
            visited.add(pos)
            pos = int(at[pos])
            length += 1
        if pos != first or length < 3:
            raise ValueError("Malformed long cycle in composed HIGH Index permutation")
        counts[length] = counts.get(length, 0) + 1
    if sum(length * count for length, count in counts.items()) != len(dirty):
        raise ValueError("Index cycle lengths do not account for all changed positions")
    return counts


def verify_permutation(flags, sources, destinations, before, after, owners, *, resident=None, invalidations=None, invalidated_slots=None, cycle_histograms_out=None):
    """Replay each simultaneous 512-element assignment plus its displaced-row closure."""
    history = before.shape[1]
    two_cycle_moves = []
    for row, owner in enumerate(owners):
        at = np.arange(history, dtype=np.int32)
        location = at.copy()
        offset = 0
        cleared = 0
        invalid_slots = set()
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
                if invalidated_slots is not None:
                    invalid_slots.update(map(int, freed[freed < resident]))
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
        paired = at[at[dirty]] == dirty
        two_cycle_moves.append(int(np.count_nonzero(paired)))
        if cycle_histograms_out is not None:
            cycle_histograms_out.append(_cycle_histogram(at, dirty, paired))
        if invalidations is not None:
            invalidations.append(cleared)
        if invalidated_slots is not None:
            invalidated_slots.append(sorted(invalid_slots))
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
    if "cycle_histograms" in profile:
        hist = profile["cycle_histograms"]
        hist_moves = [sum(k * n for k, n in row.items()) for row in hist]
        hist_pairs = [2 * row.get(2, 0) for row in hist]
        if hist_moves != counts or hist_pairs != list(two_cycle_moves) or sum(counts) != int(moved.sum()):
            raise ValueError("Index cycle lengths disagree with verified move or pair counts")
    if local := profile.get("row_partitions"):
        # Unlike the global 64-lane planner, the new schedule independently divides
        # each row's exchanges, long-cycle tuples and invalidations among its four Q lanes.
        pairs = np.asarray(two_cycle_moves, dtype=np.int64) // 2
        singles = np.asarray(counts) - 2*pairs
        def partition(items):
            return np.diff(np.arange(cores // batch + 1)[:, None]*items // (cores // batch), axis=0).ravel()
        expected_pairs = partition(pairs).reshape(4, batch)
        expected_singles = partition(singles).reshape(4, batch)
        eligible = np.zeros(batch, dtype=bool)
        if local.get("single_owner", False):
            # RQI2: move all long-cycle originals on the last lane when that
            # lane can afford them. The other work consists of atomic swap pairs.
            total = 2*pairs + singles
            eligible = (singles > 0) & (singles <= 1024) & (4*singles <= total)
            pair_cuts = np.minimum(pairs, np.arange(5)[:, None]*total//4//2)
            expected_pairs[:, eligible] = np.diff(pair_cuts, axis=0)[:, eligible]
            expected_singles[:, eligible] = 0
            expected_singles[-1, eligible] = singles[eligible]
        if (not np.array_equal(local["pairs"], expected_pairs.ravel())
                or not np.array_equal(local["singles"], expected_singles.ravel())
                or not np.array_equal(local["invalidations"], partition(np.asarray(invalidations, dtype=np.int64)))
                or not np.array_equal(moved, (2*expected_pairs + expected_singles).ravel())):
            raise ValueError("Row-local HIGH transfer or invalidation counters disagree with persistent Index oracle")
        profile["row_gathers"] = int(np.count_nonzero((singles > 0) & ~eligible))
        profile["single_owner_rows"] = int(np.count_nonzero(eligible))
        profile["persistent_moves_checked"] = True
        return
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


def overlap_schedule_timing(rows, marks, *, marker):
    """One diagnostic replay. Only nonempty final Get calls contribute transport endpoints."""
    row_local = marker != "PIH1"
    live = rows[:, 1] > 0
    origin, index_end = rows[:, 10], rows[:, 8]
    index_start = marks[:, 3 if row_local else 11]
    start, returned, waited = (rows[live, k] for k in (6, 7, 9))
    last_start = marks[live, 4] if row_local else start
    n_calls = marks[:, 8] if row_local else live.astype(np.uint64)
    if (not np.any(live) or min(origin) == 0 or np.any(index_start < origin)
            or np.any(index_end < index_start) or np.any(start < origin[live])
            or np.any(last_start < start) or np.any(returned < last_start) or np.any(waited < returned)
            or np.any(n_calls[~live] != 0) or min(n_calls[live]) < 1
            or max(n_calls[live]) > (2 if row_local else 1)):
        raise ValueError("Invalid per-lane Index interval or nonempty final Post/Wait timestamps")
    begin = int(origin.min())
    last_lane = int(np.flatnonzero(live)[np.argmax(waited)])
    result = dict(
        format=marker,
        final_api_calls=int(n_calls.sum()), nonempty_lanes=int(live.sum()),
        first_post_begin=int(start.min())-begin, last_post_begin=int(last_start.max())-begin,
        last_post_return=int(returned.max())-begin, last_final_wait_return=int(waited.max())-begin,
        longest_lane_index=int((index_end-index_start).max()),
        index_completion_spread=int(index_end.max())-int(index_end.min()),
        last_index_minus_last_final_wait=int(index_end.max())-int(waited.max()),
        blocking_final_lanes=int(np.count_nonzero(rows[live, 2] & 1)),
        last_wait_lane=last_lane, last_wait_lane_get=int(rows[last_lane, 1]),
        last_wait_lane_post_return=int(rows[last_lane, 7])-begin,
        max_lane_get=int(rows[:, 1].max()), max_row_get=int(rows[:, 1].reshape(4, 16).sum(axis=0).max()),
    )
    # Streaming writes slots 0/1 around its dispatcher in both production
    # schedules. Compare these with the original's outer role envelope.
    entry, exit_time = marks[:, 0], marks[:, 1]
    if row_local or np.any(entry):
        if (np.any(entry == 0) or np.any(entry > origin) or np.any(exit_time < index_end)
                or np.any(exit_time[live] < waited)):
            raise ValueError("Missing or inconsistent streaming outer role timestamps")
        outer = int(entry.min())
        result["outer_entry"] = dict(first_post=int(start.min()) - outer, last_wait=int(waited.max()) - outer,
                                     last_role_exit=int(exit_time.max()) - outer)
    return result


def early_post_timing(rows, marks):
    """RQI4 times observed on a successful graph, relative to outer streaming entry."""
    outer = int(marks[:, 0].min())
    # Q1 is the sole writer for the four descriptor-ready publications in a row.
    published = marks[16:32, 24:28]
    plan, tnd = rows[32:48, 4], marks[:16, 22]
    observed, validated, joined = marks[:, 20], marks[:, 21], marks[:, 23]
    live = rows[:, 1] > 0
    if (not np.any(live) or np.any(published < marks[16:32, :1]) or np.any(published[:, 1:] < published[:, :-1])
            or np.any(published[:, -1] > marks[16:32, 1]) or np.any(plan < marks[32:48, 0])
            or np.any(plan > joined[32:48]) or np.any(tnd < marks[:16, 0]) or np.any(tnd > joined[:16])
            or np.any(observed < marks[:, 0]) or np.any(observed > validated)
            or not np.array_equal(rows[:, 5], validated) or np.any(joined < validated)
            or np.any(joined > marks[:, 3]) or np.any(rows[live, 6] < validated[live])):
        raise ValueError("Missing or unordered RQI4 descriptor, local Post or row-work markers")
    # A reader may observe the flushed descriptor before its publisher timestamps
    # the return from that flush. Only compare absolute endpoints, don't force
    # happens-before on those two readings from different AIVs.
    row_id = np.flatnonzero(live) % 16
    post = rows[live, 6]
    return dict(first_query_ready=int(published.min()) - outer, last_query_ready=int(published.max()) - outer,
                first_query_observed=int(observed.min()) - outer, last_query_observed=int(observed.max()) - outer,
                last_plan=int(plan.max()) - outer, last_tnd=int(tnd.max()) - outer,
                lanes_post_before_row_join=int(np.count_nonzero(post < joined[live])),
                lanes_post_before_row_plan_or_tnd=int(np.count_nonzero(post < np.maximum(plan, tnd)[row_id])),
                nonempty_lanes=int(live.sum()))


def retire_timing(rows, marks):
    """Bound software completion calls by their own lane clock."""
    begin, end = marks[:16, 28], marks[:16, 29]
    retire = marks[:, 30]
    live = np.flatnonzero(rows[:, 1])
    if (not len(live) or np.any(marks[16:, 28:30]) or np.any(begin < rows[:16, 8])
            or np.any(end < begin) or np.any(retire[:16] < end)
            or np.any(retire < rows[:, 8]) or np.any(rows[:, 9] < retire)):
        raise ValueError("Invalid RQI5 prefetch CQ and final RetirePair timestamps")
    last = live[np.argmax(rows[live, 9])]
    # The latest-returning lane can differ from the one with the longest call.
    final = rows[live, 9] - retire[live]
    return dict(prefetch_p50=float(np.median(end-begin)), prefetch_max=int((end-begin).max()),
                final_p50=float(np.median(final)), final_max=int(final.max()),
                last_lane=int(last), last_lane_final=int(rows[last, 9]-retire[last]))


def final_poll_work(rows, marks):
    """Count final CQ entry reads and explicit pacing on successful HIGH replays."""
    probes = marks[:, 31] & np.uint64(0xffffffff)
    paced = marks[:, 31] >> np.uint64(32)
    live = rows[:, 1] != 0
    calls = marks[:, 8]
    async_only = (rows[:, 2] & 1) == 0
    if (not np.any(live) or np.any((probes == 0) & (paced != 0))
            or np.any(probes[~live]) or np.any(probes[live & async_only] < calls[live & async_only])):
        raise ValueError("Invalid RQI6 final CQ probe totals for the submitted checkpoints")
    sampled = probes != 0
    # A fallback lane can contain both a blocking and an async Post. Its
    # checkpoint count alone cannot say how many reads found an empty CQ.
    exact_empty = int(probes.sum() - calls.sum()) if np.all(async_only[live]) else None
    active = np.flatnonzero(live)
    latest = active[np.argmax(rows[active, 9])]
    return dict(latest_lane_probes=int(probes[latest]), latest_lane_paced_ticks=int(paced[latest]),
                probes_total=int(probes.sum()), empty_probes=exact_empty,
                polled_lanes=int(sampled.sum()), paced_lanes=int(np.count_nonzero(paced)),
                probes_p50=float(np.median(probes[sampled])) if np.any(sampled) else 0.,
                probes_max=int(probes.max()), paced_ticks_total=int(paced.sum()),
                paced_ticks_max=int(paced.max()))


def check_get_reassignment(counts, marks, canonical):
    """Reconcile every transport lane against the independent per-request Q descriptor counts."""
    if canonical is None or np.shape(canonical) != (16, 4):
        raise ValueError("Missing canonical final descriptor counts for RQI3")
    source, take, give, donor = (marks[:, k].reshape(4, 16) for k in (20, 21, 22, 23))
    canonical = np.asarray(canonical).T
    # Check ranges before unsigned arithmetic. Each row may split one donor's
    # suffix onto one helper; neither the Index events nor descriptor lists move.
    if (not np.array_equal(canonical, source) or np.any(canonical > 2048)
            or np.any(give > source) or np.any(take > 2048)
            or np.any((take > 0) & ((source == 0) | (marks[:, 8].reshape(4, 16) != 2)))
            or np.any((take > 0) & (give > 0)) or np.any((take == 0) != (donor == 0))
            or np.any((give > 0).sum(axis=0) > 1) or np.any((take > 0).sum(axis=0) > 1)
            or not np.array_equal(take.sum(axis=0), give.sum(axis=0))
            or not np.array_equal(counts.reshape(4, 16), source - give + take)):
        raise ValueError("Invalid row-local final Get reassignment or per-lane completion counts")
    lanes, requests = np.nonzero(take)
    if not np.array_equal(donor[lanes, requests], np.argmax(give[:, requests], axis=0) + 1):
        raise ValueError("Final Get helper read an unexpected donor Q")
    return dict(helpers=len(lanes), transferred=int(take.sum()), max_canonical_lane=int(canonical.max()))


def decode_high_index(records, detail, *, wait_first, final_get, work, cold_plan_expected=False, schedule_timing=False, canonical_counts=None):
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
    row_markers = {0x52514931: "RQI1", 0x52514932: "RQI2", 0x52514933: "RQI3", 0x52514934: "RQI4", 0x52514935: "RQI5", 0x52514936: "RQI6"}
    marker = row_markers.get(int(marks[0, 2]))
    if marker is not None and np.all(marks[:, 2] == marks[0, 2]):
        if wait_first or not schedule_timing:
            raise ValueError("Row-local timing requires the full-graph HIGH overlap comparison")
        if np.any(marks[:, 9] != marks[:, 10]) or np.any(marks[:, 10] != 2*marks[:, 16] + marks[:, 17]):
            raise ValueError("Incomplete row-local Index transfer counts")
        result["moved"] = marks[:, 10].tolist()
        result["row_partitions"] = dict(pairs=marks[:, 16].tolist(), singles=marks[:, 17].tolist(),
                                        invalidations=marks[:, 19].tolist(), single_owner=marker != "RQI1")
        schedule = overlap_schedule_timing(rows, marks, marker=marker)
        if marker == "RQI6":
            schedule["poll_work"] = final_poll_work(rows, marks)
        if marker in ("RQI5", "RQI6"):
            schedule["retire"] = retire_timing(rows, marks)
        if marker in ("RQI4", "RQI5", "RQI6"):
            schedule["early_post"] = early_post_timing(rows, marks)
        if marker == "RQI3":
            schedule["get_reassignment"] = check_get_reassignment(rows[:, 1], marks, canonical_counts)
        # block=q*16+request. RQI4+ timestamps local descriptor validation
        # before Post and Q0 TND. Earlier versions timestamp row-role work
        # and validation before their rendezvous. Keep the raw JSON keys stable.
        ready = rows[:, 5].reshape(4, 16)
        entry = rows[:, 10].reshape(4, 16)
        if np.any(ready < entry) or np.any(ready.ravel() > marks[:, 3]):
            raise ValueError("Invalid row-local validation timestamp")
        spans = ready.max(axis=0) - ready.min(axis=0)
        schedule["validation_by_role_p50"] = np.median(ready-entry, axis=1).tolist()
        schedule["validation_spread_max"] = int(spans.max())
        schedule["lane_index_p50"] = float(np.median(rows[:, 8]-marks[:, 3]))
        # Every lane timestamps RetirePair return, including a lane with no Get.
        # q0 exits only after joining the four Done flags and publishing the row.
        retired = rows[:, 9].reshape(4, 16)
        last_retired = retired.max(axis=0)
        owner_exit = marks[:16, 1]
        if np.any(retired.ravel() < rows[:, 8]) or np.any(owner_exit < last_retired):
            raise ValueError("Invalid row retirement or owner exit timestamp")
        tail = owner_exit - last_retired
        schedule["row_retire_to_exit_p50"] = float(np.median(tail))
        schedule["row_retire_to_exit_max"] = int(tail.max())
        moved = marks[:, 10].reshape(4, 16)
        schedule["max_moves_per_lane"] = int(moved.max())
        schedule["max_row_moves"] = int(moved.sum(axis=0).max())
        schedule["moved_total"] = int(moved.sum())
        schedule["index_help"] = survey_index_helpers(rows, marks)
        result["schedule_timing"] = schedule
        return result
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
    if schedule_timing:
        if wait_first:
            raise ValueError("Expected simultaneous Index and transport")
        result["schedule_timing"] = overlap_schedule_timing(rows, marks, marker="PIH1")
    return result
