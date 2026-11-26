# SPDX-License-Identifier: Apache-2.0
"""Summarize a separate BS>16 window trace; timed graph measurements stay independent."""
from statistics import mean, median


def _distinct_row_overlap(previous, current, mhz):
    """Wall time when the posting lanes of different rows are servicing adjacent epochs."""
    intersections = []
    for row, (begin, end) in enumerate(current):
        for other, (old_begin, old_end) in enumerate(previous):
            left, right = max(begin, old_begin), min(end, old_end)
            if other != row and left < right:
                intersections.append((left, right))
    covered, last = 0, 0
    for left, right in sorted(intersections):
        covered += max(0, right - max(left, last))
        last = max(last, right)
    return covered / mhz


def summarize(events, li, token_bytes, *, winners=None, sources=None, sibling_checks=None, score_spans=None):
    rows = li['per_query_cycle_stamps']
    mhz = li['clock_mhz']
    if mhz <= 0 or not rows or any(len(epoch) != len(rows) for epoch in events):
        raise ValueError('Window timestamps and LI geometry disagree')
    origin = min(q[2] for row in rows for q in row)
    finish = (max(q[3] for row in rows for q in row) - origin) / mhz
    if (winners is None) != (sources is None):
        raise ValueError("Final physical winners and the ordered prefetch journal must be supplied together")
    if sibling_checks is not None and (len(sibling_checks) != len(rows)
            or any(len(checks) != len(events) - 1 for checks in sibling_checks)):
        raise ValueError("Sibling timestamps and window geometry disagree")
    by_epoch, posts, useful_posts = [], [], []
    overlaps, previous = [], []
    cursors = [0] * len(rows)
    selected = [set(map(int, row)) for row in winners] if winners is not None else None
    for epoch, records in enumerate(events):
        starts, ends, rank_service, prior_wait = [], [], [], []
        sibling_exposed, sibling_poll = [], []
        service = []
        useful_count = 0
        for row, record in enumerate(records):
            start, end, wait_start = record[0], record[1], record[4]
            issued, doorbell, after_doorbell = record[10:13]
            after_wait = record[13]
            if record[3] >> 32:
                raise ValueError(f'Skipped service at epoch={epoch} row={row}; active-window stamps required')
            if not origin <= wait_start <= after_wait <= start <= end or not record[3] & 1:
                raise ValueError(f'Incomplete LI window stamp at epoch={epoch} row={row}')
            starts.append((wait_start - origin) / mhz)
            ends.append((end - origin) / mhz)
            rank_service.append((end - after_wait) / mhz)
            service.append((after_wait, end))
            if epoch:
                prior_end = events[epoch - 1][row][1]
                prior_wait.append(max(0, prior_end - wait_start) / mhz)
                if sibling_checks is not None:
                    arrived, joined = sibling_checks[row][epoch - 1]
                    if not origin <= arrived <= joined <= end or joined < prior_end:
                        raise ValueError(f'Invalid sibling predecessor check at epoch={epoch} row={row}')
                    sibling_exposed.append(max(0, prior_end - arrived) / mhz)
                    sibling_poll.append((joined - arrived) / mhz)
            if issued:
                if not start <= doorbell <= after_doorbell <= end:
                    raise ValueError(f'Post outside service at epoch={epoch} row={row}')
                at = (doorbell - origin) / mhz
                posts.append((at, issued))
                if selected is not None:
                    before = cursors[row]
                    ids = sources[row][before:before + issued]
                    if len(ids) != issued or any(i < 0 for i in ids):
                        raise ValueError("Incomplete or invalid prefetch journal")
                    hits = sum(i in selected[row] for i in ids)
                    if hits:
                        useful_posts.append((at, hits))
                    cursors[row] += issued
                    useful_count += hits
        if epoch:
            overlaps.append(_distinct_row_overlap(previous, service, mhz))
        previous = service
        by_epoch.append(dict(epoch=epoch, earliest_window_start_us=min(starts),
                             last_window_end_us=max(ends), posted_tokens=sum(r[10] for r in records),
                             completed_rows=len(records), rows_with_zero_post=sum(r[10] == 0 for r in records),
                             rows_arrived_before_previous_end=sum(v > 0 for v in prior_wait),
                             mean_exposed_previous_service_us=mean(prior_wait) if prior_wait else 0.,
                             mean_check_to_rank_us=mean((r[13] - r[4]) / mhz for r in records),
                             mean_rank_and_service_us=mean(rank_service)))
        if selected is not None:
            by_epoch[-1]["used_by_final_topk"] = useful_count
        if sibling_exposed:
            by_epoch[-1].update(sibling_rows_arrived_before_previous_end=sum(v > 0 for v in sibling_exposed),
                                mean_sibling_exposed_previous_us=mean(sibling_exposed),
                                max_sibling_exposed_previous_us=max(sibling_exposed),
                                mean_sibling_check_us=mean(sibling_poll))
        if epoch and score_spans is not None:
            for lane, spans in score_spans.items():
                overlaps_post = []
                for row, row_spans in enumerate(spans):
                    begin, end = row_spans[epoch - 1]
                    if not 0 < begin <= end <= records[row][1]:
                        raise ValueError(f'Invalid score interval at epoch={epoch} row={row} lane={lane}')
                    prev = events[epoch - 1][row]
                    overlaps_post.append(max(0, min(prev[1], end) - max(prev[0], begin)) / mhz)
                by_epoch[-1][f"{lane}_score_overlaps_previous_service_rows"] = sum(v > 0 for v in overlaps_post)
                by_epoch[-1][f"{lane}_mean_score_overlap_previous_service_us"] = mean(overlaps_post)
    posts.sort()
    gaps = [b[0] - a[0] for a, b in zip(posts, posts[1:])]
    sensitivity = {}
    for gbps in (8, 16, 32, 64):
        # Work-conserving FIFO with one common payload rate. Doorbells are
        # observed. The link, its protocol and CQ completion times are modeled.
        end_at = busy_before = peak = 0.
        for at, tokens in posts:
            start_at = max(at, end_at)
            end_at = start_at + tokens * token_bytes / (gbps * 1000.)
            busy_before += max(0., min(end_at, finish) - min(start_at, finish))
            peak = max(peak, end_at - at)
        sensitivity[gbps] = dict(busy_during_li_us=busy_before, idle_during_li_us=finish - busy_before,
                                 drain_after_li_us=max(0., end_at - finish), peak_queued_service_us=peak)
    result = dict(li3_span_us=finish, nonempty_posts=len(posts),
                first_post_us=posts[0][0] if posts else None,
                last_post_us=posts[-1][0] if posts else None,
                median_inter_post_us=median(gaps) if gaps else None, longest_inter_post_us=max(gaps, default=None),
                adjacent_epochs_distinct_row_service_overlap_us=overlaps,
                adjacent_epochs_arrival_envelope_overlap_us=[
                    max(0., a['last_window_end_us'] - b['earliest_window_start_us']) for a, b in zip(by_epoch, by_epoch[1:])],
                epochs=by_epoch, ideal_link_by_payload_GBps=sensitivity)
    if selected is not None:
        total = sum(n for _, n in useful_posts)
        result["first_useful_post_us"] = min((t for t, _ in useful_posts), default=None)
        result["mean_useful_lead_to_li_end_us"] = (
            sum((finish - t) * n for t, n in useful_posts) / total if total else None)
    return result


def split_role_intervals(clocks, events, li, *, shared_final=False):
    """Trace only. Pair each row's three V0 score spans with all four V1 service spans."""
    from math import isfinite

    queries = li["per_query_cycle_stamps"]
    mhz = li["clock_mhz"]
    batch = len(queries)
    if (not batch or not isfinite(mhz) or mhz <= 0 or len(events) != 4
            or any(len(window) != batch for window in events) or len(clocks) < batch * 64):
        raise ValueError("Incomplete split-role clock geometry")
    per_row = []
    li_ends, window_ends = [], []
    for row, row_queries in enumerate(queries):
        offset = row * 64
        score = [tuple(map(int, clocks[offset + 2 * e:offset + 2 * e + 2])) for e in range(3)]
        offset += 16
        service = [tuple(map(int, clocks[offset + 3 * e:offset + 3 * e + 3])) for e in range(4)]
        li_begin, li_end = map(int, row_queries[0][2:4])
        if (len(row_queries) != 4 or not 0 < li_begin < li_end
                or any(tuple(q[2:4]) != (li_begin, li_end) for q in row_queries)):
            raise ValueError(f"Invalid split-role LI3 bounds at row {row}")
        if any(not 0 < begin < end or (e and begin < score[e - 1][1])
               for e, (begin, end) in enumerate(score)):
            raise ValueError(f"Unordered V0 score clocks at row {row}")
        for e, (wait, ready, done) in enumerate(service):
            event = events[e][row]
            if not (li_begin <= wait <= ready <= event[4] <= event[0] <= event[1] <= done <= li_end
                    and (e or wait == ready)
                    and (not e or service[e - 1][2] <= wait and score[e - 1][1] <= ready)):
                raise ValueError(f"Unordered V1 ready or service clock at row {row} window {e}")
        # A fast V0 can be more than one epoch ahead. Intersect with every V1
        # service span in this row. These spans cannot overlap with each other.
        overlaps = [[max(0, min(end, done) - max(begin, ready)) / mhz
                     for _, ready, done in service] for begin, end in score]
        li_ends.append(li_end)
        window_ends.append(service[-1][2])
        per_row.append(dict(row=row, score_cycles=score, wait_ready_done_cycles=service,
                            score_us=[(end - begin) / mhz for begin, end in score],
                            wait_ready_us=[(ready - wait) / mhz for wait, ready, _ in service[1:]],
                            # Publication follows score end. The remainder includes the signal,
                            # polling, and timestamp reads even when V0 finished earlier.
                            wait_for_unfinished_score_us=[
                                max(0, score[e][1] - service[e + 1][0]) / mhz for e in range(3)],
                            service_us=[(done - ready) / mhz for _, ready, done in service],
                            score_service_overlap_matrix_us=overlaps,
                            score_overlapped_by_service_us=[sum(matches) for matches in overlaps],
                            post_window_to_li3_end_us=(li_end - service[-1][2]) / mhz))
        if shared_final:
            v0_start, v0_end = map(int, clocks[row * 64 + 6:row * 64 + 8])
            v1_start, v1_end, joined = map(int, clocks[offset + 12:offset + 15])
            event = events[3][row]
            if not (max(score[-1][1], events[2][row][1]) <= v0_start < v0_end <= joined
                    and service[-1][1] <= v1_start < v1_end <= joined <= event[0]
                    and int(event[13]) == v1_start and int(event[14]) == joined):
                raise ValueError(f"Invalid shared final ranking clocks at row {row}")
            per_row[-1]["shared_final"] = dict(v0_rank_us=(v0_end - v0_start) / mhz,
                v1_rank_us=(v1_end - v1_start) / mhz, v1_join_us=(joined - v1_end) / mhz,
                rank_overlap_us=max(0, min(v0_end, v1_end) - max(v0_start, v1_start)) / mhz,
                v0_rank_cycles=[v0_start, v0_end], v1_rank_cycles=[v1_start, v1_end], joined_cycle=joined)
    # Resolve coincident last finishes to the lowest row ID and record how many tied.
    last_li3, last_window = max(li_ends), max(window_ends)
    return dict(clock_mhz=mhz, rows=per_row, last_li3_row=li_ends.index(last_li3),
                tied_last_li3_rows=li_ends.count(last_li3),
                last_window_row=window_ends.index(last_window),
                tied_last_window_rows=window_ends.count(last_window))
