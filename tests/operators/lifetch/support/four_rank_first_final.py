# SPDX-License-Identifier: Apache-2.0
"""Host decoder for one private, first final (post-TopK) CQE per nonempty Q."""


def decode(stamps, prefixes, observations, scheduler, mhz):
    if not len(stamps) == len(prefixes) == len(observations) == len(scheduler["rows"]):
        raise ValueError("Invalid first-final observation row count")
    origin = min(q[0] for row in stamps for q in row)
    result = []
    for row, (times, detail, obs) in enumerate(zip(stamps, prefixes, observations)):
        if not len(times) == len(detail) == len(obs) == 4:
            raise ValueError("Invalid first-final observation Q count")
        for q, (t, c, x) in enumerate(zip(times, detail, obs), 1):
            if len(x) != 32 or any(x[18:]):
                raise ValueError("Invalid first-final observation layout")
            if not t[20]:
                if any(x) or c[0]:
                    raise ValueError("Spurious CQ sample for empty final GET")
                continue
            start, end, consumer, depth, post_word, read_begin, read_end, read_consumer, pending, read_word, visible, ready_word, commit_begin, commit_end, new_consumer, polls, completed, error = x[:18]
            if not (c[0] >= 1 and depth >= 2 and pending == 1 and read_consumer == consumer and new_consumer == consumer + 1
                    and polls >= 1 and completed == 1 and not error
                    and 0 < t[15] <= t[4] <= start <= end <= t[16] <= c[10] <= read_begin <= read_end <= visible
                    <= commit_begin <= commit_end <= c[11]):
                raise ValueError(f"Inconsistent first-final consumer, timestamps or status at R{row} Q{q}")
            def status(word):
                if ((word >> 2) & 1) == ((consumer // depth) & 1):
                    return "pending"
                return "error" if word >> 16 else "success"
            a, b = status(post_word), status(read_word)
            if (status(ready_word) != "success" or "error" in (a,b) or
                    (a == "success" and b != "success") or
                    (b == "success") != (polls == 1) or (b == "success") != (visible == read_end)):
                raise ValueError(f"Private CQ owner, status or polling sequence changed at R{row} Q{q}")
            # A stale owner can be observed at any instant inside its read interval.
            # Its begin is a conservative lower bound, the first successful read's end an upper bound.
            lower = max([t[15]] + ([start] if a == "pending" else []) + ([read_begin] if b == "pending" else []))
            upper = end if a == "success" else visible
            if lower > upper:
                raise ValueError(f"Inconsistent CQ visibility bound at R{row} Q{q}")
            def us(tick):
                return round((tick-origin)/mhz, 3)
            sched = scheduler["rows"][row]
            result.append(dict(row=row, query=q, team=sched["team"], predecessor=sched["predecessor"],
                               wave=sched["wave_on_team"], first_prefix_tokens=c[1], final_tokens=t[20],
                               after_post_owner_status=a, first_drain_owner_status=b, ring_consumer=consumer, depth=depth,
                               poll_count=polls, after_post_word=post_word, first_drain_word=read_word, success_word=ready_word,
                               interval_after_post_read_us=[us(start),us(end)],
                               first_drain_read_us=[us(read_begin),us(read_end)],
                               availability_lower_upper_us=[us(lower),us(upper)], observed_success_in_drain_us=us(visible),
                               consumer_commit_begin_end_us=[us(commit_begin),us(commit_end)],
                               drain_to_observed_us=round((visible-read_begin)/mhz,3),
                               first_observed_to_commit_us=round((commit_end-upper)/mhz,3),
                               original_first_drain_us=round((c[11]-c[10])/mhz,3)))
    return result
