# SPDX-License-Identifier: Apache-2.0
"""Decode non-consuming reads of the single window-CQ owned by Q2."""
from b24_tail_archive import require


def decode_visibility(stamps, detail, visibility):
    require(len(visibility) == len(stamps) == len(detail), "Missing CQ samples")
    rows = []
    for row, (times, final, snapshots) in enumerate(zip(stamps, detail, visibility)):
        require(len(snapshots) == 4 and all(len(v) == 32 for v in snapshots), "Invalid row shape")
        require(not any(any(snapshots[q]) for q in (0, 2, 3)), "A non-owner wrote window CQ data")
        raw = snapshots[1]
        count = final[1][0]
        require(0 <= count <= 4, "Invalid number of final batches")
        anchors = [times[1][16]] + [final[1][11 + 4*i] for i in range(count)]
        observations = []
        for step, anchor in enumerate(anchors):
            lo, hi, cursor, pending, status = raw[5*step:5*step+5]
            require(anchor <= lo <= hi and lo > 0 and 0 <= pending <= 5, "Broken final/Index time bound")
            require((pending == 0 and status == 3) or (pending > 0 and status in (1, 2)),
                    "Invalid lease, invisible owner, or failed CQE")
            require(not observations or observations[-1]["end_tick"] <= anchor, "Overlapping checkpoints")
            observations.append(dict(begin_tick=lo, end_tick=hi, cursor=cursor, pending=pending,
                                     all_visible=status in (2, 3)))
        cursor, pending = observations[0]["cursor"], observations[0]["pending"]
        require(all(v["cursor"] == cursor and v["pending"] == pending for v in observations),
                "Prefetch CQ changed before its single original consumer")
        require(all(not a["all_visible"] or b["all_visible"] for a, b in zip(observations, observations[1:])),
                "Already visible completions disappeared")
        require(not any(raw[5*len(anchors):25]) and not any(raw[30:]), "Unexpected extra samples")
        lo, hi, after, still_pending, status = raw[25:30]
        require(times[1][24] <= lo <= hi <= times[1][17] and
                observations[-1]["end_tick"] <= times[1][24], "Post-drain time outside Q2 interval")
        require(after == cursor+pending and still_pending == 0 and status == 3,
                "Original prefetch drain did not advance by the exact pending count")
        rows.append(dict(row=row, query=2, final_cq_batches=count, observations=observations,
                         after_original_drain_tick=hi, cursor_after=after))
    return rows
