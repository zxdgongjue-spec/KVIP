"""Decode independent producer and Fetch intervals; clock units are device ticks."""

import numpy as np


def decode(times, events, *, incremental):
    if times.ndim != 3 or times.shape[1:] != (9, 8):
        raise ValueError(f"Unexpected eight-window producer trace {times.shape}")
    if events.ndim != 3 or events.shape[0] != 9 or events.shape[2] != 128:
        raise ValueError(f"Unexpected Fetch trace {events.shape}")
    records = np.ascontiguousarray(events).view("<u8").reshape(*events.shape[:2], 16)
    # Event contains begin/end, packed core/epoch, complete/skipped, waitBegin,
    # scan/candidate/post ticks, bitmap bytes, candidate count, posted count.
    if not np.all(records[:, :, 3] & 0xffffffff) or np.any(records[:, :, 1] < records[:, :, 0]):
        raise ValueError("Unfinished or reversed Fetch interval")
    for intervals in (times[:, 0, 0], times[:, 1:, 1:3] if incremental else times[:, :0]):
        if np.any(intervals <= 0):
            raise ValueError("Missing producer interval")

    def per_window(a):
        return np.asarray(a.max(axis=1), dtype=np.int64).tolist()

    parts = times[:, 0, 3:7]
    if incremental and parts.any() and not np.array_equal(parts.sum(axis=1), times[:, 0, 0]):
        raise ValueError("Seed subintervals do not cover the same query")

    return dict(
        # Split one query's observed seed time, preserving additivity.
        seed_slowest_parts=times[int(times[:, 0, 0].argmax()), 0, 3:7].tolist(),
        seed_max_ticks=int(times[:, 0, 0].max()),
        merge_max_by_cold_window=per_window(times[:, 1:, 1].T),
        publish_max_by_cold_window=per_window(times[:, 1:, 2].T),
        wait_max_by_epoch=per_window(records[:, :, 0] - records[:, :, 4]),
        service_max_by_epoch=per_window(records[:, :, 1] - records[:, :, 0]),
        candidate_max_by_epoch=per_window(records[:, :, 6]),
        post_max_by_epoch=per_window(records[:, :, 7]),
        inspected_by_epoch=records[:, :, 9].sum(axis=1).tolist(),
    )
