"""Independent replay of admission from per-query snapshots and oracle flags."""

import numpy as np


def replay_row(flags, row, *, resident, rounds, window_limit=480, candidate_service=False):
    consumed = np.zeros(4, dtype=np.uint32)
    fetched = []
    known = set()
    counts, suppressed = [0] * rounds, [0] * rounds
    order = [int(x) for x in row[214 : 214 + rounds]]
    active = [e for e in range(rounds) if order[e]]
    if sorted(order[e] for e in active) != list(range(1, len(active) + 1)):
        raise AssertionError("service order is not a unique sequence")
    for epoch in sorted(active, key=lambda e: order[e]):
        end = np.array(row[80 + epoch * 4 : 84 + epoch * 4], dtype=np.uint32)
        # An unclaimed row retains its zeroed snapshot. No cursor was consumed.
        if not end.any():
            continue
        if np.any(end < consumed) or np.any(end > flags.shape[1]):
            raise AssertionError("ready/cursor moved backwards or beyond history")
        first = int(np.maximum(consumed, resident).min())
        last = max(resident, int(end.max()))
        count = dropped = 0
        step = max(1, last - first) if candidate_service else 1024
        for base in range(first, last, step):
            stop = min(last, base + step)
            eligible = np.zeros(stop - base, dtype=bool)
            for q in range(4):
                lo, hi = max(base, int(consumed[q])), min(stop, int(end[q]))
                if hi > lo:
                    eligible[lo - base : hi - base] |= flags[q, lo:hi]
            candidates = [base + int(i) for i in np.flatnonzero(eligible) if base + int(i) not in known]
            available = min(
                window_limit,
                max(0, int(row[208]) - len(fetched)),
                max(0, int(row[209]) - len(fetched)),
                max(0, int(row[210]) - len(fetched)),
            )
            admitted = candidates[:available]
            fetched.extend(admitted)
            known.update(admitted)
            count += len(admitted)
            dropped += len(candidates) - len(admitted)
        counts[epoch] = count
        suppressed[epoch] = dropped
        consumed = end
    if counts != list(map(int, row[16 : 16 + rounds])):
        raise AssertionError(f"per-window admission mismatch: expected {counts}, got {row[16 : 16 + rounds]}")
    if suppressed != list(map(int, row[48 : 48 + rounds])):
        raise AssertionError("suppression count mismatch")
    if len(fetched) != int(row[9]) or len(fetched) != int(row[8]):
        raise AssertionError("slot/request accounting mismatch")
    if sum(suppressed) != int(row[10]) or not np.array_equal(consumed, row[12:16]):
        raise AssertionError("suppression/cursor accounting mismatch")
    return fetched


def replay_contract(scores, hot, row, candidate_capacity, rounds, window_limit=480, cold_ids=None):
    """Replay cold-position snapshots, then map admissions to logical IDs."""
    q, length = scores.shape
    flags = np.zeros((4, candidate_capacity), dtype=bool)
    cold = np.asarray([t for t in range(length) if t not in hot] if cold_ids is None else cold_ids, dtype=np.int32)
    for i in range(q):
        visible = max(0, length - q + i + 1)
        ids = [t for t in hot if t < visible]
        threshold = np.partition(scores[i, ids], -2048)[-2048] if len(ids) >= 2048 else -np.inf
        flags[i, : len(cold)] = (cold < visible) & (scores[i, cold] >= threshold)
    positions = replay_row(flags, row, resident=0, rounds=rounds, window_limit=window_limit, candidate_service=True)
    return cold[positions].tolist()
