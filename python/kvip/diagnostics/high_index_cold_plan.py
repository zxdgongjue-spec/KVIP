# SPDX-License-Identifier: Apache-2.0
"""Check the device's per-request cold-plan choice against the published event stream."""

import numpy as np


MAGIC = 0x43503231
COUNT_FLAGS = (0, 12, 13, 14, 15)


def decode_choices(marks, expected):
    values = marks[:, 23:27]
    if not np.any(values):
        if expected:
            raise ValueError("Missing HIGH cold-plan strategy markers on the pinned v2 build")
        return None
    batch = len(marks) // 4
    owners = slice(2 * batch, 3 * batch)
    strategy = values[owners]
    if (np.any(values[:2 * batch]) or np.any(values[3 * batch:])
            or np.any(strategy[:, 0] != MAGIC) or np.any(~np.isin(strategy[:, 1], (1, 2)))
            or np.any(strategy[:, 2] > 10240) or np.any(strategy[:, 3] > marks[owners, 21])):
        raise ValueError("Invalid HIGH cold-plan strategy/source count/active time")
    return dict(mode=strategy[:, 1].tolist(), probe_phase_events=strategy[:, 2].tolist(),
                probe_active_cycles=strategy[:, 3].tolist())


def expected_choice(sources, destinations, counts, resident, history):
    """Read exactly the five phase ranges. Raw IDs describe the original sources."""
    if resident > 16384 or history > 210944:
        return False, 0, None
    seen = set()
    visits = {}
    submitted = 0
    final_offset = 2048
    for phase, size in enumerate(counts):
        begin = 0 if phase == 0 else final_offset
        if phase:
            final_offset += size
        submitted += size  # The shim submits the complete phase, even when one chunk rejects it.
        for start in range(begin, begin + size, 512):
            stop = min(start + 512, begin + size)
            src = sources[start:stop]
            dst = destinations[start:stop]
            if (np.any(src < resident) or np.any(src >= history) or np.any(dst < 0)
                    or np.any(dst >= resident) or len(np.unique(dst)) != len(dst)):
                return False, submitted, None
            raw = set(map(int, src))
            if len(raw) != len(src) or not raw.isdisjoint(seen):
                return False, submitted, None
            seen.update(raw)
            for slot in map(int, dst):
                visits[slot] = visits.get(slot, 0) + 1
    # One independent cycle per victim: initial hot token followed by that
    # victim's cold tokens in arrival order. Exactly one visit gives a swap.
    return True, submitted, (len(seen) + len(visits), sum(n == 1 for n in visits.values()))


def verify_choices(profile, flags, sources, destinations, resident, history):
    cold = profile["cold_plan"]
    row_moves = profile["preplan"]["row_moves"]
    row_pairs = profile["preplan"]["row_pairs"]
    fast_rows, fast_events, fallback_rows, fallback_events = [], 0, [], 0
    for row, mode in enumerate(cold["mode"]):
        counts = list(map(int, flags[row, COUNT_FLAGS, 1]))
        eligible, submitted, cycle_counts = expected_choice(sources[row], destinations[row], counts,
                                                            resident, history)
        if (mode == 1) != eligible or cold["probe_phase_events"][row] != submitted:
            raise ValueError(f"HIGH cold-plan choice differs from CPU eligibility on row {row}")
        if eligible:
            if cycle_counts != (row_moves[row], row_pairs[row]):
                raise ValueError(f"HIGH fast-planner final permutation size/cycles differ on row {row}")
            fast_rows.append(row)
            fast_events += sum(counts)
        else:
            fallback_rows.append(row)
            fallback_events += sum(counts)
    cold.update(cpu_eligibility_checked=True, fast_rows=fast_rows, fallback_rows=fallback_rows,
                fast_events=fast_events, fallback_events=fallback_events)
