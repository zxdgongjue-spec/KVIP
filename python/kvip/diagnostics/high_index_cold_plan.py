# SPDX-License-Identifier: Apache-2.0
"""Check the device's per-request cold-plan choice against the published event stream."""

import numpy as np


MAGIC = 0x43503231
TOUCHED_MAGIC = 0x48543331
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
    magic = int(strategy[0, 0])
    if ((expected in ("CP21","HT31") and magic != {"CP21": MAGIC, "HT31": TOUCHED_MAGIC}[expected])
            or magic not in (MAGIC, TOUCHED_MAGIC) or np.any(values[:2 * batch]) or np.any(values[3 * batch:])
            or np.any(strategy[:, 0] != magic) or np.any(~np.isin(strategy[:, 1], (1, 2, 3) if magic == TOUCHED_MAGIC else (1, 2)))
            or np.any(strategy[:, 2] > 10240) or np.any(strategy[:, 3] > marks[owners, 21])):
        raise ValueError("Invalid HIGH cold-plan strategy/source count/active time")
    if (magic == TOUCHED_MAGIC and
            np.any((strategy[:, 1] != 2) & (strategy[:, 3] != marks[owners, 21]))):
        raise ValueError("Touched planner timestamps disagree with total active cycles")
    return dict(marker="HT31" if magic == TOUCHED_MAGIC else "CP21", mode=strategy[:, 1].tolist(), probe_phase_events=strategy[:, 2].tolist(),
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


def touched_choice(sources, destinations, counts, resident, history):
    """Independently classify the ordered, per-chunk direct/touched/general eligibility."""
    if resident > 16384 or history > 210944:
        return 2, 0
    mode, processed, tested = 1, 0, 0
    seen = set()
    offset = 0
    for phase, count in enumerate(counts):
        if phase == 1:
            offset = 2048
        for begin in range(offset, offset + count, 512):
            end = min(begin+512, offset+count)
            size = end-begin
            if processed+size > 3072:
                return 2, tested
            processed += size
            if mode == 1:
                tested += size
                src, dst = sources[begin:end], destinations[begin:end]
                positions = set(src.tolist()) | set(dst.tolist())
                if (np.any(src < resident) or np.any(src >= history) or np.any(dst < 0)
                        or np.any(dst >= resident) or len(positions) != 2*size
                        or not positions.isdisjoint(seen)):
                    mode = 3
                else:
                    seen.update(positions)
        offset += count
    return mode, tested


def verify_choices(profile, flags, sources, destinations, resident, history):
    cold = profile["cold_plan"]
    row_moves = profile["preplan"]["row_moves"]
    row_pairs = profile["preplan"]["row_pairs"]
    fast_rows, fast_events, fallback_rows, fallback_events = [], 0, [], 0
    direct_rows, touched_rows = [], []
    for row, mode in enumerate(cold["mode"]):
        counts = list(map(int, flags[row, COUNT_FLAGS, 1]))
        if cold.get("marker") == "HT31":
            strategy, checked = touched_choice(sources[row], destinations[row], counts, resident, history)
            if strategy != mode or cold["probe_phase_events"][row] != checked:
                raise ValueError(f"HIGH touched-plan choice differs from CPU eligibility on row {row}")
            if strategy == 2:
                fallback_rows.append(row)
                fallback_events += sum(counts)
            else:
                (direct_rows if mode == 1 else touched_rows).append(row)
                if mode == 1 and (2*sum(counts), sum(counts)) != (row_moves[row], row_pairs[row]):
                    raise ValueError(f"HIGH disjoint swap count differs on row {row}")
                fast_rows.append(row)
                fast_events += sum(counts)
            continue
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
    if cold.get("marker") == "HT31":
        cold.update(direct_rows=direct_rows, touched_rows=touched_rows, general_rows=fallback_rows)
    cold.update(cpu_eligibility_checked=True, fast_rows=fast_rows, fallback_rows=fallback_rows,
                fast_events=fast_events, fallback_events=fallback_events)
