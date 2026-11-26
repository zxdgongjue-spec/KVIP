"""BS32 diagnostic-only tile census. Device unit is GetSystemCycle ticks.

A separate graph replay records one residue of the cold tiles. Cycling every
residue gives each tile one observation while retaining its real neighbours.
The dense pass exposes simultaneous work; only its same-replay wall buckets add.
"""

from collections import defaultdict
import math
from statistics import mean

import numpy as np


ROLES = ("AIC", "AIV0", "AIV1")
PHASES = ("peer_wait", "cold_read", "score", "output")
PERIOD = 16


def _write_pointer(state, pointer):
    import torch

    binding = state.forward.terminal.workspace
    view = binding[120:128]
    view.copy_(torch.tensor(list(int(pointer).to_bytes(8, "little")), device=view.device, dtype=torch.uint8))


def bind(state, row):
    import torch

    shape = state.rotating.runtime_shape
    terminal = state.forward.terminal
    if (shape.batch, shape.liCores, shape.rounds, shape.serviceLanes) != (32, 32, 4, 32) or not 0 <= row < 32:
        raise ValueError("Cold tile profiling needs BS32, four windows and one scoring group per row")
    if terminal.workspace.numel() != 128 or terminal.workspace[120:128].cpu().any().item():
        raise ValueError("Expected unused reserved[8] in the terminal binding")
    limit = (int(shape.history) + 127) // 128
    data = torch.zeros(16 + 3 * (limit + 3) * 16, device=state.device, dtype=torch.int64)
    data[:2].copy_(torch.tensor([row, limit], device=state.device, dtype=torch.int64))
    # Retain even if completion is uncertain and the replay worker quarantines its transport.
    terminal.cold_profile_storage = data
    _write_pointer(state, data.data_ptr())
    return data


def _windows(events, row):
    records = np.asarray(events).reshape(4, 32, 16)[:, row]
    for epoch, record in enumerate(records):
        if int(record[2]) >> 32 != epoch or int(record[3]) & 0xffffffff != 1:
            raise ValueError(f"Missing completed cold window at row {row} epoch {epoch}")
    bounds = [(int(records[ep - 1, 1]), int(records[ep, 4])) for ep in (1, 2, 3)]
    if any(not 0 < a < b for a, b in bounds) or any(b > c for (_, b), (c, _) in zip(bounds, bounds[1:])):
        raise ValueError(f"Inconsistent cold window clock for R{row}")
    return bounds


def _wall_buckets(bounds, observations):
    """Partition real wall time, clipping all lanes to the posting owner's cold intervals."""
    totals = defaultdict(int)
    for left, right in bounds:
        edges = defaultdict(lambda: [0] * 4)
        edges[left]
        edges[right]
        for phase, start, stop in observations:
            a, b = max(left, start), min(right, stop)
            if a < b:
                edges[a][phase] += 1
                edges[b][phase] -= 1
        last, active = left, [0] * 4
        for tick, delta in sorted(edges.items()):
            labels = [p for p, n in zip(PHASES, active) if n]
            totals["+".join(labels) or "other"] += tick - last
            active = [a + d for a, d in zip(active, delta)]
            last = tick
        if any(active):
            raise ValueError("Unclosed phase range")
    return totals


def _read_pass(data, bounds, expected_ranges, mhz, *, dense):
    row, limit, stride, residue = map(int, data[:4])
    if not math.isfinite(mhz) or mhz <= 0 or stride < 1 or not 0 <= residue < stride:
        raise ValueError("Invalid clock or tile selection")
    records = data[16:].reshape(3, limit + 3, 16)
    roles = {}
    intervals = []
    for role, role_records in zip(ROLES, records):
        totals = np.zeros(4)
        counts = np.zeros(4, dtype=int)
        brackets = np.zeros(4)
        drain_us = 0.0
        read_tiles, clock_violations = [], 0
        for epoch, (lower, upper) in enumerate(expected_ranges):
            footer = role_records[limit + epoch]
            if (int(footer[8]), int(footer[9])) != (lower, upper):
                raise ValueError(f"Wrong tile range R{row} {role} W{epoch+1}")
            tiles = np.arange(lower, upper)
            selected = tiles[tiles % stride == residue]
            marks = role_records[selected, :8].reshape(-1, 4, 2)
            # Every selected tile waits on its peer, computes, and writes.
            # K/scale reads occur only at real bank refills.
            active = marks[:, :, 0] != 0
            if not active[:, [0, 2, 3]].all() or not np.array_equal(active, marks[:, :, 1] != 0):
                raise ValueError(f"Missing tile timestamps R{row} {role} W{epoch+1}")
            if np.any(role_records[tiles[tiles % stride != residue], :8]):
                raise ValueError("Unexpected samples outside the selected residue")
            read_tiles.extend(map(int, selected[active[:, 1]]))
            delta = marks[:, :, 1] - marks[:, :, 0]
            if np.any(delta < 0):
                raise ValueError("Hardware observation clock reversed")
            if np.any(marks[:, 0, 1] > marks[:, 2, 0]) or np.any(marks[:, 2, 1] > marks[:, 3, 0]):
                raise ValueError(f"Invalid wait/score/write causal order for {role}")
            for phase in range(4):
                starts, ends = marks[:, phase, :].T
                if dense:
                    intervals.extend((phase, int(a), int(b)) for a, b in zip(starts, ends) if a < b)
            if role != "AIC":
                # Reject cross-core overlaps if the clocks violate the actual
                # Fixpipe -> ready flag -> V-lane edge. Same-lane deltas still stand.
                clock_violations += int(np.sum(marks[:, 0, 1] < records[0, selected, 7]))
            first, last = map(int, footer[:2])
            if not 0 < first <= last:
                raise ValueError("Missing window drain handshake")
            drain_us += (last - first) / mhz
            if dense:
                intervals.append((0, first, last))
            f = footer[[10, 2, 4, 6]]
            empty = footer[[11, 3, 5, 7]] - f
            if np.any(f <= 0) or np.any(empty < 0):
                raise ValueError("Missing empty pipe fence calibration")
            counts += active.sum(axis=0)
            totals += delta.sum(axis=0) / mhz
            brackets += active.sum(axis=0) * empty / mhz
        roles[role] = dict(raw_us=totals, count=counts, bracket_us=brackets, drain_us=drain_us,
                           reads=read_tiles, clock_violations=clock_violations)
    aligned = all(v["clock_violations"] == 0 for v in roles.values())
    buckets = _wall_buckets(bounds, intervals) if dense and aligned else {}
    return roles, {k: v / mhz for k, v in buckets.items()}


def collect(state, replay_once, mhz, row):
    """replay_once restores the native inputs and clears external diagnostic records."""
    import torch

    shape = state.rotating.runtime_shape
    if not math.isfinite(mhz) or mhz <= 0:
        raise ValueError("Missing hardware clock calibration")
    cap = (int(shape.history) + 127) // 128
    real_end = min(int(shape.history), int(state.workspace.key_lengths.cpu().numpy()[row]),
                   int(state.rotating.query_ends.cpu().numpy().reshape(32, 4)[row].max()))
    start = int(shape.resident) // 128
    stop = (real_end + 127) // 128
    length = stop - start
    if not 3 <= length <= cap:
        raise ValueError("Invalid cold page count")
    # large_li::Window uses floor boundaries. Check all three against the device footer.
    boundaries = [start + i * length // 3 for i in range(4)]
    data = bind(state, row)
    held = state.forward.terminal
    combined = {role: dict(raw_us=np.zeros(4), count=np.zeros(4, dtype=int), bracket_us=np.zeros(4),
                           drains=[], reads=[]) for role in ROLES}
    sparse_wall = []
    dense_buckets = {}
    for stride, residue in [(0, 0)] + [(PERIOD, r) for r in range(PERIOD)] + [(1, 0)]:
        # Header is read at kernel entry. Data and external event writes precede replay on this stream.
        data[2:4].copy_(torch.tensor([stride, residue], device=state.device, dtype=torch.int64))
        data[16:].zero_()
        replay_once()
        events = state.rotating.events.cpu().numpy().view("<u8")
        bounds = _windows(events, row)
        wall = sum(b - a for a, b in bounds) / mhz
        if not stride:
            control = wall
            continue
        raw = data.cpu().numpy()
        details, buckets = _read_pass(raw, bounds, list(zip(boundaries, boundaries[1:])), mhz, dense=stride == 1)
        if stride == 1:
            dense_wall, dense_buckets = wall, buckets
            if buckets and not math.isclose(sum(buckets.values()), dense_wall, abs_tol=1e-6):
                raise ValueError("Dense three-lane wall buckets do not close")
            break
        sparse_wall.append(wall)
        for role in ROLES:
            for key in ("raw_us", "count", "bracket_us"):
                combined[role][key] += details[role][key]
            combined[role]["drains"].append(details[role]["drain_us"])
            combined[role]["reads"].extend(details[role]["reads"])
    _write_pointer(state, 0)
    torch.npu.current_stream(state.device).synchronize()
    del held.cold_profile_storage
    per_role = {}
    for role, values in combined.items():
        count = values["count"]
        # BS32/Q4 uses a 256-token K bank, one scale bank per 8 or 16 tiles.
        # The first tile of each span refills both banks independently of its address.
        periods = (2,) if role == "AIC" else (8, 16)
        refill_patterns = [[t for a, b in zip(boundaries, boundaries[1:]) for t in range(a, b, step)]
                           for step in periods]
        if sorted(values["reads"]) != details[role]["reads"] or details[role]["reads"] not in refill_patterns:
            raise ValueError(f"Cold K/scale refill coverage mismatch for {role}")
        if list(count[[0, 2, 3]]) != [length] * 3 or not 0 < count[1] <= length:
            raise ValueError(f"Incomplete tile census for {role}")
        # No clipping: near or below the timer floor is itself useful information.
        per_role[role] = dict(
            observations=dict(zip(PHASES, map(int, count))),
            raw_us=dict(zip(PHASES, values["raw_us"].tolist())),
            estimated_empty_brackets_us=dict(zip(PHASES, values["bracket_us"].tolist())),
            above_empty_us=dict(zip(PHASES, (values["raw_us"] - values["bracket_us"]).tolist())),
            window_drain_mean_us=mean(values["drains"]),
        )
    return dict(row=row, tile_count=length, tiles_by_window=boundaries, sample_period=PERIOD, clock_mhz=mhz,
                control_cold_us=control, sparse_cold_mean_us=mean(sparse_wall), sparse_cold_per_pass_us=sparse_wall,
                dense_cold_us=dense_wall, dense_wall_buckets_us=dense_buckets, sparse_per_role_us=per_role,
                dense_cross_core_clock_violations=sum(v["clock_violations"] for v in details.values()))


def format_report(group, result):
    group = str(group).removeprefix("G")
    prefix = f'G{group} BS32 R{result["row"]}'
    control = result["control_cold_us"]
    sparse = result["sparse_cold_mean_us"]
    dense = result["dense_cold_us"]
    archived = result.get("archive_cold_us")
    archive_label = "" if archived is None else f" archive_cold_us={archived:.2f}"
    lines = [f'{prefix} cold_tiles={result["tile_count"]} samples_per_tile_per_role=1 '
             f'cold_us control/sparse/dense={control:.2f}/{sparse:.2f}/{dense:.2f} '
             f'sparse_delta_pct={(sparse/control-1)*100:+.2f} dense_delta_pct={(dense/control-1)*100:+.2f}'
             + archive_label
             + f' sparse_min_max_us={min(result["sparse_cold_per_pass_us"]):.2f}/{max(result["sparse_cold_per_pass_us"]):.2f}']
    for role, details in result["sparse_per_role_us"].items():
        def fields(values):
            return "/".join(f"{values[p]:.2f}" for p in ("cold_read", "score", "peer_wait", "output"))
        lines.append(f'{prefix} {role} sparse_read/score/peer_wait/output_raw_us={fields(details["raw_us"])} '
                     f'above_empty_us={fields(details["above_empty_us"])} '
                     f'window_drain_mean_us={details["window_drain_mean_us"]:.2f}')
    parts = result["dense_wall_buckets_us"]
    if not parts:
        lines.append(f'{prefix} dense_cross_role_overlap=UNAVAILABLE '
                     f'violated_ready_edges={result["dense_cross_core_clock_violations"]}')
        return lines
    exclusive = {p: parts.get(p, 0.) for p in PHASES}
    mixed = sum(v for k,v in parts.items() if "+" in k)
    lines.append(f'{prefix} dense_only_exclusive_us '
                 + " ".join(f"{p}={v:.2f}" for p,v in exclusive.items())
                 + f' overlapping_categories={mixed:.2f} other={parts.get("other",0):.2f}')
    return lines
