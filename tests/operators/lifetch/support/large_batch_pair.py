# SPDX-License-Identifier: Apache-2.0
"""Measure the strong serial and fused Q4 paths on identical expanded requests."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

import numpy as np

from large_batch_input import FIXED_BATCHES, GROUPS, SYNTHETIC_GROUP, input_digest, load, load_synthetic
from large_batch_reference import compute_reference, reference_identity, stage_serial
from reporting import atomic_json
from workloads import ReplayPagePlan, configuration

CALLS = 10
WARMUP = 4


def check_scores(inputs, selected, serial):
    # The native oracle is evaluated on this very batch. Independently, both
    # paths' normal validators check every strict winner against the full scores.
    exact = 0
    for b in range(inputs.geometry["batch"]):
        for q in range(4):
            got, ref = selected[b, q], serial[b, q]
            if np.array_equal(got, ref):
                exact += 1
                continue
            keys = inputs.logical_score_keys[b, q]
            cutoff = np.min(keys[ref])
            delta = np.setxor1d(got, ref, assume_unique=True)
            if not np.all(keys[delta] == cutoff):
                raise ValueError(f"Row {b} Q{q + 1} TopK disagrees above or below native cutoff")
    return exact


KERNEL_SUFFIX = r"(?:(?:_[0-9]+)|(?:_mix_ai[cv])|(?:_ai[cv]))*"


def fixed_stage_records(events, scope, *, fused, batch, strict_bounds=False):
    """Read raw, scope-relative CANN records, including both sides of MIX tasks."""
    from decimal import Decimal

    spans = [e for e in events if e.get("ph") == "X" and e.get("name") == scope]
    if len(spans) != 1:
        raise ValueError(f"Expected one profiler scope {scope}, found {len(spans)}")
    base = Decimal(str(spans[0]["ts"]))
    duration = float(spans[0]["dur"])
    names = (
        {"LI1": ("v9_li_large_hot",), "LI23": ("v9_li_large_threshold_cold",), "tail": ("v9_parallel_terminal_large",)}
        if fused
        else {
            "li": ("quantlightningindexer",),
            "lidu_prepare": ("a5lidufusedbatchgetpreparemtp",),
            "fetch": ("kvafexecutebatchgetaivkernelv2", "kvafexecutebatchgetaivkernelv3"),
        }
    )
    if batch == 16 and fused:
        names = {"pr30_li": ("v9_li_kernel",), "tail": ("v9_parallel_terminal",)}
    elif fused and batch < 16:
        names = {"LI1": ("v9_li_fixed_hot",), "LI2": ("v9_li_kernel_1",),
                 "LI3": ("v9_li_fixed_cold",), "tail": ("v9_parallel_terminal",)}
    names_re = {key: re.compile("(?:" + "|".join(aliases) + ")" + KERNEL_SUFFIX, re.I) for key, aliases in names.items()}
    stages = {key: [] for key in names}
    hardware_pids = {
        e.get("pid")
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Ascend Hardware"
    }
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if any(label in cat for label in ("cpu", "runtime", "python", "annotation")):
            continue
        args = e.get("args") or {}
        task_type = str(args.get("Task Type", "")).upper()
        symbol = e.get("name", "")
        key = next((k for k, p in names_re.items() if p.fullmatch(symbol)), None)
        compute = (
            "kernel" in cat
            or task_type.startswith(("KERNEL", "MIX_AI"))
            or task_type in ("AI_CORE", "AI_VECTOR_CORE", "AIC", "AIV", "AICPU")
        )
        if not key and not compute:
            # On the CANN hardware plane, named model/fence, event or memory
            # tasks are not runnable compute kernels. An unlabeled hardware
            # task cannot be silently counted as an inter-kernel gap.
            if e.get("pid") not in hardware_pids or task_type.startswith(
                ("MODEL_", "NOTIFY", "EVENT", "MEMCPY", "MEMSET", "STREAM", "LABEL", "PROFILING")
            ):
                continue
        start = float(Decimal(str(e["ts"])) - base)
        stop = start + float(e.get("dur", 0))
        if stop <= 0 or start >= duration:
            continue
        if not 0 <= start < stop <= duration:
            if strict_bounds:
                raise ValueError(f"Invalid device interval in {scope}: {symbol!r} [{start}, {stop}]")
            continue
        if key is None:
            raise ValueError(f"Unattributed device task {symbol!r} ({task_type or cat}) in {scope}")
        stream = args.get("Physic Stream Id", args.get("Stream Id", e.get("tid")))
        task = args.get("Task Id")
        variant = re.search(r"(?:_mix_|_)(ai[cv])(?=_|$)", symbol.lower())
        role = (
            variant.group(1)
            if variant
            else {"MIX_AIC": "aic", "MIX_AIV": "aiv", "KERNEL_MIX_AIC": "aic", "KERNEL_MIX_AIV": "aiv"}.get(task_type)
        )
        stages[key].append((start, stop, role, (e.get("pid"), stream, task), symbol))
    return stages


def fixed_invocation_intervals(stages, scope):
    """Validate the launches of exactly one invocation; merge paired AIC/AIV records."""
    if "pr30_li" in stages:
        stages = stages.copy()
        # With a profiler-supplied "_1" suffix the hot kernel name can equal
        # the threshold kernel's raw name. Recover the original three launches
        # in stream order, then verify each symbol and any mixed-task pair.
        launches = []
        for record in sorted(stages.pop("pr30_li"), key=lambda r: r[:2]):
            if not launches or record[0] >= max(r[1] for r in launches[-1]):
                launches.append([])
            launches[-1].append(record)
        if len(launches) != 3:
            raise ValueError(f"{scope}: expected three separate PR30 LI launches, got {len(launches)}")
        for i, (name, launch) in enumerate(zip(("v9_li_kernel", "v9_li_kernel_1", "v9_li_kernel_2"), launches), 1):
            if any(not re.fullmatch(name + KERNEL_SUFFIX, r[4], re.I) for r in launch):
                raise ValueError(f"{scope}: wrong PR30 LI{i} kernel, expected {name}, got {launch}")
            stages[f"LI{i}"] = launch
    intervals = {}
    for key, records in stages.items():
        if len(records) == 1:
            intervals[key] = [records[0][:2]]
        elif len(records) == 2:
            a, b = records
            if {a[2], b[2]} != {"aic", "aiv"} or None in a[3] or a[3] != b[3] or max(a[0], b[0]) >= min(a[1], b[1]):
                raise ValueError(f"Ambiguous repeated {key} launches in {scope}: {records}")
            intervals[key] = [(min(a[0], b[0]), max(a[1], b[1]))]
        else:
            raise ValueError(f"{scope}: expected one attributable {key} launch, got {records}")
    return intervals


def large_account(events, scope, elapsed, *, fused=True, batch=32):
    """Recognize CANN device names and task types in one restored graph."""
    stages = fixed_stage_records(events, scope, fused=fused, batch=batch)
    return dict(intervals_us=fixed_invocation_intervals(stages, scope), event_us=elapsed)


def fixed_kernel_stages(stages, fused):
    """Compute durations on the device clock, without assigning a graph event."""
    expected = ("LI1", "LI23", "tail") if fused else ("li", "lidu_prepare", "fetch")
    if fused and "LI2" in stages:
        expected = ("LI1", "LI2", "LI3", "tail")
    if set(stages) != set(expected) or any(len(stages[k]) != 1 for k in expected):
        raise ValueError(f"Unexpected production kernel sequence: {stages}")
    intervals = [stages[k][0] for k in expected]
    if any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a, b in intervals) or any(
        b > c + 0.001 for (_, b), (c, _) in zip(intervals, intervals[1:])
    ):
        raise ValueError(f"Unordered production kernel boundaries: {intervals}")
    active = {k: b - a for k, (a, b) in zip(expected, intervals)}
    span = intervals[-1][1] - intervals[0][0]
    sample = dict(
        LI=sum(active[k] for k in expected[:-1]) if fused else active["li"],
        tail=active["tail"] if fused else active["lidu_prepare"] + active["fetch"],
        span=span,
        inter_kernel_gap=span - sum(active.values()),
    )
    sample.update({k: active[k] for k in (expected[:-1] if fused else ("lidu_prepare", "fetch"))})
    return sample


def stage_sample(accounting, fused):
    """Separate active kernels, inter-kernel gaps, and graph event overhead."""
    if "error" in accounting:
        raise ValueError(f"Device attribution failed: {accounting}")
    sample = fixed_kernel_stages(accounting["intervals_us"], fused)
    event = float(accounting["event_us"])
    if not math.isfinite(event) or event <= 0 or event < sample["span"] - 2:
        raise ValueError(f"Inconsistent graph event {event} and kernel span {sample['span']}")
    return dict(sample, graph=event, event_minus_span=event - sample["span"])


def verify_abc(slots, ids, bounds, resident, stride):
    """An emitted ABC hit has a live logical token and its fixed circular slot."""
    tail = slots >= resident
    if not tail.any():
        return
    lo, hi = bounds[:, :1, None], bounds[:, 1:, None]
    width = stride - resident
    if width <= 0 or np.any(tail & ((ids < lo) | (ids >= hi) | (slots != resident + ids % width))):
        raise ValueError("ABC output differs from its logical range or circular slot")


def verify_assignments(start, layout, before_valid, valid, sources, destinations):
    """Reconstruct the full admission permutation, including released source slots."""
    src, dst = np.asarray(sources, dtype=np.int64), np.asarray(destinations, dtype=np.int64)
    if (
        src.shape != dst.shape
        or src.ndim != 1
        or np.any(src < 0)
        or np.any(dst < 0)
        or np.any(src >= len(start))
        or np.any(dst >= len(valid))
        or len(np.unique(src)) != len(src)
        or len(np.unique(dst)) != len(dst)
    ):
        raise ValueError("Duplicate or out-of-range source/destination in terminal journal")
    released = src[~np.isin(src, dst, assume_unique=True)]
    displaced = dst[~np.isin(dst, src, assume_unique=True)]
    expected = start.copy()
    expected[dst] = start[src]
    expected[released] = start[displaced]
    expected_valid = before_valid.copy()
    expected_valid[released[released < len(valid)]] = 0
    expected_valid[dst] = 1
    if not np.array_equal(expected, layout) or not np.array_equal(expected_valid, valid):
        raise ValueError("Ordered terminal journal and committed full Index/validity disagree")


def check_ordered_index(state, start, before_valid):
    src = state.forward.events_src.cpu().numpy()
    dst = state.forward.events_dst.cpu().numpy()
    counts = state.rotating.change_count.cpu().numpy()
    prefetch = state.rotating.rows[:, 9].cpu().numpy()
    final = state.rotating.final_posts[:, :, 0].cpu().numpy().sum(axis=1)
    layout = state.storage.logical_ids[1:].cpu().numpy()
    valid = state.storage.hot_valid[1:].cpu().numpy()
    if (
        src.ndim != 2
        or src.shape != dst.shape
        or src.shape[0] != state.batch
        or not np.array_equal(counts, prefetch + final)
        or np.any(counts < 0)
        or np.any(counts > src.shape[1])
    ):
        raise ValueError("Ordered terminal journal count differs from prefetch and final admissions")
    for row, count in enumerate(counts):
        verify_assignments(start[row], layout[row], before_valid[row], valid[row], src[row, :count], dst[row, :count])
    return state.batch


def mutable_buffers(state, fused):
    """Save the mutable tensors restored by reset() after preparing one frozen round.

    Queue state and transport leases are live throughout. The query, key lengths,
    tail bounds, address-table validity and other read-only inputs stay bound.
    """
    if fused:
        targets = [
            getattr(state.storage, name)[1:] for name in ("keys", "scales", "addresses", "logical_ids", "hot_valid")
        ]
        targets.extend((state.output, state.rotating.lru_slots, state.rotating.reserved_hot))
    else:
        targets = [
            getattr(state.native, name) for name in ("forward", "reverse", "cursors", "output", "preferred", "bitmap")
        ]
    return targets


def mutable_snapshots(state, fused):
    return [(target, target.clone()) for target in mutable_buffers(state, fused)]


def read_selected(state, inputs, fused):
    """Verify public IDs and the Index slots used by each Q, outside its event."""
    batch = state.batch
    shape = (batch, 4, 2048)
    if fused:
        state.check_completion()
        terminal = state.forward.terminal
        counts = state.rotating.final_posts[:, :, 0].cpu().numpy()
        from correctness import validate_terminal_receipts

        status = terminal.owner.core_status.cpu().numpy().view(np.uint64)
        validate_terminal_receipts(counts, status, terminal.queue_views)
        prefetch = state.rotating.rows[:, 9].cpu().numpy()
        if np.any(prefetch < 0) or np.any(prefetch > 2048):
            raise ValueError("Prefetch exceeded the per-request reservation")
        traffic = dict(prefetch=int(prefetch.sum()), final=int(counts.sum()))
        raw = state.workspace.topk.cpu().numpy().reshape(shape)
        unpack = state.workspace.inputs.unpack_indices.cpu().numpy().reshape(-1)
        if np.any(raw < 0) or np.any(raw >= state.history) or not np.array_equal(np.sort(unpack), np.arange(batch * 4)):
            raise ValueError("Invalid raw fused TopK physical positions or query map")
        logical = state.input_layout[np.arange(batch)[:, None, None], raw]
        expected = logical.reshape(-1, 2048)[unpack].reshape(shape)
        emitted = state.update.logical_output.cpu().numpy().reshape(shape)
        if not np.array_equal(np.sort(expected, axis=-1), np.sort(emitted, axis=-1)):
            raise ValueError("Published and raw fused TopK sets disagree")
        slots = state.update.slot_output.cpu().numpy().reshape(shape)
        if np.any(slots < 0) or np.any(slots >= state.stride):
            raise ValueError("Fused TopK output contains an invalid slot")
        rows = np.arange(1, batch + 1)[:, None, None]
        safe = np.minimum(slots, state.resident - 1)
        valid = state.storage.hot_valid.cpu().numpy()[rows, safe] != 0
        mapped = state.storage.logical_ids.cpu().numpy()[rows, safe]
        if np.any((slots < state.resident) & valid & (mapped != emitted)):
            raise ValueError("Committed hot-slot Index differs from its emitted token")
        verify_abc(slots, emitted, inputs.tail_bounds, state.resident, state.stride)
        # An Index-only exchange may retire a source hot slot while this call
        # still reads its old bytes. Only accept that exact pre-call mapping,
        # and read the complete KV bytes for every such (slot, token) pair.
        retired = (slots < state.resident) & ~valid
        traffic["retired_hot_outputs_verified"] = int(np.count_nonzero(retired))
        for row in np.flatnonzero(retired.any(axis=(1, 2))):
            old_slots = slots[row][retired[row]]
            old_ids = emitted[row][retired[row]]
            if not np.all(inputs.captured_hot_valid[row, old_slots]) or not np.array_equal(
                state.input_layout[row, old_slots], old_ids
            ):
                raise ValueError(f"Row {row} output used an invalidated slot without the original hot token")
            positions = np.unique(old_slots)
            index = state.torch.from_numpy(positions.astype(np.int64)).to(state.device)
            actual = state.output[row + 1, index].cpu().numpy()
            expected_bytes = state.pages.token_payload(row, state.input_layout[row, positions])
            if not np.array_equal(actual, expected_bytes):
                raise ValueError(f"Row {row} selected an invalidated slot with incorrect KV bytes")
    else:
        counts = state.native.counts.cpu().numpy()
        if (
            counts.shape != (batch + 1,)
            or np.any(counts[:-1] < 0)
            or np.any(counts[:-1] > 8192)
            or counts[-1] != np.sum(counts[:-1])
        ):
            raise ValueError("Native per-row fetch counts disagree with the total")
        traffic = dict(fetch=int(counts[-1]))
        emitted = state.selected.cpu().numpy().reshape(shape)
        if np.any(emitted < 0) or np.any(emitted >= state.history):
            raise ValueError("Strong serial produced an invalid token")
        slots = state.native.forward.cpu().numpy()[np.arange(batch)[:, None, None], emitted]
        exported = state.native.destinations.cpu().numpy().reshape(shape)
        if (
            np.any(slots < 0)
            or np.any(slots >= state.stride)
            or not np.array_equal(np.sort(slots, axis=-1), np.sort(exported, axis=-1))
        ):
            raise ValueError("Strong serial output slots differ from its committed token map")
    sorted_ids = np.sort(emitted, axis=-1)
    valid = np.take_along_axis(inputs.logical_score_valid, sorted_ids, axis=2)
    if (
        np.any(sorted_ids >= inputs.query_ends[..., None])
        or not np.all(valid)
        or np.any(np.diff(sorted_ids, axis=-1) == 0)
    ):
        raise ValueError("TopK contains duplicate or non-causal tokens")
    return sorted_ids.astype(np.int32), traffic


def li_intervals(words, batch, mhz):
    if not mhz or not math.isfinite(mhz) or mhz <= 0:
        raise ValueError("Missing NPU clock calibration for LI diagnostics")
    stamps = words[64 * batch : 128 * batch].reshape(batch, 4, 16)[:, :, :4]
    if np.any(stamps == 0) or np.any(stamps[:, :, 0] >= stamps[:, :, 1]) or np.any(stamps[:, :, 2] >= stamps[:, :, 3]):
        raise ValueError("Invalid per-Q LI2/LI3 diagnostic timestamps")
    from diagnostics import union

    details = {}
    active = []
    for col, label in ((0, "LI2"), (2, "LI3")):
        intervals = [(int(a), int(b)) for a, b in stamps[:, :, col : col + 2].reshape(-1, 2)]
        merged = union(intervals)
        active.append(merged)
        details[label + "_active_us"] = sum(b - a for a, b in merged) / mhz
        details[label + "_wall_span_us"] = (max(b for _, b in merged) - min(a for a, _ in merged)) / mhz
    both = union(active[0] + active[1])
    details["overlap_us"] = details["LI2_active_us"] + details["LI3_active_us"] - sum(b - a for a, b in both) / mhz
    return dict(clock_mhz=mhz, spans=details, per_query_cycle_stamps=stamps.tolist())


def last_row_windows(events, li):
    """Partition the last row owner's LI3 interval, using stamps from the same diagnostic replay."""
    stamps = np.asarray(li["per_query_cycle_stamps"], dtype=np.int64)
    batch = len(stamps)
    words = np.asarray(events, dtype=np.uint64)
    # The first two uint64s are service entry/exit. The pad holds rank entry/join.
    if words.size != 4 * batch * 16 or stamps.shape != (batch, 4, 4):
        return {"available": False, "reason": "window_record_shape"}
    events = words.reshape(4, batch, 16)
    row = int(stamps[:, 0, 3].argmax())
    start, finish = map(int, stamps[row, 0, 2:4])
    windows = events[:, row]
    if any(int(record[2]) >> 32 != epoch or (int(record[3]) & 0xffffffff) != 1
           for epoch, record in enumerate(windows)):
        return {"available": False, "reason": "window_epoch_or_completion"}
    previous = int(windows[0, 1])
    if not 0 < start <= previous <= finish:
        return {"available": False, "reason": "initial_window_clock"}
    segments = dict(initial_window=previous - start, cold_to_window=0, predecessor_wait=0, rank_and_join=0,
                    service_and_post=0)
    for record in windows[1:]:
        wait, rank, joined, service_end = map(int, record[[4, 13, 14, 1]])
        if not previous <= wait <= rank <= joined <= service_end <= finish:
            return {"available": False, "reason": "cold_window_clock"}
        segments["cold_to_window"] += wait - previous
        segments["predecessor_wait"] += rank - wait
        segments["rank_and_join"] += joined - rank
        segments["service_and_post"] += service_end - joined
        previous = service_end
    segments["retire"] = finish - previous
    mhz = li["clock_mhz"]
    return dict(available=True, row=row, span_us=(finish - start) / mhz,
                segments_us={name: ticks / mhz for name, ticks in segments.items()})


def tail_team_intervals(stamps, mhz):
    # One row holds one four-Q team until its last Q retires. Queue waits are
    # inside this interval; the time between row assignments is outside it.
    if stamps.ndim != 3 or stamps.shape[1:] != (4, 32) or len(stamps) not in FIXED_BATCHES or not mhz > 0:
        raise ValueError("Invalid terminal diagnostic shape or clock")
    starts, finishes = stamps[:, :, 0], stamps[:, :, 10]
    if np.any(starts <= 0) or np.any(finishes <= starts):
        raise ValueError("Incomplete terminal row timestamps")
    row_starts, row_finishes = starts.min(axis=1), finishes.max(axis=1)
    duration = row_finishes - row_starts
    span = int(row_finishes.max() - row_starts.min())
    last = int(row_finishes.argmax())
    return dict(
        observed_span_us=span / mhz,
        mean_assigned_teams=float(duration.sum()) / span,
        row_mean_us=float(duration.mean()) / mhz,
        row_max_us=int(duration.max()) / mhz,
        last_row=last,
        last_q=int(finishes[last].argmax()) + 1,
    )


def measured_replays(ex, proofs, serial, torch, states, sequence, config, directory, *, fused):
    import torch_npu
    from torch.profiler import record_function
    from reporting import timing_summary

    if len(states) != 1 or len(sequence.rounds) != 1 or config.warmup != WARMUP:
        raise ValueError("Expected one restorable graph and one frozen capture")
    (state,) = states
    inputs = sequence.inputs
    stream = torch.npu.current_stream(state.device)
    capture = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    resources = None
    pending = False

    state.reset()
    snapshots = mutable_snapshots(state, fused)
    initial_policies = deepcopy(state.policies)
    stream.synchronize()

    def restore():
        state.policies = deepcopy(initial_policies)
        for target, initial in snapshots:
            target.copy_(initial)
        torch.npu.current_stream(state.device).synchronize()

    class ObservedGraph:
        def replay(self):
            nonlocal pending
            pending = True
            graph.replay()

        def completed(self):
            nonlocal pending
            pending = False

    replay = ObservedGraph()
    if fused:
        terminal = state.forward.terminal
        if terminal is None or np.unique(terminal.queue_views).size != 4 * state.batch:
            raise ValueError("The fused operator must bind four independent terminal queues per request")
        if len(terminal.queue_views) != 4 * state.batch:
            raise ValueError("Unexpected number of per-query terminal queues")
        proofs["jettys_each"] = int(terminal.owner.context["jettys_per_core"])
        proofs["bound_private_queues"] = len(terminal.queue_views)
        expected_workers, expected_jettys = (32, (5 * state.batch + 31) // 32) if state.batch > 16 else (40, 2)
        if int(terminal.owner.context["worker_count"]) != expected_workers or proofs["jettys_each"] != expected_jettys:
            raise ValueError("Unexpected fused queue resource geometry")
        state.forward.set_diagnostics(False)
    selected_calls, measurements, traffic = [], [], []
    exact = 0
    try:
        with ExitStack() as stack:
            if fused:
                stack.enter_context(state.lease.capture_scope())
            stack.enter_context(torch.npu.stream(capture))
            pending = True
            if not fused:
                resources = stack.enter_context(
                    ex.capture_native_get(state.native.owner.session.runtime, capture, state.forward)
                )
                restore()
            with torch.npu.graph(graph, stream=capture):
                state.forward()
            capture.synchronize()
            pending = False
        if fused and hasattr(state.lease, "retain_graph"):
            state.lease.retain_graph(graph)
        for _ in range(WARMUP):
            restore()
            replay.replay()
            stream.synchronize()
            replay.completed()
            if fused:
                state.check_completion()
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        elapsed = []
        scopes = [f"large_batch_graph_{i:02d}" for i in range(CALLS)]
        with profiler:
            for i, scope in enumerate(scopes):
                restore()
                begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                with record_function(scope):
                    begin.record(stream)
                    replay.replay()
                    end.record(stream)
                    end.synchronize()
                    replay.completed()
                elapsed.append(begin.elapsed_time(end) * 1000)
                if i == 0:
                    state.validate(inputs)
                ids, requests = read_selected(state, inputs, fused)
                traffic.append(requests)
                if fused:
                    exact += check_scores(inputs, ids, serial[i])
                elif i:
                    check_scores(inputs, ids, selected_calls[0])
                selected_calls.append(ids)
        trace = Path(directory) / "measured_trace.json"
        profiler.export_chrome_trace(str(trace.resolve()))
        data = json.loads(trace.read_text())
        events = data["traceEvents"] if isinstance(data, dict) else data
        for scope, event_us in zip(scopes, elapsed):
            account = large_account(events, scope, event_us, fused=fused, batch=state.batch)
            measurements.append(stage_sample(account, fused))
        if fused:
            mhz = state.forward.calibrate_counter_mhz()
            state.forward.set_diagnostics(True)
            restore()
            # Confirm the restored device state, take a private copy of the old
            # logical map, and mark all journal entries stale until the kernel
            # writes the actual prefetch and terminal assignments.
            before = state.storage.logical_ids[1:].cpu().numpy().copy()
            before_valid = state.storage.hot_valid[1:].cpu().numpy().copy()
            if not np.array_equal(before, inputs.captured_layout) or not np.array_equal(
                before_valid, inputs.captured_hot_valid
            ):
                raise ValueError("Diagnostic graph did not start from the frozen capture")
            state.forward.events_src.fill_(-1)
            state.forward.events_dst.fill_(-1)
            state.forward.times.zero_()
            terminal.times.zero_()
            stream.synchronize()
            replay.replay()
            stream.synchronize()
            replay.completed()
            state.check_completion()
            diagnostic_ids, _ = read_selected(state, inputs, True)
            check_scores(inputs, diagnostic_ids, serial[0])
            state.validate(inputs)
            proofs["ordered_Index_diagnostic_rows"] = check_ordered_index(state, before, before_valid)
            if state.batch > 16:
                handoff = terminal.li_ready[:, :4, 0].cpu().numpy()
                if np.any((handoff < 1) | (handoff > 2 + 2048)):
                    raise ValueError("Incomplete large-batch LI-to-terminal classification handoff")
                early = handoff >= 2
                proofs["prior_classified_queries"] = int(early.sum())
                proofs["prior_classified_rows"] = int(np.all(early, axis=1).sum())
                proofs["separate_LI_diagnostic"] = li_intervals(state.forward.times.cpu().numpy(), state.batch, mhz)
            # Keep the existing instrumented replay's raw per-request records.
            # The two arrays use different device ABIs and are timed only once.
            proofs["separate_window_event_u64"] = state.rotating.events.cpu().numpy().view("<u8").tolist()
            stamps = terminal.times.cpu().numpy()
            proofs["separate_terminal_query_stamps"] = stamps.tolist()
            proofs["separate_terminal_teams"] = tail_team_intervals(stamps, mhz)
            state.forward.set_diagnostics(False)
    except BaseException as error:
        if pending:
            ex.NativeFetchSession.quarantine("large_batch_stream_completion_unknown", error)
        raise
    finally:
        if not pending:
            graph.reset()
            if resources is not None:
                resources.close()

    proofs.update(ids=np.stack(selected_calls), samples=measurements, traffic=traffic, exact=exact)
    timing = timing_summary([v["graph"] for v in measurements])
    return dict(correctness="PASS", failures=[], rounds={inputs.source_round: timing}, all_rounds=timing)


def verify_physical_device(device, expected):
    """Ask ACL to resolve the process-visible ordinal without opening the device."""
    import ctypes as c

    acl = c.CDLL("libascendcl.so")
    acl.aclGetRecentErrMsg.restype = c.c_char_p

    def check(name, status):
        if status != 0:
            message = (acl.aclGetRecentErrMsg() or b"").decode(errors="replace")
            raise RuntimeError(f"{name}: rc={status} {message}")

    check("aclInit", acl.aclInit(None))
    logical, physical = c.c_int32(), c.c_int32()
    check(
        "aclrtGetLogicDevIdByUserDevId",
        acl.aclrtGetLogicDevIdByUserDevId(device, c.byref(logical)),
    )
    check(
        "aclrtGetPhyDevIdByLogicDevId",
        acl.aclrtGetPhyDevIdByLogicDevId(logical.value, c.byref(physical)),
    )
    if physical.value != expected:
        raise ValueError(
            f"CANN device {device} (logical {logical.value}) maps to physical NPU {physical.value}, "
            f"expected {expected}"
        )
    print(f"DEVICE user={device} logical={logical.value} physical={physical.value}", flush=True)


def selected_device(device, physical_npu):
    if device is None and physical_npu is None:
        npu = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if npu not in ("0", "3", "4") or os.environ.get("PEER_PHYSICAL_NPU", npu) != npu:
            raise ValueError("Set PEER_PHYSICAL_NPU and ASCEND_RT_VISIBLE_DEVICES to the selected physical NPU")
        return 0, int(npu)
    if device is None or physical_npu is None or device < 0 or physical_npu < 0:
        raise ValueError("Pass both --device and --physical-npu with nonnegative IDs")
    # Keep the probe's ACL initialization lifetime separate from torch_npu's.
    subprocess.run(
        [
            sys.executable, "-B", "-c",
            "from large_batch_pair import verify_physical_device; "
            "import sys; verify_physical_device(int(sys.argv[1]), int(sys.argv[2]))",
            str(device), str(physical_npu),
        ],
        cwd=Path(__file__).resolve().parent,
        check=True,
    )
    return device, physical_npu


def run(args):
    device, physical_npu = selected_device(args.device, args.physical_npu)
    import execution as ex
    from functools import partial
    import kvaf
    import kvip
    import torch
    import torch_npu

    repo = Path(__file__).resolve().parents[4]
    revision = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "-uno"], text=True).strip():
        raise ValueError("The checked-out measurement revision has tracked modifications")
    sdk_python = args.kvaf_root.resolve() / "build-npu/python"
    if not Path(kvaf.__file__).resolve().is_relative_to(sdk_python) or not Path(kvip.__file__).resolve().is_relative_to(
        repo / "python"
    ):
        raise ValueError("Unexpected KVAF or KVIP Python module path")
    kernel = args.kvaf_root / "build-npu/bin/kvaf_execute_batch_get_kernels.o"
    with kernel.open("rb") as stream:
        sdk_kernel_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.arm == "fused" and args.serial_result is None:
        raise ValueError("The fused arm requires the matching strong serial result")
    synthetic = args.synthetic_fixture is not None
    if synthetic:
        sequence, donors = load_synthetic(args.synthetic_fixture.resolve(), args.batch)
    else:
        sequence, donors = load(args.trace.resolve(), args.group, args.round, args.batch, args.cohort, args.prepared)
    inputs = sequence.inputs
    pages = ReplayPagePlan(sequence)
    fingerprint = input_digest(inputs, pages, donors)
    if args.device is not None:
        # Establish the checked device before get_device_name can lazily initialize Torch NPU.
        torch.npu.set_device(device)
    identity = dict(
        group=args.group,
        batch=args.batch,
        native=revision,
        sequence_sha256=fingerprint,
        physical_npu=physical_npu,
        host=os.uname().nodename,
        calls=CALLS,
        warmup=WARMUP,
        hardware=torch.npu.get_device_name(torch.device(f"npu:{device}")),
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        cann=str(getattr(torch.version, "cann", "")),
        kvaf_get_kernel_sha256=sdk_kernel_sha,
        physical_history=inputs.geometry["history"],
        token_bytes=inputs.geometry["token_bytes"],
        registered_pages=pages.page_count,
        query_end_minmax=[int(inputs.query_ends.min()), int(inputs.query_ends.max())],
        donor_rows=donors,
        protocol="one_restored_graph_per_device_event_with_profiler",
    )
    if synthetic:
        identity["input_kind"] = "synthetic"
    label = "synthetic" if synthetic else f"G{args.group}"
    print(
        f"INPUT {label} BS{args.batch} sha256={fingerprint} donor_rows={donors} pages={pages.page_count} "
        f"Q_ends_min/max={inputs.query_ends.min()}/{inputs.query_ends.max()}",
        flush=True,
    )
    if args.packed_ten:
        if args.serial_result is None or args.reuse_serial_archive or args.compare_archived_pair:
            raise ValueError("Packed measurements require the matching single-call serial oracle")
        identity["protocol"] = "ten_independent_states_one_graph_one_event_v1"
    if args.reuse_serial_archive:
        if args.arm != "serial":
            raise ValueError("Only the completed serial arm can be reused")
        stage_serial(args.reuse_serial_archive, args.output, identity, repo)
        return
    if not args.kvaf_server:
        raise ValueError("A KVAF server is required for a live measurement")
    fused = args.arm == "fused"
    serial = None
    previous = None
    if args.packed_ten:
        from fixed_q4_packed_archive import load_archived_oracle
        serial = load_archived_oracle(args.serial_result, identity, repo)
    elif fused:
        base = identity
        if args.compare_archived_pair:
            base, previous = compute_reference(args.serial_result.parent, identity, args.output)
        expected, reference = reference_identity(args.serial_result, base, repo)
        old = json.loads((args.serial_result / "ARM_RESULT.json").read_text())
        path = args.serial_result / "selected.npy"
        if (
            old["identity"] != expected
            or old["arm"] != "serial"
            or old["get_workers"] != 2
            or old["checked_queries"] != CALLS * args.batch * 4
        ):
            raise ValueError("Strong serial was measured on a different input, build, device or protocol")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != old["selection_sha256"]:
                raise ValueError("Strong serial output checksum mismatch")
        serial = np.load(path, mmap_mode="r", allow_pickle=False)
        if serial.shape != (CALLS, args.batch, 4, 2048):
            raise ValueError("Strong serial output shape mismatch")
    cfg = configuration(
        [
            "bench" if args.packed_ten else "diagnose",
            "--transport",
            "urma",
            *(["--input", "synthetic"] if synthetic else [
                "--input", "real", "--trace-root", str(args.trace),
                "--group", str(args.group), "--round-range", "1:1",
            ]),
            "--device",
            str(device),
            "--kvaf-root",
            str(args.kvaf_root),
            "--kvaf-server",
            args.kvaf_server,
            "--prefetch-rank-start",
            "1024",
            "--prefetch-tokens",
            "1024",
            "--intermediate-fetches",
            "4",
            "--warmup",
            str(WARMUP),
            "--iterations",
            str(CALLS) if args.packed_ten else "1",
            *([] if args.packed_ten else ["--level", "1"]),
            "--output",
            str(args.output),
        ]
    )
    proofs = {}
    scratch_existed = (args.output / "export_only_prof_dir").exists()
    if args.packed_ten:
        from fixed_q4_packed_worker import measure_packed, save_packed
        benchmark = partial(measure_packed, ex, proofs, serial)
    else:
        benchmark = partial(measured_replays, ex, proofs, serial)
    result = ex.run_path(
        cfg, args.output / "path", sequence, pages, fused=fused, measure=benchmark,
        preserve_initial_lru=synthetic,
    )
    workers = (32 if args.batch > 16 else 40) if fused else 2
    if result["correctness"] != "PASS" or result["failures"] or result["runtime_workers"] != workers:
        raise ValueError(f"{args.arm}: production replay failed or used incorrect worker count")
    if args.packed_ten:
        save_packed(args, identity, inputs, proofs, workers, scratch_existed)
        return
    values = proofs["samples"]
    if len(values) != CALLS:
        raise ValueError("Incomplete timed replay")
    times = {k: [mean(v[k] for v in values), median(v[k] for v in values)] for k in values[0]}
    ids = args.output / "selected.npy"
    np.save(ids, proofs.pop("ids"), allow_pickle=False)
    with ids.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    checked = CALLS * args.batch * 4
    summary = dict(
        identity=identity,
        arm=args.arm,
        mean_p50_us=times,
        get_workers=workers,
        checked_queries=checked,
        checked_rows=CALLS * args.batch,
        selection_sha256=sha,
        full_KV_Index_checked_calls=1,
        full_KV_Index_checked_rows=args.batch,
        checks="per-call TopK and output-slot Index; complete selected KV bytes on first measured call",
        **proofs,
    )
    if fused:
        if reference is not None:
            summary["serial_reference"] = reference
        summary["vs_strong_serial_saved_mean_p50_us"] = {
            k: [a - b for a, b in zip(old["mean_p50_us"][k], times[k])] for k in ("graph", "LI", "tail")
        }
        summary["vs_strong_serial_graph_gain_pct"] = [
            100 * (a - b) / a for a, b in zip(old["mean_p50_us"]["graph"], times["graph"])
        ]
        if previous is not None:
            summary["last_row_LI3_windows"] = last_row_windows(
                proofs["separate_window_event_u64"], proofs["separate_LI_diagnostic"]
            )
            summary["comparison_archive"] = str(args.serial_result.parent)
            old_times = previous["mean_p50_us"]
            summary["vs_previous_fused_saved_mean_p50_us"] = {
                k: [a - b for a, b in zip(old_times[k], times[k])] for k in ("graph", "LI1", "LI23", "LI", "tail")
            }
            summary["vs_previous_fused_graph_gain_pct"] = [
                100 * (a - b) / a for a, b in zip(old_times["graph"], times["graph"])
            ]
            summary["transfer_tokens_per_call"] = {
                name: {key: mean(call[key] for call in record["traffic"]) for key in ("prefetch", "final")}
                for name, record in (("previous", previous), ("candidate", proofs))
            }
    else:
        del summary["exact"]
    atomic_json(args.output / "ARM_RESULT.json", summary)
    if not args.keep_trace:
        (args.output / "path/measured_trace.json").unlink()
        # torch_npu creates this scratch folder in the worker's cwd. The
        # launcher gives each invocation its own newly created arm directory.
        scratch = args.output / "export_only_prof_dir"
        if (
            args.output.resolve() == Path.cwd()
            and scratch.is_dir()
            and not scratch.is_symlink()
            and not scratch_existed
        ):
            shutil.rmtree(scratch)
    tag = f"{label} BS{args.batch} {args.arm}"
    print(
        f"CHECK {tag} TopK_score=PASS queries={checked}/{checked} Index_slots=PASS "
        f"rows={CALLS * args.batch}/{CALLS * args.batch} full_KV_Index=PASS full_KV_rows={args.batch}/{args.batch}"
        + (
            f" vs_strong_serial_score=PASS exact_ID_sets={proofs['exact']}/{checked} ordered_Index_diagnostic={proofs['ordered_Index_diagnostic_rows']}/{args.batch}"
            if fused
            else ""
        )
    )
    print(f"TIMING {tag}", *(f"{k}_mean/P50_us={times[k][0]:.2f}/{times[k][1]:.2f}" for k in ("graph", "LI", "tail")))
    if args.prepared is not None and fused:
        stage_names = ("LI1", "LI2", "LI3") if args.batch <= 16 else ("LI1", "LI23")
        print(f"TIMING {tag} measured_kernels",
              *(f"{k}_mean/P50_us={times[k][0]:.2f}/{times[k][1]:.2f}" for k in stage_names))
    if previous is not None:
        print(
            f"TIMING {tag} graph_boundaries",
            *(
                f"{name}_{key}_mean/P50_us={v[key][0]:.2f}/{v[key][1]:.2f}"
                for name, v in (("previous", previous["mean_p50_us"]), ("candidate", times))
                for key in ("inter_kernel_gap", "event_minus_span")
            ),
        )
        print(
            f"TRAFFIC {tag}",
            *(
                f"{name}_prefetch/final_tokens_per_call={v['prefetch']:.2f}/{v['final']:.2f}"
                for name, v in summary["transfer_tokens_per_call"].items()
            ),
        )
    else:
        print(
            f"TRAFFIC {tag}",
            *(
                f"{k}_tokens_per_call={mean(v[k] for v in proofs['traffic']):.2f}"
                for k in (("prefetch", "final") if fused else ("fetch",))
            ),
        )
    if fused:
        if "separate_LI_diagnostic" in proofs:
            spans = proofs["separate_LI_diagnostic"]["spans"]
            print(f"DIAGNOSTIC {tag} separate_calls=1", *(f"{k}={v:.2f}" for k, v in spans.items()),
                  f"prior_classified_queries={proofs['prior_classified_queries']}/{args.batch * 4} "
                  f"prior_classified_rows={proofs['prior_classified_rows']}/{args.batch}")
        if previous is not None:
            windows = summary["last_row_LI3_windows"]
            if windows["available"]:
                print(f"DIAGNOSTIC {tag} LI3_last_row={windows['row']} span_us={windows['span_us']:.2f}",
                      *(f"{k}_us={v:.2f}" for k, v in windows["segments_us"].items()))
            else:
                print(f"DIAGNOSTIC {tag} LI3_last_row_unavailable={windows['reason']}")
            teams = proofs["separate_terminal_teams"]
            print(
                f"DIAGNOSTIC {tag} tail separate_calls=1 "
                f"mean_assigned_teams={teams['mean_assigned_teams']:.2f}/16 "
                f"row_mean/max_us={teams['row_mean_us']:.2f}/{teams['row_max_us']:.2f} "
                f"last_row={teams['last_row']} last_Q={teams['last_q']}"
            )
        print(
            f"SAVED {tag} vs_strong_serial",
            *(
                f"{k}_mean/P50_us={a:+.2f}/{b:+.2f}"
                for k, (a, b) in summary["vs_strong_serial_saved_mean_p50_us"].items()
            ),
            "graph_gain_mean/P50_pct=%+.2f/%+.2f" % tuple(summary["vs_strong_serial_graph_gain_pct"]),
        )
        if previous is not None:
            print(
                f"SAVED {tag} vs_previous_fused",
                *(
                    f"{k}_mean/P50_us={a:+.2f}/{b:+.2f}"
                    for k, (a, b) in summary["vs_previous_fused_saved_mean_p50_us"].items()
                ),
                "graph_gain_mean/P50_pct=%+.2f/%+.2f" % tuple(summary["vs_previous_fused_graph_gain_pct"]),
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("serial", "fused"), required=True)
    parser.add_argument("--packed-ten", action="store_true", help="Capture and time ten independent states in one graph")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--synthetic-fixture", type=Path, help="Prepared compact synthetic Q4 fixture directory")
    parser.add_argument("--group", type=int, choices=(*GROUPS, SYNTHETIC_GROUP), required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--batch", type=int, choices=FIXED_BATCHES, required=True)
    parser.add_argument("--cohort", type=int, default=0)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--device", type=int, help="CANN user-visible device ordinal for this worker")
    parser.add_argument("--physical-npu", type=int, help="Required physical NPU for the selected ordinal")
    parser.add_argument("--kvaf-root", type=Path, required=True)
    parser.add_argument("--kvaf-server")
    parser.add_argument("--reuse-serial-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial-result", type=Path)
    parser.add_argument("--compare-archived-pair", action="store_true")
    parser.add_argument(
        "--keep-trace", action="store_true", help="Keep the raw profiler timeline after successful attribution"
    )
    args = parser.parse_args(argv)
    if args.synthetic_fixture is not None:
        if args.group != SYNTHETIC_GROUP or args.round != 1 or args.cohort != 0:
            parser.error("Synthetic input requires group 0, round 1 and cohort 0")
        if args.prepared is not None or args.reuse_serial_archive or args.compare_archived_pair:
            parser.error("Synthetic input requires its own fixture and live serial oracle")
    elif args.group == SYNTHETIC_GROUP:
        parser.error("Group 0 is reserved for synthetic input")
    return args


if __name__ == "__main__":
    run(parse_args())
