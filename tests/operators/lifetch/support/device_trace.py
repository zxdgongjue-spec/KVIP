# SPDX-License-Identifier: Apache-2.0
"""Per-replay device timeline accounting and remote-address footprint evidence."""

import hashlib
import json
from decimal import Decimal

import numpy as np

STAGES = ("init", "li", "resolve", "fetch")


def footprint(addresses, token_bytes=656):
    a = np.asarray(addresses, dtype=np.uint64)
    unique = np.unique(a)
    result = dict(
        requests=len(a),
        unique=len(unique),
        address_set_sha256=hashlib.sha256(unique.tobytes()).hexdigest(),
        span_bytes=int(unique[-1] - unique[0]) + token_bytes if len(unique) else 0,
    )
    for size in (4096, 2097152):
        # Include straddling tokens; these are virtual regions, not physical pages.
        pages = np.unique(np.concatenate((a // size, (a + token_bytes - 1) // size)))
        result[f"regions_{size}"] = len(pages)
    gaps = np.diff(unique)
    result["sorted_gap_p50_bytes"] = float(np.median(gaps)) if len(gaps) else 0
    return result


def address_snapshot(rotating, remote, block_bases):
    rows = rotating.rows.cpu().numpy().view(np.uint32)
    mapping = rotating.mapping.cpu().numpy()
    prefetch = remote[mapping >= 0]
    blocks = rotating.blocks.cpu().numpy().reshape(len(rows), -1)
    offsets = rotating.offsets.cpu().numpy().reshape(len(rows), -1)
    final = []
    for row in range(len(rows)):
        n = int(rows[row, 247])
        ids = blocks[row, :n]
        if np.any(ids < 0) or np.any(ids >= len(block_bases)):
            raise ValueError("Final descriptor block out of range")
        final.extend((block_bases[ids] + offsets[row, :n].astype(np.uint64)).tolist())
    if len(prefetch) != int(rows[:, 9].astype(np.uint64).sum()):
        raise ValueError("Prefetch mapping count differs from requests; footprint inference invalid")
    return {
        "prefetch": footprint(prefetch),
        "final": footprint(final),
        "all": footprint(np.concatenate((prefetch, np.asarray(final, dtype=np.uint64)))),
    }


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def account(events, name, elapsed):
    scopes = [e for e in events if e.get("name") == name and e.get("ph") == "X"]
    if len(scopes) != 1:
        return dict(error=f"Expected one CPU replay scope, found {len(scopes)}")
    scope = scopes[0]
    origin = Decimal(str(scope["ts"]))
    start, end = 0.0, float(scope["dur"])
    groups = {stage: [] for stage in STAGES}
    for event in events:
        n = event.get("name", "").lower()
        cat = (event.get("cat") or "").lower()
        # CANN may omit cat. Accept exact kernel names with numeric dispatch
        # suffixes, while excluding launch APIs and explicitly tagged CPU ops.
        stage = next(
            (
                key
                for key in STAGES
                if n == f"v9_{key}_kernel"
                or (n.startswith(f"v9_{key}_kernel_") and n[len(f"v9_{key}_kernel_") :].isdigit())
            ),
            None,
        )
        if event.get("ph") != "X" or stage is None or (cat and "kernel" not in cat):
            continue
        timestamp = float(Decimal(str(event["ts"])) - origin)
        finish = timestamp + float(event.get("dur", 0))
        if not start <= timestamp or finish > end:
            continue
        groups[stage].append((timestamp, finish))
    if any(not value for value in groups.values()):
        return dict(
            error="Missing named device kernels; inspect raw trace, no ordinal guessing",
            found={k: len(v) for k, v in groups.items()},
        )
    durations = {k: sum(b - a for a, b in union(v)) for k, v in groups.items()}
    intervals = union([interval for values in groups.values() for interval in values])
    busy = sum(b - a for a, b in intervals)
    overlap = sum(durations.values()) - busy
    if overlap > 0.1:
        return dict(error="Cross-stage overlap prevents additive stage accounting", overlap_us=overlap)
    span = intervals[-1][1] - intervals[0][0]
    # Split the unexplained edge time: a constant lead or tail is replay
    # dispatch, while an internal gap would point at the stream itself.
    lead = intervals[0][0]
    tail = elapsed - intervals[-1][1] if elapsed is not None else None
    residual = elapsed - span if elapsed is not None else None
    if residual is not None and residual < -2:
        return dict(error="Event/trace interval mismatch", event_us=elapsed, kernel_span_us=span)
    return dict(
        event_us=elapsed,
        kernel_span_us=span,
        stages_us=durations,
        inter_kernel_gap_us=span - busy,
        lead_us=lead,
        tail_us=tail,
        boundary_residual_us=residual,
        note="Boundary residual includes event-to-kernel edges and clock measurement error; not proven CPU idle",
    )


def collect_core(torch, rotating, graph, reset, verify, rounds, folder):
    """Profile the existing core graph separately from primary timing."""
    import torch_npu
    from torch.profiler import record_function

    stream = torch.npu.current_stream(rotating.device)
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvop_core_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            samples.append(
                dict(
                    name=name,
                    round=step,
                    us=begin.elapsed_time(end) * 1000,
                    prefetch=int(rows[:, 9].astype(np.uint64).sum()),
                    final=int(rows[:, 247].astype(np.uint64).sum()),
                    epochs=rows[:, 16 : 16 + rotating.rounds].astype(np.uint64).sum(axis=0).tolist(),
                )
            )
    # Save events/counters before parsing, so collection never needs repeating
    # merely because a vendor trace schema differs.
    (folder / "core_samples.json").write_text(json.dumps(samples, indent=2))
    path = folder / "core_trace.json"
    profile.export_chrome_trace(str(path.resolve()))
    trace = json.loads(path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account(events, sample["name"], sample["us"])
    result = dict(samples=samples, trace=str(path))
    (folder / "core_attribution.json").write_text(json.dumps(result, indent=2))
    (folder / "core_summary.log").write_text(core_summary(result) + "\n")
    return result


def core_summary(result):
    lines = ["CORE DEVICE TRACE: separate profiled pass; all values below are from the same replay."]
    for row in result["samples"]:
        a = row["accounting"]
        if "error" in a:
            lines.append(f"round={row['round']} attribution unavailable: {a}")
            continue
        values = dict(
            a["stages_us"],
            gap=a["inter_kernel_gap_us"],
            lead=a["lead_us"],
            tail=a["tail_us"],
            boundary=a["boundary_residual_us"],
        )
        lines.append(
            f"round={row['round']} total_us={row['us']:.3f} "
            + " ".join(f"{key}_us={value:.3f}" for key, value in values.items())
            + f" sum_us={sum(values.values()):.3f} prefetch={row['prefetch']} final={row['final']}"
            + f" epochs={row['epochs']}"
        )
    return "\n".join(lines)
