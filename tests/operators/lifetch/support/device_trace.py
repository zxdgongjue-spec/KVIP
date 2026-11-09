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
    residual = elapsed - span if elapsed is not None else None
    if residual is not None and residual < -2:
        return dict(error="Event/trace interval mismatch", event_us=elapsed, kernel_span_us=span)
    return dict(
        event_us=elapsed,
        kernel_span_us=span,
        stages_us=durations,
        inter_kernel_gap_us=span - busy,
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
        values = dict(a["stages_us"], gap=a["inter_kernel_gap_us"], boundary=a["boundary_residual_us"])
        lines.append(
            f"round={row['round']} total_us={row['us']:.3f} "
            + " ".join(f"{key}_us={value:.3f}" for key, value in values.items())
            + f" sum_us={sum(values.values()):.3f} prefetch={row['prefetch']} final={row['final']}"
            + f" epochs={row['epochs']}"
        )
    return "\n".join(lines)


def account_ubmem(events, name, elapsed):
    """Attribute exact UBmem device names within a single profiled replay."""
    import re

    scopes = [event for event in events if event.get("name") == name and event.get("ph") == "X"]
    if len(scopes) != 1:
        return {"error": f"Expected one replay scope, got {len(scopes)}"}
    scope = scopes[0]
    origin = Decimal(str(scope["ts"]))
    duration = float(scope["dur"])
    stages = {stage: [] for stage in ("init", "li", "resolve", "fetch", "complete",
                                     "update_owned", "update_outputs", "apply_owned")}
    examples = []
    for event in events:
        if event.get("ph") != "X":
            continue
        kernel = str(event.get("name", "")).lower()
        category = str(event.get("cat") or "").lower()
        match = re.fullmatch(r"kvip_ubmem_(?:656|2624)_(init|li|resolve|fetch|complete)_kernel(?:_\d+)?", kernel)
        update = re.fullmatch(r"kvip_ubmem_direct_(update_owned|update_outputs|apply_owned)_kernel(?:_\d+)?", kernel)
        if not match and not update:
            if "kvip_ubmem" in kernel and len(examples) < 12:
                examples.append((kernel, category))
            continue
        if category and any(value in category for value in ("cpu", "python", "runtime")):
            continue
        start = float(Decimal(str(event["ts"])) - origin)
        end = start + float(event.get("dur", 0))
        if start < 0 or end > duration:
            continue
        stages[match.group(1) if match else update.group(1)].append((start, end))
    required = ("init", "li", "resolve", "complete", "update_owned", "update_outputs", "apply_owned")
    if any(not stages[key] for key in required):
        return {"error": "Missing named UBmem device kernels", "found": {key: len(value) for key, value in stages.items()},
                "examples": examples}
    timings = {key: sum(end - start for start, end in union(value)) for key, value in stages.items()}
    intervals = union([interval for value in stages.values() for interval in value])
    span = intervals[-1][1] - intervals[0][0]
    busy = sum(end - start for start, end in intervals)
    overlap = sum(timings.values()) - busy
    timings["update"] = sum(timings[key] for key in ("update_owned", "update_outputs", "apply_owned"))
    return {"stages_us": timings, "kernel_span_us": span, "inter_kernel_gap_us": span - busy,
            "boundary_residual_us": elapsed - span, "overlap_us": overlap,
            "fetch_separate": bool(stages["fetch"]),
            "note": "resolve includes inline fetch when fetch_separate is false; boundary includes measurement edges"}


def collect_ubmem(torch, rotating, graph, reset, verify, rounds, folder):
    """Collect a separately profiled graph replay; never included in event timing."""
    import torch_npu
    from torch.profiler import record_function

    folder.mkdir(parents=True, exist_ok=True)
    stream = torch.npu.current_stream(rotating.device)
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
    )
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvip_ubmem_trace_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay(stream)
                end.record(stream)
                end.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            samples.append({"name": name, "round": step, "us": begin.elapsed_time(end) * 1000,
                            "prefetch": int(rows[:, 9].astype(np.uint64).sum()),
                            "final": int(rows[:, 247].astype(np.uint64).sum())})
    (folder / "samples.json").write_text(json.dumps(samples, indent=2))
    trace_path = folder / "trace.json"
    profile.export_chrome_trace(str(trace_path.resolve()))
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account_ubmem(events, sample["name"], sample["us"])
        account = sample["accounting"]
        if "error" not in account and hasattr(rotating.library, "v9_fetch_workers") and not account["fetch_separate"]:
            account["error"] = "Separate fetch kernel expected but absent from trace"
    (folder / "attribution.json").write_text(json.dumps(samples, indent=2))
    for sample in samples:
        account = sample["accounting"]
        if "error" in account:
            print(f"UBMEM_TRACE_ERROR round={sample['round']} {account}; raw={trace_path}", flush=True)
            raise RuntimeError(f"UBmem trace attribution failed; raw trace saved at {trace_path}")
        print(f"UBMEM_TRACE round={sample['round']} total_us={sample['us']:.3f} "
              + " ".join(f"{key}_us={value:.3f}" for key, value in account["stages_us"].items())
              + f" span_us={account['kernel_span_us']:.3f} gap_us={account['inter_kernel_gap_us']:.3f}"
              + f" boundary_us={account['boundary_residual_us']:.3f} overlap_us={account['overlap_us']:.3f}"
              + f" fetch_separate={account['fetch_separate']} prefetch={sample['prefetch']} final={sample['final']}",
              flush=True)
    print(f"UBMEM_TRACE_FILES {folder}", flush=True)
