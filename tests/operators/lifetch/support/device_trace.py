# SPDX-License-Identifier: Apache-2.0
"""Per-replay device timeline accounting and remote-address footprint evidence."""

import hashlib
import json
import re
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


def account(events, name, elapsed, *, full=False, fused_final=False):
    scopes = [e for e in events if e.get("name") == name and e.get("ph") == "X"]
    if len(scopes) != 1:
        return dict(error=f"Expected one CPU replay scope, found {len(scopes)}")
    scope = scopes[0]
    origin = Decimal(str(scope["ts"]))
    start, end = 0.0, float(scope["dur"])
    kernels = (dict(init="v9_init_kernel", li="v9_li_kernel", resolve="v9_resolve_kernel",
                    publish="kvop_direct_update_outputs_early_kernel",
                    apply="kvop_direct_apply_owned_kernel", fetch="v9_fetch_update_kernel")
               if full else {stage: f"v9_{stage}_kernel" for stage in STAGES})
    if full and any(e.get("name", "").lower().startswith("v9_streaming_resolve_update") for e in events):
        # This kernel contains both resolver and index update; reporting separate
        # kernel durations would invent a boundary that does not exist.
        kernels = dict(init="v9_init_kernel", li="v9_li_kernel",
                       resolve_update="v9_streaming_resolve_update")
        if not fused_final:
            kernels["fetch"] = "v9_fetch_update_kernel"
    if any(e.get("name", "").lower().startswith("v9_early_fetch_kernel") for e in events):
        kernels["early_fetch"] = "v9_early_fetch_kernel"
    groups = {stage: [] for stage in kernels}
    for event in events:
        n = event.get("name", "").lower()
        cat = (event.get("cat") or "").lower()
        # CANN may omit cat. Accept exact kernel names with numeric dispatch
        # suffixes, while excluding launch APIs and explicitly tagged CPU ops.
        stage = next(
            (
                key
                for key, prefix in kernels.items()
                if n == prefix or (n.startswith(prefix + "_") and re.fullmatch(r"(?:_\d+)+", n[len(prefix) :]))
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
        intervals_us=groups,
        inter_kernel_gap_us=span - busy,
        boundary_residual_us=residual,
        note="Boundary residual includes event-to-kernel edges and clock measurement error; not proven CPU idle",
    )


def decode_final_detail(owner_words, status_words):
    """Decode per-request diagnostic counters after synchronization."""
    detail = []
    for owner, status in zip(owner_words, status_words):
        if owner[4] and owner[11]:
            detail.append(dict(ticks=[int(x) for x in owner[4:12]],
                               safe=int(owner[12]), conflict=int(owner[13]), pending=int(owner[14]),
                               kvaf_ticks=[int(x) for x in status[8:16].view(np.uint64)]))
    return detail


def collect_core(torch, rotating, graph, reset, verify, rounds, folder, *, full=False, diagnostics=None, fused_final=False):
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
            from diagnostics import algorithm_before, algorithm_after
            algorithm_initial = algorithm_before(rotating)
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvop_{'full' if full else 'core'}_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            # Read diagnostic counters after the timed graph and correctness checks.
            owner_words = rotating.owners[:len(rows) * 32].cpu().numpy().view(np.uint64).reshape(len(rows), 16)
            status_words = rotating.row_status.cpu().numpy().view(np.uint32).reshape(len(rows), 32)
            detail = decode_final_detail(owner_words, status_words)
            samples.append(
                dict(
                    name=name,
                    detail=detail,
                    round=step,
                    us=begin.elapsed_time(end) * 1000,
                    prefetch=int(rows[:, 9].astype(np.uint64).sum()),
                    final=int(rows[:, 247].astype(np.uint64).sum()),
                    epochs=rows[:, 16 : 16 + rotating.rounds].astype(np.uint64).sum(axis=0).tolist(),
                )
            )
            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)
            if diagnostics is not None:
                samples[-1]["streaming_cycles"] = diagnostics()
    # Save events/counters before parsing, so collection never needs repeating
    # merely because a vendor trace schema differs.
    label = "full" if full else "core"
    (folder / f"{label}_samples.json").write_text(json.dumps(samples, indent=2))
    path = folder / f"{label}_trace.json"
    profile.export_chrome_trace(str(path.resolve()))
    trace = json.loads(path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account(events, sample["name"], sample["us"], full=full, fused_final=fused_final)
    result = dict(samples=samples, trace=str(path))
    (folder / f"{label}_attribution.json").write_text(json.dumps(result, indent=2))
    (folder / f"{label}_summary.log").write_text(core_summary(result, label=label) + "\n")
    return result


def core_summary(result, label="core"):
    lines = [f"{label.upper()} DEVICE TRACE: separate profiled pass; all values below are from the same replay."]
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
                                     "update_owned", "update_outputs", "apply_owned", "resolve_update_fetch")}
    examples = []
    for event in events:
        if event.get("ph") != "X":
            continue
        kernel = str(event.get("name", "")).lower()
        category = str(event.get("cat") or "").lower()
        match = re.fullmatch(r"kvip_ubmem_(?:656|2624)_(init|li|resolve|fetch|complete)_kernel(?:_\d+)*", kernel)
        update = re.fullmatch(r"kvip_ubmem_direct_(update_owned|update_outputs|apply_owned)_kernel(?:_\d+)*", kernel)
        streaming = re.fullmatch(r"v9_streaming_resolve_update_(?:656|2624)(?:_\d+)?", kernel)
        if not match and not update and not streaming:
            if "kvip_ubmem" in kernel and len(examples) < 12:
                examples.append((kernel, category))
            continue
        if category and any(value in category for value in ("cpu", "python", "runtime")):
            continue
        start = float(Decimal(str(event["ts"])) - origin)
        end = start + float(event.get("dur", 0))
        if start < 0 or end > duration:
            continue
        stages["resolve_update_fetch" if streaming else (match.group(1) if match else update.group(1))].append((start, end))
    required = ("init", "li", "resolve", "complete", "update_owned", "update_outputs", "apply_owned")
    if stages["resolve_update_fetch"]:
        required = ("init", "li", "resolve_update_fetch")
    if any(not stages[key] for key in required):
        return {"error": "Missing named UBmem device kernels", "found": {key: len(value) for key, value in stages.items()},
                "examples": examples}
    timings = {key: sum(end - start for start, end in union(value)) for key, value in stages.items()}
    intervals = union([interval for value in stages.values() for interval in value])
    span = intervals[-1][1] - intervals[0][0]
    busy = sum(end - start for start, end in intervals)
    overlap = sum(timings.values()) - busy
    timings["update"] = sum(timings[key] for key in ("update_owned", "update_outputs", "apply_owned"))
    return {"stages_us": timings, "intervals_us": stages, "kernel_span_us": span, "inter_kernel_gap_us": span - busy,
            "boundary_residual_us": elapsed - span, "overlap_us": overlap,
            "fetch_separate": bool(stages["fetch"]),
            "note": "resolve includes inline fetch when fetch_separate is false; boundary includes measurement edges"}


def collect_ubmem(torch, rotating, graph, reset, verify, rounds, folder, streaming=None):
    """Collect a separately profiled graph replay; never included in event timing."""
    import torch_npu
    from torch.profiler import record_function

    folder.mkdir(parents=True, exist_ok=True)
    stream = torch.npu.current_stream(rotating.device)
    samples = []
    trace_path = folder / "trace.json"
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=rounds, repeat=1),
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace_path.resolve())),
    )
    if streaming is not None:
        streaming.counter_clock_mhz = streaming.calibrate_counter_mhz()
        streaming.set_diagnostics(True)
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            from diagnostics import algorithm_before, algorithm_after
            algorithm_initial = algorithm_before(rotating)
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvip_ubmem_trace_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay(stream)
                end.record(stream)
                end.synchronize()
                torch.npu.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            samples.append({"name": name, "round": step, "us": begin.elapsed_time(end) * 1000,
                            "prefetch": int(rows[:, 9].astype(np.uint64).sum()),
                            "final": int(rows[:, 247].astype(np.uint64).sum())})
            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)
            if streaming is not None:
                samples[-1]["streaming"] = streaming.diagnostic_cycles()
            profile.step()
    if streaming is not None:
        streaming.set_diagnostics(False)
    (folder / "samples.json").write_text(json.dumps(samples, indent=2))
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account_ubmem(events, sample["name"], sample["us"])
        account = sample["accounting"]
        if "error" not in account and hasattr(rotating.library, "v9_fetch_workers") and not account["fetch_separate"] and not account["stages_us"].get("resolve_update_fetch"):
            account["error"] = "Separate fetch kernel expected but absent from trace"
    (folder / "attribution.json").write_text(json.dumps(samples, indent=2))
    for sample in samples:
        account = sample["accounting"]
        if "error" in account:
            print(f"UBMEM_TRACE_ERROR round={sample['round']} {account}; raw={trace_path}", flush=True)
            continue  # Invalid attribution is retained, never reported as stage timing.
        print(f"UBMEM_TRACE round={sample['round']} total_us={sample['us']:.3f} "
              + " ".join(f"{key}_us={value:.3f}" for key, value in account["stages_us"].items())
              + f" span_us={account['kernel_span_us']:.3f} gap_us={account['inter_kernel_gap_us']:.3f}"
              + f" boundary_us={account['boundary_residual_us']:.3f} overlap_us={account['overlap_us']:.3f}"
              + f" fetch_separate={account['fetch_separate']} prefetch={sample['prefetch']} final={sample['final']}",
              flush=True)
    print(f"UBMEM_TRACE_FILES {folder}", flush=True)


def collect_replay(torch, state, graph, sequence, validate, config, folder):
    """Profile a continuous sequence; never restore captured state per round."""
    import torch_npu
    from torch.profiler import record_function
    from diagnostics import algorithm_before, algorithm_after
    from reporting import CorrectnessRecorder, atomic_json

    folder.mkdir(parents=True, exist_ok=True)
    pipeline, rotating = state.forward, state.rotating
    stream = torch.npu.current_stream(state.device)
    checks = CorrectnessRecorder()
    # A complete sequence is the unit of reset/warmup, as in unprofiled bench.
    for repeat in range(config.warmup):
        state.reset()
        for inputs in sequence:
            state.prepare(inputs)
            stream.synchronize()
            graph.replay()
            stream.synchronize()
            checks.check(f'warmup={repeat} source_round={inputs.source_round}',
                         lambda: validate(inputs))
    pipeline.counter_clock_mhz = pipeline.calibrate_counter_mhz()
    pipeline.set_diagnostics(True)
    trace_path = folder / 'trace.json'
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0,
                                            active=config.iterations * len(sequence.rounds), repeat=1),
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace_path.resolve())))
    try:
        with profile:
            for repeat in range(config.iterations):
                state.reset()
                for inputs in sequence:
                    state.prepare(inputs)
                    stream.synchronize()
                    before = algorithm_before(rotating)
                    # TopK and prefetch positions refer to the *pre-call* Index
                    # layout, which update mutates during this replay.
                    before['logical_ids'] = state.storage.logical_ids[1:].cpu().numpy().copy()
                    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                    name = f'real_repeat_{repeat}_source_{inputs.source_round}'
                    with record_function(name):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                        end.synchronize()
                    good = checks.check(name, lambda: validate(inputs))
                    algorithm = algorithm_after(rotating, before)
                    totals = algorithm.get('totals', {})
                    sample = dict(name=name, round=repeat, source_round=inputs.source_round,
                                  scheduler_round=inputs.scheduler_round,
                                  identities=inputs.identities, us=begin.elapsed_time(end) * 1000,
                                  correctness='PASS' if good else 'FAIL', algorithm=algorithm,
                                  prefetch=totals.get('prefetch'), final=totals.get('final'))
                    try:
                        sample['streaming_cycles'] = pipeline.diagnostic_cycles()
                    except (ValueError, AssertionError) as error:
                        sample['diagnostic_error'] = str(error)
                    samples.append(sample)
                    atomic_json(folder / 'samples.json', samples)
                    profile.step()
    finally:
        pipeline.set_diagnostics(False)
    trace = json.loads(trace_path.read_text())
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    for sample in samples:
        sample['accounting'] = (account_ubmem(events, sample['name'], sample['us'])
                               if config.transport == 'ubmem' else
                               account(events, sample['name'], sample['us'], full=True, fused_final=True))
    # Each source round has its own repeat distribution. Never pool different
    # decode rounds under the same synthetic-case summary.
    for source_round in sorted({sample['source_round'] for sample in samples}):
        atomic_json(folder / f'round-{source_round}' / 'full_attribution.json', dict(
            samples=[sample for sample in samples if sample['source_round'] == source_round]))
    return dict(correctness=checks.status, failures=checks.failures,
                rounds=len(sequence.rounds), repeats=config.iterations, samples=len(samples),
                trace_path=str(trace_path))


def reparse_ubmem(root):
    """Recompute attribution from saved traces without running a device."""
    paths = sorted(root.rglob('trace.json'))
    if not paths:
        raise ValueError(f'No saved trace.json under {root}')
    for path in paths:
        samples_path = path.with_name('samples.json')
        if not samples_path.exists():
            continue
        trace = json.loads(path.read_text())
        events = trace['traceEvents'] if isinstance(trace, dict) else trace
        samples = json.loads(samples_path.read_text())
        print(f'SOURCE {path.parent}')
        for sample in samples:
            account = account_ubmem(events, sample['name'], sample['us'])
            sample['accounting'] = account
            if 'error' in account:
                print(f"ROUND {sample['round']} INVALID {account}")
            else:
                stages = account['stages_us']
                print(f"ROUND {sample['round']} init={stages['init']:.3f} LI={stages['li']:.3f} "
                      f"fused={stages['resolve_update_fetch']:.3f} gap={account['inter_kernel_gap_us']:.3f} us")
        path.with_name('attribution.json').write_text(json.dumps(samples, indent=2))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Offline UBmem trace re-attribution')
    parser.add_argument('--reparse-ubmem', type=Path, required=True)
    reparse_ubmem(parser.parse_args().reparse_ubmem)


def account_native(events, name, elapsed):
    """Attribute native graph kernels by explicit names, never launch order.

    LIDU already includes descriptor/address preparation. Unknown kernels remain
    explicit and make coverage incomplete instead of being charged to Get.
    """
    scopes = [event for event in events if event.get('name') == name and event.get('ph') == 'X']
    if len(scopes) != 1:
        return dict(error=f'Expected one native replay scope, found {len(scopes)}')
    scope = scopes[0]
    origin = Decimal(str(scope['ts']))
    groups, unknown = {}, []
    aliases = {
        'li': ('quantlightningindexer',),
        'lidu_prepare': ('a5lidufusedbatchgetpreparemtp',),
        'fetch': ('kvafexecutebatchgetaivkernelv2',),
    }
    for event in events:
        if event.get('ph') != 'X' or 'kernel' not in (event.get('cat') or '').lower():
            continue
        begin = float(Decimal(str(event['ts'])) - origin)
        end = begin + float(event.get('dur', 0))
        if begin < 0 or end > float(scope['dur']):
            continue
        normalized = re.sub('[^a-z0-9]', '', event.get('name', '').lower())
        stage = next((stage for stage, names in aliases.items()
                      if any(normalized.startswith(prefix) for prefix in names)), None)
        if stage is None:
            unknown.append(dict(name=event['name'], start_us=begin, end_us=end))
        else:
            groups.setdefault(stage, []).append((begin, end))
    missing = sorted(set(aliases) - groups.keys())
    stages = {stage: sum(end - begin for begin, end in union(intervals))
              for stage, intervals in groups.items()}
    intervals = union([interval for values in groups.values() for interval in values]
                      + [(row['start_us'], row['end_us']) for row in unknown])
    result = dict(stages_us=stages, intervals_us=groups, unclassified=unknown,
                  event_us=elapsed, missing_stages=missing,
                  note='LIDU includes descriptor/address preparation; no separate mapping boundary inferred')
    if intervals:
        busy = sum(end - begin for begin, end in intervals)
        span = intervals[-1][1] - intervals[0][0]
        result.update(kernel_span_us=span, inter_kernel_gap_us=span-busy,
                      boundary_residual_us=elapsed-span)
    if missing or unknown or not intervals:
        result['error'] = 'Native stage coverage incomplete; raw names retained for attribution'
    elif sum(stages.values()) - busy > .1 or elapsed - span < -2:
        result['error'] = 'Native stage overlap or event/trace interval mismatch'
    return result


def collect_native(torch, graph, reset, verify, rounds, folder):
    """Profile the unchanged native graph; synchronize only around whole replays."""
    import torch_npu
    from torch.profiler import record_function
    from reporting import atomic_json

    folder.mkdir(parents=True, exist_ok=True)
    stream = torch.npu.current_stream()
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False)
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f'kvop_native_{step}'
            with record_function(name):
                begin.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            samples.append(dict(name=name, round=step, us=begin.elapsed_time(end)*1000))
            atomic_json(folder / 'native_samples.json', samples)
            verify()
    trace_path = folder / 'native_trace.json'
    profile.export_chrome_trace(str(trace_path.resolve()))
    trace = json.loads(trace_path.read_text())
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    for sample in samples:
        sample['accounting'] = account_native(events, sample['name'], sample['us'])
    result = dict(path='serial', samples=samples, trace=str(trace_path))
    atomic_json(folder / 'native_attribution.json', result)
    return result
