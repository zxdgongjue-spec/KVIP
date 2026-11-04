# SPDX-License-Identifier: Apache-2.0
"""Resolver-only graphs: original-compatible input and identical service snapshot."""

import ctypes
import json
import re

import numpy as np


def timing(values):
    return dict(p50_us=float(np.median(values)), p99_us=float(np.percentile(values, 99)))


def kernel_times(events, name):
    pattern = re.compile(re.escape(name) + r"(?:_\d+)?$")
    return [
        float(e["dur"])
        for e in events
        if e.get("ph") == "X"
        and pattern.fullmatch(e.get("name", ""))
        and (not e.get("cat") or "kernel" in e["cat"].lower())
    ]


def run(torch, args, rotating, owner, topk, selected, reset, verify, folder):
    import torch_npu
    from runner import progress

    stream = torch.npu.current_stream(rotating.device)
    capture = torch.npu.Stream(device=rotating.device)
    variants = {"original": "resolve_original", "adapted": "resolve_adapted", "fixed": "resolve"}
    for entry in variants.values():
        function = getattr(rotating.library, "v9_" + entry)
        function.argtypes, function.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint32
        rotating.functions[entry] = function

    def drain_capture():
        # Capture may record without executing; complete any submitted queues
        # before restoring inputs. Byte verification is mandatory on replays.
        rotating.finish()
        stream.synchronize()
        rows = rotating.rows.cpu().numpy()
        if np.any(rows[:, [11, 211, 248]]) or np.any(rotating.row_status.cpu().numpy()[:, 0]):
            raise AssertionError("Resolver capture transport did not complete")
        owner.mark_completed(completion_verified=True)

    # Fixed post-LI state for causal A/B: drain prefetch once, then retain its
    # exact mapping, payload and counters. Only the resolver differs thereafter.
    reset()
    rotating.launch_compute()
    rotating.wait()
    stream.synchronize()
    if np.any(rotating.rows.cpu().numpy()[:, [11, 211]]):
        raise AssertionError("Snapshot prefetch failed")
    owner.mark_completed(completion_verified=True)
    snapshot = [
        (tensor, tensor.clone())
        for tensor in (topk, rotating.mapping, rotating.rows, rotating.changes, rotating.hot_valid, owner.output)
    ]
    stream.synchronize()
    compatible_topk = torch.from_numpy(selected.reshape(tuple(topk.shape))).to(topk.device, dtype=topk.dtype)
    report = {"status": "pass", "cases": {}}
    for case, labels in (
        ("original_compatible", ("original", "adapted", "fixed")),
        ("post_li_snapshot", ("adapted", "fixed")),
    ):
        progress(folder.parent, f"Resolver {case}: capture and alternating graph timing; variants={labels}")

        def prepare(case=case):
            reset()
            rotating._call("init")
            if case == "original_compatible":
                # Original v9 has no ABC/invalid-hot/LRU destination semantics.
                # All hot entries valid, no prefetch, same final TopK/DDR data.
                rotating.reserved_hot.fill_(-1)
                topk.copy_(compatible_topk)
            else:
                for target, saved in snapshot:
                    target.copy_(saved)
            stream.synchronize()

        graphs, times, counts = {}, {label: [] for label in labels}, {}
        expected = {}
        for label in labels:
            prepare()
            graph = torch.npu.NPUGraph()
            owner.graphs.append(graph)
            with torch.npu.graph(graph, stream=capture):
                rotating._call(variants[label])
            capture.synchronize()
            drain_capture()
            graphs[label] = graph

        def replay(label, measured, prepare=prepare, graphs=graphs, times=times, counts=counts, expected=expected):
            prepare()
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record(stream)
            graphs[label].replay()
            end.record(stream)
            end.synchronize()
            if measured:
                times[label].append(begin.elapsed_time(end) * 1000)
            # Network completion and payload/TopK verification are outside timing.
            rotating.finish()
            stream.synchronize()
            verify()
            count = int(rotating.rows[:, 247].cpu().long().sum())
            if label in counts and counts[label] != count:
                raise AssertionError("Fixed snapshot produced varying final Get counts")
            counts[label] = count
            if label != "original":
                for name in ("locations", "changes"):
                    values = getattr(rotating, name).cpu().numpy()
                    if name in expected and not np.array_equal(expected[name], values):
                        raise AssertionError(f"Adapted/fixed resolver {name} differs")
                    expected[name] = values.copy()

        for step in range(args.warmup + args.iterations):
            order = labels if step % 2 == 0 else tuple(reversed(labels))
            for label in order:
                replay(label, step >= args.warmup)
        if len(set(counts.values())) != 1:
            raise AssertionError(f"Resolver A/B Get work differs: {counts}")
        results = {label: dict(graph_event=timing(times[label]), final_reads=counts[label]) for label in labels}
        report["cases"][case] = results
        # Persist event timing before parsing the hardware profiler export.
        (folder / "resolver_report.json").write_text(json.dumps(report, indent=2))
        progress(folder.parent, f"Resolver {case}: separate device-kernel profiling; byte checks passed")
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        with profiler:
            for _ in range(3):
                for label in labels:
                    replay(label, False)
        path = folder / f"resolver_{case}_trace.json"
        profiler.export_chrome_trace(str(path.resolve()))
        data = json.loads(path.read_text())
        events = data["traceEvents"] if isinstance(data, dict) else data
        for label in labels:
            durations = kernel_times(events, f"v9_{variants[label]}_kernel")
            if len(durations) != 3:
                raise ValueError(f"Expected 3 device resolver spans for {label}, found {len(durations)}; trace={path}")
            results[label]["device_kernel"] = timing(durations)
        text = summary(report)
        (folder / "resolver_summary.log").write_text(text + "\n")
        print(text, flush=True)
    return report


def summary(report):
    lines = [
        "Resolver-only PASS: preparation/LI/CQ wait/KV verification excluded from resolver timing.",
        "Original comparison: all-hot-valid/no-ABC/no-prefetch; NOT an end-to-end replacement.",
        "Adapted vs fixed: identical post-LI TopK/mapping/payload snapshot; alternating execution.",
        "Device kernel distribution: 3 separately profiled replays/version; graph events use --iterations.",
        "case | version | device_p50_us | device_p99_us | graph_event_p50_us | final_reads",
    ]
    for case, rows in report["cases"].items():
        for label, row in rows.items():
            device = row.get("device_kernel", {})
            lines.append(
                f"{case} | {label} | {device.get('p50_us', '--')} | {device.get('p99_us', '--')} | "
                f"{row['graph_event']['p50_us']:.3f} | {row['final_reads']}"
            )
    return "\n".join(lines)
