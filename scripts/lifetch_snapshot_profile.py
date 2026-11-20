#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a selected-group failure without importing the model or old transport pointers."""
import argparse
import hashlib
import importlib.util
import shutil
import json
import os
import re
from collections import Counter
from pathlib import Path
import subprocess
import struct
import sys
from contextlib import nullcontext
from types import SimpleNamespace


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")



def diagnostic_builder(root, output, shape):
    """Patch a private source copy; reserve unused rank-trace cache lines."""
    if shape["batch"] != 16 or shape["rounds"] > 8 or shape["splitQ"] != 1:
        raise ValueError("Cold scan stamps currently require BS16, splitQ=1, at most 8 windows")
    private = output / "diagnostic-source"
    for relative in ("csrc/operators/lightning_indexer_with_fetch", "csrc/operators/common"):
        shutil.copytree(root / relative, private / relative, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("build*", "__pycache__", "*.so", "*.o"))
    for relative in ("python/kvip/runtime/build.py", "csrc/operators/lidu/op_kernel/hot_slot_selection.h"):
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    def patch(relative, old, new):
        target = private / relative
        text = target.read_text()
        if text.count(old) != 1:
            raise ValueError(f"Unsupported source layout: {relative}: {old[:70]}")
        target.write_text(text.replace(old, new))
    base = "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/"
    marker = "    __aicore__ inline void RotateFetch(uint32_t epoch, uint32_t sequenceHalfWindow = 0) {"
    method = """    __aicore__ inline void StampCold(uint32_t slot) {
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        auto trace = reinterpret_cast<__gm__ uint64_t*>(ctx->incremental +
            runtimeShape_.batch*(512+4*2048*6+1024+(runtimeShape_.history+255)/256*32));
        trace[blockId_*32+slot] = GetSystemCycle();
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace+blockId_*32+slot),8);
    }
"""
    patch(base+"indexer_vector.h", marker, method+marker)
    patch(base+"indexer.h", "                vectorService.LoadSequenceSplitThresholds(batch);",
          "                vectorService.StampCold(30);\n                vectorService.LoadSequenceSplitThresholds(batch);")
    scan = "                ScanSequenceHalf(begin, end, shard, loop, scaleLoop, epoch + 1 == runtimeShape_.rounds);"
    patch(base+"indexer.h", scan,
          "                if ASCEND_IS_AIV { vectorService.StampCold(16+2*(epoch-1)); }\n"+scan+
          "\n                if ASCEND_IS_AIV { vectorService.StampCold(17+2*(epoch-1)); }")
    end = "        if ASCEND_IS_AIV {\n            vectorService.FreeEventID();\n        } else {\n            matmulService.FreeEventID();"
    patch(base+"indexer.h", end, end.replace("            vectorService.FreeEventID();",
          "            vectorService.FreeEventID();\n            if constexpr (Phase == SequenceLiPhase::Cold) vectorService.StampCold(31);"))
    spec = importlib.util.spec_from_file_location("lifetch_diagnostic_build", private / "python/kvip/runtime/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_libraries


def print_cold(raw, shape):
    rows = [struct.unpack_from("<32Q", raw, core*256) for core in range(64)]
    if any(not row[30] or row[31] < row[30] for row in rows):
        raise ValueError("Missing/nonmonotonic cold AIV stamps")
    origin = min(row[30] for row in rows)
    critical = max(range(64), key=lambda core: rows[core][31])
    row = rows[critical]
    print(f"Cold latest AIV={critical}; start={row[30]-origin}; end={row[31]-origin}; span={row[31]-row[30]} ticks")
    print("Scan window | Critical AIV scan/pipeline ticks | All AIV mean | All AIV max")
    for epoch in range(1, shape["rounds"]):
        at = 16+2*(epoch-1)
        durations = [r[at+1]-r[at] for r in rows]
        if any(r[at] == 0 or r[at+1] < r[at] for r in rows):
            raise ValueError("Missing cold scan window stamps")
        print(epoch, row[at+1]-row[at], f"{sum(durations)/64:.2f}", max(durations), sep=" | ")
    print("Scan includes pipeline waits; service windows overlap across cores. Raw per-core timestamps are retained.")


class Snapshot:
    def __init__(self, path):
        import torch
        self.root = Path(path)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != "lifetch_failure_v1" or len(self.manifest["groups"]) != 1:
            raise ValueError("Expected a single-group lifetch_failure_v1 snapshot")
        self.meta = self.manifest["groups"][0]
        self.payload = torch.load(self.root / self.meta["file"], map_location="cpu", weights_only=True)

    def tensor(self, name, phase="before"):
        import torch
        spec = self.meta["tensors"][name]
        canonical = self.meta["aliases"].get(name, name)
        raw = self.payload[phase][canonical]
        return raw.view(getattr(torch, spec["dtype"].removeprefix("torch."))).reshape(spec["shape"])

    def audit(self):
        import torch
        result = {"group": self.meta["group"], "shape": self.meta["shape"],
                  "captured_error": self.manifest["error"], "phases": {}}
        for phase in ("before", "after"):
            owners = self.tensor("inputs.row_owners", phase)
            valid = self.tensor("inputs.query_valid", phase).reshape(-1, 4)
            ends = self.tensor("inputs.query_ends", phase).reshape(-1, 4)
            topk = self.tensor("arguments.9", phase)
            history = self.meta["shape"]["history"]
            bad = ((topk < 0) | (topk >= history)) & valid.bool()[:, :, None]
            locations = bad.nonzero()[:16].tolist()
            result["phases"][phase] = {
                "status": self.tensor("update.status", phase).tolist(),
                "owners": owners.tolist(), "query_valid": valid.tolist(), "query_ends": ends.tolist(),
                "invalid_topk_count": int(bad.sum()),
                "invalid_topk_examples": [dict(row=r, q=q+1, position=p, token=int(topk[r,q,p]))
                                           for r,q,p in locations],
            }
        result["input_changed_in_operator"] = [
            name for name in ("inputs.query", "inputs.weights", "inputs.scale", "inputs.query_ends",
                              "inputs.query_valid", "inputs.row_owners", "inputs.block_table")
            if not torch.equal(self.tensor(name).view(torch.uint8), self.tensor(name, "after").view(torch.uint8))]
        return result


def collect(selector):
    import torch
    result = {}
    for prefix, obj in (("inputs", selector.inputs), ("index", selector.storage),
                        ("rotating", selector.rotating), ("update", selector.update),
                        ("forward", selector.forward), ("owner", selector.rotating.owner)):
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                result[f"{prefix}.{name}"] = value
    for i, value in enumerate(selector.rotating.arguments):
        if isinstance(value, torch.Tensor):
            result[f"arguments.{i}"] = value
    return result


def rebind_rows(value, batch, queue_args, owners, address_base, address_stride, output_stride, width):
    """Rebind Row's four pointers and retire old-session queue cursors (1024-byte ABI)."""
    import torch
    raw = value.contiguous().view(torch.uint8).reshape(-1).clone()
    for row in range(batch):
        words = raw[row * 1024:(row + 1) * 1024].view(torch.int64)
        args = queue_args[row * 2]
        owner = int(owners[row])
        words[:4] = torch.tensor([int(args[8]), int(args[9]), address_base + owner * address_stride * 8,
                                 int(args[6]) + owner * output_stride * width], dtype=torch.int64)
        # Row.pending, initialHead, finalPending and finalHead/Begin/End belong to the old session.
        for first, last in ((844, 856), (992, 996), (1000, 1024)):
            raw[row * 1024 + first:row * 1024 + last] = 0
    return raw.view(value.dtype).reshape(value.shape)


def run_case(args, snap, output):
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import acl
    import kvaf
    from kvip.adapters.kvaf import KvafGetLease, KvafQueueView
    from kvip.operators.lifetch.binding import LIFetchBinding
    from kvip.operators.lifetch.forward import LIFetchForward
    from kvip.operators.lifetch.index_update import DirectIndexUpdate
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    stream = torch.npu.Stream(device=device)
    torch.npu.set_stream(stream)
    build = args.kvaf_root / "build-npu"
    cann, headers = kvaf_build_paths(build)
    if args.detail_only:
        build_libraries = diagnostic_builder(args.kvip_root, output, snap.meta["shape"])
    library, = build_libraries(cache_dir=(output / "diagnostic-build" if args.detail_only else args.kvip_root / "build/failure-replay"), cann=cann,
                              headers=headers, soc=acl.get_soc_name())
    shape = snap.meta["shape"]
    width = shape["tokenBytes"]
    old = snap.tensor("index.addresses").numpy().view(np.uint64)
    # Equal old token addresses remain aliases; no pointer from the old process is dereferenced.
    unique = np.unique(old[old != 0])
    pages = max(1, (len(unique) + 127) // 128)
    runtime = kvaf.NpuRuntime(str(build / "bin/kvaf_execute_batch_get_kernels.o"), width, 128,
                             device_id=args.device, npu_core_count=40, jettys_per_core=2,
                             batch_put_kernel_object_path=str(build / "bin/kvaf_execute_batch_put_kernels.o"))
    runtime.link_to(args.kvaf_server)
    runtime.initialize((pages + 2) * width * 128, width * 128)
    print(f"DDR fresh session: {len(unique)} distinct token addresses; {pages} pages", flush=True)
    for first in range(0, pages, 64):
        ids = list(range(first, min(first + 64, pages)))
        token_ids = np.arange(first * 128, (first + len(ids)) * 128, dtype=np.uint64)
        # Identifiable bytes, independent of Q/Index scoring. This is not captured DDR KV.
        payload = ((token_ids[:, None] + np.arange(width)[None, :]) % 251).astype(np.uint8)
        data = torch.from_numpy(payload).to(device)
        runtime.batch_put(ids, data, [i * width * 128 for i in range(len(ids))], [width * 128] * len(ids))
        stream.synchronize()
    table = runtime.prepare_access_table(list(range(pages)))
    view = runtime.access_table_view(table, device="npu")
    bases = view.addresses.cpu().numpy().view(np.uint64)
    new = np.zeros(old.shape, dtype=np.uint64)
    occupied = old != 0
    indices = np.searchsorted(unique, old[occupied])
    new[occupied] = bases[indices // 128] + (indices % 128).astype(np.uint64) * np.uint64(width)

    allocated = {}
    def load(name):
        key = snap.meta["aliases"].get(name, name)
        if key not in allocated:
            allocated[key] = snap.tensor(name).to(device)
        return allocated[key]

    inputs = SimpleNamespace(**{n.split(".", 1)[1]: load(n) for n in snap.meta["tensors"] if n.startswith("inputs.")})
    storage = SimpleNamespace(**{n.split(".", 1)[1]: load(n) for n in snap.meta["tensors"] if n.startswith("index.")})
    storage.addresses.copy_(torch.from_numpy(new.view(np.int64)).to(device))
    hbm = load("owner.output")
    lease = KvafGetLease(torch, runtime, library, hbm, stream)
    owner = KvafQueueView(lease, hbm, view.addresses, view.valid)
    arguments = [load(f"arguments.{i}") for i in range(14)]
    # Scratch constructor baseline; captured variant restores these after binding.
    for i in (9, 12, 13):
        arguments[i].zero_()
    rotating = LIFetchBinding(
        torch, library, arguments, owner, batch=shape["batch"], history=shape["history"],
        resident=shape["resident"], intermediate_fetches=args.windows or shape["rounds"], aic_count=shape["aicCount"],
        service_aics=0 if args.windows else shape["serviceAics"], final_workers=shape["finalWorkers"],
        li_tiling="auto" if shape["splitQ"] else "request", token_bytes=width,
        prefetch_limit=args.prefetch_tokens if args.prefetch_tokens is not None else snap.meta["prefetch_limit"], prefetch_rank_start=args.prefetch_rank_start if args.prefetch_rank_start is not None else snap.meta["prefetch_rank_start"],
        query_ends=inputs.query_ends.reshape(shape["batch"],4), hot_valid=storage.hot_valid,
        row_owners=inputs.row_owners, token_addresses=storage.addresses,
        tail_bounds=load("rotating.tail_bounds"), output_stride=snap.meta["output_stride"],
    )
    update = DirectIndexUpdate(torch, keys=storage.keys, scales=storage.scales, addresses=storage.addresses,
                               logical_ids=storage.logical_ids, row_owners=inputs.row_owners,
                               rotating=rotating, query_inputs=inputs)
    forward = LIFetchForward(rotating, update)
    if not args.windows and {n: int(getattr(rotating.runtime_shape,n)) for n in shape} != shape:
        raise ValueError("Reconstructed tiling differs from captured tiling")
    captured_shape = shape
    shape = {n: int(getattr(rotating.runtime_shape,n)) for n in captured_shape}
    if shape["scanTokens"] != captured_shape["scanTokens"]:
        raise ValueError("Scan capacity changed between captured and replay configurations")
    write_json(output / "effective_config.json", dict(shape=shape, captured_shape=captured_shape,
        prefetch_tokens=args.prefetch_tokens if args.prefetch_tokens is not None else snap.meta["prefetch_limit"],
        prefetch_rank_start=args.prefetch_rank_start if args.prefetch_rank_start is not None else snap.meta["prefetch_rank_start"],
        scratch="fresh constructor state" if args.windows else "captured",
        incremental_chunk_capacity=16384,
        scan_lower_bound="previous per-Q consumed frontier; starts at resident; identical source in A/B"))
    selector = SimpleNamespace(inputs=inputs, storage=storage, rotating=rotating, update=update, forward=forward)
    targets = collect(selector)
    # These contain process pointers, descriptors, or transport completion state. Rebuild them.
    rebuilt = {"rotating.context", "rotating.bindings", "arguments.14", "update.bindings",
               "forward.bindings", "forward.workspace", "forward._context", "owner.device_args",
               "owner.core_status", "owner.addresses", "owner.valid", "owner.descriptors",
               "rotating.prefetch_status", "rotating.final_status", "rotating.fallback_addresses",
               "forward.packed_fetch", "forward.dense_shadow", "rotating.owners", "rotating.final_posts"}
    input_names = {n for n in targets if n.startswith(("inputs.", "index."))}
    input_names.update({f"arguments.{i}" for i in range(9)})
    input_names.update({"rotating.lru_slots", "rotating.reserved_hot", "rotating.tail_bounds",
                        "rotating.query_ends", "rotating.hot_valid", "rotating.row_owners",
                        "rotating.token_addresses", "owner.output"})
    queue_args = owner.device_args.cpu().view(torch.int64).reshape(-1, 14)
    prepared = []
    restored, missing = [], []
    captured = args.case.endswith("captured") and not args.windows
    for name, target in targets.items():
        if name.startswith("streaming.") or name in rebuilt:
            continue
        if name in ("index.addresses", "rotating.token_addresses"):
            value = torch.from_numpy(new.view(np.int64))
        elif name in snap.meta["tensors"] and (captured or name in input_names):
            value = snap.tensor(name)
            restored.append(name)
        else:
            value = target.detach().cpu().clone()
            if captured:
                missing.append(name)
        if name in ("rotating.rows", "forward._live_rows"):
            value = rebind_rows(value, shape["batch"], queue_args, inputs.row_owners.cpu(),
                                storage.addresses.data_ptr(), storage.addresses.stride(0),
                                snap.meta["output_stride"], width)
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Snapshot/replay tensor mismatch: {name}")
        prepared.append((name, target, value.to(device)))
    write_json(output / "bindings.json", {"restored": restored, "rebuilt": sorted(rebuilt), "missing": missing,
               "ddr": "fresh immutable synthetic identity payload; captured pointer equality preserved",
               "queue_reset_between_iterations": False,
               "row_rebinding": "four pointers and old-session pending/head fields; other row/TopK state retained"})

    def reset():
        for _, target, initial in prepared:
            target.copy_(initial)
        stream.synchronize()

    graph = None
    reset()
    if args.case.startswith("graph") and not args.detail_only:
        graph = torch.npu.NPUGraph()
        with lease.capture_scope():
            with torch.npu.graph(graph, stream=stream):
                forward()
        stream.synchronize()
        # Capture may execute kernels; never reset communication state afterwards.
    kernel_samples = {}
    samples = []
    for iteration in range(0 if args.detail_only else args.iterations + args.warmup):
        reset()
        mismatches = [name for name, target, initial in prepared if name in input_names
                      and not torch.equal(target.cpu().contiguous().view(torch.uint8),
                                          initial.cpu().contiguous().view(torch.uint8))]
        if mismatches:
            raise RuntimeError(f"Device inputs differ before launch: {mismatches}")
        measured = iteration >= args.warmup
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False) if measured else None
        with profiler if profiler is not None else nullcontext():
            if graph is None:
                forward()
            else:
                graph.replay()
            stream.synchronize()
        if measured:
            trace = output / f"trace-{iteration-args.warmup}.json"
            profiler.export_chrome_trace(str(trace.resolve()))
            summarize_kernels(trace, kernel_samples)
        status = update.status.cpu().tolist()
        topk = arguments[9].cpu()
        valid = inputs.query_valid.cpu().reshape(shape["batch"],4).bool()
        bad = ((topk < 0) | (topk >= shape["history"])) & valid[:, :, None]
        selected = topk[valid].contiguous().numpy().tobytes()
        row = dict(iteration=iteration, phase="profiled" if measured else "warmup", inputs_match=True, status=status,
                   invalid_topk=int(bad.sum()), topk_sha256=hashlib.sha256(selected).hexdigest(),
                   equals_captured_output=bool(torch.equal(topk, snap.tensor("arguments.9", "after"))))
        samples.append(row)
        write_json(output / "result.json", {"case": args.case, "samples": samples})
        print(json.dumps(row), flush=True)
        if any(status) or row["invalid_topk"]:
            torch.save({"topk": topk, "locations": rotating.locations.cpu(),
                        "rows": rotating.rows.cpu(), "query_errors": forward.query_errors.cpu()}, output / "failure.pt")
            return 1  # Worker exits without tearing down uncertain communication state.
    if not kernel_samples and not args.detail_only:
        raise RuntimeError("Profiler trace contains no recognized LI kernels; inspect trace-*.json")
    write_json(output / "kernel_times.json", kernel_samples)
    print("Kernel | mean us | min us | max us | N", flush=True)
    for name, values in sorted(kernel_samples.items()):
        print(f"{name} | {sum(values)/len(values):.2f} | {min(values):.2f} | {max(values):.2f} | {len(values)}", flush=True)
    # Separate instrumented replay: never mix these clocks with profiler durations.
    if graph is not None:
        graph.reset()
    if args.profile_only:
        lease.close()
        return 0
    forward.set_options(diagnostics=True)
    reset()
    detail_graph = torch.npu.NPUGraph()
    with lease.capture_scope():
        with torch.npu.graph(detail_graph, stream=stream):
            forward()
    stream.synchronize()
    reset()
    detail_graph.replay()
    stream.synchronize()
    # Save evidence before parsing: offline reporting must never require rerunning NPU.
    raw = rotating.events.cpu().contiguous().numpy().tobytes()
    (output / "window_events.bin").write_bytes(raw)
    write_json(output / "window_shape.json", shape)
    if args.detail_only:
        cold_raw = rotating.rank_progress.cpu().contiguous().numpy().tobytes()
        (output / "cold_scan.bin").write_bytes(cold_raw)
    status = update.status.cpu().tolist()
    topk = arguments[9].cpu()
    valid = inputs.query_valid.cpu().reshape(shape["batch"], 4).bool()
    invalid = int((((topk < 0) | (topk >= shape["history"])) & valid[:, :, None]).sum())
    write_json(output / "detail_check.json", dict(status=status, invalid_topk=invalid))
    if any(status) or invalid:
        raise RuntimeError("Instrumented replay failed completion/TopK bounds checks")
    windows = decode_windows(raw, shape)
    write_json(output / "window_ticks.json", windows)
    print_windows(windows)
    if args.detail_only:
        print_cold(cold_raw, shape)
    detail_graph.reset()
    lease.close()
    return 0


def decode_windows(raw, shape):
    stride = struct.calcsize("<QQIIII12Q")
    windows = []
    for epoch in range(shape["rounds"]):
        lanes = []
        for lane in range(shape["serviceLanes"]):
            at = (epoch * shape["serviceLanes"] + lane) * stride
            begin, end, core, ep, complete, skipped, wait, *rest = struct.unpack_from("<QQIIII12Q", raw, at)
            scan, candidate, post, bitmap, candidates, posted, db0, db1, inc0, inc1, counts = rest
            if not complete or skipped:
                continue
            if not (wait <= inc0 <= inc1 <= begin <= end) or end == 0:
                raise ValueError(f"Incomplete/nonmonotonic window stamps: epoch={epoch} lane={lane}")
            lanes.append(dict(lane=lane, core=core, wait_begin=wait,
                              incremental_begin=inc0, incremental_end=inc1, service_begin=begin, service_end=end,
                              wait_ticks=inc0-wait, incremental_ticks=inc1-inc0,
                              service_ticks=end-begin, transition_ticks=begin-inc1,
                              scan_ticks=scan, candidate_ticks=candidate, post_ticks=post,
                              topk_calls=counts & 65535, scanned=(counts >> 16) & 0xffffff,
                              candidates=counts >> 40, posted=posted))
        windows.append(dict(epoch=epoch, lanes=lanes))
    return windows


def print_windows(windows):
    lanes = [lane for window in windows for lane in window["lanes"]]
    if not lanes:
        raise ValueError("No completed window stamps")
    origin = min(lane["wait_begin"] for lane in lanes)
    print("Window critical service lane: latest service end per window; all times in device ticks.")
    print("Window | Lane/Core | Start | End | Wait | TopK | Transition | Service | Scan/filter/post | Calls | Scanned | Candidates | Posted")
    for window in windows:
        if not window["lanes"]:
            continue
        v = max(window["lanes"], key=lambda lane: lane["service_end"])
        print(window["epoch"], f"{v['lane']}/{v['core']}", v["wait_begin"]-origin, v["service_end"]-origin,
              v["wait_ticks"], v["incremental_ticks"], v["transition_ticks"], v["service_ticks"],
              f"{v['scan_ticks']}/{v['candidate_ticks']}/{v['post_ticks']}",
              v["topk_calls"], v["scanned"], v["candidates"], v["posted"], sep=" | ")
    print("Rows use one lane's matching timestamps, not independent maxima. Scan/filter/post are inside Service.")
    print("Overlapping windows must not be summed. Cold scan/pipeline intervals, when collected, are listed separately below.")


def summarize_kernels(path, result):
    trace = json.loads(path.read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    observed = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        category = (event.get("cat") or "").lower()
        observed[(name, category)] += 1
        # Match the existing LIFetch diagnostics parser: CANN exports can omit
        # cat. Exact kernel names and numeric dispatch suffixes remain usable;
        # explicitly classified CPU scopes and launch wrappers are excluded.
        if category and "kernel" not in category:
            continue
        if re.fullmatch(r"v9_(?:li_kernel|streaming_resolve_update)(?:_\d+)*", name.lower()):
            result.setdefault(name, []).append(float(event["dur"]))
    return observed


def print_trace_inventory(observed):
    print("Trace event inventory (not attributed kernel timings):")
    print("Name | Category | Count")
    # Prioritize LI-related names, then the most frequent remaining events.
    ordered = sorted(observed.items(), key=lambda item: (
        not any(word in item[0][0].lower() for word in ("li_kernel", "lightning", "streaming")),
        -item[1], item[0]))
    for (name, category), count in ordered[:20]:
        print(f"{name} | {category or '(missing)'} | {count}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--kvip-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--kvaf-root", type=Path, default=Path(__file__).resolve().parents[1] / "third_party/KVAF")
    p.add_argument("--kvaf-server", default="127.0.0.1:28885")
    p.add_argument("--device", type=int, default=4)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--timeout-seconds", type=int, default=600)
    p.add_argument("--output", type=Path)
    p.add_argument("--report-only", action="store_true",
                   help="Read existing trace JSON files from the positional path; no NPU or snapshot load")
    p.add_argument("--ab-windows", action="store_true", help="Same snapshot: 4/8 windows, original profiling plus private instrumented graph")
    p.add_argument("--windows", type=int, choices=(4,8))
    p.add_argument("--prefetch-tokens", type=int)
    p.add_argument("--prefetch-rank-start", type=int)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--detail-only", action="store_true", help="Capture window detail directly, without redoing kernel profiling")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--case", choices=("eager-captured", "graph-captured", "eager-clean", "graph-clean"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.report_only:
        paths = [args.snapshot] if args.snapshot.is_file() and args.snapshot.suffix == ".json" else [] if args.snapshot.is_file() else sorted(args.snapshot.rglob("trace-*.json"))
        detail_paths = [args.snapshot] if args.snapshot.name == "window_events.bin" else sorted(args.snapshot.rglob("window_events.bin")) if args.snapshot.is_dir() else []
        for raw in detail_paths:
            shape = json.loads(raw.with_name("window_shape.json").read_text())
            print_windows(decode_windows(raw.read_bytes(), shape))
            cold = raw.with_name("cold_scan.bin")
            if cold.exists():
                print_cold(cold.read_bytes(), shape)
        if detail_paths and not paths:
            return 0
        if not paths:
            p.error(f"No trace-*.json files under {args.snapshot}")
        samples = {}
        observed = Counter()
        for path in paths:
            observed.update(summarize_kernels(path, samples))
        if not samples:
            print_trace_inventory(observed)
            p.error("No recognized device LI kernels; event names/categories are printed above")
        print(f"Source: {args.snapshot}; traces={len(paths)}; profiling samples only")
        print("Kernel | mean us | min us | max us | N")
        for name, values in sorted(samples.items()):
            print(f"{name} | {sum(values)/len(values):.2f} | {min(values):.2f} | {max(values):.2f} | {len(values)}")
        return 0
    if args.output is None:
        p.error("--output is required for snapshot execution")
    if args.iterations < 1 or args.warmup < 0:
        p.error("iterations must be positive and warmup nonnegative")
    args.output.mkdir(parents=True, exist_ok=args.case is not None)
    sys.path[:0] = [str(args.kvip_root / "python"), str(args.kvaf_root / "build-npu/python")]
    if args.case:
        try:
            code = run_case(args, Snapshot(args.snapshot), args.output)
        except Exception as error:
            import traceback
            traceback.print_exc()
            write_json(args.output / "error.json", {"error": str(error)})
            code = 2
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    snap = Snapshot(args.snapshot)
    audit = snap.audit()
    write_json(args.output / "audit.json", audit)
    before = audit["phases"]["before"]
    print("SNAPSHOT", json.dumps(dict(group=audit["group"], shape=audit["shape"],
          prefetch_limit=snap.meta["prefetch_limit"], prefetch_rank_start=snap.meta.get("prefetch_rank_start"),
          active_rows=[dict(row=i, owner=before["owners"][i], ends=before["query_ends"][i])
                       for i,v in enumerate(before["query_valid"]) if any(v)])), flush=True)
    del snap
    if args.audit_only:
        return
    rows = []
    jobs = [(w, mode) for w in (4,8) for mode in ("profile", "detail")] if args.ab_windows else [(args.windows, "detail" if args.detail_only else "profile" if args.profile_only else "both")]
    for window, mode in jobs:
        case = "graph-captured"
        destination = args.output / (f"windows-{window}-{mode}" if args.ab_windows else case)
        destination.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), str(args.snapshot), "--case", case,
                   "--output", str(destination), "--device", str(args.device), "--kvaf-root", str(args.kvaf_root),
                   "--kvip-root", str(args.kvip_root), "--kvaf-server", args.kvaf_server,
                   "--iterations", str(args.iterations), "--warmup", str(args.warmup)]
        if mode == "detail":
            command.append("--detail-only")
        elif mode == "profile":
            command.append("--profile-only")
        if window:
            command.extend(["--windows", str(window)])
        for option, value in (("--prefetch-tokens", args.prefetch_tokens), ("--prefetch-rank-start", args.prefetch_rank_start)):
            if value is not None:
                command.extend([option, str(value)])
        print(f"RUNNING {case}; log={destination / 'worker.log'}", flush=True)
        with (destination / "worker.log").open("w") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds).returncode
            except subprocess.TimeoutExpired:
                code = 124
        result = destination / "result.json"
        row = {"case": destination.name, "exit": code, "result": json.loads(result.read_text()) if result.exists() else None}
        error_path = destination / "error.json"
        if error_path.exists():
            row["error"] = json.loads(error_path.read_text())
        rows.append(row)
        write_json(args.output / "summary.json", rows)
        if (destination / "kernel_times.json").exists():
            print((destination / "kernel_times.json").read_text(), flush=True)
        if (destination / "window_ticks.json").exists():
            print_windows(json.loads((destination / "window_ticks.json").read_text()))
        if (destination / "cold_scan.bin").exists():
            print_cold((destination / "cold_scan.bin").read_bytes(), json.loads((destination / "window_shape.json").read_text()))
        print(f"{case}: exit={code}; samples={len((row['result'] or {}).get('samples', []))}", flush=True)
        if code:
            print("FAIL:", row.get("error", {"exit": code}), flush=True)
            print("--- worker.log tail ---", flush=True)
            print("\n".join((destination / "worker.log").read_text(errors="replace").splitlines()[-40:]), flush=True)
        if code == 124:
            print("Timed out; stopping remaining device cases.", flush=True)
            break
    if args.ab_windows:
        print("AB kernel 2, original build; same snapshot, separate detailed graph passes")
        print("Windows | Mean us | Min us | Max us | N")
        for window in (4,8):
            path = args.output / f"windows-{window}-profile/kernel_times.json"
            if path.exists():
                values = [v for k, vs in json.loads(path.read_text()).items()
                          if re.fullmatch(r"v9_li_kernel_2(?:_\d+)?", k) for v in vs]
                if values:
                    print(f"{window} | {sum(values)/len(values):.2f} | {min(values):.2f} | {max(values):.2f} | {len(values)}")
    print(f"Results: {args.output.resolve()}")
    return 1 if any(row["exit"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
