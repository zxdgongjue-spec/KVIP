# SPDX-License-Identifier: Apache-2.0
"""Common real-snapshot timing; production graphs remain uninstrumented."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from kvip.diagnostics.archive import array_digest, cpu_array, file_digest, source_commit
from kvip.diagnostics.console import emit
from kvip.diagnostics.demand import DEMAND_ACCOUNTING, SnapshotDemand
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.pipeline_options import early_threshold_skipped
from kvip.diagnostics.prefetch_admission import ADMISSION_ACCOUNTING, prefetch_admission
from kvip.diagnostics.snapshot import (
    input_tensors,
    mapping_values,
    persistent_tensors,
    post_state,
    relocate_snapshot_addresses,
)


def phase(retained, name, step=None):
    """Host-only failure location; call outside the device event interval."""
    retained.setdefault("progress", {}).update(phase=name, step=step)


def buffer_addresses(bundle):
    """Record allocation metadata without reading or synchronizing the NPU."""
    rotating = getattr(bundle, "rotating", None)
    storage = getattr(bundle, "storage", None)
    forward = getattr(bundle, "forward", None)
    tensors = {"output": getattr(bundle, "output", None)}
    for name in ("keys", "scales", "addresses", "logical_ids", "hot_valid"):
        tensors[name] = getattr(storage, name, None)
    for name in ("context", "rows", "mapping", "locations", "destinations", "fallback_addresses", "row_owners"):
        tensors[name] = getattr(rotating, name, None)
    arguments = getattr(rotating, "arguments", ())
    if len(arguments) >= 14:
        tensors.update(query=arguments[0], compute=arguments[12], prefix=arguments[13], topk=arguments[9])
    for name in ("workspace", "dense_shadow", "packed_fetch", "flags"):
        tensors["streaming_" + name] = getattr(forward, name, None)
    result = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        start = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        result[name] = dict(
            address=hex(start), end=hex(start + size), bytes=size, shape=list(tensor.shape), dtype=str(tensor.dtype)
        )
    if rotating is not None and "compute" in result:
        shape = rotating.runtime_shape
        size = int(shape.batch) * 4 * int(shape.history) * 2
        start = int(result["compute"]["address"], 16)
        result["scoreGM"] = dict(
            address=hex(start),
            end=hex(start + size),
            bytes=size,
            shape=[int(shape.batch), 4, int(shape.history)],
            dtype="uint16",
            alias_of="compute",
        )
    return result


def statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timing samples must be finite positive microseconds")
    return dict(
        count=len(values),
        mean_us=float(values.mean()),
        p50_us=float(np.median(values)),
        min_us=float(values.min()),
        max_us=float(values.max()),
        samples_us=values.tolist(),
    )


def fused_reads(torch, rotating, update, transport, width, demand=None, *, window_limit=None, active_rows=None,
                admission_controls=None, skip_exhausted_prefetch=False, configured_row_limit=None):
    rows = cpu_array(torch, rotating.rows)
    status = cpu_array(torch, rotating.row_status)
    posts = cpu_array(torch, rotating.final_posts)
    if np.any(rows[:, [11, 211, 248]]) or np.any(status[:, 0]) or np.any(posts[:, :, 1:4]):
        raise AssertionError("Fusion has errors or outstanding transport work")
    if np.any(cpu_array(torch, update.status)):
        raise AssertionError("Index update rejected the completed fusion")
    prefetch, final = int(rows[:, 9].astype(np.int64).sum()), int(rows[:, 247].astype(np.int64).sum())
    if transport == "ubmem":
        completed = cpu_array(torch, rotating.ubmem_status)
        if np.any(completed[:, 0]) or not np.array_equal(completed[:, 1:4:2], completed[:, 2:5:2]):
            raise AssertionError("UBmem submitted/completed counts disagree")
        if (prefetch, final) != (int(completed[:, 1].sum()), int(completed[:, 3].sum())):
            raise AssertionError("Resolver and UBmem transfer counters disagree")
        if int(completed[:, 5].sum()) != (prefetch + final) * width:
            raise AssertionError("UBmem payload-byte accounting mismatch")
    mapping = mapping_values(torch, rotating, rows)
    selected = cpu_array(torch, rotating.arguments[9])
    resident = rotating.hot_valid.shape[1]
    unused = 0
    for row, topk in zip(mapping, selected):
        # ABC mappings use resident+8192+offset and were already resident;
        # only the 2048 reserved hot-slot mapping range represents prefetch.
        fetched = np.flatnonzero((row >= resident) & (row < resident + 2048))
        unused += int(np.count_nonzero(~np.isin(fetched, topk.reshape(-1))))
    if unused > prefetch:
        raise AssertionError("Unused prefetch exceeds completed prefetch transfers")
    result = dict(
        prefetch=prefetch,
        final=final,
        transfers=prefetch + final,
        bytes=(prefetch + final) * width,
        unused_prefetch=unused,
        unused_prefetch_bytes=unused * width,
        useful_transfers=prefetch + final - unused,
        useful_transfer_bytes=(prefetch + final - unused) * width,
    )
    if demand is not None:
        result.update(demand.classify_prefetch(mapping, selected))
        classified = result["prefetch_classified"]
        if classified > prefetch:
            raise AssertionError("Prefetch mappings exceed completed transfer count")
        result["prefetch_unclassified"] = prefetch - classified
    if window_limit is not None:
        result["prefetch_admission"] = prefetch_admission(
            rows, window_limit=window_limit, rounds=rotating.rounds, active_rows=active_rows,
            admission_controls=admission_controls,
            configured_row_limit=configured_row_limit,
        )
    if skip_exhausted_prefetch:
        events = cpu_array(torch, rotating.events)
        if transport != "urma" or events.dtype != np.uint8 or events.shape[-1] != 128:
            raise ValueError("Skipped-prefetch accounting requires the URMA Event ABI")
        result["capacity_skipped_row_windows"] = int(events.view(np.uint32)[..., 7].sum(dtype=np.uint64))
        result["suppression_scope"] = "Only scanned row/windows; capacity-skipped candidates were not enumerated"
    return result


def prepare_fused_state(bundle, archive, restored, torch, device, transport):
    selector, rotating, update = bundle.selector, bundle.rotating, bundle.update
    seeds = []
    for name, target in (persistent_tensors(selector) | input_tensors(selector)).items():
        seed = (
            torch.from_numpy(relocate_snapshot_addresses(archive, restored["new_base"]).copy()).to(device)
            if name == "addresses"
            else archive.tensor("pre." + name, torch, device)
        )
        if target.shape != seed.shape or target.dtype != seed.dtype:
            raise ValueError(f"Replay binding differs from captured tensor: {name}")
        seeds.append((target, seed))
    if getattr(rotating, "threshold_bounds", None) is not None:
        seeds.append((rotating.threshold_bounds, archive.tensor("pre.threshold_bounds", torch, device)))
    oracle = DecodeOracle(archive)
    demand = SnapshotDemand(archive)
    expected = archive.array("post.logical_output").reshape(archive.manifest["geometry"]["batch"] * 4, 2048)
    delta = restored["new_base"] - restored["old_base"]
    active_rows = np.any(archive.array("pre.query_ends").reshape(-1, 4) > 0, axis=1)
    score_reference = (
        archive.array("post.score_keys") if archive.manifest.get("content") == "real_li_synthetic_kv" else None
    )

    def reset():
        for target, seed in seeds:
            target.copy_(seed)
        # Production init resets round-local work; never rewind transport queues.

    def read_counts():
        if getattr(bundle, "auto_replay", None) is not None:
            bundle.auto_replay.observe()
        reads = fused_reads(
            torch,
            rotating,
            update,
            transport,
            archive.manifest["geometry"]["token_bytes"],
            demand,
            window_limit=bundle.tuning["window_limit"],
            active_rows=active_rows,
            admission_controls=(bundle.auto_replay.latest["controls"] if getattr(bundle, "auto_replay", None)
                                else bundle.tuning.get("admission_controls")),
            skip_exhausted_prefetch=bundle.tuning.get("skip_exhausted_prefetch", False),
            configured_row_limit=(bundle.auto_replay.state.full_controls[0]
                                  if getattr(bundle, "auto_replay", None) else None),
        )

        logical = cpu_array(torch, update.logical_output)
        if early_threshold_skipped(bundle.tuning) and reads["prefetch"] != 0:
            raise AssertionError("Low-miss skip issued speculative prefetch reads")
        ordered = logical.reshape(expected.shape).astype("<i4")
        if (
            array_digest(np.sort(ordered, axis=-1)) != bundle.selection_comparison["selected_set_sha256"]
            or array_digest(ordered) != bundle.selection_comparison["ordered_output_sha256"]
        ):
            raise AssertionError("Fused final ordered selection changed during timed replay")
        reads.update(demand.classify(logical))
        return reads

    def verify():
        if score_reference is not None:
            from kvip.diagnostics.score_trace import read_score_keys

            bundle.score_comparison = verify_score_keys(
                read_score_keys(torch, rotating), score_reference, archive.array("pre.query_ends")
            )
        from kvip.diagnostics.selection import verify_fused_scores

        selected_expected, selection = verify_fused_scores(archive, cpu_array(torch, rotating.arguments[9]))
        if bundle.selection_comparison and any(
            selection[key] != bundle.selection_comparison[key]
            for key in ("selected_set_sha256", "ordered_output_sha256")
        ):
            raise AssertionError("Fused ordered TopK selection changed across eager/graph runs")
        bundle.selection_comparison = selection
        reads = read_counts()

        from kvip.diagnostics.threshold_check import verify_initial_threshold

        bundle.threshold_comparison = verify_initial_threshold(torch, bundle, archive)
        state = post_state(torch, selector)
        oracle.verify(state, address_delta=delta, expected_topk=selected_expected)
        reads.update(demand.classify(state["logical_output"]))
        return reads

    bundle.reset, bundle.verify, bundle.read_counts = reset, verify, read_counts
    bundle.seeds, bundle.oracle = seeds, oracle
    bundle.selection_comparison = {}


def verify_score_keys(actual, expected, query_ends):
    """Check valid score bits outside timing; future/padding positions are undefined."""
    if actual.shape != expected.shape or actual.dtype != np.uint16 or expected.dtype != np.uint16:
        raise AssertionError("Replayed LI score layout differs from the captured real input")
    ends = np.asarray(query_ends).reshape(actual.shape[:2])
    compared = 0
    for row, query in np.ndindex(ends.shape):
        end = int(ends[row, query])
        if not 0 <= end <= actual.shape[2]:
            raise AssertionError("Captured query end exceeds the LI score row")
        mismatch = np.count_nonzero(actual[row, query, :end] != expected[row, query, :end])
        if mismatch:
            raise AssertionError(f"Real LI score keys differ: row={row} query={query} positions={mismatch}")
        compared += end
    return dict(status="bit_exact", compared_scores=compared, included_in_timing=False)


def measure(torch, kvaf, stream, bundle, args, retained, runtime=None):
    """Time the complete current graph without per-stage instrumentation."""
    graphs = retained.setdefault("graphs", [])
    reset, verify = bundle.reset, bundle.verify
    lease = bundle.lease

    def capture(call):
        graph = torch.npu.NPUGraph() if args.mode == "serial" else kvaf.AclGraph()
        graphs.append(graph)
        if lease is not None and hasattr(lease, "retain_graph"):
            lease.retain_graph(graph)
        if args.mode == "serial":
            # Native LI/gather allocate Torch outputs during capture. Use the
            # Torch graph pool, explicitly on the same stream as native Get.
            with torch.npu.graph(graph, stream=stream):
                call()
        else:
            graph.capture_begin(stream)
            call()
            graph.capture_end()
        return graph

    def replay(graph):
        if args.mode == "serial":
            graph.replay()
        else:
            graph.replay(stream)

    phase(retained, "eager_reset")
    reset()
    phase(retained, "eager_forward_and_sync")
    bundle.forward()
    stream.synchronize()
    phase(retained, "eager_verify")
    verify()
    phase(retained, "capture_prepare")
    reset()
    stream.synchronize()
    phase(retained, "graph_capture_and_fence")
    with lease.capture_scope() if lease is not None else nullcontext():
        graph = capture(bundle.forward)
    phase(retained, "first_replay_reset")
    reset()  # Graph capture may execute the body.
    phase(retained, "first_replay_and_sync")
    replay(graph)
    stream.synchronize()
    phase(retained, "first_replay_verify")
    verify()
    if getattr(args, "probe_only", False):
        from kvip.diagnostics.topk_probe import decode

        phase(retained, "topk_probe_reset")
        reset()
        bundle.forward.times.zero_()
        bundle.forward.set_options(diagnostics=True)
        stream.synchronize()
        phase(retained, "topk_probe_replay")
        replay(graph)
        stream.synchronize()
        phase(retained, "topk_probe_verify")
        verify()
        result = dict(topk_probe=decode(cpu_array(torch, bundle.forward.times), bundle.forward.batch))
        bundle.forward.set_options(diagnostics=False)
        stream.synchronize()
        for graph in reversed(graphs):
            graph.reset()
        return result
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    retained["events"] = [begin, end]
    samples, reads, auto_samples = [], [], []
    auto = getattr(bundle, "auto_replay", None)
    for step in range(args.warmup + args.iterations):
        phase(retained, "iteration_reset", step)
        reset()
        if auto is not None:
            auto.begin(step=max(0, step - args.warmup + 1), learning=step >= args.warmup)
        stream.synchronize()
        phase(retained, "warmup_replay" if step < args.warmup else "timed_replay", step)
        begin.record(stream)
        replay(graph)
        end.record(stream)
        end.synchronize()
        phase(retained, "iteration_read_counts", step)
        elapsed = begin.elapsed_time(end) * 1000
        counts = bundle.read_counts()
        if step >= args.warmup:
            samples.append(float(elapsed))
            reads.append(counts)
            if auto is not None:
                auto_samples.append(auto.record(counts))
    phase(retained, "final_verify")
    verify()  # Full score/Index/KV oracle after the final measured replay.
    result = dict(timing=statistics(samples), reads_samples=reads)
    if auto is not None:
        config = cpu_array(torch, auto.state.config)
        result["auto_route"] = dict(
            status="pass", samples=auto_samples, low_steps=sum(row["mode"] == "low" for row in auto_samples),
            high_steps=sum(row["mode"] == "high" for row in auto_samples),
            low_budget=int(config[5]), high_budget=int(config[6]), budget_units="Get_bytes * 1024 / active_score_positions",
            feedback_source="Previous successful device completion; no oracle counts uploaded",
            capture_and_warmup_learning=False, decision_and_feedback_in_timing=True,
            metadata_h2d_in_timing=False, snapshot_restored_each_step=True,
            cold_start_us=samples[0], steady_timing=statistics(samples[1:]) if len(samples) > 1 else None,
            scope="One frozen input with evolving device feedback, not a live multi-step request trajectory")
        if getattr(args, "check_auto_route", False):
            phase(retained, "auto_route_lifecycle")
            result["auto_route"]["lifecycle"] = check_auto_route(torch, stream, bundle, lambda: replay(graph), args.iterations)
    if (getattr(args, "check_mode_switch", False) and auto is None
            and early_threshold_skipped(getattr(bundle, "tuning", {}))):
        phase(retained, "mode_switch_check")
        result["mode_switch_check"] = check_mode_switch(torch, stream, bundle, lambda: replay(graph))
    diagnostic_count = getattr(args, "diagnostic_replays", 0)
    if diagnostic_count and args.mode == "fused":
        from kvip.diagnostics.native_profile import decode_profile

        diagnostic_samples = []
        for step in range(diagnostic_count):
            phase(retained, "diagnostic_reset", step)
            reset()
            bundle.forward.times.zero_()
            bundle.forward.set_options(diagnostics=True)
            stream.synchronize()
            phase(retained, "diagnostic_replay_and_sync", step)
            replay(graph)
            stream.synchronize()
            phase(retained, "diagnostic_verify", step)
            diagnostic_reads = verify()
            diagnostic_samples.append(
                dict(
                    **decode_profile(
                        cpu_array(torch, bundle.forward.times),
                        batch=bundle.forward.batch,
                        workers=bundle.forward.q_workers,
                        resolver_cores=bundle.forward.resolver_cores,
                        transport=args.transport,
                        threshold_markers=not bool(getattr(args, "upstream_root", None))
                        and args.transport == "urma"
                        and getattr(bundle, "tuning", {}).get("recent_profile", True),
                        early_prefetch_drain=getattr(bundle, "tuning", {}).get("early_prefetch_drain", False),
                    ),
                    reads=diagnostic_reads,
                )
            )
            # Only after successful completion; failures retain all owners and
            # the enabled profile buffer for the existing fatal-exit path.
            phase(retained, "diagnostic_disable", step)
            bundle.forward.set_options(diagnostics=False)
            stream.synchronize()
        result["native_diagnostics"] = dict(
            included_in_timing=False,
            units="raw_device_cycles",
            count=len(diagnostic_samples),
            samples=diagnostic_samples,
            note="Separate instrumented replay with barriers/cache operations; intervals overlap and cannot be summed into graph latency or converted to microseconds without a measured clock.",
        )
    if getattr(args, "profile_dir", None) is not None:
        from kvip.diagnostics.system_profile import collect_system_profiles

        variant = retained.get("progress", {}).get("variant", args.threshold_policy)
        result["system_profiles"] = collect_system_profiles(
            torch=torch,
            reset=reset,
            replay=lambda: replay(graph),
            synchronize=stream.synchronize,
            verify=verify,
            directory=args.profile_dir,
            case=f"{args.transport}-{args.mode}-{variant}",
            metrics=args.profile_metrics,
            metadata=retained["profile_metadata"],
            retained=retained,
            checkpoint=lambda name, step: phase(retained, name, step),
        )
    phase(retained, "graph_reset")
    for graph in reversed(graphs):
        graph.reset()
    return result


def allocate_context(args, archive, retained):
    """Import real DDR once; case graphs and leases are owned separately."""
    build = args.build_dir.resolve()
    sys.path.insert(0, str(build / "python"))
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    from kvip.diagnostics.ddr import import_pages
    from kvip.diagnostics.replay_runtime import create_runtime
    from kvip.runtime.build import kvaf_build_paths
    from kvip.runtime.device import query_device_resources
    from kvip.operators.lifetch_ubmem.build import kvaf_cann

    if not Path(kvaf.__file__).resolve().is_relative_to(build / "python/kvaf"):
        raise RuntimeError(f"Wrong KVAF module: {kvaf.__file__}")
    if not torch.npu.is_available():
        raise RuntimeError("Real NPU required; no CPU/simulator timing fallback")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    stream = torch.npu.Stream(device=device)
    torch.npu.set_stream(stream)
    retained["stream"] = stream
    geometry, ddr = archive.manifest["geometry"], archive.manifest["ddr"]
    cann = kvaf_cann(build) if args.transport == "ubmem" else kvaf_build_paths(build)[0]
    available = query_device_resources(cann, args.device)["effective_aic"]
    if available < geometry["aic_count"]:
        raise ValueError("Replay device has fewer AICs than captured geometry")
    aics = geometry["aic_count"]
    phase(retained, "runtime_create")
    runtime = create_runtime(args, archive, kvaf, torch, device, stream, cann, aics)
    retained["runtime"] = runtime
    phase(retained, "runtime_link")
    runtime.link_to(args.kvaf_server)
    phase(retained, "runtime_initialize")
    runtime.initialize(ddr["region_bytes"], ddr["slot_bytes"])
    phase(retained, "ddr_import")
    restored = import_pages(
        runtime, archive.directory / "ddr", ddr, torch_module=torch, target_transport=args.transport
    )
    retained["restored"] = restored
    if restored["access_table"] is None:
        raise ValueError("Real-history replay requires published DDR pages")
    phase(retained, "access_table")
    table = runtime.access_table_view(restored["access_table"], device="npu")
    retained["table"] = table
    return SimpleNamespace(
        build=build,
        kvaf=kvaf,
        torch=torch,
        device=device,
        stream=stream,
        cann=cann,
        aics=aics,
        runtime=runtime,
        restored=restored,
        table=table,
    )


def measure_case(args, archive, context, retained):
    """Release a successful case after graph reset; retain everything on failure."""
    from kvip.diagnostics.replay_runtime import get_kernel_path, runtime_worker_config

    build, kvaf, torch = context.build, context.kvaf, context.torch
    device, stream, cann, aics = context.device, context.stream, context.cann, context.aics
    runtime, restored, table = context.runtime, context.restored, context.table
    case_retained = {"progress": retained.setdefault("progress", {})}
    retained["active_case"] = case_retained
    phase(retained, "bundle_create")
    if args.mode == "fused":
        from kvip.diagnostics.fused_replay import create_fused

        tuning = getattr(args, "fused_tuning", None)
        if tuning is None:
            tuning = getattr(args, "ubmem_tuning", None)
        bundle = create_fused(
            torch,
            runtime,
            archive,
            restored,
            table,
            build,
            args.soc,
            device,
            stream,
            cann,
            aics,
            transport=args.transport,
            tuning=tuning,
            split_index_stages=getattr(args, "split_index_stages", False),
            upstream=bool(getattr(args, "upstream_root", None)),
            pipeline_options=getattr(args, "pipeline_options", None),
            q_workers=getattr(args, "pipeline_q_workers", 2),
            topk_probe=getattr(args, "probe_only", False),
            admission_controls=getattr(args, "admission_controls", None),
        )
        case_retained["bundle"] = bundle
        case_retained["buffers"] = buffer_addresses(bundle)
        retained["buffers"] = case_retained["buffers"]
        retained["buffers_variant"] = retained["progress"].get("variant")
        phase(retained, "prepare_fused_state")
        prepare_fused_state(bundle, archive, restored, torch, device, args.transport)
        if bundle.tuning.get("auto_route"):
            from kvip.diagnostics.auto_replay import AutoReplay

            bundle.auto_replay = AutoReplay(torch, bundle, archive)
    else:
        from kvip.diagnostics.serial_replay import create_serial

        bundle = create_serial(
            torch,
            runtime,
            archive,
            restored,
            table,
            build,
            args.soc,
            device,
            stream,
            cann,
            aics,
            transport=args.transport,
        )
        case_retained["bundle"] = bundle
    if getattr(args, "profile_dir", None) is not None:
        library = getattr(bundle, "library", None)
        libraries = getattr(bundle, "libraries", (() if library is None else (library,)))
        case_retained["profile_metadata"] = dict(
            snapshot=str(archive.directory),
            snapshot_sha256=file_digest(archive.directory / "manifest.json"),
            content=archive.manifest.get("content"),
            geometry=archive.manifest["geometry"],
            production_source=getattr(args, "production_source", None)
            or dict(
                root=str(getattr(args, "upstream_root", None) or Path(__file__).resolve().parents[3]),
                commit=getattr(args, "upstream_revision", None) or source_commit(),
            ),
            diagnostics_commit=source_commit(),
            transport=args.transport,
            mode=args.mode,
            device=args.device,
            soc=args.soc,
            cann=str(cann),
            kvaf_module=str(Path(kvaf.__file__).resolve()),
            kvaf_build=str(build),
            graph_backend="torch.npu.NPUGraph" if args.mode == "serial" else "kvaf.AclGraph",
            tuning=dict(getattr(bundle, "tuning", {})),
            pipeline_options=getattr(args, "pipeline_options", None),
            pipeline_q_workers=getattr(args, "pipeline_q_workers", None),
            runtime_configuration=runtime_worker_config(args, archive, aics),
            libraries=[dict(path=str(path), sha256=file_digest(path)) for path in libraries],
        )
    results = measure(torch, kvaf, stream, bundle, args, case_retained, runtime=runtime)
    phase(retained, "result_metadata")
    upstream_root = getattr(args, "upstream_root", None)
    if upstream_root:
        from kvip.diagnostics.upstream import loaded_lidu_artifacts, provenance

        production_source = provenance(upstream_root, args.upstream_revision)
        production_source["lidu"] = dict(
            getattr(args, "lidu_provenance", {}),
            used_in_this_case=args.mode == "serial",
            loaded_artifacts=loaded_lidu_artifacts(),
            build_provenance="Source equality permits sharing; the installed binary was not rebuilt or independently attested",
        )
    else:
        production_source = dict(
            kind="feature_checkout", root=str(Path(__file__).resolve().parents[3]), commit=source_commit()
        )
    library = getattr(bundle, "library", None)
    libraries = getattr(bundle, "libraries", (() if library is None else (library,)))
    result = dict(
        schema_version=2,
        status="pass",
        transport=args.transport,
        mode=args.mode,
        snapshot=str(archive.directory),
        content=archive.manifest.get("content", "full"),
        input_validation=archive.manifest.get("oracle", {}),
        snapshot_sha256=file_digest(archive.directory / "manifest.json"),
        replay_selection=archive.manifest.get("replay_selection"),
        scope="Query_Index_ready_to_LI_TopK_Fetch_cache_state_and_TND_slot_outputs",
        geometry=archive.manifest["geometry"],
        device=args.device,
        soc=args.soc,
        kvaf_module=str(Path(kvaf.__file__).resolve()),
        kvaf_build=str(build),
        library=str(library) if library is not None else None,
        library_sha256=file_digest(library) if library is not None else None,
        libraries=[dict(path=str(path), sha256=file_digest(path)) for path in libraries],
        kvip_commit=production_source["commit"],
        diagnostics_commit=source_commit(),
        capture_kvip_commit=archive.manifest.get("provenance", {}).get("kvip_commit", "unknown"),
        production_source=production_source,
        baseline_configuration="Unified replay workers/budget, not default vLLM service configuration or TPOT",
        **runtime_worker_config(args, archive, aics),
        warmup=args.warmup,
        address_relocation=restored["new_base"] - restored["old_base"],
        fidelity=archive.manifest["fidelity"],
        exclusions="Model, input preparation/import/packing, DDR Put, state restoration and correctness readbacks",
        interpretation="Isolated Index-sharing group graph latency; not DP service latency or TPOT",
        timing_boundary="Device inputs ready through all required device preparation, recent scoring/merge (if enabled), LI/TopK, Fetch, Index update and TND outputs. Not full service CPU/scheduler latency.",
        allocation_boundary=(
            "Current upstream UBmem HAL/FABRIC allocation; original THP/NUMA physical placement is not restored or measured"
            if args.transport == "ubmem"
            else "Current URMA CPU allocation; page policy is supplied to the local CPU service, physical coverage is not measured"
        ),
        graph_backend="torch.npu.NPUGraph" if args.mode == "serial" else "kvaf.AclGraph",
        selection_comparison=bundle.selection_comparison,
        demand_accounting=DEMAND_ACCOUNTING,
        representation=(
            archive.manifest["fidelity"]
            if archive.manifest.get("content") in ("synthetic_token_replay", "real_li_synthetic_kv")
            else "Native logical paged Index and LIDU maps reconstructed outside timing; original serial map allocation padding was not captured."
            if args.mode == "serial"
            else "Original hot-first Index, KV bytes, packed inputs and relative DDR addresses restored."
        ),
        **results,
    )
    if prediction := retained.get("seed_ub_eligibility"):
        result["seed_ub_eligibility"] = prediction
    if args.mode == "fused":
        if score_reference_result := getattr(bundle, "score_comparison", None):
            result["score_comparison"] = score_reference_result
        result["tuning"] = dict(bundle.tuning)
        result["threshold_comparison"] = bundle.threshold_comparison
        result["prefetch_admission_accounting"] = ADMISSION_ACCOUNTING
        result["reserved_metadata"] = dict(
            dense_bytes_per_group=int(archive.array("pre.reserved_hot").nbytes),
            compact_lru_bytes_per_group=int(archive.array("pre.lru_slots").nbytes),
            restored_from_snapshot=True,
            h2d_in_timing=False,
            note="Current production reserved-hot table is an input; CPU construction/H2D is excluded from device-chain timing.",
        )
    if args.transport == "urma" and args.mode == "serial":
        kernel = get_kernel_path(args)
        result["get_kernel"] = dict(path=str(kernel), sha256=file_digest(kernel))
    # measure() reset every captured graph first. A failing reset or lease
    # close leaves the active case and shared arena retained by the CLI.
    if bundle.lease is not None:
        phase(retained, "lease_close")
        bundle.lease.close()
    case_retained.clear()
    del retained["active_case"]
    return result


def release_context(context):
    """Called only after every case graph and output lease completed."""
    context.runtime.release_access_table(context.restored["access_table"])
    context.runtime.shutdown()


def study_cases(args):
    """Keep each study's baseline explicit when production defaults advance."""
    study = getattr(args, "study", "threshold")
    if study == "post-threshold":
        return {name: {"recent_threshold": True} for name in args.pipeline_variants}
    if study == "overlap-victims":
        return {
            name: dict(
                recent_threshold=True,
                recent_pipeline=True,
                recent_unified_topk=True,
                recent_skip=False,
                recent_threshold_only=True,
                recent_profile=False,
                recent_seed_ub=False,
                recent_local_victims=True,
                recent_register_kth=True,
                recent_overlap_victims=overlap,
            )
            for name, overlap in (("overlap", True), ("register", False))
        }
    if study == "register-kth":
        # Measure the candidate first, then the established baseline, in one arena.
        return {
            name: dict(
                recent_threshold=True,
                recent_pipeline=True,
                recent_unified_topk=True,
                recent_skip=False,
                recent_threshold_only=True,
                recent_profile=False,
                recent_seed_ub=False,
                recent_local_victims=True,
                recent_register_kth=register,
            )
            for name, register in (("register", True), ("local", False))
        }
    if study == "local-victims":
        return {
            name: dict(
                recent_threshold=True,
                recent_pipeline=True,
                recent_unified_topk=True,
                recent_skip=False,
                recent_threshold_only=True,
                recent_profile=False,
                recent_seed_ub=False,
                recent_local_victims=local,
            )
            for name, local in (("release", False), ("local", True))
        }
    if study == "seed-ub":
        return {
            name: dict(
                recent_threshold=True,
                recent_pipeline=True,
                recent_unified_topk=True,
                recent_skip=False,
                recent_threshold_only=True,
                recent_profile=profile,
                recent_seed_ub=seed,
            )
            for name, profile, seed in (("current", True, False), ("release", False, False), ("seed", False, True))
        }
    if study == "threshold-only":
        return {
            name: dict(
                recent_threshold=True,
                recent_pipeline=True,
                recent_unified_topk=True,
                recent_skip=False,
                recent_threshold_only=threshold_only,
            )
            for name, threshold_only in (("before", False), ("after", True))
        }
    if study == "skip":
        return {
            name: dict(recent_threshold=True, recent_pipeline=True, recent_unified_topk=True, recent_skip=skip)
            for name, skip in (("before", False), ("after", True))
        }
    if study == "pipeline":
        return {
            name: dict(recent_threshold=True, recent_pipeline=pipeline, recent_unified_topk=topk, recent_skip=False)
            for name, pipeline, topk in (("before", False, False), ("pipeline", True, False), ("unified", True, True))
        }
    return {policy: {"recent_threshold": policy == "recent", "recent_skip": True} for policy in ("resident", "recent")}


def check_auto_route(torch, stream, bundle, replay, last_step):
    """Same captured graph, real packed-row reorder, device-owned feedback."""
    from kvip.diagnostics.auto_replay import RowPermutation, requests_for

    auto = bundle.auto_replay
    original = auto.archive
    pointers = buffer_addresses(bundle)
    rows = []

    def run(name, step, *, learning=True, generation=1, reject=False, archive=None, expected_mode=None):
        stream.synchronize()
        selected = original if archive is None else archive
        if selected is not auto.archive:
            prepare_fused_state(bundle, selected, bundle.restored, torch, bundle.rotating.device, "urma")
            auto.archive = selected
        if buffer_addresses(bundle) != pointers:
            raise AssertionError("Auto lifecycle changed a captured input address")
        requests = requests_for(selected, generation=generation, step=step)
        if reject:
            requests[:, 3] = 3  # Safe identity/Q mismatch; never inject a transport fault.
        auto.begin(step=step, learning=learning, requests=requests)
        bundle.reset()
        stream.synchronize()
        replay()
        stream.synchronize()
        reads = bundle.verify()
        observation = auto.record(reads)
        if (not learning or reject or name in ("duplicate", "backward", "old-generation")) and observation["learned_owners"]:
            raise AssertionError("Rejected lifecycle step learned new feedback")
        if expected_mode is not None and observation["mode"] != expected_mode:
            raise AssertionError("Prediction fault did not exercise the requested graph mode")
        rows.append(dict(name=name, **observation))

    run("learning-disabled", last_step + 1, learning=False)
    run("duplicate", last_step)
    run("backward", max(0, last_step - 1))
    run("identity-rejected", last_step + 1, reject=True)
    run("resume", last_step + 1)
    run("new-generation", 1, generation=2)
    run("old-generation", last_step + 2)
    run("new-generation-resume", 2, generation=2)
    order = np.arange(bundle.rotating.batch)[::-1]
    run("packed-row-reorder", 3, generation=2, archive=RowPermutation(original, order))
    run("restore-row-order", 4, generation=2)
    # Deliberately corrupt only the predicted demand, after every headline
    # sample. These fault injections are not the normal learning mechanism.
    auto.inject_prediction_fault(0)
    run("prediction_fault_injection_low", 5, generation=2, expected_mode="low")
    run("prediction_fault_recovery_low", 6, generation=2)
    auto.inject_prediction_fault(8192)
    run("prediction_fault_injection_high", 7, generation=2, expected_mode="high")
    run("prediction_fault_recovery_high", 8, generation=2)
    return dict(status="pass", graph_captures=1, captured_addresses_unchanged=True,
                sequence=rows, included_in_timing=False, snapshot_restored_before_each=True,
                feedback_reset=False, real_packed_row_reorder=True,
                prediction_fault_injection=True,
                scope="Untimed prediction fault injection and identity rejection, not a real transport fault; frozen payload restored, not consecutive service cache evolution")


def check_mode_switch(torch, stream, bundle, replay):
    """Exercise one captured graph; controls/H2D and snapshot restore are untimed."""
    from kvip.diagnostics.threshold_check import poison_early_thresholds, verify_early_threshold_sentinel

    rotating = bundle.rotating
    original = list(bundle.tuning["admission_controls"])
    if original != [0, 64, 0, 0]:
        raise AssertionError("Mode-switch check must start and end in light-low")
    address = rotating.admission_controls.data_ptr()
    buffers = buffer_addresses(bundle)
    rows = []
    for quota in (0, 512, 0):
        stream.synchronize()
        controls = [quota, 64, 0, 0]
        rotating.set_admission_controls(torch.tensor([controls] * rotating.batch, dtype=torch.int32, device="cpu"))
        if rotating.admission_controls.data_ptr() != address or buffer_addresses(bundle) != buffers:
            raise AssertionError("Mode switch changed a captured tensor address")
        bundle.tuning["admission_controls"] = controls
        bundle.reset()
        if quota == 0:
            poison_early_thresholds(bundle)
        stream.synchronize()
        replay()
        stream.synchronize()
        if quota == 0:
            verify_early_threshold_sentinel(torch, bundle)
        reads = bundle.verify()
        expected = "not_computed" if quota == 0 else "bit_exact"
        if bundle.threshold_comparison["status"] != expected or (quota == 0 and reads["prefetch"] != 0):
            raise AssertionError("Same-graph replay did not validate the requested mode")
        rows.append(dict(quota=quota, prefetch=reads["prefetch"], final=reads["final"],
                         threshold=bundle.threshold_comparison["status"],
                         threshold_sentinel_unchanged=True if quota == 0 else None,
                         ordered_output_sha256=bundle.selection_comparison["ordered_output_sha256"]))
    return dict(status="pass", controls_address=address, captured_addresses_unchanged=True,
                graph_captures=1, sequence=rows, included_in_timing=False,
                snapshot_restored_before_each=True,
                scope="Mode switching and round-state reset only; not consecutive service-state evolution")


def verify_case_equivalence(cases, *, same_threshold):
    first = next(iter(cases.values()))
    computed = [case for case in cases.values() if not early_threshold_skipped(case.get("tuning", {}))]
    threshold_reference = computed[0] if computed else first
    for case in cases.values():
        if (
            case["selection_comparison"]["ordered_output_sha256"]
            != first["selection_comparison"]["ordered_output_sha256"]
        ):
            raise AssertionError("Seed A/B changed the final ordered TopK outputs")
        if same_threshold:
            if case["threshold_comparison"]["final_keys"] != first["threshold_comparison"]["final_keys"]:
                raise AssertionError("Seed execution optimization changed exact threshold values")
            if early_threshold_skipped(case.get("tuning", {})):
                if case["threshold_comparison"].get("status") != "not_computed" or case["threshold_comparison"]["initial_keys"] is not None:
                    raise AssertionError("Skipped early threshold must be reported as not computed")
            elif case["threshold_comparison"]["initial_keys"] != threshold_reference["threshold_comparison"]["initial_keys"]:
                raise AssertionError("Seed execution optimization changed exact threshold values")


def run(args, archive, retained):
    if getattr(args, "study", "threshold") == "seed-ub":
        from kvip.diagnostics.seed_eligibility import predict_seed_eligibility

        prediction = predict_seed_eligibility(archive)
        retained["seed_ub_eligibility"] = prediction
        emit(
            f"SEED_UB_ELIGIBILITY predicted={prediction['predicted_eligible_queries']}/{prediction['total_queries']} "
            f"fallback={prediction['predicted_fallback_queries']} not_runtime_observation"
        )
        if not prediction["predicted_eligible_queries"]:
            raise ValueError("Seed-UB has no eligible queries; refusing an all-fallback comparison")
    phase(retained, "context_allocate")
    retained["geometry"] = getattr(archive, "manifest", {}).get("geometry", {})
    completed = retained.setdefault("completed_cases", {})
    context = allocate_context(args, archive, retained)
    execution_study = getattr(args, "study", "threshold") in (
        "pipeline",
        "skip",
        "threshold-only",
        "seed-ub",
        "local-victims",
        "register-kth",
        "overlap-victims",
        "post-threshold",
    )
    if args.mode == "fused" and (args.threshold_policy == "both" or execution_study):
        cases = {}
        for name, tuning in study_cases(args).items():
            if args.study == "post-threshold":
                from kvip.diagnostics.pipeline_options import admission_for, options_for, q_workers_for

                args.pipeline_options = options_for(name)
                args.pipeline_q_workers = q_workers_for(name)
                args.admission_controls = admission_for(name)
            retained["progress"].update(variant=name, tuning=dict(tuning))
            phase(retained, "case_begin")
            emit(f"CASE_BEGIN {args.transport}-{args.mode}-{name}")
            args.fused_tuning = tuning
            cases[name] = measure_case(args, archive, context, retained)
            completed[name] = cases[name]
            emit(f"CASE_PASS {args.transport}-{args.mode}-{name} p50_us={cases[name]['timing']['p50_us']:.3f}")
        phase(retained, "case_equivalence")
        verify_case_equivalence(cases, same_threshold=execution_study)
        result = dict(
            status="pass",
            transport=args.transport,
            mode=args.mode,
            study=getattr(args, "study", "threshold"),
            cases=cases,
            execution_order=list(cases),
        )
        if getattr(args, "topk_probe", False):
            from kvip.diagnostics.pipeline_options import options_for

            probe_args = SimpleNamespace(**vars(args))
            probe_args.pipeline_options = options_for("topk")
            probe_args.pipeline_q_workers = 2
            probe_args.admission_controls = None
            probe_args.check_mode_switch = False
            probe_args.probe_only = True
            probe_args.fused_tuning = {"recent_threshold": True}
            retained["progress"].update(variant="topk-probe")
            probe_result = measure_case(probe_args, archive, context, retained)
            reference = next(iter(cases.values()))
            for key in ("selected_set_sha256", "ordered_output_sha256"):
                if probe_result["selection_comparison"][key] != reference["selection_comparison"][key]:
                    raise ValueError("TopK probe changed the final ordered selection")
            for key in ("initial_keys", "final_keys"):
                if probe_result["threshold_comparison"][key] != reference["threshold_comparison"][key]:
                    raise ValueError("TopK probe changed the threshold")
            result["topk_probe_report"] = probe_result
    else:
        retained["progress"]["variant"] = (
            f"upstream-{args.mode}"
            if getattr(args, "upstream_root", None)
            else args.threshold_policy
            if args.mode == "fused"
            else "serial"
        )
        result = measure_case(args, archive, context, retained)
        completed[retained["progress"]["variant"]] = result
    phase(retained, "context_release")
    release_context(context)
    return result
