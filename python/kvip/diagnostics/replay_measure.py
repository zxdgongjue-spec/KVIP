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


def fused_reads(torch, rotating, update, transport, width, demand=None, *, window_limit=None, active_rows=None):
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
            rows, window_limit=window_limit, rounds=rotating.rounds, active_rows=active_rows
        )
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
        reads = fused_reads(
            torch,
            rotating,
            update,
            transport,
            archive.manifest["geometry"]["token_bytes"],
            demand,
            window_limit=bundle.tuning["window_limit"],
            active_rows=active_rows,
        )

        logical = cpu_array(torch, update.logical_output)
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
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    retained["events"] = [begin, end]
    samples, reads = [], []
    for step in range(args.warmup + args.iterations):
        phase(retained, "iteration_reset", step)
        reset()
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
    phase(retained, "final_verify")
    verify()  # Full score/Index/KV oracle after the final measured replay.
    result = dict(timing=statistics(samples), reads_samples=reads)
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
        )
        case_retained["bundle"] = bundle
        case_retained["buffers"] = buffer_addresses(bundle)
        retained["buffers"] = case_retained["buffers"]
        retained["buffers_variant"] = retained["progress"].get("variant")
        phase(retained, "prepare_fused_state")
        prepare_fused_state(bundle, archive, restored, torch, device, args.transport)
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


def verify_case_equivalence(cases, *, same_threshold):
    first = next(iter(cases.values()))
    for case in cases.values():
        if (
            case["selection_comparison"]["ordered_output_sha256"]
            != first["selection_comparison"]["ordered_output_sha256"]
        ):
            raise AssertionError("Seed A/B changed the final ordered TopK outputs")
        if same_threshold and any(
            case["threshold_comparison"][key] != first["threshold_comparison"][key]
            for key in ("initial_keys", "final_keys")
        ):
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
    )
    if args.mode == "fused" and (args.threshold_policy == "both" or execution_study):
        cases = {}
        for name, tuning in study_cases(args).items():
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
