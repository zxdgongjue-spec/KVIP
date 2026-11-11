# SPDX-License-Identifier: Apache-2.0
"""Common real-snapshot timing; production graphs remain uninstrumented."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from kvip.diagnostics.archive import array_digest, cpu_array, file_digest, source_commit
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
    else:
        for value in (rotating.prefetch_status, rotating.final_status):
            words = cpu_array(torch, value).view(np.uint64)
            if np.any(words[:, 0] != 0x4B5641464247444F) or np.any(words[:, 2]):
                raise AssertionError("URMA completion status failed")
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
    compact_reserved = getattr(bundle, "tuning", {}).get("compact_reserved", False)
    seeds = []
    for name, target in (persistent_tensors(selector) | input_tensors(selector)).items():
        if name == "reserved_hot" and compact_reserved:
            continue
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
    saved_tags = archive.array("post.selected_tags")
    saved_payload = archive.array("post.selected_payload")
    delta = restored["new_base"] - restored["old_base"]
    active_rows = np.any(archive.array("pre.query_ends").reshape(-1, 4) > 0, axis=1)
    reserved_expected = archive.array("pre.reserved_hot")

    def reset():
        for target, seed in seeds:
            target.copy_(seed)
        if compact_reserved:
            # Poison outside timing. The captured init must rebuild the table;
            # retaining a correct table from the preceding replay cannot pass.
            rotating.reserved_hot.fill_(-2)
        # Production init resets round-local work; never rewind transport queues.

    def verify():
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
        if compact_reserved:
            actual_reserved = cpu_array(torch, rotating.reserved_hot)
            if not np.array_equal(actual_reserved[active_rows], reserved_expected[active_rows]):
                raise AssertionError("Device reserved-hot reconstruction differs from the captured inverse map")
        state = post_state(torch, selector)
        checked = oracle.verify(state, address_delta=delta, expected_topk=expected)
        positions = {tuple(tag): i for i, tag in enumerate(checked["selected_tags"])}
        if len(positions) != len(saved_tags):
            raise AssertionError("Replay selected a different logical payload set")
        for i, tag in enumerate(saved_tags):
            position = positions.get(tuple(tag))
            if position is None or not np.array_equal(saved_payload[i], checked["selected_payload"][position]):
                raise AssertionError("Replay KV bytes differ from the captured service output")
        reads.update(demand.classify(state["logical_output"]))
        return reads

    bundle.reset, bundle.verify = reset, verify
    bundle.seeds, bundle.oracle = seeds, oracle
    bundle.selection_comparison = dict(
        status="exact",
        selected_set_sha256=array_digest(np.sort(expected.astype("<i4"), axis=-1)),
        compared_queries=len(expected),
        changed_queries=0,
        total_replacements=0,
        min_overlap_fraction=1.0,
        reference="Captured fusion TopK sets; equality enforced on every replay",
    )


def profile_urma_get(runtime, stream, reset, verify, replay_stages):
    """One instrumented replay after headline timing, without guessing a clock rate."""
    reset()
    stream.synchronize()
    runtime.start_batch_get_kernel_profile()
    failed = False
    try:
        replay_stages()
        stream.synchronize()
        cores = runtime.batch_get_kernel_profile()
    except BaseException:
        failed = True
        raise
    finally:
        try:
            runtime.stop_batch_get_kernel_profile()
        except BaseException:
            # Preserve the original completion/readback error; the CLI keeps
            # the full active bundle alive and exits without native cleanup.
            if not failed:
                raise
    reads = verify()
    if not cores:
        raise RuntimeError("URMA native Get profile returned no core records")
    fields = (
        "active_requests",
        "total_posts",
        "submitted_group_count",
        "completed_group_count",
        "completion_count",
        "post_active_cycles",
        "drain_wait_cycles",
    )
    totals = {name: sum(int(core[name]) for core in cores) for name in fields}
    if totals["active_requests"] != reads["final"]:
        raise AssertionError("URMA native Get profile request count differs from verified serial reads")
    maxima = {name: max(int(core[name]) for core in cores) for name in ("post_active_cycles", "drain_wait_cycles")}
    return dict(
        method="one additional instrumented stage-graph replay after headline and stage timing",
        unit="native_cycles",
        included_in_headline=False,
        caveat="Core sums and maxima are diagnostics, not additive wall latency; no cycle-to-us conversion.",
        cores=cores,
        totals=totals,
        maxima=maxima,
        active_cores=sum(int(core["active_requests"]) > 0 for core in cores),
        reads=reads,
    )


def measure(torch, kvaf, stream, bundle, args, retained, runtime=None):
    """Time one full graph; optional stage graphs are separately labelled diagnostics."""
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

    reset()
    bundle.forward()
    stream.synchronize()
    verify()
    reset()
    stream.synchronize()
    with lease.capture_scope() if lease is not None else nullcontext():
        graph = capture(bundle.forward)
    reset()  # Graph capture may execute the body.
    replay(graph)
    stream.synchronize()
    verify()
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    retained["events"] = [begin, end]
    samples, reads = [], []
    for step in range(args.warmup + args.iterations):
        reset()
        stream.synchronize()
        begin.record(stream)
        replay(graph)
        end.record(stream)
        end.synchronize()
        elapsed = begin.elapsed_time(end) * 1000
        counts = verify()
        if step >= args.warmup:
            samples.append(float(elapsed))
            reads.append(counts)
    result = dict(timing=statistics(samples), reads_samples=reads)
    if getattr(args, "measure_call_preparation", False):
        from kvip.diagnostics.call_preparation import measure_candidate_call

        result["candidate_call"] = measure_candidate_call(torch, stream, bundle, args, lambda: replay(graph), retained)
    if args.stage_iterations:
        # Capture all stages in order without resetting in between. URMA can
        # leave DMA in flight until a later stage consumes its CQ completion.
        reset()
        stream.synchronize()
        with lease.capture_scope() if lease is not None else nullcontext():
            stages = [(name, capture(call)) for name, call in bundle.stage_calls]
        reset()
        for _, stage in stages:
            replay(stage)
        stream.synchronize()
        verify()
        events = [torch.npu.Event(enable_timing=True) for _ in range(len(stages) + 1)]
        retained["events"].extend(events)
        stage_samples = {name: [] for name, _ in stages}
        stage_reads, totals = [], []
        for step in range(1 + args.stage_iterations):
            reset()
            stream.synchronize()
            events[0].record(stream)
            for index, (_, stage) in enumerate(stages):
                replay(stage)
                events[index + 1].record(stream)
            events[-1].synchronize()
            counts = verify()
            if step:
                for index, (name, _) in enumerate(stages):
                    stage_samples[name].append(events[index].elapsed_time(events[index + 1]) * 1000)
                totals.append(events[0].elapsed_time(events[-1]) * 1000)
                stage_reads.append(counts)
        result["stage_diagnostics"] = dict(
            method="sequential_stage_graphs_with_interposed_device_events",
            caveat="Extra graph submissions/events can alter scheduling and overlap; these are not additive components of the uninstrumented latency.",
            stages={name: statistics(values) for name, values in stage_samples.items()},
            total=statistics(totals),
            reads_samples=stage_reads,
        )
        if getattr(args, "transport", None) == "urma" and args.mode == "serial":
            # Stage capture rebinds native LI/slot output tensors. Reuse that
            # same stage chain, whose outputs verify() currently owns, rather
            # than replaying the earlier full graph against stale references.
            def replay_stages():
                for _, stage in stages:
                    replay(stage)

            result["native_get_profile"] = profile_urma_get(runtime, stream, reset, verify, replay_stages)
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
    from kvip.operators.lifetch_ubmem.specialization import kvaf_cann

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
    runtime = create_runtime(args, archive, kvaf, torch, device, stream, cann, aics)
    retained["runtime"] = runtime
    if args.transport == "urma":
        runtime.link_to(args.kvaf_server)
    runtime.initialize(ddr["region_bytes"], ddr["slot_bytes"])
    restored = import_pages(
        runtime, archive.directory / "ddr", ddr, torch_module=torch, target_transport=args.transport
    )
    retained["restored"] = restored
    if restored["access_table"] is None:
        raise ValueError("Real-history replay requires published DDR pages")
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
    case_retained = {}
    retained["active_case"] = case_retained
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
        )
        case_retained["bundle"] = bundle
        prepare_fused_state(bundle, archive, restored, torch, device, args.transport)
        if getattr(args, "measure_call_preparation", False):
            from kvip.diagnostics.call_preparation import CandidatePreparation

            bundle.candidate_preparation = CandidatePreparation(torch, archive, bundle)
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
        kvip_commit=source_commit(),
        capture_kvip_commit=archive.manifest.get("provenance", {}).get("kvip_commit", "unknown"),
        **runtime_worker_config(args, archive, aics),
        warmup=args.warmup,
        address_relocation=restored["new_base"] - restored["old_base"],
        fidelity=archive.manifest["fidelity"],
        exclusions="Model, input preparation/import/packing, DDR Put, state restoration and correctness readbacks",
        interpretation="Isolated four-layer group graph latency; not DP service latency or TPOT",
        timing_boundary="timing contains device graph only; optional candidate_call reports a separate host-to-output boundary",
        graph_backend="torch.npu.NPUGraph" if args.mode == "serial" else "kvaf.AclGraph",
        selection_comparison=bundle.selection_comparison,
        demand_accounting=DEMAND_ACCOUNTING,
        representation=(
            archive.manifest["fidelity"]
            if archive.manifest.get("content") == "synthetic_token_replay"
            else "Native logical paged Index and LIDU maps reconstructed outside timing; original serial map allocation padding was not captured."
            if args.mode == "serial"
            else "Original hot-first Index, KV bytes, packed inputs and relative DDR addresses restored."
        ),
        **results,
    )
    if args.mode == "fused":
        result["tuning"] = dict(bundle.tuning)
        result["prefetch_admission_accounting"] = ADMISSION_ACCOUNTING
        result["reserved_metadata"] = dict(
            dense_bytes_per_group=int(archive.array("pre.reserved_hot").nbytes),
            compact_lru_bytes_per_group=int(archive.array("pre.lru_slots").nbytes),
            reconstruction_checked=bool(bundle.tuning.get("compact_reserved", False)),
            h2d_in_timing=False,
            h2d_in_candidate_call="candidate_call" in results,
            note="Compact LRU upload already exists in serving; only dense reserved-hot H2D is eliminated.",
        )
    if args.transport == "urma" and args.mode == "serial":
        kernel = get_kernel_path(args)
        result["get_kernel"] = dict(path=str(kernel), sha256=file_digest(kernel))
    # measure() reset every full/stage graph first. A failing reset or lease
    # close leaves the active case and shared arena retained by the CLI.
    if bundle.lease is not None:
        bundle.lease.close()
    case_retained.clear()
    del retained["active_case"]
    return result


def release_context(context):
    """Called only after every case graph and output lease completed."""
    context.runtime.release_access_table(context.restored["access_table"])
    context.runtime.shutdown()


def run(args, archive, retained):
    context = allocate_context(args, archive, retained)
    result = measure_case(args, archive, context, retained)
    release_context(context)
    return result
