"""ABI 11 standalone benchmark: CPU contract model or real NPU/KVAF execution."""

import argparse
import copy
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from benchmark_record import render_record
from contract_fixture import cases, make_fixture, manifest, to_device
from contract_index_reference import ColdOrder, cache_metrics, inspect_cache, verify_cache
from contract_lifecycle import Lifecycle
from contract_payload import payload
from contract_reference import ContractError, frozen_hot, plan_row, select_topk
from contract_simulator import Simulator, seed_output
from contract_timing import (
    completion_diagnostics,
    decode_device_stages,
    decode_kernel_timeline,
    decode_service_stages,
    summarize_device_stages,
    summarize_service_stages,
)
from contract_verify import score_workspace_diagnostic, snapshot, verify_skipped, verify_success
from kvip.v9.contract import Status
from kvip.v9.device_resources import query_device_resources
from kvip.v9.geometry import final_aivs, li_geometry, native_kvaf_workers
from protocol import replay_contract


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-order", choices=("alternate", "native-first", "fused-first"), default="alternate")
    p.add_argument("--transport", choices=("kvaf", "npu-memory"), default="kvaf")
    p.add_argument("--backend", choices=("cpu", "npu"), default="npu")
    p.add_argument("--batches", default="1")
    p.add_argument("--queries", default="4")
    p.add_argument("--history", type=int, default=16384, help="Initial candidate length; need not be aligned")
    p.add_argument("--resident", type=int, default=12288)
    p.add_argument("--tail", type=int, default=129)
    p.add_argument("--cache-miss-rates", help="Optional exact per-query target percentages, e.g. 0,10,20")
    p.add_argument("--case-suite", choices=("contract", "full", "lifecycle", "performance"), default="contract")
    p.add_argument(
        "--prefetch-decoys", type=int, default=0, help="Performance suite: extra cold threshold hits per query"
    )
    p.add_argument("--state-mode", choices=("snapshot", "stateful"), default="snapshot")
    p.add_argument(
        "--index-storage",
        choices=("owned", "derived"),
        default="owned",
        help="owned: one hot/paged-cold payload; derived: compatibility comparison",
    )
    p.add_argument(
        "--same-length-rewrite",
        action="store_true",
        help="Stateful: alternate Q4 growth with a same-length Key/scale rewrite",
    )
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--execution", choices=("eager", "graph"), default="eager")
    p.add_argument(
        "--phase-timing",
        action="store_true",
        help="NPU eager only: add an event after init to separate init/LI/resolve/fetch latency",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prefetch-limit", type=int, default=2048)
    p.add_argument("--window-limit", type=int, default=480)
    p.add_argument("--intermediate-fetches", type=int, default=4)
    p.add_argument("--li-tiling", choices=("auto", "q4"), default="auto")
    p.add_argument("--final-strategy", choices=("replicated",), default="replicated")
    p.add_argument("--final-aivs", type=int, choices=(0, 1, 2, 4), default=0)
    p.add_argument("--service-aics", type=int, default=0)
    p.add_argument("--fetch-post-mode", choices=("batch_mixed", "batch_simt", "per_batch_simd"), default="batch_mixed")
    p.add_argument("--publish-tiles", type=int, choices=(1, 2, 4, 8, 16), default=1)
    p.add_argument("--intermediate-profiling", choices=("on", "off"), default="on")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--final-submit-limit-us", type=float, default=None)
    p.add_argument("--native-baseline", choices=("on", "off"), default="on")
    p.add_argument("--qsfa", choices=("auto", "required", "off"), default="auto")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--aic-count", type=int, default=0)
    p.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    p.add_argument("--soc", default="Ascend950PR_9599")
    p.add_argument("--kvaf-server")
    p.add_argument("--kvaf-root", type=Path)
    p.add_argument("--output", type=Path, default=Path("v9_contract_results"))
    a = p.parse_args(argv)
    if a.transport == "npu-memory" and (a.native_baseline != "off" or a.case_suite == "performance"):
        p.error("NPU memory transport requires --native-baseline off and a correctness suite")
    a.batch_values = [int(x) for x in a.batches.split(",")]
    a.query_values = [int(x) for x in a.queries.split(",")]
    a.miss_rates = None if a.cache_miss_rates is None else [float(x) for x in a.cache_miss_rates.split(",")]
    if a.miss_rates is not None and any(not 0 <= r <= 100 for r in a.miss_rates):
        p.error("cache miss rates must be percentages in 0..100")
    if any(not 1 <= b <= 32 for b in a.batch_values) or any(q != 4 for q in a.query_values):
        p.error("requires batch 1..32 and fixed MTP3/Q4")
    if (
        a.history < 1
        or a.tail < 0
        or a.prefetch_decoys < 0
        or a.resident not in (6144, 10240, 12288)
        or not 0 <= a.prefetch_limit <= 2048
        or not 1 <= a.window_limit <= 480
        or a.steps < 1
        or a.layers < 1
        or a.warmup < 0
        or a.iterations < 1
    ):
        p.error("invalid lengths, resident budget or iteration counts")
    if a.backend == "npu" and a.transport == "kvaf" and not a.kvaf_server:
        p.error("--kvaf-server is required for real NPU execution")
    if a.backend == "cpu" and a.execution == "graph":
        p.error("CPU simulation cannot validate ACL Graph execution")
    if a.phase_timing and (a.backend != "npu" or a.execution != "eager"):
        p.error("--phase-timing requires --backend npu --execution eager")
    if a.case_suite == "lifecycle" and (a.steps < 3 or a.layers < 2):
        p.error("lifecycle requires at least three steps and two layers")
    if a.case_suite == "lifecycle":
        a.state_mode = "stateful"
    if a.backend == "npu" and a.index_storage == "owned" and a.state_mode == "stateful" and a.execution == "graph":
        p.error("owned stateful layout diagnostics require eager execution")
    if a.prefetch_decoys and a.case_suite != "performance":
        p.error("--prefetch-decoys requires --case-suite performance")
    if a.case_suite == "performance" and a.state_mode != "snapshot":
        p.error("controlled performance cases require snapshot mode; use lifecycle for stateful correctness")
    if a.prefetch_decoys and a.miss_rates and 0 in a.miss_rates:
        p.error("zero miss rate cannot have strictly non-winning threshold decoys")
    return a


def snapshot_mapping_epoch(state_mode, owned, arrays, seed):
    """Only a paired owned snapshot restore certifies a reusable mapping version."""
    if state_mode != "snapshot" or not owned:
        return 0
    return int(all(np.array_equal(arrays[name], seed[name]) for name in ("forward", "reverse")))


def measurement_schedule(stateful, steps, warmup, iterations):
    """Validate a complete call before admitting any latency samples."""
    if stateful:
        return [(i, "preflight" if i == 0 else "measurement") for i in range(steps)]
    return [(-1, "preflight")] + [(i, "warmup" if i < warmup else "measurement") for i in range(warmup + iterations)]


def workload_summary(case, step, execution):
    rows = step.get("per_request_scan_and_fetch", [])
    rates = step.get("per_query_miss_fraction", [])
    return {
        "controlled_scores": case.controlled,
        "target_miss_percent": None if case.target_miss_fraction is None else case.target_miss_fraction * 100,
        "actual_miss_percent": float(np.mean(rates) * 100) if rates else None,
        "prefetch_decoys_per_query": case.prefetch_decoys,
        "q4_unique_misses": sum(step.get("logical_misses", [])),
        "windows_requested": execution["intermediate_fetches_requested"],
        "windows_compiled": execution["intermediate_fetches"],
        "windows_effective": (
            min(execution["intermediate_fetches"], max(1, (max(row["cold_tokens"] for row in rows) + 127) // 128))
            if rows
            else None
        ),
        "queries_per_aic": execution["queries_per_aic"],
        **{
            name: sum(row[name] for row in rows)
            for name in (
                "threshold_candidate_union",
                "prefetch_submitted_and_completed",
                "supplement_submitted_and_completed",
                "retained_non_topk_prefetch",
                "discarded_prefetch",
            )
        },
    }


def phase_elapsed_us(start, init_end, final_start, final_end, end):
    """Read already-completed same-stream event intervals; never synchronize here."""
    events = (start, init_end, final_start, final_end, end)
    return {
        name: events[i].elapsed_time(events[i + 1]) * 1000 for i, name in enumerate(("init", "li", "resolve", "fetch"))
    }


def prepare_step(a, g, arrays, step, scores=None):
    if a.state_mode != "stateful":
        return
    candidate = a.history + (step // 2 if a.same_length_rewrite else step) * g.queries
    arrays["candidate_lens"].fill(candidate)
    arrays["logical_lens"].fill(candidate + a.tail)
    if a.same_length_rewrite and step % 2:
        token = max(0, candidate - 4)
        for b in range(g.batch):
            page = int(arrays["index_table"][b, token // 128])
            # Toggle a finite FP8 row and its scale; length alone cannot detect it.
            values = arrays["key"][page, token % 128, 0, : g.queries]
            values[:] = np.where(values == 0x58, 0x38, 0x58)
            arrays["key_scale"][page, token % 128, 0] = 1.5
            for q in range(g.queries):
                feature = int(np.flatnonzero(arrays["query"][b * g.queries + q, 0])[0])
                scores[b, q, token] = (
                    (16 if values[feature] == 0x58 else 1) * 1.5 * arrays["query_scale"][b * g.queries + q, 0]
                )
    # Caller supplies the newly appended tail; maps and cursors are feedback
    # from the previous successful invocation and are never reset here.
    out = arrays["resident_output"].reshape(-1, 656)
    for b, p in enumerate(arrays["pool_rows"]):
        s = int(arrays["budgets"][b])
        slots = np.arange(s, s + a.tail)
        physical = arrays["resident_table"][b, slots // 128] * 128 + slots % 128
        out[physical] = payload(int(p) * g.num_layers + g.layer_id, np.arange(candidate, candidate + a.tail))


def expected_failure(g, arrays, scores, case):
    if case.invalid_directory:
        return Status.DIRECTORY
    if case.profile == 2 or case.pad:
        return Status.SKIPPED
    for b, p in enumerate(arrays["pool_rows"]):
        c, s = int(arrays["candidate_lens"][b]), int(arrays["budgets"][b])
        hot = frozen_hot(arrays["forward"][p], arrays["reverse"][p], arrays["victims"][b], c, s)
        try:
            plan_row(select_topk(scores[b], g.queries), hot, arrays["victims"][b], s, int(arrays["cursors"][p]))
        except ContractError as error:
            return error.status
    return Status.OK


def check_outcome(g, before, after, scores, reservations, expected, descriptors=None):
    if expected == Status.SKIPPED:
        verify_skipped(before, after)
        return {"status": "pass", "expected": "skipped", "side_effects": False}
    if expected < 0:
        if not np.all(after["status"] < 0) or expected not in after["status"]:
            raise AssertionError(f"expected rejection {expected}, got {after['status']}")
        for name in ("forward", "reverse", "cursors"):
            if not np.array_equal(before[name], after[name]):
                raise AssertionError(f"rejected call published {name}")
        if np.any(after["destination"] != -1) or np.any(after["resident_lengths"]):
            raise AssertionError("rejected call exposed downstream attention data")
        return {
            "status": "pass",
            "expected_rejection": int(expected),
            "transfers": after["transfers"].tolist(),
            "must_retire_chain": True,
        }
    return verify_success(g, before, after, scores, reservations, descriptor_rows=descriptors)


def run_cpu(a, g, arrays, scores, case, root):
    seed_output(arrays, g)
    sim = Simulator(g, arrays)
    lifecycle = Lifecycle(g) if a.state_mode == "stateful" else None
    results = []
    steps = a.steps if a.state_mode == "stateful" else 1
    for step in range(steps):
        event = lifecycle.prepare(arrays, scores, step) if lifecycle else None
        prepare_step(a, g, arrays, step, scores)
        before = snapshot(arrays)
        current = scores[:, :, : int(arrays["candidate_lens"][0])]
        expected = expected_failure(g, arrays, current, case)
        sim.launch(
            current,
            prefetch_limit=a.prefetch_limit,
            window_limit=a.window_limit,
            rounds=li_geometry(
                g.batch, a.li_tiling, a.intermediate_fetches, a.aic_count or 32, a.service_aics, g.queries
            )["intermediate_fetches"],
        )
        reserved = [row["reservations"] for row in sim.trace]
        result = check_outcome(g, before, arrays, current, reserved, expected)
        result["lifecycle"] = event
        results.append(result)
        if lifecycle and expected == Status.OK:
            lifecycle.commit(arrays)
        if expected < 0:
            sim.launch(current)
            if not np.all(arrays["status"] == Status.POISONED):
                raise AssertionError("failed chain was reusable")
            break
    return {
        "status": "pass",
        "steps": results,
        "execution": "cpu_contract_model",
        "correctness_status": "pass",
        "hardware_validation_status": "not_run",
        "native_baseline": {"status": "skipped", "reason": "requires NPU"},
        "qsfa": {"status": "skipped", "reason": "requires NPU"},
        "performance_status": "not_measured",
        "timing": {"status": "not_measured", "reason": "CPU model is not a device benchmark"},
    }


def build(a, g, root, resources, helper):
    from kvip.v9 import native_source

    source = native_source()
    if a.transport == "npu-memory":
        from memory_build import prepare_source

        source = prepare_source(source, root / "memory_source")
    build_dir = (
        a.output.resolve()
        / ("build_memory" if a.transport == "npu-memory" else "build")
        / f"b{g.batch}-q{g.queries}-c{g.candidate_capacity}-s{g.resident_capacity}"
    )
    options = dict(
        V9_CONTRACT_MODE="ON",
        SIM_BATCH=g.batch,
        SIM_QUERIES=g.queries,
        SIM_HISTORY=g.candidate_capacity,
        SIM_RESIDENT=g.resident_capacity,
        RUN_MODE="npu",
        SOC_VERSION=a.soc,
        ASCEND_CANN_PACKAGE_PATH=a.cann,
        AIC_COUNT=resources["effective_aic"],
        LI_TILING=a.li_tiling,
        INTERMEDIATE_FETCHES=a.intermediate_fetches,
        SERVICE_AICS=a.service_aics,
        FINAL_REPLICATED=1,
        FINAL_AIVS=a.final_aivs,
        FETCH_POST_MODE=a.fetch_post_mode,
        ROTATION_PUBLISH_TILES=a.publish_tiles,
        INTERMEDIATE_PROFILING=int(a.intermediate_profiling == "on"),
        KVAF_INCLUDE=helper.find_headers(a.kvaf_root),
    )
    helper.command(
        ["cmake", "-S", source, "-B", build_dir] + [f"-D{k}={v}" for k, v in options.items()], root / "configure.log"
    )
    helper.command(["cmake", "--build", build_dir, "-j2"], root / "build.log")
    return build_dir / "libv9_bridge.so", {k: str(v) for k, v in options.items()}


def run_npu(a, g, arrays, scores, case, root):
    # Lazy imports keep CPU contract checks independent of the Ascend stack.
    import native_support as helper
    from contract_native import NativeBaseline
    from contract_qsfa import check_qsfa
    from contract_session import ContractSession

    if a.transport == "npu-memory":
        from memory_session import MemorySession as ContractSession
    from kvip.v9 import ContractLaunch

    torch, registration, _ = helper._load_native_operators("c8_offload")
    if a.qsfa == "required" and getattr(torch.ops._C_ascend, "npu_dsa_a5_sparse_tail_attention_c8", None) is None:
        raise RuntimeError("QSFA binding is missing; load its standalone registration or use --qsfa off")
    import torch_npu

    torch.npu.set_device(a.device)
    device = torch.device(f"npu:{a.device}")
    resources = query_device_resources(a.cann, a.device, a.aic_count)
    library, options = build(a, g, root, resources, helper)
    workers = (
        g.batch * (final_aivs(g.batch, a.li_tiling, resources["effective_aic"], a.final_aivs, g.queries) + 1) + 1
    ) // 2
    owner = native_owner = None
    report = {
        "status": "running",
        "transport": a.transport,
        "steps": [],
        "build_options": options,
        "device_resources": resources,
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "native_baseline": {"status": "skipped", "reason": "disabled or rejection case"},
        "qsfa": {"status": "skipped", "reason": "disabled or rejection case"},
        "execution": a.execution,
        "native_configuration": {
            "kvaf_workers": native_kvaf_workers(g.batch, resources["effective_aic"], g.queries),
            "fused_kvaf_workers": workers,
            "transport_policy": "native fixed auto/default; independent of fused tuning",
            "layout_query": "TND",
            "sparse_mode": 3,
            "queries": g.queries,
            "batch": g.batch,
            "history": a.history,
            "resident": a.resident,
            "tail": a.tail,
            "seed": a.seed,
            "execution": a.execution,
            "phase_events": a.phase_timing,
            "measurement_order": a.measurement_order,
        },
        "software": {"torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown")},
    }
    try:
        tensors = to_device(torch, arrays, g, device)
        owner = ContractSession.open(
            torch, device, g, tensors["resident_output"], server=a.kvaf_server, kvaf_root=a.kvaf_root, workers=workers
        )
        owner.seed_resident(arrays)
        tensors["resident_output"] = owner.output
        tensors["address_table"] = owner.ddr_block_addresses
        tensors["valid_table"] = owner.ddr_valid
        tensors["valid_table"].bitwise_and_(torch.from_numpy(arrays["valid_table"]).to(device))
        if case.invalid_directory:
            d = (int(arrays["pool_rows"][0]) * g.num_layers + g.layer_id) * g.qcap
            tensors["valid_table"][d : d + 1].zero_()
        for name in ("resident_output", "address_table", "valid_table"):
            arrays[name] = tensors[name].cpu().numpy().copy()
        baseline = None
        if a.native_baseline == "on" and expected_failure(g, arrays, scores[:, :, : a.history], case) == Status.OK:
            names = (
                ("lidu_fused_batch_get_prepare_mtp_out",) if g.queries > 1 else ("lidu_fused_batch_get_prepare_out",)
            )
            registration.require_a5_dsa_registration(names)
            native_owner = ContractSession.open(
                torch,
                device,
                g,
                tensors["resident_output"],
                server=a.kvaf_server,
                kvaf_root=a.kvaf_root,
                workers=native_kvaf_workers(g.batch, resources["effective_aic"], g.queries),
            )
            baseline = NativeBaseline(torch, torch_npu, native_owner, g, tensors)
        owner.acquire()
        launcher = ContractLaunch(
            torch,
            library,
            tensors,
            owner,
            g,
            prefetch_limit=a.prefetch_limit,
            window_limit=a.window_limit,
            li_tiling=a.li_tiling,
            intermediate_fetches=a.intermediate_fetches,
            aic_count=resources["effective_aic"],
            service_aics=a.service_aics,
            final_workers=a.final_aivs,
        )
        stream = torch.npu.current_stream(device)
        graph = native_graph = None
        if a.execution == "graph":
            # Both capture and a checked dummy replay use the device control.
            # Capture never changes the caller's resident cache or posts a WQE.
            dummy_caches = [x.cpu().numpy().copy() for x in launcher.index_caches]
            tensors["profile"].fill_(2)
            graph = torch.npu.NPUGraph()
            owner.graphs.append(graph)
            capture_stream = torch.npu.Stream(device=device)
            stream.synchronize()
            with torch.npu.graph(graph, stream=capture_stream):
                launcher.launch()
            capture_stream.synchronize()
            owner.mark_launched()
            graph.replay()
            launcher.completion()
            dummy = {
                name: tensors[name].cpu().numpy()
                for name in (
                    "forward",
                    "reverse",
                    "cursors",
                    "resident_output",
                    "topk",
                    "destination",
                    "miss_counts",
                    "resident_lengths",
                    "transfers",
                    "extra_prefetch_slots",
                    "extra_prefetch_counts",
                    "status",
                )
            }
            verify_skipped(snapshot(arrays), dummy)
            for buffer, saved in zip(launcher.index_caches, dummy_caches):
                if not np.array_equal(buffer.cpu().numpy(), saved):
                    raise AssertionError("Graph dummy modified persistent index cache")
            tensors["profile"].copy_(torch.from_numpy(arrays["profile"]).to(device))
            report["graph_dummy"] = {"status": "pass", "persistent_writes": False, "wqes": 0}
            if baseline is not None:
                baseline.restore(snapshot(arrays))
                baseline.t["profile"].fill_(2)
                stream.synchronize()
                baseline.prewarm_graph_output()
                report["native_graph_prewarm"] = {
                    "status": "pass",
                    "miss_count": 0,
                    "synchronize": True,
                    "scope": "outside capture and timing",
                }
                native_graph = torch.npu.NPUGraph()
                native_owner.graphs.append(native_graph)
                with torch.npu.graph(native_graph, stream=capture_stream):
                    baseline.launch()
                capture_stream.synchronize()
                baseline.restore(snapshot(arrays))
        initial = {n: x.copy() for n, x in arrays.items()}
        elapsed, native_elapsed, final_elapsed = [], [], []
        native_phase_samples, service_samples = [], []
        fused_api_samples, native_api_samples = [], []
        phase_samples = []
        device_samples = []
        lifecycle = Lifecycle(g) if a.state_mode == "stateful" else None
        from kvip.v9.index_cache import allocate_cache
        from kvip.v9.owned_index_cache import OwnedIndexCache

        cache_pool, layouts = {}, {}
        owned_pool, owned_payload_seed = {}, {}
        use_owned = a.index_storage == "owned"
        report["index_storage"] = "owned" if use_owned else "derived"

        def owned_backing():
            if not use_owned:
                return None
            return {
                "key": tensors["key"].view(torch.uint8).cpu().numpy(),
                "key_scale": tensors["key_scale"].cpu().numpy(),
            }

        cache_seeds, layout_seeds = {}, {}
        owner.keepalive.append(cache_seeds)
        owner.keepalive.append(cache_pool)
        restore_stream = torch.npu.Stream(device=device)
        # Snapshot mode has an additional, fully validated call even at warmup=0.
        # Stateful mode uses step zero as its preflight and measures later steps.
        for step, phase in measurement_schedule(a.state_mode == "stateful", a.steps, a.warmup, a.iterations):
            preflight = phase == "preflight"
            if a.state_mode == "snapshot":
                for name in ("forward", "reverse", "cursors", "resident_output"):
                    arrays[name] = initial[name].copy()
            event = lifecycle.prepare(arrays, scores, step) if lifecycle else None
            prepare_step(a, g, arrays, step, scores)
            prepare_start = torch.npu.Event(enable_timing=True)
            prepare_start.record(stream)
            if use_owned and a.state_mode == "stateful":
                for b, pool in enumerate(arrays["pool_rows"]):
                    previous_owner = owned_pool.get((int(pool), g.layer_id))
                    length = int(arrays["candidate_lens"][b])
                    if previous_owner is not None and length > previous_owner.length:
                        previous_owner.grow(
                            length, torch.from_numpy(arrays["index_table"][b]).to(device),
                            page_ids=tuple(int(p) for p in arrays["index_table"][b]),
                        )
            # Only ownership boundaries reconstruct the logical source. Normal
            # stateful steps keep the actual device cold bank from the last call.
            reuse_owned = use_owned and all(
                (int(pool), g.layer_id) in owned_pool
                and owned_pool[(int(pool), g.layer_id)].length == int(arrays["candidate_lens"][b])
                for b, pool in enumerate(arrays["pool_rows"])
            )
            if use_owned and not reuse_owned:
                for existing in owned_pool.values():
                    existing.restore_logical()
                owned_pool.clear()
            restore_stream.wait_stream(stream)
            cold_orders = []
            active_buffers = []
            for b, pool in enumerate(arrays["pool_rows"]):
                identity = (int(pool), g.layer_id)
                if identity not in cache_pool:
                    cache_pool[identity] = allocate_cache(torch, g, device)
                active_buffers.append(cache_pool[identity])
                hot = frozen_hot(
                    arrays["forward"][pool],
                    arrays["reverse"][pool],
                    arrays["victims"][b],
                    int(arrays["candidate_lens"][b]),
                    int(arrays["budgets"][b]),
                )
                if a.state_mode == "snapshot" and identity in layout_seeds:
                    layouts[identity] = copy.deepcopy(layout_seeds[identity])
                    with torch.npu.stream(restore_stream):
                        cache_pool[identity].copy_(cache_seeds[identity])
                layout = layouts.setdefault(identity, ColdOrder())
                previous = layout.length
                arrays["index_dirty"][b] = [
                    max(0, min(previous, int(arrays["candidate_lens"][b]) - 4)),
                    int(arrays["candidate_lens"][b]),
                ]
                cold_orders.append(layout.reconcile(hot, int(arrays["candidate_lens"][b])).copy())
                if a.state_mode == "snapshot":
                    arrays["index_dirty"][b] = [0, 0]
            with torch.npu.stream(restore_stream):
                launcher.tensors["index_cache"].copy_(
                    torch.tensor([x.data_ptr() for x in active_buffers], dtype=torch.int64, device=device)
                )
            for name in (
                "query",
                "key",
                "key_scale",
                "weights",
                "query_scale",
                "pool_rows",
                "budgets",
                "row_modes",
                "index_table",
                "index_dirty",
                "resident_table",
                "forward",
                "reverse",
                "cursors",
                "victims",
                "victim_bitmap",
                "candidate_lens",
                "logical_lens",
                "resident_output",
            ):
                if reuse_owned and name in ("key", "key_scale"):
                    if a.state_mode == "snapshot":
                        with torch.npu.stream(restore_stream):
                            tensors[name].copy_(owned_payload_seed[name])
                    continue
                source = torch.from_numpy(arrays[name])
                if name in ("query", "key"):
                    source = source.view(torch.float8_e4m3fn)
                # Caller restore is explicitly ordered before the fused stream.
                with torch.npu.stream(restore_stream):
                    tensors[name].copy_(source.to(device, dtype=tensors[name].dtype))
            restored = torch.npu.Event()
            restored.record(restore_stream)
            stream.wait_event(restored)
            if use_owned:
                setup_start, setup_end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                setup_start.record(stream)
                for b, pool in enumerate(arrays["pool_rows"]):
                    identity = (int(pool), g.layer_id)
                    length = int(arrays["candidate_lens"][b])
                    if identity not in owned_pool:
                        owned_pool[identity] = OwnedIndexCache(
                            torch,
                            g,
                            tensors["key"],
                            tensors["key_scale"],
                            tensors["index_table"][b],
                            pool=int(pool),
                            length=length,
                            budget=int(arrays["budgets"][b]),
                            forward=tensors["forward"],
                            reverse=tensors["reverse"],
                            page_ids=tuple(int(p) for p in arrays["index_table"][b]),
                        )
                        cache_pool[identity] = owned_pool[identity].buffer
                        active_buffers[b] = cache_pool[identity]
                        # A transition starts with the import's resident-slot /
                        # logical-cold ordering, not a previous compact ordering.
                        layouts[identity] = ColdOrder()
                        hot = frozen_hot(
                            arrays["forward"][pool],
                            arrays["reverse"][pool],
                            arrays["victims"][b],
                            length,
                            int(arrays["budgets"][b]),
                        )
                        cold_orders[b] = layouts[identity].reconcile(hot, length).copy()
                    elif a.state_mode == "stateful":
                        lo, hi = map(int, arrays["index_dirty"][b])
                        ids = np.arange(lo, min(hi, length), dtype=np.int64)
                        pages = arrays["index_table"][b, ids // 128]
                        owned_pool[identity].write_logical(
                            torch.from_numpy(ids).to(device),
                            torch.from_numpy(arrays["key"][pages, ids % 128, 0]).to(device),
                            torch.from_numpy(arrays["key_scale"][pages, ids % 128, 0]).to(device),
                        )
                launcher.tensors["index_cache"].copy_(
                    torch.tensor([buffer.data_ptr() for buffer in active_buffers], dtype=torch.int64, device=device)
                )
                launcher.tensors["index_dirty"].zero_()
                setup_end.record(stream)
                stream.synchronize()
                report.setdefault("caller_prepare_device_span_us", []).append(
                    prepare_start.elapsed_time(setup_end) * 1000
                )
                if not reuse_owned:
                    report["index_cache_setup_us"] = setup_start.elapsed_time(setup_end) * 1000
                if a.state_mode == "snapshot" and not cache_seeds:
                    owned_payload_seed.update({name: tensors[name].clone() for name in ("key", "key_scale")})
            if (
                a.state_mode == "snapshot"
                and not cache_seeds
                and expected_failure(g, arrays, scores, case) == Status.OK
            ):
                # First-use bank construction is reported separately. This init
                # posts no transport and is not a substitute for full preflight.
                setup_start, setup_end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                setup_start.record(stream)
                launcher.workspace["mapping_epoch"].fill_(
                    snapshot_mapping_epoch(a.state_mode, use_owned, arrays, initial)
                )
                launcher._call("init")
                setup_end.record(stream)
                stream.synchronize()
                seed_us = setup_start.elapsed_time(setup_end) * 1000
                report["h0_snapshot_seed_us"] = seed_us
                report.setdefault("index_cache_setup_us", seed_us)
                with torch.npu.stream(stream):
                    for b, pool in enumerate(arrays["pool_rows"]):
                        identity = (int(pool), g.layer_id)
                        cache_seeds[identity] = active_buffers[b].clone()
                        layout_seeds[identity] = copy.deepcopy(layouts[identity])
            # Only the paired snapshot restorer can certify this fixed version.
            # Stateful/external mapping writes keep the exact-validation route.
            epoch = snapshot_mapping_epoch(a.state_mode, use_owned, arrays, initial)
            launcher.workspace["mapping_epoch"].fill_(epoch)
            before = snapshot(arrays)
            current = scores[:, :, : int(arrays["candidate_lens"][0])]
            expected = expected_failure(g, arrays, current, case)
            unchanged_index = None
            if use_owned and expected in (Status.SKIPPED, Status.DIRECTORY):
                unchanged_index = ([buffer.cpu().numpy().copy() for buffer in active_buffers], owned_backing())
            native_first = a.measurement_order == "native-first" or (
                a.measurement_order == "alternate" and step % 2 == 0
            )

            def native_sample(expected=expected, before=before, current=current):
                if baseline is None or expected != Status.OK:
                    return None, None, None
                native_phase_us = None
                baseline.restore(before)
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                native_marks = [torch.npu.Event(enable_timing=True) for _ in range(2)] if a.phase_timing else None
                native_api_start = time.perf_counter_ns()
                start.record(stream)
                if native_graph is None:
                    baseline.launch(phase_events=native_marks)
                else:
                    native_graph.replay()
                end.record(stream)
                stream.synchronize()
                native_api_us = (time.perf_counter_ns() - native_api_start) / 1000
                native_us = start.elapsed_time(end) * 1000
                if native_marks is not None:
                    bounds = [start, *native_marks, end]
                    native_phase_us = {
                        name: bounds[i].elapsed_time(bounds[i + 1]) * 1000
                        for i, name in enumerate(("li", "prepare", "kvaf_get"))
                    }
                report["native_baseline"] = baseline.verify(before, current)
                return native_us, native_phase_us, native_api_us

            native_us, native_phase_us, native_api_us = native_sample() if native_first else (None, None, None)
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            final_start, final_end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            init_end = torch.npu.Event(enable_timing=True) if a.phase_timing else None
            layout_checks = a.state_mode == "stateful" and expected == Status.OK and graph is None
            stage_checks = preflight and expected == Status.OK and graph is None
            report["stateful_layout_diagnostics_timed"] = bool(layout_checks)
            li_scores = None

            def after_init(
                event=init_end,
                checks=stage_checks,
                inputs=before,
                buffers=active_buffers,
                orders=cold_orders,
                observe_layout=layout_checks,
            ):
                if event is not None:
                    event.record(stream)
                if checks or observe_layout:
                    stream.synchronize()
                    cache_checks = []
                    for b, pool in enumerate(inputs["pool_rows"]):
                        layout = layouts[(int(pool), g.layer_id)]
                        try:
                            raw = buffers[b].cpu().numpy()
                            # Physical order is not part of the contract. Verify
                            # payload/partition/positions BEFORE using the actual
                            # directory to replay service-window admission.
                            verify_cache(raw, g, inputs, b, backing=owned_backing())
                            header, _, _, ids, _ = inspect_cache(raw, g)
                            orders[b] = ids[g.resident_capacity : g.resident_capacity + header[3]].copy()
                            layout.hot = ids[: header[2]].tolist()
                            layout.cold = orders[b].tolist()
                            layout.length = int(header[1])
                            cache_checks.append({"request": b, "status": "pass"})
                        except AssertionError as error:
                            cache_checks.append({"request": b, "status": "failed", "error": str(error)})
                    if checks:
                        report["preflight_stages"] = {"init_index_cache": cache_checks}
                    else:
                        report.setdefault("stateful_init_checks", []).append(cache_checks)
                        if any(item["status"] != "pass" for item in cache_checks):
                            raise AssertionError("stateful init Index Cache verification failed")

            fused_api_start = time.perf_counter_ns()
            owner.mark_launched()
            start.record(stream)
            if graph is not None:
                graph.replay()
            else:
                launcher.launch_compute(
                    after_init=after_init if stage_checks or layout_checks or init_end is not None else None
                )
                if stage_checks:
                    # Init/LI diagnostics belong only to the untimed preflight.
                    # Never inspect scores after fetch: cache reconciliation reuses them.
                    stream.synchronize()
                    from contract_verify import logical_score_workspace

                    li_scores = logical_score_workspace(
                        launcher.workspace["scores"].cpu().numpy(),
                        [layouts[(int(p), g.layer_id)].hot for p in before["pool_rows"]],
                        cold_orders,
                    )
                    report["preflight_stages"]["li_scores"] = score_workspace_diagnostic(li_scores, current)
                final_start.record(stream)
                launcher.launch_final_submit()
                final_end.record(stream)
                launcher.finish()
            end.record(stream)
            stream.synchronize()
            fused_api_us = (time.perf_counter_ns() - fused_api_start) / 1000
            for name in (
                "topk",
                "destination",
                "miss_counts",
                "resident_lengths",
                "transfers",
                "extra_prefetch_slots",
                "extra_prefetch_counts",
                "status",
                "forward",
                "reverse",
                "cursors",
                "resident_output",
            ):
                arrays[name] = tensors[name].cpu().numpy().copy()
            rows = launcher.rows.cpu().numpy()
            posts = launcher.final_posts.cpu().numpy()
            row_status = launcher.row_status.cpu().numpy()
            report["last_completion"] = completion_diagnostics(
                arrays["status"],
                rows,
                posts,
                row_status,
                launcher.workspace["error_detail"].cpu().numpy(),
                launcher.workspace["plan_count"].cpu().numpy(),
                launcher.workspace["sticky"].cpu().numpy(),
            )
            # Persist failure evidence before any quarantine can terminate process.
            if np.any(arrays["status"] < 0) or np.any(rows[:, 11]) or np.any(posts[:, :, 1:4]):
                np.savez_compressed(
                    root / f"step-{step}-failure.npz",
                    **arrays,
                    **{"before_" + n: x for n, x in before.items()},
                    rows=rows,
                    posts=posts,
                )
                (root / "status.json").write_text(json.dumps(report["last_completion"]), encoding="utf-8")
                print("COMPLETION_DIAGNOSTICS " + json.dumps(report["last_completion"]), flush=True)
            launcher.completion()
            completed_step = step
            if not native_first:
                native_us, native_phase_us, native_api_us = native_sample()
            if stage_checks:
                print("PREFLIGHT_STAGES " + json.dumps(report["preflight_stages"]), flush=True)
                stages = report["preflight_stages"]
                if any(item["status"] != "pass" for item in stages["init_index_cache"]):
                    raise AssertionError(
                        "init Index Cache differs from canonical Key/scale or logical partition; see preflight_stages"
                    )
                if stages["li_scores"]["status"] != "pass":
                    raise AssertionError(
                        "LI logical scores differ from reference before resolve/fetch; see preflight_stages"
                    )
            rt = launcher.workspace["reserved_tokens"].cpu().numpy()
            rs = launcher.workspace["reserved_slots"].cpu().numpy()
            reservations = [
                dict(zip(map(int, rt[b, : int(rows[b, 8])]), map(int, rs[b, : int(rows[b, 8])])))
                for b in range(g.batch)
            ]
            if expected == Status.OK:
                for b, p in enumerate(before["pool_rows"]):
                    hot = frozen_hot(
                        before["forward"][p],
                        before["reverse"][p],
                        before["victims"][b],
                        int(before["candidate_lens"][b]),
                        int(before["budgets"][b]),
                    )
                    admitted = replay_contract(
                        current[b], hot, rows[b], g.candidate_capacity, launcher.rounds, a.window_limit, cold_orders[b]
                    )
                    if admitted != list(reservations[b]):
                        raise AssertionError("reservation order differs from actual service snapshots")
            blocks, offsets, destinations = [
                x.cpu().numpy() for x in (launcher.blocks, launcher.offsets, launcher.destinations)
            ]
            descriptors = [
                [
                    (int(blocks[b * 8192 + i]), int(offsets[b * 8192 + i]), int(destinations[b * 8192 + i]))
                    for i in range(int(rows[b, 247]))
                ]
                for b in range(g.batch)
            ]
            result = check_outcome(g, before, arrays, current, reservations, expected, descriptors)
            if unchanged_index is not None:
                cached, paged = unchanged_index
                if any(
                    not np.array_equal(saved, buffer.cpu().numpy()) for saved, buffer in zip(cached, active_buffers)
                ):
                    raise AssertionError("rejected/skipped call modified owned Index metadata or hot payload")
                if any(not np.array_equal(paged[name], value) for name, value in owned_backing().items()):
                    raise AssertionError("rejected/skipped call modified owned paged cold payload")
            if expected == Status.OK:
                result["index_cache_metrics"] = []
                for b, pool in enumerate(arrays["pool_rows"]):
                    hot = frozen_hot(
                        arrays["forward"][pool],
                        arrays["reverse"][pool],
                        arrays["victims"][b],
                        int(arrays["candidate_lens"][b]),
                        int(arrays["budgets"][b]),
                    )
                    cache_snapshot = active_buffers[b].cpu().numpy()
                    verify_cache(cache_snapshot, g, arrays, b, backing=owned_backing())
                    header, _, _, ids, _ = inspect_cache(cache_snapshot, g)
                    layout = layouts[(int(pool), g.layer_id)]
                    layout.hot = ids[: header[2]].tolist()
                    layout.cold = ids[g.resident_capacity : g.resident_capacity + header[3]].tolist()
                    layout.length = int(header[1])
                    result["index_cache_metrics"].append(cache_metrics(cache_snapshot, g))
                result["index_cache"] = "keys/scales/partition/positions checked"
            device_stages = decode_device_stages(row_status)
            kernel_spans = None
            if expected == Status.OK:
                kernel_ticks, kernel_spans = decode_kernel_timeline(
                    launcher.workspace["kernel_timeline"].cpu().numpy(),
                    g.batch,
                    launcher.li_cores,
                    launcher.final_workers,
                )
                device_stages["ticks"].update(kernel_ticks)
            result.update(
                device_stages=device_stages,
                step=step,
                phase=phase,
                lifecycle=event,
                deferred_requests=int(np.count_nonzero(row_status[:, 3])),
                layer=g.layer_id,
                transfers=arrays["transfers"].tolist(),
                native_us=native_us,
                native_phase_us=native_phase_us,
                fused_us=start.elapsed_time(end) * 1000,
                kernel_spans=kernel_spans,
            )
            if init_end is not None:
                result["phase_us"] = phase_elapsed_us(start, init_end, final_start, final_end, end)
            report["steps"].append(result)
            if lifecycle and expected == Status.OK:
                lifecycle.commit(arrays)
            if expected == Status.OK and a.qsfa != "off":
                completed_step = None  # A later QSFA failure need not leave the device readable.
                report["qsfa"] = check_qsfa(torch, tensors, g, before, arrays)
                if a.qsfa == "required" and report["qsfa"]["status"] != "pass":
                    raise RuntimeError("required QSFA operator is unavailable")
            if preflight:
                report["correctness_preflight"] = {
                    "status": "pass",
                    "qsfa": report.get("qsfa", {}).get("status", "off"),
                }
            if phase == "measurement":
                device_samples.append(device_stages)
                elapsed.append(result["fused_us"])
                fused_api_samples.append(fused_api_us)
                if "phase_us" in result:
                    phase_samples.append(result["phase_us"])
                if native_us is not None:
                    native_elapsed.append(native_us)
                    native_api_samples.append(native_api_us)
                if native_phase_us is not None:
                    native_phase_samples.append(native_phase_us)
                if a.phase_timing and a.intermediate_profiling == "on":
                    service_samples.append(decode_service_stages(launcher.events.cpu().numpy(), launcher.rounds))
                if graph is None:
                    final_elapsed.append(final_start.elapsed_time(final_end) * 1000)
            if expected < 0:
                # Sticky failure is checked without clearing persistent workspace.
                owner.mark_launched()
                launcher.launch()
                launcher.completion()
                if not np.all(tensors["status"].cpu().numpy() == Status.POISONED):
                    raise AssertionError("failed V9 chain accepted a replay")
                break
        report["timing"] = {
            "api_wall": {
                "scope": (
                    "fixed prepared inputs: submit/replay through stream completion; "
                    "excludes fixture restore, validation and initialization"
                ),
                "framework_end_to_end": False,
                "fused_p50_us": float(np.median(fused_api_samples)) if fused_api_samples else None,
                "native_p50_us": float(np.median(native_api_samples)) if native_api_samples else None,
                "fused_samples_us": fused_api_samples,
                "native_samples_us": native_api_samples,
            },
            "fused_p50_us": float(np.median(elapsed)) if elapsed else None,
            "fused_p95_us": float(np.percentile(elapsed, 95)) if elapsed else None,
            "native_p50_us": float(np.median(native_elapsed)) if native_elapsed else None,
            "native_p95_us": float(np.percentile(native_elapsed, 95)) if native_elapsed else None,
            "final_submit_p50_us": float(np.median(final_elapsed)) if final_elapsed else None,
            "includes_final_cq_and_publication": True,
            "reset_and_validation_timed": False,
            "baseline_execution": a.execution,
            "native_chain": "LI -> LIDU MTP prepare_out -> KVAF batch_get",
            "score_layout": "physical_hot_aligned128_then_cold; final IDs mapped to logical",
            "comparison_protocol": {
                "same_execution_mode": True,
                "original_v9_execution": "graph",
                "matches_original_execution": a.execution == "graph",
                "phase_events_inside_timing": a.phase_timing,
                "semantic_difference": "current TND/sparse3 causal Q4; original BSND/sparse0",
            },
            "native_over_fused_speedup": float(np.median(native_elapsed) / np.median(elapsed))
            if native_elapsed and elapsed
            else None,
            "samples": len(elapsed),
            "index_storage": report["index_storage"],
            "mapping_validation": "paired_snapshot_epoch" if use_owned and a.state_mode == "snapshot" else "exact",
            "stateful_layout_diagnostics_timed": report.get("stateful_layout_diagnostics_timed", False),
            "correctness_preflight": report.get("correctness_preflight", {}).get("status"),
            "snapshot_restore": "KV maps/payload and matching Index Cache" if a.state_mode == "snapshot" else None,
            "index_cache_setup_excluded": a.state_mode == "snapshot",
            "ownership_setup_us": report.get("index_cache_setup_us"),
            "h0_snapshot_seed_us": report.get("h0_snapshot_seed_us"),
            "caller_prepare_scope": "separate: snapshot restore/input copies, Index growth or ownership transition",
        }
        report["timing"]["phase_timing_enabled"] = a.phase_timing
        report["timing"]["resolve_includes_index_maintenance"] = True
        report["timing"]["index_maintenance_executor"] = "final_worker_0_after_submit"
        report["timing"]["device_stages"] = summarize_device_stages(device_samples)
        report["timing"]["index_work_last_step"] = device_samples[-1]["index_work"] if device_samples else []
        if phase_samples:
            report["timing"]["phases"] = {
                name: {
                    "p50_us": float(np.median([s[name] for s in phase_samples])),
                    "p95_us": float(np.percentile([s[name] for s in phase_samples], 95)),
                    "samples": len(phase_samples),
                }
                for name in phase_samples[0]
            }
        if native_phase_samples:
            report["timing"]["native_phase_samples_us"] = native_phase_samples
            report["timing"]["native_phases"] = {
                name: {
                    "p50_us": float(np.median([v[name] for v in native_phase_samples])),
                    "p95_us": float(np.percentile([v[name] for v in native_phase_samples], 95)),
                    "samples": len(native_phase_samples),
                }
                for name in native_phase_samples[0]
            }
        report["timing"]["li_service_windows"] = summarize_service_stages(service_samples)
        if a.transport == "npu-memory":
            report["timing"]["diagnostic_only"] = True
            report["timing"]["native_over_fused_speedup"] = None
        report["performance_status"] = "not_applicable_memory_transport" if a.transport == "npu-memory" else "not_gated"
        if a.final_submit_limit_us is not None:
            if not final_elapsed:
                raise ValueError("explicit final-submit gate requires eager execution")
            report["performance_status"] = "pass" if np.median(final_elapsed) < a.final_submit_limit_us else "fail"
        report["status"] = "pass"
        report["correctness_status"] = "pass"
        report["hardware_validation_status"] = (
            "pass"
            if report["native_baseline"]["status"] == "pass" and report["qsfa"]["status"] == "pass"
            else "partial"
        )
        report["source_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parent.iterdir()
            if p.is_file() and p.suffix in (".cpp", ".h", ".inc", ".py", ".txt")
        }
        if report["performance_status"] == "fail":
            report["status"] = "failed_explicit_performance_gate"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        evidence = {"after_" + n: x for n, x in arrays.items()}
        if "before" in locals():
            evidence.update({"before_" + n: x for n, x in before.items()})
        # Read device diagnostics only after verified fused completion. Keep
        # these failure-only transfers outside every measured/captured path.
        if "completed_step" in locals() and completed_step == step:
            evidence.update(rows=rows, posts=posts, reference_scores=current)
            if li_scores is not None:
                evidence["li_scores_before_fetch"] = li_scores
            try:
                for name in (
                    "reserved_tokens",
                    "reserved_slots",
                    "eligible_victims",
                    "eligible_counts",
                    "hot_thresholds",
                    "threshold_valid",
                    "hot_counts",
                    "hot_tokens",
                ):
                    evidence["workspace_" + name] = launcher.workspace[name].cpu().numpy().copy()
            except Exception as diagnostic_error:
                report["diagnostic_error"] = repr(diagnostic_error)
                (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(root / "failure-state.npz", **evidence)
        if owner is not None and owner.in_flight:
            owner.quarantine("v9_benchmark_unverified_completion", error)
        raise
    finally:
        if owner is not None:
            owner.close()
        if native_owner is not None:
            native_owner.close()
    return report


def main(argv=None):
    a = parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    run_root = a.output.resolve() / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(exist_ok=False)
    reports = []
    for batch in a.batch_values:
        for queries in a.query_values:
            for case in cases(a.case_suite, a.miss_rates, a.prefetch_decoys):
                for layer in range(a.layers):
                    root = run_root / f"b{batch}-q{queries}-{case.name}-layer{layer}"
                    root.mkdir()
                    candidate = a.history + (a.steps * queries if a.state_mode == "stateful" else 0)
                    g, arrays, scores = make_fixture(
                        batch=batch,
                        queries=queries,
                        candidate=candidate,
                        budget=a.resident,
                        tail=a.tail,
                        layers=a.layers,
                        layer=layer,
                        seed=a.seed,
                        case=case,
                    )
                    arrays["candidate_lens"].fill(a.history)
                    arrays["logical_lens"].fill(a.history + a.tail)
                    (root / "manifest.json").write_text(
                        json.dumps(
                            manifest(g, case)
                            | {
                                "seed": a.seed,
                                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    # Static input snapshot is sufficient to reproduce payload
                    # from pool/layer/token; large output bytes are saved on error.
                    np.savez_compressed(
                        root / "fixture.npz",
                        **{n: x for n, x in arrays.items() if n not in ("resident_output", "topk", "destination")},
                    )
                    try:
                        report = (run_cpu if a.backend == "cpu" else run_npu)(a, g, arrays, scores, case, root)
                    except Exception as error:
                        report = (
                            json.loads((root / "report.json").read_text(encoding="utf-8"))
                            if (root / "report.json").exists()
                            else {}
                        )
                        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
                    if report.get("steps"):
                        device_aic = report.get("device_resources", {}).get("effective_aic", a.aic_count or 32)
                        report["workload"] = workload_summary(
                            case,
                            report["steps"][-1],
                            li_geometry(
                                batch, a.li_tiling, a.intermediate_fetches, device_aic, a.service_aics, queries
                            ),
                        )
                    report.setdefault(
                        "source_sha256",
                        {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).resolve().parent.iterdir()
                            if p.is_file() and p.suffix in (".cpp", ".h", ".inc", ".py", ".txt")
                        },
                    )
                    report.update(
                        batch=batch,
                        queries=queries,
                        case=case.name,
                        layer=layer,
                        backend=a.backend,
                        state_mode=a.state_mode,
                    )
                    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    (root / "data_record.md").write_text(render_record([report], vars(a)), encoding="utf-8")
                    reports.append(report)
                    (run_root / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
                    (run_root / "data_record.md").write_text(render_record(reports, vars(a)), encoding="utf-8")
                    print(
                        json.dumps({k: report[k] for k in ("status", "batch", "queries", "case", "layer", "backend")}),
                        flush=True,
                    )
                    if "workload" in report:
                        print("WORKLOAD " + json.dumps(report["workload"]), flush=True)
                    if "timing" in report:
                        print("TIMING " + json.dumps(report["timing"]), flush=True)
    print(f"data_record={run_root / 'data_record.md'}", flush=True)
    print(f"report={run_root / 'summary.json'}", flush=True)
    if any(r["status"] != "pass" for r in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
