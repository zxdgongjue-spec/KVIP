"""Graph A/B using the original LI tiling, service adapters and AIV maintenance."""

import json
import traceback
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from device_trace import collect_core, core_summary
from index_update import DirectCacheUpdate
from kvip.operators.lifetch.forward import LIFetchForward
from kvip.runtime.loader import load_lifetch_library
from prefetch_audit import actual_reads, completion_failures, score_rows, selected_bounds
from runner import progress
from reporting import CorrectnessRecorder, atomic_json, timing_summary

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "csrc/operators/lightning_indexer_with_fetch"
NATIVE_BITMAP_WORDS = 4176


def prepare_native_bitmap(torch, native):
    """Adapt the historical fixture before graph capture and timing."""
    native.bitmap = torch.zeros((native.batch, NATIVE_BITMAP_WORDS), dtype=torch.int32, device=native.bitmap.device)


def seed_native_bitmap(torch, native, row, bitmap):
    """Copy LRU's capacity-sized bitmap into the fixed service ABI row."""
    native.bitmap[row].zero_()
    native.bitmap[row, : len(bitmap)].copy_(torch.as_tensor(bitmap, device=native.bitmap.device))


def run(args, run_dir):
    from attention import check_sfa
    from contract_payload import payload
    from kvip.operators.lifetch import binding as launch_module
    from kvip.runtime.device import query_device_resources
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry
    from kvip.cache.lru import SlotLRU
    from native_batchget_support import NativeFetchSession
    from native_mtp import NativeMtp
    from native_support import _load_native_operators, find_headers
    from kvip.runtime.build import build_libraries
    from real_fetch_inputs import batch_inputs, native_li, prepare_full_chain_fixture
    from real_fetch_session import RealFetchSession
    from execution import native_log

    torch, registration, _ = _load_native_operators("c8_offload")
    import torch_npu

    registration.require_a5_dsa_registration(("lidu_fused_batch_get_prepare_mtp_out",))
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    resources = query_device_resources(args.cann, args.device, 0)
    aics = resources["effective_aic"]

    class FiniteSession(NativeFetchSession):
        @classmethod
        def open(cls, *pos, **kw):
            kw["shared_pool"] = getattr(args, "shared_pool", None)
            kw["payload_factory"] = lambda t, b, h: t.from_numpy(np.stack([payload(i, np.arange(h)) for i in range(b)]))
            return super().open(*pos, **kw)

    class Owner(RealFetchSession):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            self.support = SimpleNamespace(**vars(self.support))
            self.support._session_class = lambda: FiniteSession

    summaries = []
    failed = False
    original_args = args
    cases = [
        (b, args.history, args.resident, args.index_heads, args.intermediate_fetches)
        for b in map(int, args.batches.split(","))
    ]
    if args.runtime_shapes:
        cases = [
            (1, 65536, 8192, 64, 4),
            (8, 131072, 16384, 32, 8),
            (16, 65536, 8192, 32, 4),
            (1, 131072, 16384, 32, 8),
        ]
    shared_build = run_dir / "universal-build"
    built_digest = None
    for case, (batch, history, resident, heads, windows) in enumerate(cases):
        args = SimpleNamespace(
            **{
                **vars(original_args),
                "history": history,
                "resident": resident,
                "index_heads": heads,
                "intermediate_fetches": windows,
            }
        )
        folder = run_dir / f"case{case}-b{batch}-prototype-graph"
        folder.mkdir()
        stage = "fixture preparation"
        progress(run_dir, f"BS={batch}: {stage}; logs={folder}")
        with native_log(folder / "native.log"):
            common, data, build = (folder / "fixture", folder / "input", shared_build)
            pool = getattr(args, 'shared_pool', None)
            fixture_key = ('urma', args.history, args.resident, args.miss_percent, args.index_heads, getattr(args, 'seed', 20261104))
            if pool is not None and fixture_key in pool.fixtures:
                cached_path, fixture_info = pool.fixtures[fixture_key]
                common.symlink_to(cached_path.resolve(), target_is_directory=True)
            else:
                fixture_info = prepare_full_chain_fixture(
                    common, args.history, args.resident, args.miss_percent,
                    index_heads=args.index_heads, seed=getattr(args, 'seed', 20261104))
                if pool is not None:
                    pool.fixtures[fixture_key] = (common, fixture_info)
            geometry = li_geometry(batch, "auto", args.intermediate_fetches, aics, 0)
            batch_inputs(common, data, batch, args.history, geometry["rotating_aic"])
            weights = np.fromfile(data / "arg2.bin", np.float32)
            (weights.view(np.uint32) >> 16).astype(np.uint16).tofile(data / "native_weights_bf16.bin")
            options = dict(
                RUN_MODE="npu",
                SOC_VERSION=args.soc,
                ASCEND_CANN_PACKAGE_PATH=args.cann,
                KVAF_INCLUDE=find_headers(args.kvaf_root),
            )
            if getattr(args, "optimized_resolve", False):
                options.update(KVIP_FUSE_QUERY_WORK="ON", KVIP_OUTPUT_WORKER="ON")
            if not shared_build.exists():
                progress(run_dir, f"BS={batch}: resolve shared LIFetch build cache; logs={folder}")
                (library,) = build_libraries(
                    cache_dir=args.kvaf_root / "build-npu/lifetch",
                    cann=args.cann, headers=options["KVAF_INCLUDE"], soc=args.soc)
                shared_build.symlink_to(library.parent.resolve(), target_is_directory=True)
            progress(run_dir, f"BS={batch}: REUSE LIFetch build={shared_build.resolve()}; logs={folder}")
            import hashlib

            # Fail before allocating/populating DDR if shared dependencies cannot load.
            load_lifetch_library(build / "libv9_bridge.so")

            digest = hashlib.sha256((build / "libv9_bridge.so").read_bytes()).hexdigest()
            if built_digest is not None and digest != built_digest:
                raise AssertionError("Binary changed between runtime shapes")
            built_digest = digest
            progress(
                run_dir,
                f"UNIVERSAL binary_sha256={digest} batch={batch} H={history} R={resident} G={heads} windows={windows}",
            )
            queue_views = batch * (final_aivs(batch, "auto", aics, args.final_workers or 0) + 1)
            if args.q_workers:
                # Two contiguous private Jettys for each (Q, worker), after the
                # prefetch pairs; the original final views may be reused.
                queue_views = max(queue_views, 2 * batch + 8 * args.q_workers)
            workers = (queue_views + 1) // 2
            progress(run_dir, f"BS={batch}: DDR Put and transport initialization; logs={folder}")
            owner = Owner.open(
                torch,
                device,
                batch=batch,
                history=args.history,
                resident=args.resident,
                server=args.kvaf_server,
                kvaf_root=args.kvaf_root,
                workers=workers,
            )
            try:
                progress(run_dir, f"BS={batch}: graph A/B, AIV maintenance and stateful checks; logs={folder}")
                report = run_device(
                    args,
                    batch,
                    folder,
                    torch,
                    torch_npu,
                    device,
                    owner,
                    data,
                    build,
                    native_li,
                    NativeMtp,
                    launch_module,
                    SlotLRU,
                    check_sfa,
                    aics,
                )
                failed |= report.get("correctness", "PASS") != "PASS"
                report.update(
                    batch=batch, history=args.history, resident=args.resident, miss_percent=args.miss_percent,
                    index_heads=args.index_heads, seed=args.seed if hasattr(args, 'seed') else 20261104,
                    fixture=fixture_info,
                    resources=resources,
                    resource_geometry=geometry,
                    final_workers=final_aivs(batch, "auto", aics, args.final_workers or 0),
                    build_options={k: str(v) for k, v in options.items()},
                )
                (folder / "report.json").write_text(json.dumps(report, indent=2))
            except BaseException:
                (folder / "error.log").write_text(traceback.format_exc())
                raise
            finally:
                owner.close()
        if args.fetch_only_compare:
            lines = [f"BS={batch} Fetch-only graph: exact fused Q worker; native descriptors prepacked outside timing; byte checks PASS.",
                     "query | AIV | jetty/AIV | entries | safe/conflict | bytes | native p50/p99 us | native GB/s | fused p50/p99 us | fused GB/s | change"]
            results = report['fetch_only']
            for query in ('Q1', 'Q2', 'Q3', 'Q4', 'Q4_all'):
                old = next(r for r in results if r['query'] == query and r['version'] == 'native')
                new = next(r for r in results if r['query'] == query and r['version'] == 'fused')
                delta = (new['p50_us'] / old['p50_us'] - 1) * 100
                native_bw = f"{old['effective_GBps']:.2f}" if old['effective_GBps'] is not None else '--'
                fused_bw = f"{new['effective_GBps']:.2f}" if new['effective_GBps'] is not None else '--'
                lines.append(f"{query} | {old['aiv']} | 2 | {old['count']} | {old['safe_count']}/{old['conflict_count']} | {old['bytes']} | "
                             f"{old['p50_us']:.3f}/{old['p99_us']:.3f} | {native_bw} | "
                             f"{new['p50_us']:.3f}/{new['p99_us']:.3f} | {fused_bw} | {delta:+.2f}%")
            for r in results:
                lines.extend(r.get('stages', []))
                if r['version'] == 'fused_before':
                    after = next(x for x in results if x['query'] == r['query'] and x['version'] == 'fused')
                    lines.append(f"FIX A/B {r['query']}: before={r['p50_us']:.3f} after={after['p50_us']:.3f} us change={(after['p50_us']/r['p50_us']-1)*100:+.2f}%")
            lines.append('Q1-Q4 are individual queries; Q4_all runs all four concurrently. Zero-entry cases measure launch/control only.')
            line = '\n'.join(lines)
            summaries.append(line)
            (run_dir / 'cases.log').write_text('\n'.join(summaries) + '\n')
            print(line, flush=True)
            continue
        times = report["timing"]
        line = (
            f"binary_sha256={built_digest} early_fetch={args.early_fetch} streaming={args.streaming} q_workers={args.q_workers}\nBS={batch} prototype {report.get('correctness', 'PASS')} index_heads={args.index_heads} windows={geometry['intermediate_fetches']} requested_windows={args.intermediate_fetches} history={args.history} hot={args.resident} topk=2048 Q=4 graph=True\n"
            + "graph_p50_us="
            + ", ".join((f"{k}:{v['p50_us']:.3f}" for k, v in times.items()))
            + f"\nfull_speedup={times['native_chain']['p50_us'] / times['prototype_with_update']['p50_us']:.3f}x"
            + "; full=init + LI/Fetch/resolve + service output mapping + Index update; fixture restore/verification and SFA/LRU CPU excluded\n"
            + f"SFA_max_abs={report['sfa']['max_abs']:.6g}; stateful_rounds={report['stateful_rounds']} "
            + f"native_unique_misses={report['native_misses_per_request']}"
        )
        line += "\nfixture=" + str(fixture_info["hot_background"]) + "; prefetch_predicate=score>resident_threshold"
        line += "; actual_miss_pct_per_query=" + json.dumps(
            [round(count * 100 / 2048, 4) for count in fixture_info["final_misses_per_query"]]
        )
        native_reads = sum(report["native_misses_per_request"])
        line += f"\nactual_reads native: total={native_reads} unused=0 (final selected misses)"
        for label, samples in report.get("split_samples", {}).items():
            ranges = {key: [min(x[key] for x in samples), max(x[key] for x in samples)]
                      for key in ("final_reads", "nonconflicting_reads", "conflicting_reads")}
            line += f"\nfinal_split {label}: min_max=" + json.dumps(ranges)
        for label, samples in report.get("read_samples", {}).items():
            ranges = {key: [min(row[key] for row in samples), max(row[key] for row in samples)] for key in samples[0]}
            line += f"\nactual_reads {label}: min_max=" + json.dumps(ranges)
        if report.get("full_trace"):
            line += "\n" + core_summary(report["full_trace"], label="full")
        if report.get("cache_update"):
            line += "\ncache_update=" + json.dumps(report["cache_update"])
        if report.get("wall_costs"):
            line += "\nBENCH WALL COSTS seconds=" + json.dumps(report["wall_costs"])
        summaries.append(line)
        (run_dir / "cases.log").write_text("\n".join(summaries) + "\n")
        print(line, flush=True)
    progress(run_dir, "all prototype graph cases completed")
    return int(failed)


def run_device(
    args,
    b,
    folder,
    torch,
    torch_npu,
    device,
    owner,
    data,
    build,
    native_li,
    NativeMtp,
    launch_module,
    SlotLRU,
    check_sfa,
    aics,
):
    from real_fetch_inputs import install_remote_addresses

    h, r = (args.history, args.resident)
    install_remote_addresses(data, owner.remote_token_addresses.cpu().numpy(), b, h)
    arguments = [
        torch.from_numpy(np.fromfile(data / ("native_weights_bf16.bin" if i == 2 else f"arg{i}.bin"), np.uint8)).to(
            device
        )
        for i in range(14)
    ]
    arguments[1] = arguments[1].repeat(b)
    arguments[4] = arguments[4].repeat(b)
    arguments[7].view(torch.int32).copy_(torch.arange(b * h // 128, device=device, dtype=torch.int32))
    fixture = dict(
        query=arguments[0].view(torch.float8_e4m3fn).reshape(b, 4, args.index_heads, 128),
        key=arguments[1].view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
        actual_query=torch.arange(1, b + 1, device=device, dtype=torch.int32) * 4,
        li_query_lengths=torch.full((b,), 4, device=device, dtype=torch.int32),
        actual_key=torch.full((b,), h, device=device, dtype=torch.int32),
        block_table=arguments[7].view(torch.int32).reshape(b, -1),
    )
    fixture["native_inputs"] = dict(
        weights=arguments[2].view(torch.bfloat16).reshape(b, 4, args.index_heads),
        query_scale=arguments[3].view(torch.float32).reshape(b, 4, args.index_heads),
        key_scale=arguments[4].view(torch.float32).reshape(-1, 128, 1),
    )
    stream = torch.npu.current_stream(device)
    capture = torch.npu.Stream(device=device)
    native = NativeMtp(torch, device, owner, fixture, source_tokens=h, padded_tokens=h)
    prepare_native_bitmap(torch, native)
    state = {}
    correctness = CorrectnessRecorder()
    boundary_phases = {}
    completed_timings = {}
    initial_selected = native_li(torch_npu, fixture).cpu().reshape(b, -1).numpy()
    logical_scores = np.load(folder / "fixture/logical_scores.npy")
    physical_order = np.load(folder / "fixture/physical_to_logical.npy")
    scores = logical_scores[:, physical_order]
    expected_ids = np.stack([np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in scores])
    for row in initial_selected.reshape(b, 4, 2048):
        if not np.array_equal(np.sort(row, axis=-1), expected_ids):
            raise AssertionError("native LI differs from independent fixture score oracle")
    initial_orders = []
    policies = []
    for row in range(b):
        policy = SlotLRU()
        policy.resize(r, r)
        policy.commit(initial_selected[row][initial_selected[row] < r])
        order, slot = ([], int(policy.head[0]))
        for _ in range(r):
            order.append(slot)
            slot = int(policy.next[0, slot])
        initial_orders.append(order)
        policies.append(policy)
        native.preferred[row].copy_(torch.from_numpy(policy.slots[0]).to(device))
        seed_native_bitmap(torch, native, row, policy.bitmap[0])

    def native_call():
        state["native_topk"] = native_li(torch_npu, fixture)
        native.launch(state["native_topk"])

    def native_check():
        counts = native.verify(state["native_topk"])
        expected = [len(set(int(t) for t in ids if t >= r)) for ids in initial_selected]
        if counts[:-1] != expected:
            raise AssertionError(f"native Get work differs from logical misses: {counts[:-1]} != {expected}")
        state["native_misses"] = counts[:-1]

    bench_only = getattr(args, "mode", None) == "bench"
    wall_costs = {}
    full_trace = {}
    read_samples = {}
    split_samples = {}

    def measure(call, reset, check, read_label=None, timing_name=None):
        costs = dict(reset_s=0., validation_s=0., replay_wait_s=0., completion_s=0., reads_s=0.)
        raw_check, raw_reset = check, reset
        def check():
            began = time.perf_counter()
            try:
                return correctness.check(state["check_phase"], raw_check)
            finally:
                costs['validation_s'] += time.perf_counter() - began
        def reset():
            began = time.perf_counter()
            raw_reset()
            stream.synchronize()
            costs['reset_s'] += time.perf_counter() - began
        state["check_phase"] = f"timing {read_label or getattr(call, '__name__', 'operation')}"
        reset()
        call()
        stream.synchronize()
        check()
        reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        owner.graphs.append(graph)
        with torch.npu.graph(graph, stream=capture):
            call()
        capture.synchronize()
        graph.replay()
        stream.synchronize()
        check()
        samples = []
        for step in range(args.warmup + args.iterations):
            reset()
            stream.synchronize()
            start, end = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            began = time.perf_counter()
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            costs['replay_wait_s'] += time.perf_counter() - began
            if step >= args.warmup:
                samples.append(start.elapsed_time(end) * 1000)
            if not bench_only or step == args.warmup + args.iterations - 1:
                check()
            elif read_label == "prototype_with_update":
                began = time.perf_counter()
                verify(True, completion_only=True)
                costs['completion_s'] += time.perf_counter() - began
            elif bench_only:
                began = time.perf_counter()
                probe = owner.session.runtime.acquire_fused_batch_get_context(native.output, stream)
                owner.session.runtime.release_fused_batch_get_context(probe, stream)
                costs["completion_s"] += time.perf_counter() - began
            if read_label and step >= args.warmup and (not bench_only or step == args.warmup + args.iterations - 1):
                began = time.perf_counter()
                split_samples.setdefault(read_label, []).append(dict(state["completion"]))
                read_samples.setdefault(read_label, []).append(
                    actual_reads(
                        rotating.mapping.cpu().numpy(),
                        topk.cpu().numpy(),
                        rotating.rows.cpu().numpy().view(np.uint32),
                        r,
                    )
                )
                costs['reads_s'] += time.perf_counter() - began
        wall_costs[timing_name or read_label or 'operation'] = costs
        atomic_json(folder / 'bench_wall_costs.json', wall_costs)
        print(f'BENCH WALL COSTS {timing_name or read_label}: {json.dumps(costs)}', flush=True)
        measured = timing_summary(samples)
        completed_timings[timing_name or read_label or getattr(call, '__name__', 'operation')] = measured
        atomic_json(folder / 'completed_timings.json', dict(
            batch=b, miss_percent=args.miss_percent, timing=completed_timings, correctness=correctness.status, failures=correctness.failures))
        if read_label == "prototype_with_update" and args.comparison_output:
            reset()
            stream.synchronize()
            graph.replay()
            stream.synchronize()
            check()
            from transport_snapshot import save
            save(args.comparison_output, state["ids"], state["slots"], owner.output.cpu().numpy(),
                 rotating.mapping.cpu().numpy(), rotating.lru_slots.cpu().numpy(),
                 rotating.rows.cpu().numpy().view(np.uint32), owner.remote_token_addresses.cpu().numpy())
        if read_label == "prototype_with_update" and args.trace_rounds:
            progress(folder.parent, f"BS={b}: full graph profiler ({args.trace_rounds} replays); logs={folder}")
            if args.streaming:
                pipeline.set_diagnostics(True)
                stream.synchronize()
            def trace_reset():
                reset()
                if args.streaming:
                    pipeline.times.zero_()
            try:
                full_trace.update(collect_core(torch, rotating, graph, trace_reset, check, args.trace_rounds, folder,
                                               full=True, diagnostics=pipeline.diagnostic_cycles if args.streaming else None,
                                               fused_final=args.fused_final))
            finally:
                if args.streaming:
                    pipeline.set_diagnostics(False)
                    stream.synchronize()
        if timing_name == 'native_chain' and args.trace_rounds:
            from device_trace import collect_native
            collect_native(torch, graph, reset, check, args.trace_rounds, folder)
        graph.reset()
        owner.graphs.remove(graph)
        return measured

    native.reset()
    native_call()
    stream.synchronize()
    native_check()
    timings = {"native_chain": measure(native_call, native.reset, native_check, timing_name="native_chain")}
    # The previous native chain has prepared stable LIDU descriptors and maps.
    # Restore only the output so that the captured graph contains KVAF Get alone.
    if not bench_only:
        timings["native_get_only"] = measure(
            native.fetch, lambda: native.output.copy_(native.initial_payload), native_check, timing_name="native_get_only")
    reference = state["native_topk"].cpu().reshape(b, 4, 2048).numpy()
    owner.acquire()
    rotating = launch_module.LIFetchBinding(
        torch,
        build / "libv9_bridge.so",
        arguments,
        owner,
        batch=b,
        history=h,
        resident=r,
        aic_count=aics,
        intermediate_fetches=args.intermediate_fetches,
        final_workers=args.final_workers or 0,
        prefetch_limit=args.prefetch_tokens,
        output_stride=owner.output.shape[1],
    )
    topk = arguments[9].view(torch.int32).reshape(b, 4, 2048)
    prefix = 4 * 2048 * 6 + 1024 + 4 * h * 2
    tail = h * 8 + 8192 * 64 + 4 * 128
    addresses = arguments[13][b * prefix :].view(torch.int64).reshape(b, tail // 8)[:, :h]
    update = DirectCacheUpdate(
        torch,
        arguments[1].reshape(b, h, 128),
        arguments[4].view(torch.float32).reshape(b, h),
        addresses,
        owner.output,
        batch=b,
        history=h,
        hot=r,
        topk=topk,
        locations=rotating.locations,
        rotating=rotating,
    )
    if args.streaming:
        from streaming_forward import StreamingForward
        pipeline = StreamingForward(rotating, update, torch, args.prefetch_tokens, fused_final=args.fused_final, q_workers=args.q_workers)
        pipeline.legacy_layout = args.legacy_fetch_layout
        pipeline.parallel_finalize = args.parallel_fetch_finalize
        pipeline.prepare_metadata = args.prepare_update_metadata
        pipeline.set_diagnostics(False)
    else:
        pipeline = LIFetchForward(rotating, update)
    counter_clock_mhz = (pipeline.calibrate_counter_mhz()
                         if args.streaming and args.q_workers and not args.fetch_only_compare else None)
    if counter_clock_mhz is not None:
        pipeline.counter_clock_mhz = counter_clock_mhz
        print(f"GetSystemCycle calibration: {counter_clock_mhz:.2f} MHz (independent long graph event)", flush=True)
    saved = [arguments[i].clone() for i in (1, 4, 13)]
    update.set_order(initial_orders)
    order_snapshot = (rotating.lru_slots.clone(), rotating.reserved_hot.clone())

    def reset():
        if owner.in_flight:
            raise RuntimeError("previous replay completion not checked")
        for i, seed in zip((1, 4, 13), saved):
            arguments[i].copy_(seed)
        update.logical.copy_(update.identity)
        rotating.hot_valid.fill_(1)
        if bench_only:
            rotating.lru_slots.copy_(order_snapshot[0])
            rotating.reserved_hot.copy_(order_snapshot[1])
        else:
            update.set_order(initial_orders)
        owner.reset()
        owner.mark_launched()

    def full():
        pipeline()

    def verify(updated=False, persistent=True, completion_only=False):
        rows = rotating.rows.cpu().numpy()
        posts = rotating.final_posts.cpu().numpy()
        statuses = rotating.row_status.cpu().numpy()
        cq = [status.cpu().view(torch.uint64).numpy() for status in (rotating.prefetch_status, rotating.final_status)]
        failures = completion_failures(rows, posts, statuses, *cq)
        if failures:
            phase = state.get("check_phase", "graph replay")
            path = folder / "completion_failure.json"
            try:
                bounds = selected_bounds(topk.cpu().numpy(), h)
            except Exception as exc:
                bounds = dict(error=f"{type(exc).__name__}: {exc}")
            try:
                scores = score_rows(arguments[12][:4 * h * 2].cpu().numpy(), h)
            except Exception as exc:
                scores = dict(error=f"{type(exc).__name__}: {exc}")
            path.write_text(json.dumps(dict(phase=phase, requests=failures, selected_bounds=bounds,
                                            first_request_score_rows=scores), indent=2))
            print(f"COMPLETION FAILURE: phase={phase}; first={json.dumps(failures[0])}; full={path}", flush=True)
            print(f"LI OUTPUT BOUNDS: {json.dumps(bounds)}", flush=True)
            print(f"LI SCORE WORDS request=0: {json.dumps(scores)}", flush=True)
            raise RuntimeError(
                f"completion failed phase={phase}; bad_requests={len(failures)}; "
                f"first={json.dumps(failures[0])}; full={path}"
            )
        if not updated:
            destinations = rotating.destinations.cpu().numpy().reshape(-1)
            lru_slots = rotating.lru_slots.cpu().numpy()
            for row_index in range(b):
                # Compare physical hot slots, independent of resolver's inverse map.
                used_slots = set(lru_slots[row_index, : int(rows[row_index, 8])].tolist())
                for post in posts[row_index]:
                    count, start, safe = int(post[0]), int(post[4]), int(post[5])
                    if safe > count:
                        raise AssertionError("Invalid nonconflicting descriptor count")
                    for index, destination in enumerate(destinations[start : start + count]):
                        slot = int(destination) % owner.output.shape[1]
                        if (slot in used_slots) != (index >= safe):
                            raise AssertionError(
                                f"Final split mismatch: request={row_index} descriptor={start + index}"
                            )
        state["completion"] = dict(
            final_reads=int(statuses[:, 1].astype(np.uint64).sum()),
            nonconflicting_reads=int(posts[:, :, 5].astype(np.uint64).sum()),
            conflicting_reads=int((posts[:, :, 0].astype(np.uint64) - posts[:, :, 5].astype(np.uint64)).sum()),
            native_deferred_rows=int(np.count_nonzero(statuses[:, 3])),
            completed_checkpoints=int(statuses[:, 4].astype(np.uint64).sum()),
        )
        owner.mark_completed(completion_verified=True)
        if updated:
            update.check_capacity()
        if completion_only:
            return
        selected = topk.cpu().numpy()
        if np.any(selected < 0) or np.any(selected >= h):
            raise RuntimeError("TopK contains out-of-range positions")
        logical = state.get("before_logical", np.broadcast_to(np.arange(h), (b, h)))
        ids = np.take_along_axis(logical, selected.reshape(b, -1), axis=1).reshape(b, 4, 2048)
        slots = rotating.locations.reshape(b, -1).cpu().long()
        if (slots < 0).any() or (slots >= (r if persistent else owner.output.shape[1])).any():
            raise RuntimeError("Fetch slot outside cache")
        state["ids"], state["slots"] = (ids, slots.numpy())
        if not np.array_equal(np.sort(ids, axis=-1), np.sort(reference, axis=-1)):
            raise AssertionError("TopK changed")
        if updated:
            if not np.array_equal(update.logical_output.cpu().numpy().reshape(b, 4, 2048), ids):
                raise AssertionError("Service logical TopK output differs from pre-update IDs")
            if not torch.equal(update.slot_output.cpu().reshape(b, -1).long(), slots):
                raise AssertionError("Service SFA slot output differs from Fetch destinations")
        reference_device = getattr(owner.session, 'reference', None)
        output = owner.output if reference_device is not None else owner.output.cpu()
        for row in range(b):
            if slots[row].min() < 0 or slots[row].max() >= (r if persistent else owner.output.shape[1]):
                raise AssertionError("slot outside cache")
            if reference_device is not None:
                actual = output[row, slots[row].to(device)]
                expected = reference_device[row, torch.from_numpy(ids[row].reshape(-1)).to(device)]
            else:
                actual = output[row, slots[row]]
                expected = owner.session.cpu_payload[row, ids[row].reshape(-1)]
            if not torch.equal(actual, expected):
                actual, expected = actual.cpu(), expected.cpu()
                output = output.cpu()
                mismatches = (actual != expected).reshape(actual.shape[0], -1).any(dim=1)
                positions = mismatches.nonzero().flatten()[:4].tolist()
                lru = rotating.lru_slots[row].cpu().tolist()
                changes = rotating.changes[row].cpu().tolist()
                details = []
                for pos in positions:
                    token, slot = (int(selected[row].reshape(-1)[pos]), int(slots[row, pos]))
                    ordinal = lru.index(slot) if slot in lru else -1
                    details.append(
                        dict(
                            query=pos // 2048,
                            topk_pos=pos % 2048,
                            token=token,
                            slot=slot,
                            victim_ordinal=ordinal,
                            slot_owner=changes[ordinal] if ordinal >= 0 else None,
                        )
                    )
                if args.streaming_fetch:
                    from streaming_fetch_diagnostics import summarize_mismatches
                    report = summarize_mismatches(
                        row, selected[row], ids[row], slots[row].numpy(), output[row].numpy(),
                        owner.session.cpu_payload[row].numpy(), rotating.destinations.cpu().numpy(),
                        rotating.fallback_addresses.cpu().numpy(), posts[row], lru,
                        rotating.mapping[row].cpu().numpy(), owner.output.shape[1], r,
                        owner.remote_token_addresses[row].cpu().numpy())
                    report["phase"] = state.get("check_phase", "unknown")
                    report["initial_output_matches"] = [
                        bool(torch.equal(owner.initial_output[row, item["slot"]].cpu(),
                                         owner.session.cpu_payload[row, item["expected_token"]]))
                        for item in report["examples"]
                    ]
                    report_path = folder / "streaming_fetch_kv_mismatch.json"
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    try:
                        from streaming_fetch_diagnostics import audit_queue_writes
                        destinations = rotating.destinations.cpu().numpy().reshape(-1)
                        remote = rotating.fallback_addresses.cpu().numpy().reshape(-1).view(np.uint64)
                        report["actual_sq_writers"], report["final_wqe_comparison"] = audit_queue_writes(
                            owner, rows, posts, row, [item["slot"] for item in report["examples"]],
                            destinations, remote, q_workers=args.q_workers)
                    except Exception as exc:
                        report["sq_audit_error"] = f"{type(exc).__name__}: {exc}"
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"STREAMING FETCH KV AUDIT: {json.dumps(report)}; details={report_path}", flush=True)
                raise AssertionError(
                    f"KV byte mismatch: batch={row} mismatched_tokens={int(mismatches.sum())} examples={details}"
                )
        state["ids"], state["slots"] = (ids, slots.numpy())

    def verify_all_valid_hot():
        """Catch stale KV beyond the current TopK, including displaced slots."""
        valid = rotating.hot_valid.cpu().numpy().reshape(b, r)
        logical = update.logical.cpu().long()
        reference_device = getattr(owner.session, 'reference', None)
        hot_kv = owner.output[:, :r] if reference_device is not None else owner.output[:, :r].cpu()
        for row in range(b):
            mask = torch.from_numpy(valid[row] != 0)
            if reference_device is not None:
                expected = reference_device[row, logical[row, :r][mask].to(device)]
                actual = hot_kv[row, mask.to(device)]
            else:
                expected = owner.session.cpu_payload[row, logical[row, :r][mask]]
                actual = hot_kv[row, mask]
            if not torch.equal(actual, expected):
                raise AssertionError(f"valid hot slot has mismatched Index/KV owner: request={row}")

    def check_pipeline(orders, cold_start=False, label="regular"):
        state["check_phase"] = label
        reset()
        update.set_order(orders)
        if cold_start:
            rotating.hot_valid.zero_()
        before_valid = rotating.hot_valid.cpu().numpy().copy()
        before_layout = update.logical.cpu().numpy()[:, :r].copy()
        full()
        stream.synchronize()
        def validate_pipeline():
            verify(True)
            verify_all_valid_hot()
            # Validate every Index field against its logical owner, including rows
            # outside current TopK. This checks the actual staged update, not an
            # independently executed old planner.
            logical = update.logical.cpu().long()
            seeds = [saved[0].reshape(b, h, 128).cpu(), saved[1].view(torch.float32).reshape(b, h).cpu()]
            address_seed = saved[2][b * prefix:].view(torch.int64).reshape(b, tail // 8)[:, :h].cpu()
            for row in range(b):
                if not torch.equal(torch.sort(logical[row]).values, torch.arange(h)):
                    raise AssertionError("Index logical owners are not a permutation")
                for name, tensor, seed in zip(("Index", "scales", "addresses"), update.inputs[:3], seeds + [address_seed]):
                    if not torch.equal(tensor[row].cpu(), seed[row, logical[row]]):
                        raise AssertionError(f"Staged update field mismatch: {name}, request={row}")
        passed = correctness.check(label, validate_pipeline)
        from correctness import boundary_evidence
        boundary_phases[label] = boundary_evidence(
            state["ids"], before_valid, before_layout,
            rotating.rows[:, 9].cpu().numpy(), rotating.final_posts[:, :, 0].cpu().numpy())
        atomic_json(folder / "boundary_coverage.json", boundary_phases)
        if cold_start:
            from correctness import transfer_coverage
            boundary_phases['transfer'] = transfer_coverage(
                rotating.final_posts.cpu().numpy(), rotating.destinations.cpu().numpy(),
                rotating.lru_slots.cpu().numpy(), owner.output.shape[1], r)
            atomic_json(folder / "boundary_coverage.json", boundary_phases)
        changes = rotating.changes.cpu().numpy()
        result = dict(transport=state["completion"],
            first_window_fetches=int(rotating.rows[:, 16].cpu().sum()),
            hot_changes=int(np.count_nonzero((changes >= 0) & (changes < r))),
            cold_changes=int(np.count_nonzero(changes >= r)))
        print(f"CHECK {'PASS' if passed else 'FAIL'} {label}: {json.dumps(result)}; full KV/Index/ownership verified", flush=True)
        return result

    snapshot_changes = check_pipeline(initial_orders)
    cold_start_changes = (None if bench_only else
                          check_pipeline(initial_orders, cold_start=True, label="cold-start"))
    unlucky = []
    for row in range(b):
        selected_hot = list(dict.fromkeys(initial_selected[row].reshape(-1).tolist()))
        selected_hot = [x for x in selected_hot if x < r]
        seen = set(selected_hot)
        unlucky.append(selected_hot + [x for x in initial_orders[row] if x not in seen])
    hot_case = (dict(hot_changes=None, first_window_fetches=None) if bench_only else
                check_pipeline(unlucky, label="hot-victim-collision"))
    restored_hot = hot_case["hot_changes"]
    if not bench_only and args.prefetch_tokens and args.miss_percent < 100:
        def check_collision_coverage():
            assert hot_case["first_window_fetches"] and restored_hot, (
                f"First window did not Fetch former hot candidates into LRU slots: {hot_case}")
        correctness.check("hot-victim-collision coverage", check_collision_coverage)

    if getattr(args, "check_suite", None) == "boundary":
        reset()
        ends = rotating.query_ends.clone()
        validity = update.output_valid.clone()
        protected = (*update.inputs[:4], rotating.hot_valid, owner.output)
        snapshots = [tensor.clone() for tensor in protected]
        rotating.query_ends.zero_()
        update.output_valid.zero_()
        full()
        stream.synchronize()
        def check_padding():
            rows = rotating.rows.cpu().numpy()
            posts = rotating.final_posts.cpu().numpy()
            statuses = rotating.row_status.cpu().numpy()
            cq = [x.cpu().view(torch.uint64).numpy() for x in (rotating.prefetch_status, rotating.final_status)]
            if completion_failures(rows, posts, statuses, *cq) or update.status.cpu().count_nonzero():
                raise RuntimeError('Padding execution did not complete successfully')
            owner.mark_completed(completion_verified=True)
            assert not rows[:, 9].any() and not posts[:, :, 0].any(), 'Padding issued fetches'
            assert all(torch.equal(tensor, saved_tensor) for tensor, saved_tensor in zip(protected, snapshots)), 'Padding changed persistent state'
            assert (update.logical_output.cpu() == -1).all() and (update.slot_output.cpu() == -1).all(), 'Padding retained outputs'
        passed = correctness.check('padding', check_padding)
        boundary_phases['padding'] = 'PASS' if passed else 'FAIL'
        rotating.query_ends.copy_(ends)
        update.output_valid.copy_(validity)
        atomic_json(folder / 'boundary_coverage.json', boundary_phases)

    if getattr(args, "check_suite", None) == "boundary":
        from correctness import require_boundary
        correctness.check("boundary trigger coverage", lambda: require_boundary(args.check_case, boundary_phases))

    if args.fetch_only_compare:
        # Capture a regular real resolver snapshot. Prefetch has completed;
        # the isolated comparison intentionally excludes LI and CQ join.
        state["check_phase"] = "fetch-only snapshot"
        reset()
        full()
        stream.synchronize()
        verify(True)
        verify_all_valid_hot()
        from fetch_only_ab import run as run_fetch_only
        return dict(status="pass", fetch_only=run_fetch_only(torch, owner, rotating, pipeline, args))

    def init_reset():
        reset()
        for tensor in (
            rotating.rows,
            rotating.progress,
            rotating.events,
            rotating.owners,
            rotating.mapping,
            rotating.changes,
            rotating.row_status,
            rotating.count,
        ):
            tensor.fill_(1)
        header = b * (4 * 2048 * 6 + 1024)
        arguments[13][header : header + b * h].fill_(255)

    def init_check():
        owner.mark_completed(completion_verified=True)
        if (
            rotating.rows[:, [8, 9, 10, 211, 247, 248]].cpu().count_nonzero()
            or rotating.progress.cpu().count_nonzero()
            or rotating.events.cpu().count_nonzero()
            or rotating.owners.cpu().count_nonzero()
            or rotating.count.cpu().count_nonzero()
            or (not (rotating.mapping.cpu() == -1).all())
            or (not (rotating.changes.cpu() == -1).all())
        ):
            raise AssertionError("Round init retained poisoned workspace")
        header = b * (4 * 2048 * 6 + 1024)
        bitmap = arguments[13][header : header + b * h].cpu().reshape(b, 4, h // 4)
        if not (bitmap == 255).all():
            raise AssertionError("Round init unexpectedly rewrote producer-owned predicates")

    if not bench_only:
        timings["round_init"] = measure(lambda: rotating._call("init"), init_reset, init_check, timing_name="round_init")
    timings["prototype_with_update"] = measure(full, reset, lambda: verify(True), read_label="prototype_with_update")
    if not bench_only:
        reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        owner.graphs.append(graph)
        with torch.npu.graph(graph, stream=capture):
            full()
        capture.synchronize()
        for step in range(3):
            next_orders = []
            if step:
                owner.mark_launched()
            state["before_logical"] = update.logical.cpu().numpy().copy()
            graph.replay()
            stream.synchronize()
            verify(True)
            if args.streaming:
                verify_all_valid_hot()
            for row, p in enumerate(policies):
                touched = state["slots"][row]
                p.commit(touched[touched < r].astype(np.int32))
                order = []
                slot = int(p.head[0])
                for _ in range(r):
                    order.append(slot)
                    slot = int(p.next[0, slot])
                next_orders.append(order)
            update.set_order(next_orders)
        graph.reset()
        owner.graphs.remove(graph)
    forward = np.full((b, h), -1, np.int32)
    logical = update.logical.cpu().numpy()
    for row in range(b):
        forward[row, logical[row, :r]] = np.arange(r)
        forward[row, state["ids"][row].reshape(-1)] = state["slots"][row]
    after = {"topk": state["ids"].reshape(b * 4, 1, 2048), "forward": forward}
    table = torch.arange(b * (r + 8192) // 128, device=device, dtype=torch.int32).reshape(b, -1)
    tensors = dict(
        query=fixture["query"],
        resident_output=owner.output,
        forward=torch.from_numpy(forward).to(device),
        pool_rows=torch.arange(b, device=device, dtype=torch.int32),
        topk=torch.from_numpy(after["topk"]).to(device),
        resident_table=table,
        cumulative_query=fixture["actual_query"],
        logical_lens=torch.full((b,), h, device=device, dtype=torch.int32),
    )
    geom = SimpleNamespace(batch=b, queries=4, num_layers=1, layer_id=0, resident_blocks=b * (r + 8192) // 128)
    sfa = check_sfa(torch, tensors, geom, {"pool_rows": np.arange(b), "budgets": np.full(b, r + 8192)}, after,
                    output_path=args.comparison_sfa)
    fetch_only = None
    if getattr(args, "diagnostic_fetch_only", False):
        # A separate replay supplies the exact final descriptors. No standalone
        # preparation or tracing enters the full-graph performance samples.
        from fetch_only_ab import run as run_fetch_only

        reset()
        full()
        stream.synchronize()
        verify(True)
        verify_all_valid_hot()
        fetch_only = run_fetch_only(torch, owner, rotating, pipeline, args)
        atomic_json(folder / "fetch_only.json", dict(status="PASS", rows=fetch_only))
    return dict(
        status=correctness.status.lower(),
        correctness=correctness.status,
        correctness_failures=correctness.failures,
        boundary_coverage=boundary_phases,
        counter_clock_mhz=counter_clock_mhz,
        timing=timings, wall_costs=wall_costs,
        full_trace=full_trace,
        fetch_only=fetch_only,
        read_samples=read_samples,
        split_samples=split_samples,
        sfa=sfa,
        stateful_rounds=0 if bench_only else 3,
        native_misses_per_request=state["native_misses"],
        cache_update=dict(
            swap_capacity_per_request=update.max_swaps,
            producer_cold_changes=snapshot_changes["cold_changes"],
            hot_kv_copy_payload_bytes=0,
            cold_start_changes_checked=cold_start_changes,
            prefetch_tokens=args.prefetch_tokens,
            former_hot_changes_checked=restored_hot,
            first_window_fetches_checked=hot_case["first_window_fetches"],
            scope=("Bench regular checks; boundary/stateful checks belong to check mode" if bench_only else
                   "Actual streaming pipeline verified; no temporary KV slots; zero KV copy"),
        ),
        kernel="KVIP/csrc/operators/lightning_indexer_with_fetch; preserved core tiling, direct LRU interface",
        scope="fixed Q4 shared history; graph kernel and AIV maintenance separately; no ABC growth",
    )
