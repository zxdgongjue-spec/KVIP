"""Graph A/B using the original LI tiling, service adapters and AIV maintenance."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from device_trace import collect_core, core_summary
from index_update import DirectCacheUpdate, permutation_records
from kvip.operators.lifetch.forward import LIFetchForward
from kvip.runtime.loader import load_lifetch_library
from prefetch_audit import actual_reads, completion_failures
from runner import progress

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "csrc/operators/lightning_indexer_with_fetch"
NATIVE_BITMAP_WORDS = 4176
AUTO_CORE_TRACE_ROUNDS = 3


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
    from native_support import _load_native_operators, command, find_headers
    from real_fetch_inputs import batch_inputs, native_li, prepare_full_chain_fixture
    from real_fetch_session import RealFetchSession
    from test_lifetch import native_log

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
            kw["payload_factory"] = lambda t, b, h: t.from_numpy(np.stack([payload(i, np.arange(h)) for i in range(b)]))
            return super().open(*pos, **kw)

    class Owner(RealFetchSession):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            self.support = SimpleNamespace(**vars(self.support))
            self.support._session_class = lambda: FiniteSession

    summaries = []
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
        stage = "LIFetch build"
        progress(run_dir, f"BS={batch}: {stage}; logs={folder}")
        with native_log(folder / "native.log"):
            common, data, build = (folder / "fixture", folder / "input", shared_build)
            fixture_info = prepare_full_chain_fixture(
                common, args.history, args.resident, args.miss_percent, index_heads=args.index_heads
            )
            geometry = li_geometry(batch, "auto", args.intermediate_fetches, aics, 0)
            batch_inputs(common, data, batch, args.history, geometry["rotating_aic"])
            weights = np.fromfile(data / "arg2.bin", np.float32)
            (weights.view(np.uint32) >> 16).astype(np.uint16).tofile(data / "native_weights_bf16.bin")
            options = dict(
                RUN_MODE="npu",
                RESOLVER_COMPARE=int(args.resolver_compare),
                SOC_VERSION=args.soc,
                ASCEND_CANN_PACKAGE_PATH=args.cann,
                KVAF_INCLUDE=find_headers(args.kvaf_root),
            )
            if not shared_build.exists():
                command(
                    ["cmake", "-S", SOURCE, "-B", build, *[f"-D{k}={v}" for k, v in options.items()]],
                    folder / "configure.log",
                )
                command(["cmake", "--build", build, "-j2"], folder / "build.log")
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
            workers = (batch * (final_aivs(batch, "auto", aics, 0) + 1) + 1) // 2
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
                report.update(
                    fixture=fixture_info,
                    resources=resources,
                    resource_geometry=geometry,
                    build_options={k: str(v) for k, v in options.items()},
                )
                (folder / "report.json").write_text(json.dumps(report, indent=2))
                if args.li_only_varlen:
                    line = (
                        f"V9_LI_VARLEN_PASS batch={batch} history={history} resident={resident} "
                        f"index_heads={heads} rounds={len(report['rounds'])}; "
                        "scope=v9_init+v9_li+v9_wait; resolver/final-fetch/Index-update not tested"
                    )
                    summaries.append(line)
                    (run_dir / "cases.log").write_text("\n".join(summaries) + "\n")
                    print(line, flush=True)
                    continue
            except BaseException:
                (folder / "error.log").write_text(traceback.format_exc())
                raise
            finally:
                owner.close()
        if args.resolver_compare:
            from resolver import summary

            line = summary(report)
            summaries.append(line)
            (run_dir / "cases.log").write_text("\n".join(summaries) + "\n")
            print(line, flush=True)
            continue
        times = report["timing"]
        line = (
            f"binary_sha256={built_digest}\nBS={batch} prototype PASS index_heads={args.index_heads} windows={geometry['intermediate_fetches']} requested_windows={args.intermediate_fetches} history={args.history} hot={args.resident} topk=2048 Q=4 graph=True\n"
            + "graph_p50_us="
            + ", ".join((f"{k}:{v['p50_us']:.3f}" for k, v in times.items()))
            + f"\ncore_speedup={times['native_chain']['p50_us'] / times['prototype_core']['p50_us']:.3f}x"
            + f" full_speedup={times['native_chain']['p50_us'] / times['prototype_with_update']['p50_us']:.3f}x"
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
        for label, samples in report.get("read_samples", {}).items():
            ranges = {key: [min(row[key] for row in samples), max(row[key] for row in samples)] for key in samples[0]}
            line += f"\nactual_reads {label}: min_max=" + json.dumps(ranges)
        if report.get("core_trace"):
            line += "\n" + core_summary(report["core_trace"])
        if report.get("cache_update"):
            line += "\ncache_update=" + json.dumps(report["cache_update"])
        summaries.append(line)
        (run_dir / "cases.log").write_text("\n".join(summaries) + "\n")
        print(line, flush=True)
    progress(run_dir, "all prototype graph cases completed")
    return 0



def run_v9_li_varlen_only(args, b, torch, torch_npu, device, owner, arguments, fixture, native_li, launch_module, aics):
    """Exercise only v9_init + v9_li + v9_wait with per-request Q/K lengths."""
    from kvip.adapters.vllm.inputs import QueryInputs

    history, resident = args.history, args.resident
    inputs = QueryInputs(
        torch, b, device, pin_memory=True, history=history, index_heads=args.index_heads
    )
    query_lengths = torch.zeros((b,), dtype=torch.int32, device=device)
    key_lengths = torch.zeros((b,), dtype=torch.int32, device=device)
    inputs.bind_actual_lengths(query_lengths, key_lengths)

    # These are the exact pointers consumed as P(5)/P(6) by v9_li_kernel.
    arguments[5] = query_lengths
    arguments[6] = key_lengths
    query_ends = inputs.query_ends.view(b, 4)
    q_ptr, k_ptr, ends_ptr = (
        query_lengths.data_ptr(), key_lengths.data_ptr(), query_ends.data_ptr()
    )

    owner.acquire()
    rotating = launch_module.LIFetchBinding(
        torch,
        owner._lifetch_library_path,
        arguments,
        owner,
        batch=b,
        history=history,
        resident=resident,
        aic_count=aics,
        intermediate_fetches=args.intermediate_fetches,
        prefetch_limit=0,
        output_stride=owner.output.shape[1],
        query_ends=query_ends,
    )
    topk = arguments[9].view(torch.int32).reshape(b, 4, 2048)
    stream = torch.npu.current_stream(device)

    scenarios = (
        {
            "name": "mixed_qk_round1",
            "query_counts": (4, 2, 1, 0),
            "query_starts": (0, 4, 6, 7),
            "key_lengths": (8192, 6144, 2048, 0),
        },
        {
            "name": "mixed_qk_round2",
            "query_counts": (1, 0, 3, 2),
            "query_starts": (0, 1, 1, 4),
            "key_lengths": (4096, 0, 8192, 6144),
        },
    )

    summaries = []
    for scenario in scenarios:
        counts = scenario["query_counts"]
        expected_k = scenario["key_lengths"]
        packed_positions = [
            key_len - 1
            for count, key_len in zip(counts, expected_k)
            for _ in range(count)
        ]
        inputs.prepare(
            counts,
            query_starts=scenario["query_starts"],
            query_owners=tuple(range(b)),
        )
        inputs.bind_positions(
            torch.tensor(packed_positions, dtype=torch.int64, device=device)
        )
        stream.synchronize()

        actual_q = tuple(int(x) for x in query_lengths.cpu().tolist())
        actual_k = tuple(int(x) for x in key_lengths.cpu().tolist())
        if actual_q != counts or actual_k != expected_k:
            raise AssertionError(
                f"{scenario['name']}: actual Q/K lengths "
                f"{actual_q}/{actual_k} != {counts}/{expected_k}"
            )

        fixture["li_query_lengths"] = query_lengths
        fixture["actual_key"] = key_lengths
        native_output = native_li(torch_npu, fixture).reshape(b, 4, 2048)
        torch.npu.synchronize()

        # The zero-miss fixture guarantees every native TopK is resident.
        for row, q_len in enumerate(counts):
            active = native_output[row, :q_len]
            if active.numel() and (
                bool(torch.any(active < 0).item()) or bool(torch.any(active >= resident).item())
            ):
                raise AssertionError(
                    f"{scenario['name']}: native TopK escaped the resident range at row {row}"
                )

        topk.fill_(0x5A5A5A5A)
        rotating.reset()
        owner.reset()
        owner.mark_launched()
        rotating.launch_compute()
        rotating.wait()
        stream.synchronize()

        # v9_wait drains the LI-stage prefetch CQ. No final queue is submitted
        # in this LI-only test, so only the prefetch status is authoritative.
        status_words = rotating.prefetch_status.cpu().contiguous().view(torch.uint64).numpy()
        expected_magic = 0x4B5641464247444F
        if status_words.shape[0] != b * 2:
            raise AssertionError(f"unexpected prefetch status rows: {status_words.shape}")
        if not bool(np.all(status_words[:, 0] == expected_magic)) or not bool(
            np.all(status_words[:, 2] == 0)
        ):
            raise AssertionError(f"{scenario['name']}: LI prefetch CQ status is invalid")
        rows = rotating.rows.cpu().numpy()
        if bool(np.any(rows[:, 11] != 0)) or bool(np.any(rows[:, 211] != 0)):
            raise AssertionError(f"{scenario['name']}: LI row error or pending WQE remains")

        owner.mark_completed(completion_verified=True)

        active_queries = 0
        padded_queries = 0
        for row, q_len in enumerate(counts):
            for query_index in range(q_len):
                actual = torch.sort(topk[row, query_index]).values
                expected = torch.sort(native_output[row, query_index]).values
                if not torch.equal(actual, expected):
                    raise AssertionError(
                        f"{scenario['name']}: v9 LI/native TopK mismatch "
                        f"at row={row}, query={query_index}"
                    )
                active_queries += 1
            padding = topk[row, q_len:]
            if padding.numel():
                if not bool(torch.all(padding == -1).item()):
                    raise AssertionError(
                        f"{scenario['name']}: v9 LI did not clear padded Q outputs at row {row}"
                    )
                padded_queries += 4 - q_len

        if (
            query_lengths.data_ptr() != q_ptr
            or key_lengths.data_ptr() != k_ptr
            or inputs.query_ends.data_ptr() != ends_ptr
        ):
            raise AssertionError("Q/K/query-end buffer addresses changed between forwards")

        summaries.append(
            {
                "name": scenario["name"],
                "query_lengths": list(actual_q),
                "key_lengths": list(actual_k),
                "active_queries_checked": active_queries,
                "padded_queries_checked": padded_queries,
                "native_vs_v9_li": "PASS",
                "prefetch_cq": "PASS",
                "result": "PASS",
            }
        )

    return {
        "test": "custom_v9_li_variable_actual_qk_lengths",
        "soc": args.soc,
        "batch": b,
        "history": history,
        "resident": resident,
        "index_heads": args.index_heads,
        "topk": 2048,
        "prefetch_limit": 0,
        "length_buffers_stable": True,
        "rounds": summaries,
        "executed_stages": ["v9_init", "v9_li", "v9_wait"],
        "excluded_stages": ["v9_resolve", "v9_fetch", "Index cache update"],
        "result": "PASS",
    }


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
    if getattr(args, "li_only_varlen", False):
        return run_v9_li_varlen_only(
            args, b, torch, torch_npu, device, owner, arguments, fixture,
            native_li, launch_module, aics,
        )
    stream = torch.npu.current_stream(device)
    capture = torch.npu.Stream(device=device)
    native = NativeMtp(torch, device, owner, fixture, source_tokens=h, padded_tokens=h)
    prepare_native_bitmap(torch, native)
    state = {}
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

    core_trace = {}
    read_samples = {}

    def measure(call, reset, check, trace_core=False, read_label=None):
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
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            if step >= args.warmup:
                samples.append(start.elapsed_time(end) * 1000)
            check()
            if read_label and step >= args.warmup:
                read_samples.setdefault(read_label, []).append(
                    actual_reads(
                        rotating.mapping.cpu().numpy(),
                        topk.cpu().numpy(),
                        rotating.rows.cpu().numpy().view(np.uint32),
                        r,
                    )
                )
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
        if trace_core:
            core_trace.update(
                collect_core(torch, rotating, graph, reset, check, args.trace_rounds or AUTO_CORE_TRACE_ROUNDS, folder)
            )
        graph.reset()
        owner.graphs.remove(graph)
        return dict(p50_us=float(np.median(samples)), p99_us=float(np.percentile(samples, 99)), samples=len(samples))

    native.reset()
    native_call()
    stream.synchronize()
    native_check()
    timings = {"native_chain": measure(native_call, native.reset, native_check)}
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
        prefetch_limit=args.prefetch_tokens,
        output_stride=owner.output.shape[1],
        row_owners=torch.arange(b, dtype=torch.int32, device=device) if args.resolver_compare else None,
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
        library=build / "libkvop_cache_update.so",
        topk=topk,
        locations=rotating.locations,
        rotating=rotating,
    )
    pipeline = LIFetchForward(rotating, update)
    saved = [arguments[i].clone() for i in (1, 4, 13)]

    def reset():
        if owner.in_flight:
            raise RuntimeError("previous replay completion not checked")
        for i, seed in zip((1, 4, 13), saved):
            arguments[i].copy_(seed)
        update.logical.copy_(update.identity)
        rotating.hot_valid.fill_(1)
        update.set_order(initial_orders)
        owner.reset()
        owner.mark_launched()

    def core():
        rotating.launch()

    def full():
        pipeline()

    def verify(updated=False, persistent=True):
        rows = rotating.rows.cpu().numpy()
        posts = rotating.final_posts.cpu().numpy()
        statuses = rotating.row_status.cpu().numpy()
        cq = [status.cpu().view(torch.uint64).numpy() for status in (rotating.prefetch_status, rotating.final_status)]
        failures = completion_failures(rows, posts, statuses, *cq)
        if failures:
            phase = state.get("check_phase", "graph replay")
            path = folder / "completion_failure.json"
            path.write_text(json.dumps(dict(phase=phase, requests=failures), indent=2))
            print(f"COMPLETION FAILURE: phase={phase}; first={json.dumps(failures[0])}; full={path}", flush=True)
            raise AssertionError(
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
        selected = topk.cpu().numpy()
        logical = state.get("before_logical", np.broadcast_to(np.arange(h), (b, h)))
        ids = np.take_along_axis(logical, selected.reshape(b, -1), axis=1).reshape(b, 4, 2048)
        if not np.array_equal(np.sort(ids, axis=-1), np.sort(reference, axis=-1)):
            raise AssertionError("TopK changed")
        slots = rotating.locations.reshape(b, -1).cpu().long()
        if updated:
            if not np.array_equal(update.logical_output.cpu().numpy().reshape(b, 4, 2048), ids):
                raise AssertionError("Service logical TopK output differs from pre-update IDs")
            if not torch.equal(update.slot_output.cpu().reshape(b, -1).long(), slots):
                raise AssertionError("Service SFA slot output differs from Fetch destinations")
        output = owner.output.cpu()
        for row in range(b):
            if slots[row].min() < 0 or slots[row].max() >= (r if persistent else owner.output.shape[1]):
                raise AssertionError("slot outside cache")
            actual = output[row, slots[row]]
            expected = owner.session.cpu_payload[row, ids[row].reshape(-1)]
            if not torch.equal(actual, expected):
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
                raise AssertionError(
                    f"KV byte mismatch: batch={row} mismatched_tokens={int(mismatches.sum())} examples={details}"
                )
        state["ids"], state["slots"] = (ids, slots.numpy())

    if args.resolver_compare:
        from resolver import run

        return run(torch, args, rotating, owner, topk, initial_selected, reset, verify, folder)

    def check_direct_update(orders, cold_start=False, label="regular"):
        state["check_phase"] = f"{label}: fetch before Index update"
        reset()
        update.set_order(orders)
        if cold_start:
            rotating.hot_valid.zero_()
        core()
        stream.synchronize()
        verify()
        state["check_phase"] = f"{label}: after Index update"
        source_data = [t.cpu().clone() for t in update.inputs[:4]]
        changes = rotating.changes.cpu().numpy()
        lru = rotating.change_slots.cpu().numpy()
        expected = [t.clone() for t in source_data]
        moved_rows = 0
        expected_valid = rotating.hot_valid.cpu().numpy().copy()
        for row in range(b):
            records = permutation_records(lru[row], changes[row], h)
            moved_rows += sum((dst != src for dst, src in records))
            written = set(lru[row, changes[row] >= 0].tolist())
            for dst, src in records:
                if dst < r and dst != src and (dst not in written):
                    expected_valid[row, dst] = 0
            for dst, src in records:
                for out, old in zip(expected, source_data):
                    out[row, dst] = old[row, src]
        update(topk, rotating.locations)
        verify(True)
        for name, actual, wanted in zip(("Index", "scales", "addresses", "logical"), update.inputs[:4], expected):
            if not torch.equal(actual.cpu(), wanted):
                raise AssertionError(f"Producer-driven update mismatch: {name}")
        valid = rotating.hot_valid.cpu().numpy()
        if not np.array_equal(valid, expected_valid):
            raise AssertionError("Index update hot validity mismatch")
        logical = update.logical.cpu().long()
        hot_kv = owner.output[:, :r].cpu()
        for row in range(b):
            resident = torch.from_numpy(valid[row] != 0)
            if not torch.equal(hot_kv[row, resident], owner.session.cpu_payload[row, logical[row, :r][resident]]):
                raise AssertionError("Hot Index/KV ownership mismatch")
        print(f"CHECK PASS {label}: {json.dumps(state['completion'])}; KV/Index/ownership verified", flush=True)
        return dict(
            transport=state["completion"],
            moved_index_rows=moved_rows,
            first_window_fetches=int(rotating.rows[:, 16].cpu().sum()),
            hot_changes=int(np.count_nonzero((changes >= 0) & (changes < r))),
            cold_changes=int(np.count_nonzero(changes >= r)),
        )

    snapshot_changes = check_direct_update(initial_orders)
    cold_start_changes = check_direct_update(initial_orders, cold_start=True, label="cold-start")
    unlucky = []
    for row in range(b):
        selected_hot = list(dict.fromkeys(initial_selected[row].reshape(-1).tolist()))
        selected_hot = [x for x in selected_hot if x < r]
        seen = set(selected_hot)
        unlucky.append(selected_hot + [x for x in initial_orders[row] if x not in seen])
    hot_case = check_direct_update(unlucky, label="hot-victim-collision")
    restored_hot = hot_case["hot_changes"]
    if not hot_case["first_window_fetches"] or not restored_hot:
        raise AssertionError(f"First window did not Fetch former hot candidates into LRU slots: {hot_case}")

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

    timings["round_init"] = measure(lambda: rotating._call("init"), init_reset, init_check)
    timings["prototype_core"] = measure(core, reset, verify, trace_core=True, read_label="prototype_core")
    timings["prototype_with_update"] = measure(full, reset, lambda: verify(True), read_label="prototype_with_update")
    reset()
    header = b * (4 * 2048 * 6 + 1024)
    arguments[13][header : header + b * h].fill_(255)
    core()
    stream.synchronize()
    verify()

    def update_reset():
        reset()
        core()
        stream.synchronize()

    timings["aiv_cache_update"] = measure(lambda: update(topk, rotating.locations), update_reset, lambda: verify(True))

    def plan_check():
        verify()
        update.check_capacity()

    timings["aiv_update_plan"] = measure(update.plan, update_reset, plan_check)

    def apply_reset():
        update_reset()
        update.plan()
        stream.synchronize()

    timings["aiv_update_move"] = measure(update.apply, apply_reset, lambda: verify(True))
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
    return dict(
        status="pass",
        timing=timings,
        core_trace=core_trace,
        read_samples=read_samples,
        sfa=sfa,
        stateful_rounds=3,
        native_misses_per_request=state["native_misses"],
        cache_update=dict(
            swap_capacity_per_request=update.max_swaps,
            index_swap_payload_bytes=snapshot_changes["moved_index_rows"] * 128,
            producer_cold_changes=snapshot_changes["cold_changes"],
            hot_kv_copy_payload_bytes=0,
            moved_index_rows=snapshot_changes["moved_index_rows"],
            field_read_write_bytes=snapshot_changes["moved_index_rows"] * (128 + 4 + 8 + 8) * 2,
            update_effective_GBps=snapshot_changes["moved_index_rows"]
            * (128 + 4 + 8 + 8)
            * 2
            / timings["aiv_cache_update"]["p50_us"]
            / 1000,
            update_target_20us=timings["aiv_cache_update"]["p50_us"] < 20,
            update_copy_workers_per_request=4,
            cold_start_changes_checked=cold_start_changes,
            prefetch_tokens=args.prefetch_tokens,
            former_hot_changes_checked=restored_hot,
            first_window_fetches_checked=hot_case["first_window_fetches"],
            scope="Producer records include final hot fallback; no temporary KV slots; zero KV copy",
        ),
        kernel="KVIP/csrc/operators/lightning_indexer_with_fetch; preserved core tiling, direct LRU interface",
        scope="fixed Q4 shared history; graph kernel and AIV maintenance separately; no ABC growth",
    )
