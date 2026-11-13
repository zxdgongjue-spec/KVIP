"""Graph A/B using the original LI tiling, service adapters and AIV maintenance."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from device_trace import collect_core, core_summary
from index_update import DirectCacheUpdate, permutation_records
from native_miss_oracle import native_miss_oracle
from actual_k_admission import parse_actual_key_lengths, score_oracle_for_actual_k
from actual_k_transitions import transition_schedule
from persistent_actual_k import (
    persistent_growth_schedule,
    score_oracle_for_persistent_state,
)
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

    if args.li_only_varlen:
        import torch
        import torch_npu  # noqa: F401
    else:
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
            actual_lengths = parse_actual_key_lengths(
                getattr(args, "actual_key_lengths", None), batch,
                args.history, args.resident,
            )
            fixture_info = prepare_full_chain_fixture(
                common, args.history, args.resident, args.miss_percent,
                index_heads=args.index_heads,
                cold_probe_per_query=int(getattr(args, "cold_prefetch_probe", False)),
                native_union_fit=bool(getattr(args, "native_union_fit", False)),
                actual_key_lengths=actual_lengths,
            )
            geometry = li_geometry(batch, getattr(args, "li_tiling", "auto"), args.intermediate_fetches, aics, 0)
            batch_inputs(common, data, batch, args.history, geometry["rotating_aic"])
            weights = np.fromfile(data / "arg2.bin", np.float32)
            (weights.view(np.uint32) >> 16).astype(np.uint16).tofile(data / "native_weights_bf16.bin")
            options = dict(
                RUN_MODE="npu",
                RESOLVER_COMPARE=int(args.resolver_compare),
                SOC_VERSION=args.soc,
                ASCEND_CANN_PACKAGE_PATH=args.cann,
                KVAF_INCLUDE=find_headers(args.kvaf_root),
                V9_INTERMEDIATE_PROFILING=int(getattr(args, "epoch_audit_profile", False)),
                V9_ACTIVE_K_MAP_CLEAR=int(getattr(args, "active_k_map_clear", False)),
                V9_Q2_TRUNK8K_CONTROL=int(getattr(args, "li_q2_trunk8k_control", False)),
                V9_S2_SCORE_PROBE=int(getattr(args, "li_s2_score_probe", False)),
                V9_S2_SCORE_CORES=int(getattr(args, "li_s2_score_cores", 1)),
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
                    if getattr(args, "li_s2_score_probe", False):
                        score = report["score_probe"]
                        line = (
                            f"P0_C5A_SCORE_PASS cores={score['aic_groups']} "
                            f"Q=1 K={score['actual_k']} "
                            f"raw_score_equal={score['raw_score_equal']} "
                            f"topk_unchanged={score['topk_unchanged']} "
                            f"p50_us={score['p50_us']:.3f} "
                            f"p95_us={score['p95_us']:.3f}"
                        )
                    else:
                        line = (
                            f"V9_LI_VARLEN_PASS batch={batch} history={history} resident={resident} "
                            f"index_heads={heads} li_tiling={report['li_tiling']} "
                            f"li_cores={report['li_cores']} rotation_epochs={report['rotation_epochs']} "
                            f"scenarios={len(report['rounds'])}; "
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
        if report.get("dynamic_k_transition_report"):
            line += "\nP0_A2_DYNAMIC_K_PASS=" + json.dumps(report["dynamic_k_transition_report"])
        if report.get("persistent_actual_k_report"):
            line += "\nP0_A2_2B_PERSISTENT_ACTUAL_K_PASS=" + json.dumps(
                report["persistent_actual_k_report"]
            )
        if fixture_info.get("actual_key_admission"):
            line += "\nP0_A2_ACTUAL_K=" + json.dumps(fixture_info["actual_key_admission"]["key_lengths"])
            line += (
                "; scope=per_request_actual_K; "
                + (
                    "dynamic_K_workspace_replay=PASS_long_short_long_reset"
                    if report.get("dynamic_k_transition_report")
                    else "dynamic_K_workspace_replay=PASS_persistent_growth_no_reset"
                    if report.get("persistent_actual_k_report")
                    else "dynamic_K_workspace_replay=not_tested"
                )
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



def run_v9_li_varlen_only(
    args, b, torch, torch_npu, device, owner, arguments, fixture,
    native_li, launch_module, build, aics, folder,
):
    """Exercise only v9_init + v9_li + v9_wait with per-request Q/K lengths."""
    from kvip.adapters.vllm.inputs import QueryInputs
    from stage_diagnostics import record_stage, run_synchronized_stage
    from epoch_audit import audit_rotation_events, format_rotation_table
    from prefetch_payload_audit import (
        audit_prefetch_mapping, verify_prefetch_payload,
    )

    history, resident = args.history, args.resident
    stage_log = Path(folder) / "li-varlen-stage.jsonl"
    record_stage(
        stage_log,
        "setup",
        "test",
        "begin",
        batch=b,
        history=history,
        resident=resident,
        index_heads=args.index_heads,
        aic_count=aics,
        soc=args.soc,
        device=args.device,
        prefetch_limit=args.prefetch_tokens,
    )
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
        build / "libv9_bridge.so",
        arguments,
        owner,
        batch=b,
        history=history,
        resident=resident,
        aic_count=aics,
        intermediate_fetches=args.intermediate_fetches,
        prefetch_limit=args.prefetch_tokens,
        li_tiling=getattr(args, "li_tiling", "auto"),
        output_stride=owner.output.shape[1],
        query_ends=query_ends,
        enable_lru_slots=not getattr(args, "cold_prefetch_probe", False),
    )
    topk = arguments[9].view(torch.int32).reshape(b, 4, 2048)
    stream = torch.npu.current_stream(device)
    if getattr(args, "li_s2_score_probe", False):
        from s2_score_device import run_s2_score_probe
        return run_s2_score_probe(
            args=args, torch=torch, device=device, owner=owner,
            arguments=arguments, fixture=fixture, native_li=native_li,
            torch_npu=torch_npu, rotating=rotating, topk=topk, stream=stream,
            inputs=inputs, query_lengths=query_lengths, key_lengths=key_lengths,
        )

    if getattr(args, "prefetch_payload_audit", False) and not getattr(
        args, "cold_prefetch_probe", False
    ):
        # Deterministic, injective *physical* victim slot mapping. The regular
        # LI-only fixture does not initialize this LRU policy array. Without
        # this seed, PhysicalSlot() can map every prefetch to physical slot 0.
        # This changes only test metadata, never the KVAF/LI production code.
        if resident != 2048:
            raise ValueError("P1-C0 identity victim fixture requires resident=2048")
        rotating.lru_slots.copy_(
            torch.arange(2048, dtype=torch.int32, device=device)
            .unsqueeze(0).expand(b, 2048)
        )
        stream.synchronize()
        record_stage(
            stage_log, "setup", "victim_slot_fixture", "initialized",
            mapping="identity_per_request", victim_slots=2048,
            expected_physical_slots_unique=True,
        )

    quarter = max(resident, (history // 4 // 128) * 128)
    half = max(resident, (history // 2 // 128) * 128)
    three_quarter = max(resident, ((history * 3 // 4) // 128) * 128)
    scenarios = (
        {
            "name": "mixed_qk_round1",
            "query_counts": (4, 2, 1, 0),
            "query_starts": (0, 4, 6, 7),
            "key_lengths": (history, three_quarter, quarter, 0),
        },
        {
            "name": "mixed_qk_round2",
            "query_counts": (1, 0, 3, 2),
            "query_starts": (0, 1, 1, 4),
            "key_lengths": (half, 0, history, three_quarter),
        },
    )
    if getattr(args, "li_core_sweep", False):
        scenarios += (
            {
                "name": "balanced_full_q4",
                "query_counts": (4, 4, 4, 4),
                "query_starts": (0, 4, 8, 12),
                "key_lengths": (history,) * 4,
            },
            {
                "name": "mixed_k_full_q4",
                "query_counts": (4, 4, 4, 4),
                "query_starts": (0, 4, 8, 12),
                "key_lengths": (quarter, half, three_quarter, history),
            },
            {
                "name": "extreme_k_full_q4",
                "query_counts": (4, 4, 4, 4),
                "query_starts": (0, 4, 8, 12),
                "key_lengths": (resident, resident, resident, history),
            },
        )

    if getattr(args, "li_s2_slope_sweep", False):
        # P0-C4: all Q/K work is in request 0; empty requests isolate the
        # critical long S2 path. History, Resident, Graph and buffers remain
        # fixed; only actual-K and actual-Q vary. No cold winners in this
        # diagnostic: additional cold correctness probes are separate.
        scenarios = tuple(
            {
                "name": f"s2_slope_q{qcount}_k{key_len // 1024}k",
                "query_counts": (qcount, 0, 0, 0),
                "query_starts": (0, qcount, qcount, qcount),
                "key_lengths": (key_len, 0, 0, 0),
            }
            for qcount in (1, 4)
            for key_len in (resident, quarter, half, three_quarter, history)
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
        record_stage(
            stage_log,
            scenario["name"],
            "round",
            "begin",
            query_counts=list(counts),
            query_starts=list(scenario["query_starts"]),
            expected_key_lengths=list(expected_k),
            packed_position_count=len(packed_positions),
        )
        run_synchronized_stage(
            stage_log,
            scenario["name"],
            "bind_runtime_lengths",
            lambda: (
                inputs.prepare(
                    counts,
                    query_starts=scenario["query_starts"],
                    query_owners=tuple(range(b)),
                ),
                inputs.bind_positions(
                    torch.tensor(packed_positions, dtype=torch.int64, device=device)
                ),
            ),
            stream.synchronize,
        )

        actual_q = tuple(int(x) for x in query_lengths.cpu().tolist())
        actual_k = tuple(int(x) for x in key_lengths.cpu().tolist())
        if actual_q != counts or actual_k != expected_k:
            record_stage(
                stage_log,
                scenario["name"],
                "length_validation",
                "failed",
                actual_query_lengths=list(actual_q),
                actual_key_lengths=list(actual_k),
            )
            raise AssertionError(
                f"{scenario['name']}: actual Q/K lengths "
                f"{actual_q}/{actual_k} != {counts}/{expected_k}"
            )
        record_stage(
            stage_log,
            scenario["name"],
            "length_validation",
            "completed",
            actual_query_lengths=list(actual_q),
            actual_key_lengths=list(actual_k),
            query_lengths_ptr=q_ptr,
            key_lengths_ptr=k_ptr,
            query_ends_ptr=ends_ptr,
        )

        fixture["li_query_lengths"] = query_lengths
        fixture["actual_key"] = key_lengths
        native_output = run_synchronized_stage(
            stage_log,
            scenario["name"],
            "native_li",
            lambda: native_li(torch_npu, fixture).reshape(b, 4, 2048),
            torch.npu.synchronize,
        )

        cold_probe = bool(getattr(args, "cold_prefetch_probe", False))
        native_cold_winners = 0
        for row, q_len in enumerate(counts):
            for q in range(q_len):
                ids = native_output[row, q]
                if bool(torch.any(ids < 0).item()) or bool(torch.any(ids >= expected_k[row]).item()):
                    raise AssertionError(
                        f"{scenario['name']}: native LI emitted invalid ID in row={row} Q={q}"
                    )
                cold = ids[ids >= resident]
                want_cold = int(cold_probe and expected_k[row] > resident)
                if int(cold.numel()) != want_cold:
                    raise AssertionError(
                        f"{scenario['name']}: native cold winners={cold.numel()} "
                        f"expected={want_cold} row={row} Q={q}"
                    )
                if want_cold:
                    wanted_id = resident + 16 * q
                    if not bool(torch.any(cold == wanted_id).item()):
                        raise AssertionError(
                            f"{scenario['name']}: missing expected cold winner "
                            f"token={wanted_id}, row={row} Q={q}"
                        )
                native_cold_winners += int(cold.numel())
        record_stage(
            stage_log, scenario["name"], "native_cold_topk", "verified",
            cold_probe=cold_probe,
            cold_winners=native_cold_winners,
        )

        def reset_v9_round():
            topk.fill_(0x5A5A5A5A)
            rotating.reset()
            owner.reset()

        run_synchronized_stage(
            stage_log,
            scenario["name"],
            "reset_v9_state",
            reset_v9_round,
            stream.synchronize,
        )
        owner.mark_launched()
        record_stage(
            stage_log,
            scenario["name"],
            "v9_launch",
            "owner_marked_in_flight",
        )
        run_synchronized_stage(
            stage_log,
            scenario["name"],
            "v9_init",
            lambda: rotating._call("init"),
            stream.synchronize,
        )
        run_synchronized_stage(
            stage_log,
            scenario["name"],
            "v9_li",
            lambda: rotating._call("li"),
            stream.synchronize,
        )
        run_synchronized_stage(
            stage_log,
            scenario["name"],
            "v9_wait",
            lambda: rotating._call("wait"),
            stream.synchronize,
        )

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

        skipped_lanes = 0
        completed_lanes = 0
        epoch_audit_filename = None
        epoch_observed_candidates = None
        epoch_selected_for_post = None
        if args.prefetch_tokens:
            shape = rotating.runtime_shape
            audit = audit_rotation_events(
                rotating.events.cpu().numpy(),
                batch=int(shape.batch),
                split_q=int(shape.splitQ),
                li_cores=int(shape.liCores),
                rounds=int(shape.rounds),
                service_aics=int(shape.serviceAics),
                service_lanes=int(shape.serviceLanes),
                history=int(shape.history),
                resident=int(shape.resident),
                profiling_enabled=bool(getattr(args, "epoch_audit_profile", False)),
            )
            audit["scenario"] = scenario["name"]
            audit["binary_event_profiling"] = bool(getattr(args, "epoch_audit_profile", False))
            epoch_audit_filename = f"epoch-audit-{scenario['name']}.json"
            (Path(folder) / epoch_audit_filename).write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            (Path(folder) / f"epoch-audit-{scenario['name']}.txt").write_text(
                format_rotation_table(audit)
            )
            if audit["errors"]:
                raise AssertionError(
                    f"{scenario['name']}: Rotation Event audit failed; "
                    f"see {epoch_audit_filename}: {audit['errors'][:3]}"
                )
            summary = audit["summary"]
            completed_lanes = summary["complete_lanes"]
            skipped_lanes = summary["skipped_lanes"]
            epoch_observed_candidates = summary["candidate_count_observed"]
            epoch_selected_for_post = summary["selected_for_post_count_observed"]
            record_stage(
                stage_log,
                scenario["name"],
                "rotation_epochs",
                "verified",
                complete_lanes=completed_lanes,
                skipped_lanes=skipped_lanes,
                rounds=rotating.rounds,
                service_lanes=rotating.service_lanes,
                profiling_enabled=bool(getattr(args, "epoch_audit_profile", False)),
                candidate_count_observed=epoch_observed_candidates,
                selected_for_post_count_observed=epoch_selected_for_post,
                audit_file=epoch_audit_filename,
            )
        record_stage(
            stage_log,
            scenario["name"],
            "prefetch_completion",
            "verified",
            status_rows=int(status_words.shape[0]),
            row_error_words=rows[:, 11].tolist(),
            row_pending_words=rows[:, 211].tolist(),
        )
        owner.mark_completed(completion_verified=True)
        record_stage(
            stage_log,
            scenario["name"],
            "owner_state",
            "marked_completed",
            completion_verified=True,
        )

        payload_audit_filename = None
        payload_checked_tokens = None
        payload_checked_bytes = None
        payload_cq_groups = None
        payload_hot_candidates = None
        payload_cold_candidates = None
        if getattr(args, "prefetch_payload_audit", False):
            if epoch_selected_for_post is None or epoch_selected_for_post <= 0:
                raise AssertionError(
                    f"{scenario['name']}: P1-C0 expected nonzero profiled SelectedForPost"
                )
            payload_audit_filename = f"prefetch-payload-{scenario['name']}.json"
            payload_path = Path(folder) / payload_audit_filename
            plan = audit_prefetch_mapping(
                rotating.mapping.cpu().numpy(),
                rotating.lru_slots.cpu().numpy(),
                rows,
                status_words,
                actual_k,
                resident=resident,
                output_stride=int(owner.output.shape[1]),
                rounds=int(rotating.rounds),
                selected_for_post=epoch_selected_for_post,
                use_lru_slots=not getattr(args, "cold_prefetch_probe", False),
            )
            plan["scenario"] = scenario["name"]
            plan["fixture_victim_policy"] = "identity_per_request"
            payload_path.write_text(json.dumps(plan, indent=2) + "\n")
            if plan["errors"]:
                record_stage(
                    stage_log, scenario["name"], "prefetch_mapping", "failed",
                    audit_file=payload_audit_filename,
                    errors=plan["errors"][:4],
                )
                raise AssertionError(
                    f"{scenario['name']}: Row/CQ/mapping mismatch; "
                    f"see {payload_audit_filename}: {plan['errors'][:3]}"
                )
            record_stage(
                stage_log, scenario["name"], "prefetch_mapping", "verified",
                audit_file=payload_audit_filename,
                requests=plan["summary"]["requests"],
                mapping_count=plan["summary"]["mapping_count"],
                cq_completed_groups=plan["summary"]["cq_completed_groups"],
            )

            def read_hbm(row_index, slots):
                indices = torch.as_tensor(slots, dtype=torch.int64, device=device)
                return owner.output[row_index].index_select(0, indices).cpu().numpy()

            def read_ddr(row_index, tokens):
                source = owner.session.cpu_payload[row_index]
                indices = torch.as_tensor(
                    tokens, dtype=torch.int64, device=source.device
                )
                return source.index_select(0, indices).cpu().numpy()

            plan = verify_prefetch_payload(
                plan,
                read_hbm=read_hbm,
                read_ddr=read_ddr,
                token_bytes=int(owner.output.shape[-1]),
            )
            payload_path.write_text(json.dumps(plan, indent=2) + "\n")
            payload_summary = plan["summary"]
            if cold_probe:
                if payload_summary["cold_candidate_count"] < 1:
                    plan["errors"].append(
                        "P1-C1: no real cold KV prefetch was observed"
                    )
                if payload_summary["resident_candidate_count"]:
                    plan["errors"].append(
                        "P1-C1: cold-only diagnostic unexpectedly fetched resident tokens"
                    )
                for row_info in plan["rows"]:
                    for record in row_info["mappings"]:
                        allowed = {
                            resident + 16 * q
                            for q in range(counts[row_info["row"]])
                            if expected_k[row_info["row"]] > resident
                        }
                        if record["token"] not in allowed:
                            plan["errors"].append(
                                f"row={row_info['row']}: unexpected cold token "
                                f"{record['token']} not in {sorted(allowed)}"
                            )
                plan["summary"]["validation"] = (
                    "PASS" if not plan["errors"] else "FAIL"
                )
                payload_path.write_text(json.dumps(plan, indent=2) + "\n")
            if plan["errors"]:
                record_stage(
                    stage_log, scenario["name"], "prefetch_payload", "failed",
                    audit_file=payload_audit_filename,
                    errors=plan["errors"][:4],
                )
                raise AssertionError(
                    f"{scenario['name']}: DDR/HBM payload mismatch; "
                    f"see {payload_audit_filename}: {plan['errors'][:3]}"
                )
            payload_checked_tokens = payload_summary["payload_tokens_checked"]
            payload_checked_bytes = payload_summary["payload_bytes_checked"]
            payload_cq_groups = payload_summary["cq_completed_groups"]
            payload_hot_candidates = payload_summary["resident_candidate_count"]
            payload_cold_candidates = payload_summary["cold_candidate_count"]
            record_stage(
                stage_log, scenario["name"], "prefetch_payload", "verified",
                audit_file=payload_audit_filename,
                payload_tokens_checked=payload_checked_tokens,
                payload_bytes_checked=payload_checked_bytes,
                cq_completed_groups=payload_cq_groups,
                resident_candidate_count=payload_hot_candidates,
                cold_candidate_count=payload_cold_candidates,
            )

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

        li_perf = None
        if getattr(args, "li_core_perf", False):
            if args.prefetch_tokens != 0:
                raise AssertionError("P0-C1 LI-only profiling requires Prefetch=0")
            # P0-C1 R1: the old one-LI Event interval was highly
            # multimodal (0.26/1.9/3.8 ms) and can include Host submission
            # gaps between start.record(), replay(), and end.record().
            # Capture multiple *sequential* LI launches into ONE NPU Graph.
            # The device stays busy long enough for End to be queued before
            # the work finishes; divide the device interval by graph_repeats.
            # This is a throughput-amortized LI cost, NOT standalone LI
            # single-request service latency. Prefetch=0 disables Rotation.
            graph_repeats = int(args.li_core_perf_graph_repeats)
            graph = torch.npu.NPUGraph()
            owner.graphs.append(graph)
            capture_stream = torch.npu.Stream(device=device)
            stream.synchronize()
            with torch.npu.graph(graph, stream=capture_stream):
                for _ in range(graph_repeats):
                    rotating._call("li")
            capture_stream.synchronize()
            graph.replay()
            stream.synchronize()
            # Replayed TopK must remain equivalent to the already-checked
            # Native result before timing is allowed to contribute samples.
            for row, q_len in enumerate(counts):
                for query_index in range(q_len):
                    actual = torch.sort(topk[row, query_index]).values
                    expected = torch.sort(native_output[row, query_index]).values
                    if not torch.equal(actual, expected):
                        raise AssertionError(
                            f"{scenario['name']}: captured LI Graph TopK mismatch "
                            f"row={row} Q={query_index}"
                        )
                if topk[row, q_len:].numel() and not bool(
                    torch.all(topk[row, q_len:] == -1).item()
                ):
                    raise AssertionError(f"{scenario['name']}: captured Graph padded output mismatch")

            # Warm up the captured N-LI graph before timing. Each measured
            # sample represents exactly graph_repeats sequential LI launches.
            for _ in range(args.warmup):
                graph.replay()
            stream.synchronize()
            samples_us = []
            graph_elapsed_us = []
            for step in range(args.iterations):
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                end_event.synchronize()
                elapsed_us = float(start_event.elapsed_time(end_event) * 1000.0)
                graph_elapsed_us.append(elapsed_us)
                samples_us.append(elapsed_us / graph_repeats)
            stream.synchronize()
            # Multiple Graph replays must be non-destructive, not just the
            # first replay checked above. Verify after the final sample too.
            for row, q_len in enumerate(counts):
                for query_index in range(q_len):
                    if not torch.equal(
                        torch.sort(topk[row, query_index]).values,
                        torch.sort(native_output[row, query_index]).values,
                    ):
                        raise AssertionError(
                            f"{scenario['name']}: repeated LI Graph TopK mismatch "
                            f"row={row} Q={query_index}"
                        )
                if topk[row, q_len:].numel() and not bool(
                    torch.all(topk[row, q_len:] == -1).item()
                ):
                    raise AssertionError(f"{scenario['name']}: repeated Graph padded output mismatch")
            if not samples_us or not all(np.isfinite(samples_us)):
                raise AssertionError("P0-C1 device timing samples missing or invalid")
            li_perf = dict(
                scope="device_event_graph_repeated_v9_li_amortized",
                timing_method="graph_in_graph_sequential_li_repeats",
                warmup=args.warmup,
                iterations=args.iterations,
                graph_repeats=graph_repeats,
                samples_us=samples_us,
                raw_graph_samples_us=graph_elapsed_us,
                p50_us=float(np.median(samples_us)),
                p95_us=float(np.percentile(samples_us, 95)),
                minimum_us=float(np.min(samples_us)),
                maximum_us=float(np.max(samples_us)),
                initial_graph_topk_verified_against_native=True,
                prefetch_disabled=True,
                li_cores=int(rotating.runtime_shape.liCores),
                split_q=int(rotating.runtime_shape.splitQ),
                q2_trunk8k_control=bool(getattr(args, "li_q2_trunk8k_control", False)),
            )
            print(
                "P0_C1_LI_GRAPH_TIMING "
                f"scenario={scenario['name']} "
                f"tiling={getattr(args, 'li_tiling', 'auto')} "
                f"cores={li_perf['li_cores']} "
                f"graph_repeats={graph_repeats} "
                f"p50_us={li_perf['p50_us']:.3f} "
                f"p95_us={li_perf['p95_us']:.3f}",
                flush=True,
            )
            record_stage(
                stage_log, scenario["name"], "li_graph_timing", "completed",
                **{k: v for k, v in li_perf.items() if k != "samples_us"},
            )
        record_stage(
            stage_log,
            scenario["name"],
            "round",
            "passed",
            active_queries_checked=active_queries,
            padded_queries_checked=padded_queries,
            length_buffers_stable=True,
            epoch_completed_lanes=completed_lanes,
            epoch_skipped_lanes=skipped_lanes,
            epoch_audit_file=epoch_audit_filename,
            epoch_candidate_count_observed=epoch_observed_candidates,
            epoch_selected_for_post_count_observed=epoch_selected_for_post,
            payload_audit_file=payload_audit_filename,
            payload_tokens_checked=payload_checked_tokens,
            payload_bytes_checked=payload_checked_bytes,
            cq_completed_groups=payload_cq_groups,
            resident_prefetch_candidates=payload_hot_candidates,
            cold_prefetch_candidates=payload_cold_candidates,
            native_cold_topk_winners=native_cold_winners,
        )
        summaries.append(
            {
                "name": scenario["name"],
                "query_lengths": list(actual_q),
                "key_lengths": list(actual_k),
                "active_queries_checked": active_queries,
                "padded_queries_checked": padded_queries,
                "native_vs_v9_li": "PASS",
                "li_core_perf": li_perf,
                "prefetch_cq": "PASS",
                "epoch_completed_lanes": completed_lanes,
                "epoch_skipped_lanes": skipped_lanes,
                "epoch_audit_file": epoch_audit_filename,
                "epoch_candidate_count_observed": epoch_observed_candidates,
                "epoch_selected_for_post_count_observed": epoch_selected_for_post,
                "prefetch_payload_audit_file": payload_audit_filename,
                "prefetch_payload_tokens_checked": payload_checked_tokens,
                "prefetch_payload_bytes_checked": payload_checked_bytes,
                "prefetch_cq_completed_groups": payload_cq_groups,
                "resident_prefetch_candidates": payload_hot_candidates,
                "cold_prefetch_candidates": payload_cold_candidates,
                "native_cold_topk_winners": native_cold_winners,
                "result": "PASS",
            }
        )

    return {
        "test": "custom_v9_li_variable_actual_qk_lengths",
        "soc": args.soc,
        "li_tiling": getattr(args, "li_tiling", "auto"),
        "li_q2_trunk8k_control": bool(getattr(args, "li_q2_trunk8k_control", False)),
        "li_core_sweep": bool(getattr(args, "li_core_sweep", False)),
        "li_s2_slope_sweep": bool(getattr(args, "li_s2_slope_sweep", False)),
        "li_core_perf": bool(getattr(args, "li_core_perf", False)),
        "li_cores": int(rotating.runtime_shape.liCores),
        "rotation_epochs": int(rotating.runtime_shape.rounds),
        "batch": b,
        "history": history,
        "resident": resident,
        "index_heads": args.index_heads,
        "topk": 2048,
        "prefetch_limit": args.prefetch_tokens,
        "epoch_event_profiling": bool(getattr(args, "epoch_audit_profile", False)),
        "prefetch_payload_audit": bool(getattr(args, "prefetch_payload_audit", False)),
        "cold_prefetch_probe": bool(getattr(args, "cold_prefetch_probe", False)),
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
    actual_lengths = parse_actual_key_lengths(
        getattr(args, "actual_key_lengths", None), b, h, r
    )
    # arguments[6] is the exact P(6) pointer consumed by v9_li and v9_init.
    # Alias its original 4-byte-per-request tensor as int32 for Native QLI.
    actual_key_buffer = arguments[6].view(torch.int32).reshape(b)
    if actual_lengths is not None:
        actual_key_buffer.copy_(torch.tensor(actual_lengths, dtype=torch.int32, device=device))
    fixture = dict(
        query=arguments[0].view(torch.float8_e4m3fn).reshape(b, 4, args.index_heads, 128),
        key=arguments[1].view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
        actual_query=torch.arange(1, b + 1, device=device, dtype=torch.int32) * 4,
        li_query_lengths=torch.full((b,), 4, device=device, dtype=torch.int32),
        actual_key=actual_key_buffer,
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
            native_li, launch_module, build, aics, folder,
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
    key_lengths = actual_lengths or (h,) * b
    if actual_lengths is None:
        expected_ids = np.stack([
            np.flatnonzero(row >= np.partition(row, -2048)[-2048])
            for row in scores
        ])
        expected_by_row = np.broadcast_to(expected_ids, (b, 4, 2048))
    else:
        per_request_admission, expected_by_row = score_oracle_for_actual_k(
            logical_scores, physical_order, key_lengths, r
        )
        (folder / "actual-k-oracle.json").write_text(json.dumps(dict(
            fixed_history=h, actual_key_lengths=list(key_lengths),
            requests=per_request_admission,
        ), indent=2))
    for row, want in zip(initial_selected.reshape(b, 4, 2048), expected_by_row):
        if not np.array_equal(np.sort(row, axis=-1), want):
            raise AssertionError("native LI differs from independent per-K score oracle")
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

    # Freeze the pre-launch Native LIDU classification oracle outside timings.
    # The input bitmap reserves potential eviction slots. A selected resident
    # token in those slots is a Miss too; cold-token count alone undercounts
    # Native Get work. The four Qs share one unique Miss set per request.
    initial_maps = native.initial.cpu().numpy()
    initial_victim_bitmaps = native.bitmap.cpu().numpy()
    native_miss_expected = [
        native_miss_oracle(
            initial_selected[row], initial_maps[row],
            initial_victim_bitmaps[row], r
        )
        for row in range(b)
    ]

    def native_call():
        state["native_topk"] = native_li(torch_npu, fixture)
        native.launch(state["native_topk"])

    def native_check():
        counts = native.verify(state["native_topk"])
        expected = [entry["effective_unique_misses"] for entry in native_miss_expected]
        if counts[:-1] != expected:
            raise AssertionError(
                "native Get work differs from initial-map/victim-bitmap oracle: "
                f"actual={counts[:-1]} expected={expected} "
                f"detail={native_miss_expected}"
            )
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
                        actual_key_lengths=actual_lengths,
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
    query_ends = None
    if actual_lengths is not None:
        query_ends = torch.tensor(
            [[k] * 4 for k in key_lengths], dtype=torch.int32, device=device
        ).contiguous()
        if actual_key_buffer.data_ptr() != arguments[6].data_ptr():
            raise AssertionError("Native/V9 actual key length storage was copied")
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
        query_ends=query_ends,
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
            or (not (rotating.changes.cpu() == -1).all())
        ):
            raise AssertionError("Round init retained poisoned workspace")
        # P0-A2: shortened Mapping clear intentionally preserves the suffix.
        # Poison with +1 before v9_init, then require every valid K prefix
        # cleared to -1 while ineligible [K, History) remains exactly +1.
        # With the optimization OFF (or no actual K), require a full clear.
        observed_mapping = rotating.mapping.cpu().numpy()
        if actual_lengths is not None and args.active_k_map_clear:
            for row, effective_k in enumerate(actual_lengths):
                if not np.all(observed_mapping[row, :effective_k] == -1):
                    raise AssertionError(
                        f"Partial round init failed to clear valid K: row={row} K={effective_k}"
                    )
                if not np.all(observed_mapping[row, effective_k:] == 1):
                    raise AssertionError(
                        f"Partial round init overwrote out-of-K suffix: row={row} K={effective_k}"
                    )
        elif not np.all(observed_mapping == -1):
            raise AssertionError("Full round init retained poisoned Mapping")
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
    dynamic_k_transition_report = None
    persistent_actual_k_report = None
    if getattr(args, "actual_k_transition_check", False):
        if actual_lengths is None or query_ends is None or not args.active_k_map_clear:
            raise ValueError("Dynamic-K transition requires valid K and Active-K reset")
        schedule = transition_schedule(key_lengths, h, r)
        dynamic_k_transition_report = []
        old_key_ptr = actual_key_buffer.data_ptr()
        old_ends_ptr = rotating.query_ends.data_ptr()
        # Same captured full Graph and V9 workspace; restore Index/KV baseline
        # per stage, but deliberately keep Mapping across all K transitions.
        for label, new_k in schedule:
            _, expected_step = score_oracle_for_actual_k(
                logical_scores, physical_order, new_k, r
            )
            reset()
            reference[...] = expected_step
            device_k = torch.tensor(new_k, dtype=torch.int32, device=device)
            actual_key_buffer.copy_(device_k)
            rotating.query_ends.copy_(device_k[:, None].expand(b, 4))
            # Force stale-positive Mapping outside short K (not an actual
            # Prefetch). The next long stage must clear it before using K.
            for row, k in enumerate(new_k):
                if k < h:
                    rotating.mapping[row, k:].fill_(1)
            stream.synchronize()
            if actual_key_buffer.data_ptr() != old_key_ptr or rotating.query_ends.data_ptr() != old_ends_ptr:
                raise AssertionError("Dynamic-K metadata pointer identity changed")
            before_map = rotating.mapping.cpu().numpy().copy()
            state["check_phase"] = "P0-A2.2a " + label
            state["before_logical"] = update.logical.cpu().numpy().copy()
            graph.replay()
            stream.synchronize()
            verify(True)
            after_map = rotating.mapping.cpu().numpy()
            for row, k in enumerate(new_k):
                if not np.array_equal(after_map[row, k:], before_map[row, k:]):
                    raise AssertionError(
                        f"Dynamic-K Mapping suffix changed: {label} row={row} K={k}"
                    )
            read_check = actual_reads(
                after_map, topk.cpu().numpy(),
                rotating.rows.cpu().numpy().view(np.uint32), r,
                actual_key_lengths=new_k,
            )
            dynamic_k_transition_report.append(dict(
                stage=label, key_lengths=list(new_k),
                graph_and_workspace_reused=True,
                index_kv_restored_each_stage=True,
                topk_kv_index_and_completion_pass=True,
                stale_mapping_suffix_untouched=True,
                prefetch_reconciled=True,
                prefetch_requests=read_check["prefetch_hot"] + read_check["prefetch_cold"],
                stale_out_of_k_entries=read_check["ignored_out_of_K_mapping"],
            ))
            print(f"P0_A2_DYNAMIC_K_STAGE_PASS {label}: K={list(new_k)}", flush=True)
        state.pop("check_phase", None)
        # Last stage restores initial K, preserving the subsequent SFA test.
        if tuple(actual_key_buffer.cpu().tolist()) != tuple(key_lengths):
            raise AssertionError("Dynamic-K final stage did not restore initial K")

    # P0-A2.2b: one baseline reset, then monotonic K growth while the
    # Index rows, scales, DDR addresses, hot KV and LRU state remain live.
    if getattr(args, "actual_k_persistent_check", False):
        if actual_lengths is None or query_ends is None or not args.active_k_map_clear:
            raise ValueError("Persistent Actual-K check requires valid K and Active-K reset")
        schedule = persistent_growth_schedule(key_lengths, h, r)
        if tuple(schedule[0][1]) != tuple(key_lengths):
            raise AssertionError("Persistent Actual-K schedule does not start at fixture K")

        # A single reset establishes the baseline. These immutable CPU
        # snapshots are the reference for every later in-place Index move.
        reset()
        base_index_keys = update.inputs[0].detach().cpu().numpy().copy()
        base_index_scales = update.inputs[1].detach().cpu().numpy().copy()
        base_index_addresses = update.inputs[2].detach().cpu().numpy().copy()
        base_payload = owner.session.cpu_payload.detach().cpu().numpy()
        base_rows = np.arange(h, dtype=np.int64)

        # Recreate the Python LRU oracle at the same initial state as reset().
        persistent_policies = []
        for row in range(b):
            policy = SlotLRU()
            policy.resize(r, r)
            policy.commit(initial_selected[row][initial_selected[row] < r])
            persistent_policies.append(policy)
        update.set_order(initial_orders)

        old_key_ptr = actual_key_buffer.data_ptr()
        old_ends_ptr = rotating.query_ends.data_ptr()
        first_k = schedule[0][1]
        actual_key_buffer.copy_(torch.tensor(first_k, dtype=torch.int32, device=device))
        rotating.query_ends.copy_(
            torch.tensor(first_k, dtype=torch.int32, device=device)[:, None].expand(b, 4)
        )
        # Positive poison models stale entries left beyond the active K.
        # Each later growth step must clear the newly admitted interval, while
        # the still-inactive suffix remains byte-for-byte untouched.
        for row, key in enumerate(first_k):
            rotating.mapping[row, key:].fill_(1)
        stream.synchronize()
        if actual_key_buffer.data_ptr() != old_key_ptr or rotating.query_ends.data_ptr() != old_ends_ptr:
            raise AssertionError("Persistent Actual-K metadata pointer identity changed")

        persistent_stages = []
        previous_k = None
        for stage_index, (label, new_k) in enumerate(schedule):
            device_k = torch.tensor(new_k, dtype=torch.int32, device=device)
            actual_key_buffer.copy_(device_k)
            rotating.query_ends.copy_(device_k[:, None].expand(b, 4))
            stream.synchronize()
            if tuple(actual_key_buffer.cpu().tolist()) != tuple(new_k):
                raise AssertionError(f"{label}: Actual-K buffer did not update")
            if tuple(tuple(x) for x in rotating.query_ends.cpu().tolist()) != tuple((k,) * 4 for k in new_k):
                raise AssertionError(f"{label}: QueryEnd buffer did not update")

            # Snapshot current physical->baseline row map BEFORE this stage's
            # Index Update. Native LI and the independent oracle both consume
            # the Index cache produced by the previous stage.
            current_map_before = update.logical.detach().cpu().numpy().astype(np.int64, copy=True)
            oracle_reports, oracle_selected = score_oracle_for_persistent_state(
                logical_scores, physical_order, current_map_before, new_k, r
            )
            fixture["actual_key"] = actual_key_buffer
            native_selected = native_li(torch_npu, fixture).detach().cpu().reshape(b, 4, 2048).numpy()
            for row in range(b):
                for query in range(4):
                    if not np.array_equal(
                        np.sort(native_selected[row, query]),
                        oracle_selected[row, query],
                    ):
                        raise AssertionError(
                            f"{label}: Native LI differs from persistent-state score oracle "
                            f"row={row} query={query} K={new_k[row]}"
                        )

            # verify() compares logical token IDs, not transient physical
            # TopK row numbers. Capture the map used by this round.
            logical_reference = np.take_along_axis(
                current_map_before,
                native_selected.reshape(b, -1),
                axis=1,
            ).reshape(b, 4, 2048)
            reference[...] = logical_reference
            state["before_logical"] = current_map_before.copy()
            state["check_phase"] = "P0-A2.2b " + label
            before_mapping = rotating.mapping.detach().cpu().numpy().copy()

            # Probe v9_init independently at the current K. The init-only
            # replay touches workspace Mapping, not Index/KV state. It must
            # clear [0,K) and preserve the poisoned [K,H) suffix.
            if stage_index:
                owner.mark_launched()
            rotating._call("init")
            stream.synchronize()
            owner.mark_completed(completion_verified=True)
            init_mapping = rotating.mapping.detach().cpu().numpy()
            for row, key in enumerate(new_k):
                if not np.all(init_mapping[row, :key] == -1):
                    raise AssertionError(
                        f"{label}: v9_init failed to clear active Mapping prefix row={row} K={key}"
                    )
                if not np.all(init_mapping[row, key:] == 1):
                    raise AssertionError(
                        f"{label}: v9_init changed poisoned out-of-K suffix row={row} K={key}"
                    )

            # No reset() is called between growth stages. Re-arm the same
            # queue lease and replay the already-captured full pipeline.
            owner.mark_launched()
            graph.replay()
            stream.synchronize()
            verify(True)

            current_map_after = update.logical.detach().cpu().numpy().astype(np.int64, copy=True)
            if any(not np.array_equal(np.sort(row), base_rows) for row in current_map_after):
                raise AssertionError(f"{label}: Index logical map is no longer a permutation")
            changed_rows = int(np.count_nonzero(current_map_after != current_map_before))
            if changed_rows == 0:
                raise AssertionError(f"{label}: Index Update did not change persistent Index state")

            # Every source TopK row and every hot destination is inside [0,K).
            # Index Update must not move state across the current actual-K edge.
            after_mapping = rotating.mapping.detach().cpu().numpy()
            for row, key in enumerate(new_k):
                if not np.array_equal(current_map_after[row, key:], current_map_before[row, key:]):
                    raise AssertionError(
                        f"{label}: Index Update crossed actual-K boundary row={row} K={key}"
                    )
                if not np.array_equal(after_mapping[row, key:], before_mapping[row, key:]):
                    raise AssertionError(
                        f"{label}: out-of-K Mapping suffix changed row={row} K={key}"
                    )

            # Prove the complete Index tuple moved as one logical record:
            # Key bytes, scale, DDR address and logical identity must all agree.
            current_keys = update.inputs[0].detach().cpu().numpy()
            expected_keys = base_index_keys[np.arange(b)[:, None], current_map_after]
            current_scales = update.inputs[1].detach().cpu().numpy()
            expected_scales = base_index_scales[np.arange(b)[:, None], current_map_after]
            current_addresses = update.inputs[2].detach().cpu().numpy()
            expected_addresses = base_index_addresses[np.arange(b)[:, None], current_map_after]
            if not np.array_equal(current_keys, expected_keys):
                raise AssertionError(f"{label}: Index key rows do not match persistent logical map")
            if not np.array_equal(current_scales, expected_scales):
                raise AssertionError(f"{label}: Index scales do not match persistent logical map")
            if not np.array_equal(current_addresses, expected_addresses):
                raise AssertionError(f"{label}: DDR addresses do not match persistent logical map")

            # Every still-valid Hot KV slot must belong to the logical Index
            # row at that slot; invalid slots may be refilled on this step.
            hot_valid = rotating.hot_valid.detach().cpu().numpy().astype(bool)
            hot_output = owner.output[:, :r].detach().cpu().numpy()
            for row in range(b):
                expected_hot = base_payload[row, current_map_after[row, :r][hot_valid[row]]]
                if not np.array_equal(hot_output[row, hot_valid[row]], expected_hot):
                    raise AssertionError(f"{label}: persistent Hot KV ownership mismatch row={row}")

            read_check = actual_reads(
                after_mapping,
                topk.detach().cpu().numpy(),
                rotating.rows.detach().cpu().numpy().view(np.uint32),
                r,
                actual_key_lengths=new_k,
            )
            expected_ignored = sum(h - key for key in new_k)
            if read_check["ignored_out_of_K_mapping"] != expected_ignored:
                raise AssertionError(
                    f"{label}: stale out-of-K entries were consumed or overwritten "
                    f"actual={read_check['ignored_out_of_K_mapping']} expected={expected_ignored}"
                )

            # Carry the actual LRU victim order forward to the next Decode step.
            next_orders = []
            for row, policy in enumerate(persistent_policies):
                touched = state["slots"][row]
                policy.commit(touched[touched < r].astype(np.int32))
                order = []
                slot = int(policy.head[0])
                for _ in range(r):
                    order.append(slot)
                    slot = int(policy.next[0, slot])
                next_orders.append(order)
            update.set_order(next_orders)

            if previous_k is not None and any(
                new < old for old, new in zip(previous_k, new_k)
            ):
                raise AssertionError(f"{label}: Actual-K regressed")
            persistent_stages.append(dict(
                stage=label,
                key_lengths=list(new_k),
                newly_admitted_tokens=(
                    [0] * b if previous_k is None
                    else [int(new_k[i] - previous_k[i]) for i in range(b)]
                ),
                monotonic_growth=True,
                graph_and_workspace_reused=True,
                reset_between_stages=False,
                index_kv_restored_between_stages=False,
                native_li_independent_oracle_pass=True,
                index_key_scale_address_permutation_pass=True,
                index_rows_changed=changed_rows,
                active_k_init_prefix_clear_pass=True,
                active_k_boundary_preserved=True,
                hot_kv_ownership_pass=True,
                topk_slot_payload_cq_pass=True,
                out_of_k_mapping_untouched=True,
                prefetch_requests=read_check["prefetch_hot"] + read_check["prefetch_cold"],
                stale_out_of_k_entries=read_check["ignored_out_of_K_mapping"],
                oracle=oracle_reports,
            ))
            print(
                f"P0_A2_2B_PERSISTENT_STAGE_PASS {label}: "
                f"K={list(new_k)} changed_index_rows={changed_rows}",
                flush=True,
            )
            previous_k = tuple(new_k)

        state.pop("check_phase", None)
        final_k = (h,) * b
        if tuple(actual_key_buffer.cpu().tolist()) != final_k:
            raise AssertionError("Persistent Actual-K sequence did not finish at full History")
        if tuple(tuple(x) for x in rotating.query_ends.cpu().tolist()) != tuple((h,) * 4 for _ in range(b)):
            raise AssertionError("Persistent Actual-K final QueryEnd is not full History")
        persistent_actual_k_report = dict(
            status="PASS",
            mode="monotonic_per_request_growth",
            reset_count=1,
            reset_between_stages=0,
            graph_reused=True,
            workspace_reused=True,
            actual_key_pointer_stable=actual_key_buffer.data_ptr() == old_key_ptr,
            query_ends_pointer_stable=rotating.query_ends.data_ptr() == old_ends_ptr,
            index_kv_restored_between_stages=False,
            final_key_lengths=list(final_k),
            stages=persistent_stages,
        )

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
        logical_lens=actual_key_buffer,
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
        actual_key_lengths=list(key_lengths),
        dynamic_k_transition_report=dynamic_k_transition_report,
        persistent_actual_k_report=persistent_actual_k_report,
        scope="fixed H and BSND K, Q4 shared TopK; static K for timing; "
              "optional A2.2a reset-between-K replay and A2.2b monotonic-K "
              "persistent Index/KV/LRU replay",
    )
