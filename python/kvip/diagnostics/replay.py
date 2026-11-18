# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen real-Q/Index input through the current serial/fused paths."""

import argparse
import json
import os
from pathlib import Path
import traceback

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.console import NativeLog, emit, flush_output, print_replay_failure
from kvip.diagnostics.pipeline_options import DEFAULT_VARIANTS, VARIANTS, is_auto_variant
from kvip.diagnostics.replay_journal import ReplayJournal
from kvip.diagnostics.system_profile import PROFILE_METRICS

_RETAINED = []


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--transport", choices=("urma", "ubmem"), required=True)
    p.add_argument("--mode", choices=("serial", "fused"), required=True)
    p.add_argument("--threshold-policy", choices=("resident", "recent", "both"), default="resident")
    p.add_argument(
        "--study",
        choices=(
            "threshold",
            "pipeline",
            "skip",
            "threshold-only",
            "seed-ub",
            "local-victims",
            "register-kth",
            "overlap-victims",
            "post-threshold",
        ),
        default="threshold",
    )
    p.add_argument("--upstream-root", type=Path, help="Use production modules from this exact clean checkout")
    p.add_argument("--upstream-revision", help="Full pinned upstream commit, required with upstream-root")
    p.add_argument("--kvaf-server")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--soc", required=True)
    p.add_argument("--serial-workers", type=int)
    p.add_argument(
        "--pipeline-variants",
        nargs="+",
        default=None,
        choices=tuple(VARIANTS),
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument(
        "--diagnostic-replays",
        type=int,
        default=0,
        help="Separate instrumented fused graph replays after headline timing",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--check-mode-switch", action="store_true",
                   help="Untimed light-low replay: one captured graph, quotas 0/512/0, restore before each")
    p.add_argument("--check-auto-route", action="store_true",
                   help="Untimed device-feedback lifecycle checks on the auto variant")
    p.add_argument("--verify-consumer-work", action="store_true",
                   help="Check HIGH Index events, final Get descriptors and persistent Index outside timing")
    p.add_argument("--high-post-index", action="store_true",
                   help="Enable the full-AIV terminal Index on naturally selected eight-window HIGH steps")
    p.add_argument("--frozen-high-tail-library", type=Path,
                   help="Run three orders on one frozen descriptor stream with the same live URMA queues")
    p.add_argument("--post-index-wait-first", action="store_true",
                   help="Offline same-binary control: drain CQ before the complete Index update")
    p.add_argument("--calibrate-counter", action="store_true",
                   help="After headline timing, calibrate device cycles with three independent long graph events")
    p.add_argument("--profile-dir", type=Path, help="Independent system traces, outside headline timing")
    p.add_argument("--profile-graph-event", action="store_true",
                   help="Record the complete graph Event inside the same Level0 trace")
    p.add_argument("--profile-metrics", nargs="+", choices=PROFILE_METRICS, default=None)
    p.add_argument(
        "--topk-probe",
        action="store_true",
        help="After all headline cases, run one separately built pruning-counter graph without timing",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.device < 0 or args.diagnostic_replays < 0:
        p.error("Require nonnegative device/warmup and positive iterations")
    if args.output.exists():
        p.error("Output already exists")
    if args.profile_metrics is not None and args.profile_dir is None:
        p.error("profile-metrics requires profile-dir")
    if args.profile_graph_event and (args.profile_dir is None or args.profile_metrics not in (None, ["none"])):
        p.error("profile-graph-event requires a Level0 timeline (--profile-dir, metric none)")
    if args.profile_dir is not None:
        if args.topk_probe or args.diagnostic_replays:
            p.error("profile-dir cannot be combined with topk-probe or diagnostic-replays")
        if args.profile_dir.exists() and not args.profile_dir.is_dir():
            p.error("profile-dir must be a directory")
        args.profile_metrics = args.profile_metrics or ["none"]
        if len(set(args.profile_metrics)) != len(args.profile_metrics):
            p.error("profile-metrics must be unique")
    if not args.kvaf_server:
        p.error("A local KVAF CPU service address is required for this runtime")
    if args.mode == "serial" and args.threshold_policy != "resident":
        p.error("Threshold policy changes only the fused path")
    if bool(args.upstream_root) != bool(args.upstream_revision):
        p.error("upstream-root and upstream-revision must be supplied together")
    if args.upstream_root and (args.study not in ("threshold", "post-threshold") or args.threshold_policy == "both"):
        p.error("Exact production replay requires threshold/post-threshold study and one threshold policy")
    if args.study == "post-threshold":
        if (
            not args.upstream_root
            or args.mode != "fused"
            or args.transport != "urma"
            or args.threshold_policy != "recent"
        ):
            p.error("post-threshold requires an exact production checkout and URMA fused recent policy")
        args.pipeline_variants = args.pipeline_variants or list(DEFAULT_VARIANTS)
        if len(set(args.pipeline_variants)) != len(args.pipeline_variants):
            p.error("Pipeline variants must be unique")
    elif args.pipeline_variants:
        p.error("pipeline-variants requires study=post-threshold")
    if args.topk_probe and args.study != "post-threshold":
        p.error("topk-probe requires study=post-threshold")
    if args.check_mode_switch and (args.study != "post-threshold" or "light-low" not in args.pipeline_variants):
        p.error("check-mode-switch requires post-threshold with light-low")
    auto = args.study == "post-threshold" and any(is_auto_variant(name) for name in args.pipeline_variants)
    if args.check_auto_route and not auto:
        p.error("check-auto-route requires post-threshold with an automatic route variant")
    if auto and args.topk_probe:
        p.error("Auto routing excludes topk-probe")
    if auto and args.profile_dir and not (
        args.pipeline_variants == ["auto-post-index"] and args.profile_metrics == ["none"]
        and args.check_auto_route and args.iterations >= 2 and not args.post_index_wait_first
    ):
        p.error("Auto timeline requires only auto-post-index, --profile-metrics none, --check-auto-route, "
                "at least two timed iterations and normal Post-Index-CQ order")
    if (args.post_index_wait_first or args.high_post_index) and (
        args.study != "post-threshold" or args.pipeline_variants != ["auto-post-index"]
    ):
        p.error("post-index-wait-first/high-post-index requires only the auto-post-index variant")
    if args.verify_consumer_work and (not args.check_auto_route or args.transport != "urma" or args.post_index_wait_first
                                      or args.pipeline_variants != ["auto-post-index"]):
        p.error("verify-consumer-work requires URMA auto-post-index, normal completion order and check-auto-route")
    if args.frozen_high_tail_library and (not args.high_post_index or args.post_index_wait_first
                                          or not args.diagnostic_replays or not args.calibrate_counter):
        p.error("frozen-high-tail-library requires high-post-index, calibration and diagnostic replays in normal order")
    if args.calibrate_counter and (args.mode != "fused" or not args.diagnostic_replays):
        p.error("calibrate-counter requires fused mode and separate diagnostic-replays")
    if args.study in ("threshold-only", "seed-ub", "local-victims", "register-kth", "overlap-victims") and (
        args.transport != "urma" or args.mode != "fused"
    ):
        p.error(f"study={args.study} requires URMA fused replay")
    if args.study in ("seed-ub", "local-victims", "register-kth", "overlap-victims") and args.diagnostic_replays:
        p.error(f"study={args.study} excludes instrumented diagnostic replays")
    args.fused_tuning = {"recent_threshold": args.threshold_policy == "recent"}
    return args


def finish_checkpoint(retained, *, status, report):
    """A journal write failure must not change the actual replay outcome."""
    try:
        retained["journal"].finish(retained["progress"], status=status, report=report)
    except (OSError, TypeError, ValueError) as error:
        emit(f"Replay completion checkpoint could not be saved: {error}")


def main(argv=None):
    args = arguments(argv)
    retained = {"progress": {"variant": "unassigned", "phase": "input_validation", "step": None}}
    retained["journal"] = ReplayJournal(args.output)
    retained["journal"].phase(retained["progress"])
    _RETAINED.append(retained)
    log = NativeLog(args.quiet, report_failure=False)
    try:
        with log:
            if args.upstream_root:
                from kvip.diagnostics.upstream import activate, verify_lidu_sources

                feature_root = Path(__file__).resolve().parents[3]
                args.lidu_provenance = verify_lidu_sources(args.upstream_root, feature_root)
                args.production_source = activate(args.upstream_root, args.upstream_revision)
            from kvip.diagnostics.ddr import validate_pages
            from kvip.diagnostics.replay_measure import run
            from kvip.diagnostics.snapshot import validate_capture_inputs

            archive = load_snapshot(args.snapshot)
            validate_capture_inputs(archive)
            validate_pages(archive.directory / "ddr", archive.manifest["ddr"])
            if archive.manifest.get("content") != "real_li_synthetic_kv":
                raise ValueError("This comparison requires real Q/Index and generated KV")
            result = run(args, archive, retained)
        retained["progress"].update(phase="result_write", step=None)
        retained["journal"].phase(retained["progress"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        finish_checkpoint(retained, status="complete", report=args.output)
        label = "upstream" if args.upstream_root else args.threshold_policy
        for name, case in result.get("cases", {label: result}).items():
            emit(
                f"REPLAY_PASS {args.transport}-{args.mode}-{name} "
                f"mean_us={case['timing']['mean_us']:.3f} p50_us={case['timing']['p50_us']:.3f}"
            )
            if case.get("mode_switch_check"):
                emit("MODE_SWITCH_PASS one_graph quotas=0/512/0 captured_addresses=unchanged "
                     "low_threshold_sentinel=unchanged "
                     "snapshot_restore_each=true untimed=true continuous_service_state=not_tested")
            if route := case.get("auto_route"):
                emit(f"AUTO_ROUTE_PASS {name} low_steps={route['low_steps']} high_steps={route['high_steps']} "
                     "feedback=device_previous_success capture_warmup_learning=false device_policy_in_timing=true")
                if route.get("lifecycle"):
                    emit("AUTO_LIFECYCLE_PASS one_graph feedback_reset=false packed_row_reorder=true "
                         "prediction_fault_injection=true rejection=identity_not_transport_fault snapshot_restore_each=true")
                    if route["lifecycle"].get("persistent_carry"):
                        emit("READ_THROUGH_CARRY_PASS one_graph low/repacked-low/high persistent_reset_between_steps=false "
                             "hot_abc_index=verified selected_KV_bytes=verified untimed=true")
    except BaseException as error:
        active = retained.get("active_case")
        result = dict(
            schema_version=2,
            status="failed",
            transport=args.transport,
            mode=args.mode,
            study=args.study,
            snapshot=str(args.snapshot),
            progress=dict(retained.get("progress", {})),
            geometry=retained.get("geometry", {}),
            buffers=active.get("buffers", {}) if active is not None else retained.get("buffers", {}),
            buffers_variant=retained.get("progress", {}).get("variant")
            if active is not None
            else retained.get("buffers_variant"),
            completed_cases=retained.get("completed_cases", {}),
            checkpoints=dict(
                cases=str(retained["journal"].directory), progress=str(retained["journal"].progress),
            ),
            profile_manifests=active.get("system_profile_manifests", []) if active is not None else [],
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
            native_log=str(log.failure_path) if log.failure_path else None,
        )
        save_error = None
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2)
        except (OSError, TypeError, ValueError) as writing_error:
            save_error = writing_error
        print_replay_failure(result, result_path=args.output, log_path=log.failure_path)
        if save_error is not None:
            emit(f"Failure JSON could not be saved: {save_error}")
        finish_checkpoint(retained, status="failed", report=args.output if save_error is None else None)
        flush_output()
        # Unknown device completion: preserve owners until process termination.
        os._exit(130 if isinstance(error, KeyboardInterrupt) else 70)


if __name__ == "__main__":
    main()
