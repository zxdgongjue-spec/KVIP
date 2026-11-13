# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen real-Q/Index input through the current serial/fused paths."""

import argparse
import json
import os
from pathlib import Path
import traceback

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.console import NativeLog, emit, flush_output, print_replay_failure
from kvip.diagnostics.pipeline_options import DEFAULT_VARIANTS, VARIANTS

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
    p.add_argument("--topk-probe", action="store_true", help="After all headline cases, run one separately built pruning-counter graph without timing")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.device < 0 or args.diagnostic_replays < 0:
        p.error("Require nonnegative device/warmup and positive iterations")
    if args.output.exists():
        p.error("Output already exists")
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
    if args.study in ("threshold-only", "seed-ub", "local-victims", "register-kth", "overlap-victims") and (
        args.transport != "urma" or args.mode != "fused"
    ):
        p.error(f"study={args.study} requires URMA fused replay")
    if args.study in ("seed-ub", "local-victims", "register-kth", "overlap-victims") and args.diagnostic_replays:
        p.error(f"study={args.study} excludes instrumented diagnostic replays")
    args.fused_tuning = {"recent_threshold": args.threshold_policy == "recent"}
    return args


def main(argv=None):
    args = arguments(argv)
    retained = {"progress": {"variant": "unassigned", "phase": "input_validation", "step": None}}
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        label = "upstream" if args.upstream_root else args.threshold_policy
        for name, case in result.get("cases", {label: result}).items():
            emit(
                f"REPLAY_PASS {args.transport}-{args.mode}-{name} "
                f"mean_us={case['timing']['mean_us']:.3f} p50_us={case['timing']['p50_us']:.3f}"
            )
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
        flush_output()
        # Unknown device completion: preserve owners until process termination.
        os._exit(130 if isinstance(error, KeyboardInterrupt) else 70)


if __name__ == "__main__":
    main()
