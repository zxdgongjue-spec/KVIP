# SPDX-License-Identifier: Apache-2.0
"""Replay one real decode group without model weights; setup/reset are untimed."""

import argparse
import json
import os
import traceback
import sys
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.console import NativeLog, compact_report, emit, flush_output, report_only
from kvip.diagnostics.snapshot import (
    relocate_snapshot_addresses,
    validate_capture_inputs,
)


# If completion becomes unknown, Python must not collect tensors or destroy a
# captured graph while kernels may still reference them. CLI failure retains
# the complete owner bundle until process exit, with no unsafe cleanup retry.
_RETAINED_RUNS = []


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--quiet", action="store_true", help="Keep key progress/results; retain native logs only on failure"
    )
    parser.add_argument("--report-only", type=Path, help="Summarize an existing result JSON without loading a runtime")
    parser.add_argument("--inspect", action="store_true", help="Verify hashes and address coverage without Torch/NPU")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc")
    parser.add_argument("--transport", choices=("ubmem", "urma"), default="ubmem")
    parser.add_argument("--mode", choices=("fused", "serial"), default="fused")
    parser.add_argument("--kvaf-server", help="Dedicated local CPU service, required for URMA replay")
    parser.add_argument("--serial-workers", type=int, help="Serial Get workers: service defaults UBmem=32, URMA=2")
    parser.add_argument(
        "--urma-serial-post-size",
        type=int,
        choices=(32, 128),
        help="URMA serial WQEs per post; default 128, 32 requires the optional post32 kernel",
    )
    parser.add_argument(
        "--urma-serial-max-inflight",
        type=int,
        help="URMA serial total inflight bound: 0 submits whole core slices; omitted keeps native default 2048",
    )
    parser.add_argument("--fused-tuning", help="JSON object of supported options for one fused replay")
    parser.add_argument(
        "--fused-variants", help="JSON list of named same-transport tuning objects sharing one DDR import"
    )
    parser.add_argument(
        "--ubmem-ablation",
        action="store_true",
        help="Three UBmem fused cases, sharing one restored DDR arena: baseline, optimized, half-prefetch",
    )
    parser.add_argument(
        "--split-index-stages",
        action="store_true",
        help="Split existing Index plan/output and apply launches for stage diagnostics",
    )
    parser.add_argument(
        "--service-ablation",
        action="store_true",
        help="Compare dense baseline vs inactive-LI skipping and compact reserved-hot construction on real rows",
    )
    parser.add_argument(
        "--active-batches",
        type=int,
        nargs="+",
        help="Derived active request counts for service-ablation (default: 1 and captured-batch)",
    )
    parser.add_argument(
        "--measure-call-preparation",
        action="store_true",
        help="Also time one group's captured-candidate CPU staging, H2D and fusion completion (not the full service step)",
    )
    parser.add_argument(
        "--stage-iterations",
        type=int,
        default=0,
        help="Separate instrumented stage runs; full graph timing remains uninstrumented",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, help="New result JSON; stdout also contains the result")
    args = parser.parse_args(argv)
    if args.report_only is not None:
        if any(
            (
                args.snapshot,
                args.build_dir,
                args.soc,
                args.output,
                args.inspect,
                args.kvaf_server,
                args.fused_tuning,
                args.fused_variants,
                args.ubmem_ablation,
                args.service_ablation,
            )
        ):
            parser.error("--report-only cannot be combined with capture/build/output or run options")
        return args
    if args.snapshot is None:
        parser.error("--snapshot is required unless --report-only is used")
    if args.device < 0 or args.warmup < 0 or args.iterations < 1:
        parser.error("device/warmup must be nonnegative and iterations positive")
    if args.stage_iterations < 0:
        parser.error("stage-iterations must be nonnegative")
    if args.fused_variants is not None:
        if (
            args.inspect
            or args.mode != "fused"
            or args.ubmem_ablation
            or args.service_ablation
            or args.fused_tuning is not None
            or args.output is None
        ):
            parser.error("--fused-variants requires fused replay, --output, and no other ablation/tuning option")
        from kvip.diagnostics.ubmem_ablation import parse_variants

        try:
            args.fused_variants = parse_variants(args.fused_variants, args.transport)
        except ValueError as error:
            parser.error(str(error))
    if args.fused_tuning is not None:
        if args.inspect or args.mode != "fused" or args.ubmem_ablation or args.service_ablation:
            parser.error("--fused-tuning requires one fused device replay, without an ablation matrix")
        from kvip.diagnostics.fused_replay import parse_fused_tuning

        try:
            args.fused_tuning = parse_fused_tuning(args.fused_tuning, args.transport)
        except ValueError as error:
            parser.error(str(error))
    if args.measure_call_preparation and (args.inspect or args.mode != "fused"):
        parser.error("--measure-call-preparation requires fused device replay")
    if (args.ubmem_ablation or args.service_ablation) and (
        args.inspect or args.transport != "ubmem" or args.mode != "fused"
    ):
        parser.error("UBmem ablation requires UBmem fused replay")
    if args.split_index_stages and (args.inspect or args.mode != "fused"):
        parser.error("--split-index-stages requires fused device replay")
    if args.ubmem_ablation and args.service_ablation:
        parser.error("Select only one ablation matrix")
    if (args.ubmem_ablation or args.service_ablation) and args.output is None:
        parser.error("Ablation requires --output to retain all case results")
    if args.active_batches and not args.service_ablation:
        parser.error("--active-batches requires --service-ablation")
    if args.serial_workers is not None and (args.mode != "serial" or not 1 <= args.serial_workers <= 48):
        parser.error("serial-workers requires serial mode and a value in 1..48")
    if args.urma_serial_post_size is not None and (args.inspect or (args.transport, args.mode) != ("urma", "serial")):
        parser.error("--urma-serial-post-size requires URMA serial device replay")
    if args.urma_serial_post_size is None:
        args.urma_serial_post_size = 128
    if args.urma_serial_max_inflight is not None:
        if args.inspect or (args.transport, args.mode) != ("urma", "serial"):
            parser.error("--urma-serial-max-inflight requires URMA serial device replay")
        from kvip.diagnostics.replay_runtime import validate_urma_serial_max_inflight

        try:
            validate_urma_serial_max_inflight(args.urma_serial_max_inflight, args.serial_workers or 2)
        except ValueError as error:
            parser.error(str(error))
    if not args.inspect and ((args.transport == "urma") != bool(args.kvaf_server)):
        parser.error("--kvaf-server is required only for URMA replay")
    if not args.inspect and (args.build_dir is None or not args.soc):
        parser.error("device replay requires --build-dir and --soc")
    if args.output is not None and args.output.exists():
        parser.error("--output exists; refusing to overwrite")
    return args


def inspect_snapshot(archive):
    from kvip.diagnostics.ddr import validate_pages
    from kvip.diagnostics.demand import DEMAND_ACCOUNTING, snapshot_demand

    ddr = archive.manifest["ddr"]
    summary = validate_pages(archive.directory / "ddr", ddr)
    relocate_snapshot_addresses(archive, ddr["region_base"])
    validate_capture_inputs(archive)
    if archive.manifest.get("oracle", {}).get("passed") is not True:
        raise ValueError("Captured service call has no verified oracle")
    geometry = archive.manifest["geometry"]
    metadata = archive.array("pre.metadata")
    owners = metadata[4].view(np.int32)[: geometry["batch"]]
    ends = archive.array("pre.query_ends").reshape(geometry["batch"], 4)
    metadata_source = archive.manifest["metadata"]
    source = {
        name: metadata_source.get(name) for name in ("capture_mode", "dp_rank", "dp_size", "group", "layer_names")
    }
    source["request_count"] = len(metadata_source.get("requests", []))
    source["vllm_commit"] = metadata_source.get("provenance", {}).get("vllm_commit")
    source["kvip_commit"] = archive.manifest.get("provenance", {}).get("kvip_commit", "unknown")
    return dict(
        status="verified_snapshot",
        geometry=geometry,
        query_ends=ends.tolist(),
        row_owners=owners.tolist(),
        active_requests=int(np.count_nonzero(np.any(ends > 0, axis=1))),
        ddr=summary,
        source=source,
        captured_fusion_demand=snapshot_demand(archive),
        demand_accounting=DEMAND_ACCOUNTING,
        scope=archive.manifest["scope"],
        fidelity=archive.manifest["fidelity"],
    )


def run(args, archive):
    retained = {}
    _RETAINED_RUNS.append(retained)
    result = _run(args, archive, retained)
    _RETAINED_RUNS[:] = [owner for owner in _RETAINED_RUNS if owner is not retained]
    return result


def _run(args, archive, retained):
    if (
        getattr(args, "ubmem_ablation", False)
        or getattr(args, "service_ablation", False)
        or getattr(args, "fused_variants", None) is not None
    ):
        from kvip.diagnostics.ubmem_ablation import run_ablation

        return run_ablation(args, archive, retained)
    from kvip.diagnostics.replay_measure import run as measure

    return measure(args, archive, retained)


def main():
    args = arguments()
    if args.report_only is not None:
        report_only(args.report_only)
        return
    logging = NativeLog(args.quiet)
    try:
        with logging:
            emit(
                f"CONFIG transport={args.transport} mode={args.mode} device={args.device} "
                f"warmup={args.warmup} iterations={args.iterations} stages={args.stage_iterations}"
            )
            emit("SNAPSHOT_CHECK_BEGIN")
            archive = load_snapshot(args.snapshot)
            inspected = inspect_snapshot(archive)
            emit("SNAPSHOT_CHECK_PASS")
            matrix = args.ubmem_ablation or args.service_ablation or args.fused_variants is not None
            if not args.inspect and not matrix:
                emit(f"CASE_BEGIN {args.transport}-{args.mode}")
            result = inspected if args.inspect else run(args, archive)
            if not args.inspect and not matrix:
                emit(f"CASE_PASS {args.transport}-{args.mode}")
    except BaseException as error:
        if logging.failure_path is None:
            traceback.print_exc()
        if args.output is not None:
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as output:
                    json.dump(
                        dict(
                            status="failed",
                            transport=args.transport,
                            mode=args.mode,
                            error=f"{type(error).__name__}: {error}",
                            failure_log=str(logging.failure_path) if logging.failure_path else None,
                        ),
                        output,
                        indent=2,
                    )
            except Exception as save_error:
                emit(f"Could not save failure JSON {args.output}: {save_error}", file=sys.stderr)
        flush_output()
        if _RETAINED_RUNS:
            # Completion may be unknown. Avoid Python destructor order while
            # native kernels/DMA can still reference the retained object graph.
            os._exit(130 if isinstance(error, KeyboardInterrupt) else 70)
        if args.quiet:
            raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None
        raise
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(text)
    if args.quiet:
        compact_report(result, path=args.output)
    elif args.ubmem_ablation or args.service_ablation or args.fused_variants is not None:
        print(f"Results: {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
