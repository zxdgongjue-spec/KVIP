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
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true", help="Verify hashes and address coverage without Torch/NPU")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc")
    parser.add_argument("--transport", choices=("ubmem", "urma"), default="ubmem")
    parser.add_argument("--mode", choices=("fused", "serial"), default="fused")
    parser.add_argument("--kvaf-server", help="Dedicated local CPU service, required for URMA replay")
    parser.add_argument("--serial-workers", type=int, help="Serial Get workers: service defaults UBmem=32, URMA=2")
    parser.add_argument(
        "--ubmem-ablation",
        action="store_true",
        help="Three UBmem fused cases, sharing one restored DDR arena: baseline, optimized, half-prefetch",
    )
    parser.add_argument(
        "--split-index-stages",
        action="store_true",
        help="Diagnose UBmem index_plan/index_outputs/index_apply instead of combined index_update",
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
    if args.device < 0 or args.warmup < 0 or args.iterations < 1:
        parser.error("device/warmup must be nonnegative and iterations positive")
    if args.stage_iterations < 0:
        parser.error("stage-iterations must be nonnegative")
    if args.measure_call_preparation and (args.inspect or args.mode != "fused"):
        parser.error("--measure-call-preparation requires fused device replay")
    if (args.ubmem_ablation or args.service_ablation or args.split_index_stages) and (
        args.inspect or args.transport != "ubmem" or args.mode != "fused"
    ):
        parser.error("UBmem ablation/split-index-stages require UBmem fused replay")
    if args.ubmem_ablation and args.service_ablation:
        parser.error("Select only one ablation matrix")
    if (args.ubmem_ablation or args.service_ablation) and args.output is None:
        parser.error("Ablation requires --output to retain all case results")
    if args.active_batches and not args.service_ablation:
        parser.error("--active-batches requires --service-ablation")
    if args.serial_workers is not None and (args.mode != "serial" or not 1 <= args.serial_workers <= 48):
        parser.error("serial-workers requires serial mode and a value in 1..48")
    if not args.inspect and ((args.transport == "urma") != bool(args.kvaf_server)):
        parser.error("--kvaf-server is required only for URMA replay")
    if not args.inspect and (args.build_dir is None or not args.soc):
        parser.error("device replay requires --build-dir and --soc")
    if args.output is not None and args.output.exists():
        parser.error("--output exists; refusing to overwrite")
    return args


def inspect_snapshot(archive):
    from kvip.diagnostics.ddr import validate_pages

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
    if getattr(args, "ubmem_ablation", False) or getattr(args, "service_ablation", False):
        from kvip.diagnostics.ubmem_ablation import run_ablation

        return run_ablation(args, archive, retained)
    from kvip.diagnostics.replay_measure import run as measure

    return measure(args, archive, retained)


def main():
    args = arguments()
    try:
        archive = load_snapshot(args.snapshot)
        inspected = inspect_snapshot(archive)
        result = inspected if args.inspect else run(args, archive)
    except BaseException as error:
        traceback.print_exc()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(
                    dict(
                        status="failed",
                        transport=args.transport,
                        mode=args.mode,
                        error=f"{type(error).__name__}: {error}",
                    ),
                    output,
                    indent=2,
                )
        sys.stdout.flush()
        sys.stderr.flush()
        if _RETAINED_RUNS:
            # Completion may be unknown. Avoid Python destructor order while
            # native kernels/DMA can still reference the retained object graph.
            os._exit(130 if isinstance(error, KeyboardInterrupt) else 70)
        raise
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(text)
    if args.ubmem_ablation or args.service_ablation:
        print(f"Results: {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
