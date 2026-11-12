# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen real-Q/Index input through the current serial/fused paths."""

import argparse
import json
import os
from pathlib import Path

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.console import NativeLog, emit, flush_output

_RETAINED = []


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--transport", choices=("urma", "ubmem"), required=True)
    p.add_argument("--mode", choices=("serial", "fused"), required=True)
    p.add_argument("--threshold-policy", choices=("resident", "recent", "both"), default="resident")
    p.add_argument("--study", choices=("threshold", "pipeline", "skip"), default="threshold")
    p.add_argument("--kvaf-server")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--soc", required=True)
    p.add_argument("--serial-workers", type=int)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.device < 0:
        p.error("Require nonnegative device/warmup and positive iterations")
    if args.output.exists():
        p.error("Output already exists")
    if not args.kvaf_server:
        p.error("A local KVAF CPU service address is required for this runtime")
    if args.mode == "serial" and args.threshold_policy != "resident":
        p.error("Threshold policy changes only the fused path")
    args.fused_tuning = {"recent_threshold": args.threshold_policy == "recent"}
    return args


def main(argv=None):
    args = arguments(argv)
    retained = {}
    _RETAINED.append(retained)
    log = NativeLog(args.quiet)
    try:
        with log:
            from kvip.diagnostics.ddr import validate_pages
            from kvip.diagnostics.replay_measure import run
            from kvip.diagnostics.snapshot import validate_capture_inputs

            archive = load_snapshot(args.snapshot)
            validate_capture_inputs(archive)
            validate_pages(archive.directory / "ddr", archive.manifest["ddr"])
            if archive.manifest.get("content") != "real_li_synthetic_kv":
                raise ValueError("This comparison requires real Q/Index and generated KV")
            result = run(args, archive, retained)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        for name, case in result.get("cases", {args.threshold_policy: result}).items():
            emit(
                f"REPLAY_PASS {args.transport}-{args.mode}-{name} "
                f"mean_us={case['timing']['mean_us']:.3f} p50_us={case['timing']['p50_us']:.3f}"
            )
    except BaseException as error:
        emit(f"REPLAY_FAILED {type(error).__name__}: {error}")
        flush_output()
        # Unknown device completion: preserve owners until process termination.
        os._exit(130 if isinstance(error, KeyboardInterrupt) else 70)


if __name__ == "__main__":
    main()
