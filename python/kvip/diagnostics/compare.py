# SPDX-License-Identifier: Apache-2.0
"""Four production paths on one real snapshot, one transport/process at a time."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CASES = (("ubmem", "serial"), ("urma", "serial"), ("ubmem", "fused"), ("urma", "fused"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--ubmem-build-dir", required=True, type=Path)
    parser.add_argument("--urma-build-dir", required=True, type=Path)
    parser.add_argument("--kvaf-server", required=True, help="Dedicated local URMA CPU service")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--stage-iterations", type=int, default=3)
    parser.add_argument("--ubmem-serial-workers", type=int, default=32)
    parser.add_argument("--urma-serial-workers", type=int, default=2)
    parser.add_argument("--result-dir", required=True, type=Path, help="New directory; four cases and one summary JSON")
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.stage_iterations < 0:
        parser.error("Invalid device/warmup/iterations/stage-iterations")
    if any(not 1 <= n <= 48 for n in (args.ubmem_serial_workers, args.urma_serial_workers)):
        parser.error("Serial worker counts must be 1..48")
    if args.result_dir.exists():
        parser.error("--result-dir already exists; choose a new directory")
    if args.ubmem_build_dir.resolve() == args.urma_build_dir.resolve():
        parser.error("URMA and UBmem require separate KVAF builds")
    return args


def command(args, transport, mode):
    build = getattr(args, transport + "_build_dir")
    result = args.result_dir / f"{transport}-{mode}.json"
    values = [
        sys.executable,
        "-B",
        "-m",
        "kvip.diagnostics.replay",
        "--snapshot",
        str(args.snapshot),
        "--transport",
        transport,
        "--mode",
        mode,
        "--build-dir",
        str(build),
        "--device",
        str(args.device),
        "--soc",
        args.soc,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--stage-iterations",
        str(args.stage_iterations),
        "--output",
        str(result),
    ]
    if transport == "urma":
        values += ["--kvaf-server", args.kvaf_server]
    if mode == "serial":
        values += ["--serial-workers", str(getattr(args, transport + "_serial_workers"))]
    return values, result


def summarize(results):
    if set(results) != {f"{transport}-{mode}" for transport, mode in CASES}:
        raise ValueError("All four cases are required")
    first = results["ubmem-serial"]
    for name, result in results.items():
        if result.get("status") != "pass":
            raise ValueError(f"Case did not pass validation: {name}")
        for key in ("snapshot_sha256", "geometry", "scope", "soc", "device", "kvip_commit"):
            if result.get(key) != first.get(key) or key not in result:
                raise ValueError(f"Incomparable {key}: {name}")
        if name != f"{result['transport']}-{result['mode']}":
            raise ValueError("Case identity mismatch")
    table = {}
    for name, result in results.items():
        timing = result["timing"]
        if not all(np.isfinite(timing[k]) and timing[k] > 0 for k in ("mean_us", "p50_us")):
            raise ValueError(f"Invalid timing: {name}")
        table[name] = dict(
            mean_us=timing["mean_us"],
            p50_us=timing["p50_us"],
            mean_read_bytes=float(np.mean([x["bytes"] for x in result["reads_samples"]])),
            runtime_workers=result["runtime_workers"],
            reads_per_round={
                key: float(np.mean([row[key] for row in result["reads_samples"]]))
                for key in ("prefetch", "final", "unused_prefetch", "cold_demand", "candidate_hot_refetch")
                if key in result["reads_samples"][0]
            },
        )
    ratios = {}
    for transport in ("ubmem", "urma"):
        serial, fused = (table[f"{transport}-{mode}"]["mean_us"] for mode in ("serial", "fused"))
        ratios[transport] = dict(serial_over_fused=serial / fused, latency_reduction_percent=(1 - fused / serial) * 100)
    return dict(
        status="pass",
        snapshot_sha256=first["snapshot_sha256"],
        geometry=first["geometry"],
        scope=first["scope"],
        table=table,
        fusion_vs_serial=ratios,
        stage_diagnostics={name: result.get("stage_diagnostics") for name, result in results.items()},
        notes=[
            "Full-path latency uses one uninstrumented production graph per case.",
            "Stage diagnostics add graph/event boundaries: use to localize cost, not as an exact decomposition.",
            "Serial uses native CANN LI plus production LIDU/Get; fused uses its own LI and Index update.",
            "The snapshot fixes inputs and pre-state, not speculative prefetch scheduling or physical pages.",
        ],
    )


def display(summary):
    print("\ncase          mean_us     p50_us   read_MiB/round   Get_workers")
    for transport, mode in CASES:
        name = f"{transport}-{mode}"
        row = summary["table"][name]
        print(
            f"{name:13s} {row['mean_us']:10.3f} {row['p50_us']:10.3f} "
            f"{row['mean_read_bytes'] / (1 << 20):16.3f} {row['runtime_workers']:13d}"
        )
        print("  reads/round: " + ", ".join(f"{key}={value:.1f}" for key, value in row["reads_per_round"].items()))
    for transport, values in summary["fusion_vs_serial"].items():
        print(
            f"{transport}: serial/fused={values['serial_over_fused']:.3f}x; "
            f"latency reduction={values['latency_reduction_percent']:.2f}%"
        )
    for name, values in summary["stage_diagnostics"].items():
        if values is None:
            continue
        ordered = sorted(values["stages"].items(), key=lambda item: item[1]["mean_us"], reverse=True)
        print(f"\n{name} instrumented stages (not the uninstrumented total):")
        for stage, timing in ordered:
            print(f"  {stage:30s} {timing['mean_us']:10.3f} us")
        print(f"  stage-chain total              {values['total']['mean_us']:10.3f} us")


def main():
    args = arguments()
    args.result_dir.mkdir(parents=True)
    results = {}
    for transport, mode in CASES:
        name = f"{transport}-{mode}"
        print(f"[snapshot-compare] {name}: restore real DDR/pre-state, then measure", flush=True)
        argv, result_file = command(args, transport, mode)
        # Keep successful native setup chatter out of the terminal. The temporary
        # spool is removed immediately, not retained as another benchmark log.
        with tempfile.TemporaryFile(mode="w+") as output:
            result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                output.seek(0)
                lines = output.readlines()
                print("".join(lines[-100:]), file=sys.stderr, end="")
                raise SystemExit(f"{name} failed (exit {result.returncode}); result: {result_file}")
        results[name] = json.loads(result_file.read_text())
        print(f"[snapshot-compare] {name}: PASS mean={results[name]['timing']['mean_us']:.3f} us", flush=True)
    summary = summarize(results)
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    display(summary)
    print(f"\nResults: {args.result_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
