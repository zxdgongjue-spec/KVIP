# SPDX-License-Identifier: Apache-2.0
"""Compare current serial chains and resident/recent fusion on one frozen input.

Each backend has its own process. A fused A/B pair shares one DDR import and
uses the same production launcher, changing only its threshold range pointer.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kvip.diagnostics.archive import file_digest, source_commit
from kvip.diagnostics.console import failure_tail


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True, help="Current dual-transport KVAF build-npu")
    p.add_argument("--kvaf-server", required=True, help="Local CPU DDR service; required by both current transports")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--soc", default="Ascend950DT_9582")
    p.add_argument("--serial-workers", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--result-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or not 1 <= args.serial_workers <= 48:
        p.error("Invalid device, warmup, iterations or workers")
    if args.result_dir.exists():
        p.error("result-dir already exists; choose a new directory")
    for root, object_name in ((args.build_dir, "kvaf_execute_batch_get_kernels.o"),):
        if root is not None and not (root / "bin" / object_name).is_file():
            p.error(f"Missing exact requested kernel: {root / 'bin' / object_name}")
    return args


def read_means(result):
    samples = result["reads_samples"]
    names = (
        "cold_miss_rate",
        "cold_demand",
        "selected_unique",
        "prefetch",
        "final",
        "unused_prefetch",
        "unused_prefetch_bytes",
        "bytes",
        "useful_transfer_bytes",
    )
    return {name: sum(x[name] for x in samples) / len(samples) for name in names}


def summarize(results):
    first = next(iter(results.values()))
    for result in results.values():
        if result.get("status") != "pass" or any(
            result[key] != first[key] for key in ("snapshot_sha256", "geometry", "scope", "soc", "device")
        ):
            raise ValueError("Comparison requires successful identical frozen inputs, scope, geometry and device")
    table = {}
    for name, result in results.items():
        reads = read_means(result)
        mean = result["timing"]["mean_us"]
        table[name] = dict(
            mean_us=mean,
            p50_us=result["timing"]["p50_us"],
            reads=reads,
            graph_payload_GBps=reads["bytes"] / mean / 1000,
            threshold=result.get("threshold_comparison"),
            selection=result["selection_comparison"],
            workers=result["runtime_workers"],
            post_size=result.get("batch_get_post_size"),
        )
    effects = {}
    for transport in ("urma", "ubmem"):
        best = transport + "-serial"
        before, after = (table[f"{transport}-fused-{policy}"] for policy in ("resident", "recent"))
        if any(
            before["selection"][key] != after["selection"][key]
            for key in ("selected_set_sha256", "ordered_output_sha256")
        ):
            raise ValueError("Threshold A/B changed the final selection")
        effects[transport] = dict(
            serial_reference=best,
            recent_vs_resident_reduction_pct=100 * (1 - after["mean_us"] / before["mean_us"]),
            recent_vs_serial_reduction_pct=100 * (1 - after["mean_us"] / table[best]["mean_us"]),
        )
    return dict(
        status="pass",
        kvip_commit=source_commit(),
        geometry=first["geometry"],
        snapshot_sha256=first["snapshot_sha256"],
        table=table,
        effects=effects,
        scope=first["scope"],
        timing_boundary=first["timing_boundary"],
    )


def print_summary(summary):
    print(
        "case                         mean_us    p50_us  cold_miss%   prefetched  unused_prefetch  total_read_MiB",
        flush=True,
    )
    for name, row in summary["table"].items():
        r = row["reads"]
        print(
            f"{name:28} {row['mean_us']:9.3f} {row['p50_us']:9.3f} {100 * r['cold_miss_rate']:10.4f} "
            f"{r['prefetch']:12.1f} {r['unused_prefetch']:16.1f} {r['bytes'] / (1 << 20):15.3f}",
            flush=True,
        )
    for transport, effect in summary["effects"].items():
        a, b = (summary["table"][f"{transport}-fused-{p}"]["threshold"] for p in ("resident", "recent"))
        print(
            f"THRESHOLD {transport} gap_mean={a['mean_gap']:.7g}->{b['mean_gap']:.7g} "
            f"equal_final={100 * a['equal_fraction']:.2f}%->{100 * b['equal_fraction']:.2f}% "
            "native_threshold=bit_exact final_scores=bit_exact final_selection=unchanged_ordered",
            flush=True,
        )
        print(
            f"GAIN {transport} recent_vs_resident={effect['recent_vs_resident_reduction_pct']:.2f}% "
            f"recent_vs_serial={effect['recent_vs_serial_reduction_pct']:.2f}% "
            f"serial_reference={effect['serial_reference']}",
            flush=True,
        )
    print(
        "TIMING complete device chain (recent scoring/merge included); excludes data restoration, CPU scheduler and input H2D.",
        flush=True,
    )


def run(args):
    args.result_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    cases = [(transport, mode, args.build_dir, 128) for transport in ("urma", "ubmem") for mode in ("serial", "fused")]
    for transport, mode, build, post_size in cases:
        name = f"{transport}-{mode}"
        output = args.result_dir / f"{name}.json"
        log = args.result_dir / f".{name}.log"
        cmd = [
            sys.executable,
            "-B",
            "-m",
            "kvip.diagnostics.replay",
            "--snapshot",
            str(args.snapshot),
            "--build-dir",
            str(build),
            "--transport",
            transport,
            "--mode",
            mode,
            "--threshold-policy",
            "both" if mode == "fused" else "resident",
            "--kvaf-server",
            args.kvaf_server,
            "--device",
            str(args.device),
            "--soc",
            args.soc,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--output",
            str(output),
        ]
        if mode == "serial":
            cmd += ["--serial-workers", str(args.serial_workers)]
        print(
            f"RUN {name}"
            + (
                " resident/recent share DDR"
                if mode == "fused"
                else f" workers={args.serial_workers}"
                + (f" post={post_size} inflight=2048" if transport == "urma" else "")
            ),
            flush=True,
        )
        with log.open("w") as stream:
            process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            failure_tail(log, message=f"{name} exit={process.returncode}", lines=45)
            raise SystemExit(1)
        report = json.loads(output.read_text())
        if mode == "fused":
            results.update({f"{name}-{key}": value for key, value in report["cases"].items()})
        else:
            results[name] = report
        log.unlink()
    summary = summarize(results)
    summary["snapshot"] = str(args.snapshot.resolve())
    summary["manifest_sha256"] = file_digest(args.snapshot / "manifest.json")
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(
        "ALLOCATOR current upstream: URMA CPU allocation / UBmem HAL allocation; each fused A/B shares one arena.",
        flush=True,
    )
    print(f"COMPARE_PASS RESULTS={args.result_dir / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    run(arguments(argv))


if __name__ == "__main__":
    main()
