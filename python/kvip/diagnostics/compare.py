# SPDX-License-Identifier: Apache-2.0
"""Compare current serial and fused chains on one frozen input.

Each backend/mode has its own process. Fused variants share one DDR import and
production launcher. Studies change the seed range or its pipeline/TopK/score-reuse work.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kvip.diagnostics.archive import file_digest, source_commit
from kvip.diagnostics.console import failure_tail, print_replay_failure


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
    p.add_argument(
        "--study", choices=("threshold", "pipeline", "skip", "upstream", "threshold-only"), default="threshold"
    )
    p.add_argument("--transports", choices=("urma", "ubmem", "both"), default="urma")
    p.add_argument("--upstream-root", type=Path)
    p.add_argument("--upstream-revision", help="Full pinned upstream commit for the upstream study")
    p.add_argument(
        "--diagnostic-replays",
        type=int,
        default=0,
        help="Separate instrumented fused replays; excluded from headline timing",
    )
    p.add_argument("--result-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if (
        args.device < 0
        or args.warmup < 0
        or args.iterations < 1
        or args.diagnostic_replays < 0
        or not 1 <= args.serial_workers <= 48
    ):
        p.error("Invalid device, warmup, iterations or workers")
    if args.result_dir.exists():
        p.error("result-dir already exists; choose a new directory")
    if args.study in ("upstream", "threshold-only"):
        if not args.upstream_root or not args.upstream_revision:
            p.error("This study requires upstream-root and upstream-revision")
        from kvip.diagnostics.upstream import validate_checkout

        try:
            args.upstream_root = validate_checkout(args.upstream_root, args.upstream_revision)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            p.error(str(error))
    elif args.upstream_root or args.upstream_revision:
        p.error("upstream-root/revision apply only to study=upstream or threshold-only")
    if args.study == "threshold-only" and args.transports != "urma":
        p.error("study=threshold-only requires --transports urma")
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
    means = {name: sum(x[name] for x in samples) / len(samples) for name in names}
    means["prefetch_waste_pct"] = 100 * means["unused_prefetch"] / means["prefetch"] if means["prefetch"] else None
    return means


def display_name(name, study):
    """Keep saved result keys stable while spelling out the actual algorithms."""
    if study in ("upstream", "threshold-only"):
        for suffix, label in (
            ("upstream-serial", "main-serial"),
            ("upstream-fused", "main-resident"),
            ("recent-before", "recent-recompute"),
            ("recent-after", "recent-kth-only" if study == "threshold-only" else "recent-score-replay"),
        ):
            if name.endswith(suffix):
                return name[: -len(suffix)] + label
    return name


def reduction(before, after):
    return 100 * (1 - after / before) if before else None


def traffic_change(before, after):
    return dict(
        unused_prefetch_reduction_pct=reduction(before["unused_prefetch"], after["unused_prefetch"]),
        total_read_bytes_reduction_pct=reduction(before["bytes"], after["bytes"]),
    )


def summarize(results, study="threshold"):
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
            display_name=display_name(name, study),
            mean_us=mean,
            p50_us=result["timing"]["p50_us"],
            reads=reads,
            graph_payload_GBps=reads["bytes"] / mean / 1000,
            threshold=result.get("threshold_comparison"),
            score_comparison=result.get("score_comparison"),
            selection=result["selection_comparison"],
            workers=result["runtime_workers"],
            post_size=result.get("batch_get_post_size"),
            tuning=result.get("tuning"),
            timing=result["timing"],
            production_source=result.get("production_source"),
            native_diagnostics=result.get("native_diagnostics"),
        )
    effects = {}
    for transport in (value for value in ("urma", "ubmem") if any(key.startswith(value + "-") for key in table)):
        best = transport + "-serial"
        if study in ("skip", "upstream", "threshold-only"):
            prefix = "recent" if study in ("upstream", "threshold-only") else "fused"
            before, after = (table[f"{transport}-{prefix}-{name}"] for name in ("before", "after"))
            changed_option = "recent_threshold_only" if study == "threshold-only" else "recent_skip"
            label = "Threshold-only A/B" if study == "threshold-only" else "Skip A/B"
            for row, enabled in ((before, False), (after, True)):
                if not row["score_comparison"] or row["score_comparison"].get("status") != "bit_exact":
                    raise ValueError(f"{label} requires bit-exact validation against all captured valid LI scores")
                expected = dict(
                    recent_threshold=True, recent_pipeline=True, recent_unified_topk=True, recent_skip=False
                )
                expected[changed_option] = enabled
                if row["tuning"] is None or any(row["tuning"].get(key) is not value for key, value in expected.items()):
                    raise ValueError(
                        f"{label} requires identical recent/pipeline/TopK settings and only {changed_option} changed"
                    )
            if {k: v for k, v in before["tuning"].items() if k != changed_option} != {
                k: v for k, v in after["tuning"].items() if k != changed_option
            }:
                raise ValueError(f"{label} changed another tuning option or prefetch budget")
            if any(
                before["selection"][key] != after["selection"][key]
                for key in ("selected_set_sha256", "ordered_output_sha256")
            ):
                raise ValueError("Skip A/B changed the final selection")
            if any(before["threshold"][key] != after["threshold"][key] for key in ("initial_keys", "final_keys")):
                raise ValueError("Skip A/B changed the threshold")
            if study in ("upstream", "threshold-only"):
                best = transport + "-upstream-serial"
                main = table[transport + "-upstream-fused"]
                for reference in (table[best], main):
                    provenance = reference.get("production_source") or {}
                    if provenance.get("kind") != "exact_upstream_checkout" or not provenance.get("commit"):
                        raise ValueError("Upstream reference must execute an exact verified upstream checkout")
                if table[best]["production_source"]["commit"] != main["production_source"]["commit"]:
                    raise ValueError("Upstream serial and fused source revisions differ")
                if not main["score_comparison"] or main["score_comparison"].get("status") != "bit_exact":
                    raise ValueError("Upstream fused requires full captured-score bit-exact verification")
                for key in ("selected_set_sha256", "ordered_output_sha256"):
                    if main["selection"][key] != before["selection"][key]:
                        raise ValueError("Upstream and recent fusion changed the final ordered TopK")
                if main["threshold"]["final_keys"] != before["threshold"]["final_keys"]:
                    raise ValueError("Upstream and recent fusion changed the final threshold")
            effects[transport] = dict(
                serial_reference=best,
                optimization_reduction_pct=100 * (1 - after["mean_us"] / before["mean_us"]),
                optimization_vs_serial_reduction_pct=100 * (1 - after["mean_us"] / table[best]["mean_us"]),
                traffic_before_to_after=traffic_change(before["reads"], after["reads"]),
            )
            if study != "threshold-only":
                effects[transport].update(
                    skip_reduction_pct=effects[transport]["optimization_reduction_pct"],
                    skip_vs_serial_reduction_pct=effects[transport]["optimization_vs_serial_reduction_pct"],
                )
            if study in ("upstream", "threshold-only"):
                effects[transport].update(
                    upstream_revision=main["production_source"]["commit"],
                    traffic_main_to_before=traffic_change(main["reads"], before["reads"]),
                    traffic_main_to_after=traffic_change(main["reads"], after["reads"]),
                    recent_before_vs_upstream_reduction_pct=100 * (1 - before["mean_us"] / main["mean_us"]),
                    recent_after_vs_upstream_reduction_pct=100 * (1 - after["mean_us"] / main["mean_us"]),
                )
            continue
        if study == "pipeline":
            before, pipeline, unified = (
                table[f"{transport}-fused-{name}"] for name in ("before", "pipeline", "unified")
            )
            for variant in (pipeline, unified):
                if any(
                    variant["selection"][key] != before["selection"][key]
                    for key in ("selected_set_sha256", "ordered_output_sha256")
                ):
                    raise ValueError("Pipeline/TopK A/B changed the final selection")
                if any(variant["threshold"][key] != before["threshold"][key] for key in ("initial_keys", "final_keys")):
                    raise ValueError("Pipeline/TopK A/B changed the threshold")
            effects[transport] = dict(
                serial_reference=best,
                pipeline_reduction_pct=100 * (1 - pipeline["mean_us"] / before["mean_us"]),
                unified_topk_incremental_reduction_pct=100 * (1 - unified["mean_us"] / pipeline["mean_us"]),
                combined_reduction_pct=100 * (1 - unified["mean_us"] / before["mean_us"]),
                combined_vs_serial_reduction_pct=100 * (1 - unified["mean_us"] / table[best]["mean_us"]),
            )
            continue
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
        study=study,
        diagnostics_commit=source_commit(),
        geometry=first["geometry"],
        snapshot_sha256=first["snapshot_sha256"],
        table=table,
        effects=effects,
        scope=first["scope"],
        timing_boundary=first["timing_boundary"],
    )


def print_summary(summary):
    print(
        "case                         mean_us    p50_us  cold_miss%   prefetched  unused_prefetch  waste%  total_read_MiB",
        flush=True,
    )
    for name, row in summary["table"].items():
        r = row["reads"]
        waste = "n/a" if r["prefetch_waste_pct"] is None else f"{r['prefetch_waste_pct']:.2f}"
        print(
            f"{row['display_name']:28} {row['mean_us']:9.3f} {row['p50_us']:9.3f} {100 * r['cold_miss_rate']:10.4f} "
            f"{r['prefetch']:12.1f} {r['unused_prefetch']:16.1f} {waste:>7} {r['bytes'] / (1 << 20):15.3f}",
            flush=True,
        )
    for transport, effect in summary["effects"].items():
        if summary.get("study") in ("upstream", "threshold-only"):
            main, recent = (
                summary["table"][f"{transport}-{name}"]["threshold"] for name in ("upstream-fused", "recent-after")
            )
            print(
                f"THRESHOLD {transport} gap_mean={main['mean_gap']:.7g}->{recent['mean_gap']:.7g} "
                f"equal_final={100 * main['equal_fraction']:.2f}%->{100 * recent['equal_fraction']:.2f}%",
                flush=True,
            )
            print(
                f"CHECK {transport} upstream={effect['upstream_revision']} all_fused_scores=bit_exact "
                "all_fused_topk=unchanged_ordered recent_initial_threshold=unchanged_bit_exact KV_Index_outputs=verified",
                flush=True,
            )
            print(
                f"GAIN {transport} after_vs_recent_before={effect['optimization_reduction_pct']:.2f}% "
                f"after_vs_upstream_fused={effect['recent_after_vs_upstream_reduction_pct']:.2f}% "
                f"after_vs_upstream_serial={effect['optimization_vs_serial_reduction_pct']:.2f}%",
                flush=True,
            )
            for pair in ("main_to_before", "main_to_after", "before_to_after"):
                change = effect["traffic_" + pair]
                values = " ".join(
                    f"{key}={value:.2f}%" if value is not None else f"{key}=n/a" for key, value in change.items()
                )
                print(f"TRAFFIC {transport} {pair} {values}", flush=True)
            print(
                "BASELINE upstream production sources; unified replay configuration, not default service TPOT.",
                flush=True,
            )
            continue
        if summary.get("study") == "skip":
            print(
                f"CHECK {transport} initial_threshold=unchanged_bit_exact final_scores=bit_exact "
                "final_topk=unchanged_ordered KV_Index_outputs=verified",
                flush=True,
            )
            print(
                f"GAIN {transport} skip={effect['skip_reduction_pct']:.2f}% "
                f"skip_vs_serial={effect['skip_vs_serial_reduction_pct']:.2f}% "
                f"serial_reference={effect['serial_reference']}",
                flush=True,
            )
            continue
        if summary.get("study") == "pipeline":
            print(
                f"CHECK {transport} initial_threshold=unchanged_bit_exact final_scores=bit_exact "
                "final_topk=unchanged_ordered KV_Index_outputs=verified",
                flush=True,
            )
            print(
                f"GAIN {transport} pipeline={effect['pipeline_reduction_pct']:.2f}% "
                f"unified_topk_increment={effect['unified_topk_incremental_reduction_pct']:.2f}% "
                f"combined={effect['combined_reduction_pct']:.2f}% "
                f"combined_vs_serial={effect['combined_vs_serial_reduction_pct']:.2f}%",
                flush=True,
            )
            continue
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
    for name, row in summary["table"].items():
        if row.get("native_diagnostics"):
            from kvip.diagnostics.native_profile import print_profile

            print_profile(row["display_name"], row["native_diagnostics"])
    print(
        "TIMING complete device chain (recent scoring/merge included); excludes data restoration, CPU scheduler and input H2D.",
        flush=True,
    )


def run(args):
    args.result_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    transports = ("urma", "ubmem") if args.transports == "both" else (args.transports,)
    cases = [
        (transport, mode, reference)
        for transport in transports
        for mode, reference in (
            (("serial", True), ("fused", True), ("fused", False))
            if args.study in ("upstream", "threshold-only")
            else (("serial", False), ("fused", False))
        )
    ]
    for transport, mode, reference in cases:
        prefix = "upstream-" if reference else "recent-" if args.study in ("upstream", "threshold-only") else ""
        name = f"{transport}-{prefix}{mode}"
        paired = mode == "fused" and not reference
        child_study = "threshold" if reference else "skip" if args.study == "upstream" else args.study
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
            str(args.build_dir),
            "--transport",
            transport,
            "--mode",
            mode,
            "--threshold-policy",
            "both" if paired else "resident",
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
            "--study",
            child_study,
        ]
        if mode == "fused" and getattr(args, "diagnostic_replays", 0):
            cmd += ["--diagnostic-replays", str(args.diagnostic_replays)]
        if reference:
            cmd += ["--upstream-root", str(args.upstream_root), "--upstream-revision", args.upstream_revision]
        if mode == "serial":
            cmd += ["--serial-workers", str(args.serial_workers)]
        print(
            f"RUN {name}"
            + (
                f" exact_upstream={args.upstream_revision}"
                if reference and mode == "fused"
                else {
                    "threshold": " resident/recent share DDR",
                    "pipeline": " before/pipeline/unified share DDR",
                    "skip": " before/after share DDR",
                    "upstream": " recompute/score-replay share DDR",
                    "threshold-only": " recompute/kth-only share DDR",
                }[args.study]
                if paired
                else f" workers={args.serial_workers}" + (" post=128 inflight=2048" if transport == "urma" else "")
            ),
            flush=True,
        )
        with log.open("w") as stream:
            process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            try:
                failure = json.loads(output.read_text())
            except (OSError, ValueError):
                failure = None
            if isinstance(failure, dict) and failure.get("status") == "failed":
                failure["comparison_log"] = str(log.resolve())
                failure["child_exit_code"] = process.returncode
                try:
                    output.write_text(json.dumps(failure, indent=2) + "\n")
                except OSError:
                    pass  # Preserve the original error and already-saved child report.
                print_replay_failure(failure, result_path=output, log_path=log)
            else:
                failure_tail(log, message=f"{name} exit={process.returncode}; no structured failure report", lines=12)
            raise SystemExit(1)
        report = json.loads(output.read_text())
        if paired:
            case_prefix = f"{transport}-recent" if args.study in ("upstream", "threshold-only") else name
            results.update({f"{case_prefix}-{key}": value for key, value in report["cases"].items()})
        else:
            results[name] = report
        log.unlink()
    summary = summarize(results, args.study)
    summary["snapshot"] = str(args.snapshot.resolve())
    summary["manifest_sha256"] = file_digest(args.snapshot / "manifest.json")
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    allocators = " / ".join(
        "URMA CPU allocation" if transport == "urma" else "UBmem HAL allocation" for transport in transports
    )
    print(f"ALLOCATOR current runtime: {allocators}; each feature fused A/B shares one arena.", flush=True)
    print(f"COMPARE_PASS RESULTS={args.result_dir / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    run(arguments(argv))


if __name__ == "__main__":
    main()
