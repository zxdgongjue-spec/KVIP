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
        "--study",
        choices=(
            "threshold",
            "pipeline",
            "skip",
            "upstream",
            "threshold-only",
            "seed-ub",
            "local-victims",
            "register-kth",
        ),
        default="threshold",
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
    if args.study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth"):
        if not args.upstream_root or not args.upstream_revision:
            p.error("This study requires upstream-root and upstream-revision")
        from kvip.diagnostics.upstream import validate_checkout

        try:
            args.upstream_root = validate_checkout(args.upstream_root, args.upstream_revision)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            p.error(str(error))
    elif args.upstream_root or args.upstream_revision:
        p.error("upstream-root/revision require a study with an exact upstream reference")
    if args.study in ("threshold-only", "seed-ub", "local-victims", "register-kth") and args.transports != "urma":
        p.error(f"study={args.study} requires --transports urma")
    if args.study in ("seed-ub", "local-victims", "register-kth") and args.diagnostic_replays:
        p.error(f"study={args.study} excludes instrumented diagnostic replays")
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
    if study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth"):
        for suffix, label in (
            ("upstream-serial", "main-serial"),
            ("upstream-fused", "main-resident"),
            ("recent-current", "current-kth"),
            ("recent-release", "release-kth"),
            ("recent-seed", "seed-ub"),
            ("recent-local", "local-victims"),
            ("recent-register", "register-kth"),
            ("recent-before", "recent-recompute"),
            ("recent-after", "recent-kth-only" if study == "threshold-only" else "recent-score-replay"),
        ):
            if name.endswith(suffix):
                return name[: -len(suffix)] + label
    return name


def reduction(before, after):
    return 100 * (1 - after / before) if before else None


def timing_spread(timing):
    """Report a transparent tail heuristic, never delete or reweight samples."""
    median, mean = timing["p50_us"], timing["mean_us"]
    samples = timing.get("samples_us")
    maximum = timing.get("max_us", max(samples) if samples else None)
    minimum = timing.get("min_us", min(samples) if samples else None)
    mean_ratio = mean / median
    maximum_ratio = maximum / median if maximum is not None else None
    reasons = []
    if mean_ratio >= 1.10:
        reasons.append("mean/p50>=1.10")
    if maximum_ratio is not None and maximum_ratio >= 1.50:
        reasons.append("max/p50>=1.50")
    return dict(
        flagged=bool(reasons),
        reasons=reasons,
        count=timing.get("count", len(samples) if samples is not None else None),
        min_us=minimum,
        max_us=maximum,
        mean_to_p50=mean_ratio,
        max_to_p50=maximum_ratio,
        high_tail_samples=sum(value >= 1.5 * median for value in samples) if samples is not None else None,
        interpretation="Heuristic timing-spread warning, not a significance test; every original sample is retained",
    )


def timing_comparisons(table, transport, study):
    def key(suffix):
        return f"{transport}-{suffix}"

    if study == "register-kth":
        pairs = [("register_kth", "recent-local", "recent-register", "step")]
        pairs += [(name, "upstream-fused", f"recent-{name}", "main") for name in ("local", "register")]
    elif study == "local-victims":
        pairs = [("local_victims", "recent-release", "recent-local", "step")]
        pairs += [(name, "upstream-fused", f"recent-{name}", "main") for name in ("release", "local")]
    elif study == "seed-ub":
        pairs = [
            ("marker_removal", "recent-current", "recent-release", "step"),
            ("seed_reuse", "recent-release", "recent-seed", "step"),
        ]
        pairs += [(name, "upstream-fused", f"recent-{name}", "main") for name in ("current", "release", "seed")]
    elif study in ("upstream", "threshold-only"):
        pairs = [
            ("after_vs_before", "recent-before", "recent-after", "step"),
            ("after", "upstream-fused", "recent-after", "main"),
            ("after_vs_serial", "upstream-serial", "recent-after", "serial_reference"),
        ]
    elif study == "skip":
        pairs = [
            ("skip", "fused-before", "fused-after", "step"),
            ("skip_vs_serial", "serial", "fused-after", "serial_reference"),
        ]
    elif study == "pipeline":
        pairs = [
            ("pipeline", "fused-before", "fused-pipeline", "step"),
            ("unified_topk_increment", "fused-pipeline", "fused-unified", "step"),
            ("combined", "fused-before", "fused-unified", "step"),
            ("combined_vs_serial", "serial", "fused-unified", "serial_reference"),
        ]
    else:
        pairs = [
            ("recent_vs_resident", "fused-resident", "fused-recent", "step"),
            ("recent_vs_serial", "serial", "fused-recent", "serial_reference"),
        ]
    result = {}
    for name, source, target, role in pairs:
        before, after = table[key(source)], table[key(target)]
        result[name] = dict(
            reference=key(source),
            candidate=key(target),
            role=role,
            p50_reduction_pct=reduction(before["p50_us"], after["p50_us"]),
            mean_reduction_pct=reduction(before["mean_us"], after["mean_us"]),
            timing_spread_flagged=before["timing_spread"]["flagged"] or after["timing_spread"]["flagged"],
        )
    return result


def print_timing_gains(transport, effect):
    for role in ("main", "step", "serial_reference"):
        comparisons = [(name, value) for name, value in effect["timing_comparisons"].items() if value["role"] == role]
        if comparisons:
            values = " ".join(
                f"{name}={value['p50_reduction_pct']:+.2f}%/{value['mean_reduction_pct']:+.2f}%"
                for name, value in comparisons
            )
            print(f"{role.upper()}_GAIN {transport} p50/mean {values}", flush=True)


def traffic_change(before, after):
    return dict(
        unused_prefetch_reduction_pct=reduction(before["unused_prefetch"], after["unused_prefetch"]),
        total_read_bytes_reduction_pct=reduction(before["bytes"], after["bytes"]),
    )


def summarize_execution_study(table, transport, study):
    if transport != "urma":
        raise ValueError(f"The {study} study supports only URMA")
    main_key = f"{transport}-upstream-fused"
    names, changed_options = {
        "seed-ub": (("current", "release", "seed"), ("recent_profile", "recent_seed_ub")),
        "local-victims": (("release", "local"), ("recent_local_victims",)),
        "register-kth": (("local", "register"), ("recent_register_kth",)),
    }[study]
    expected_keys = {main_key, *(f"{transport}-recent-{name}" for name in names)}
    if set(table) != expected_keys:
        raise ValueError(f"{study} requires only main fusion and its feature variants, no serial reference")
    main = table[main_key]
    provenance = main.get("production_source") or {}
    if provenance.get("kind") != "exact_upstream_checkout" or not provenance.get("commit"):
        raise ValueError("Main fusion must execute an exact verified upstream checkout")
    from kvip.diagnostics.replay_measure import study_cases
    from types import SimpleNamespace

    expected = study_cases(SimpleNamespace(study=study))
    variants = {name: table[f"{transport}-recent-{name}"] for name in names}
    for row in (main, *variants.values()):
        if not row["score_comparison"] or row["score_comparison"].get("status") != "bit_exact":
            raise ValueError(f"{study} requires all captured valid LI scores to be bit exact")
        if any(
            row["selection"][key] != main["selection"][key] for key in ("selected_set_sha256", "ordered_output_sha256")
        ):
            raise ValueError(f"{study} changed final ordered TopK")
        if row["threshold"]["final_keys"] != main["threshold"]["final_keys"]:
            raise ValueError(f"{study} changed final threshold")
        if row.get("native_diagnostics"):
            raise ValueError(f"{study} headline comparison excludes instrumented diagnostic replays")
    current = variants[names[0]]
    prediction = current.get("seed_ub_eligibility")
    if study == "seed-ub":
        if not prediction or not prediction.get("predicted_eligible_queries"):
            raise ValueError("Seed-UB requires nonzero predicted eligible queries")
        if any(row.get("seed_ub_eligibility") != prediction for row in variants.values()):
            raise ValueError("Seed-UB eligibility predictions differ between variants")
    common_tuning = {key: value for key, value in (current["tuning"] or {}).items() if key not in changed_options}
    for name, row in variants.items():
        if row["tuning"] is None or any(row["tuning"].get(key) is not value for key, value in expected[name].items()):
            raise ValueError(f"Wrong {study} native options: {name}")
        if {key: value for key, value in row["tuning"].items() if key not in changed_options} != common_tuning:
            raise ValueError(f"{study} changed an unrelated tuning option or budget")
        if row["threshold"]["initial_keys"] != current["threshold"]["initial_keys"]:
            raise ValueError(f"{study} changed initial threshold")
    result = dict(
        main_reference=main_key,
        upstream_revision=provenance["commit"],
        **{
            f"{name}_vs_main_reduction_pct": reduction(main["mean_us"], row["mean_us"])
            for name, row in variants.items()
        },
    )
    if study == "seed-ub":
        result.update(
            seed_ub_eligibility=prediction,
            marker_removal_reduction_pct=reduction(current["mean_us"], variants["release"]["mean_us"]),
            seed_reuse_reduction_pct=reduction(variants["release"]["mean_us"], variants["seed"]["mean_us"]),
            traffic_main_to_seed=traffic_change(main["reads"], variants["seed"]["reads"]),
        )
    elif study == "register-kth":
        result.update(
            register_kth_reduction_pct=reduction(current["mean_us"], variants["register"]["mean_us"]),
            traffic_main_to_register=traffic_change(main["reads"], variants["register"]["reads"]),
        )
    else:
        result.update(
            local_victims_reduction_pct=reduction(current["mean_us"], variants["local"]["mean_us"]),
            traffic_main_to_local=traffic_change(main["reads"], variants["local"]["reads"]),
        )
    return result


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
            graph_payload_GBps=reads["bytes"] / mean / 1000,  # Compatibility alias.
            graph_total_GBps=reads["bytes"] / mean / 1000,
            graph_useful_GBps=reads["useful_transfer_bytes"] / mean / 1000,
            payload_efficiency_pct=100 * reads["useful_transfer_bytes"] / reads["bytes"] if reads["bytes"] else None,
            hardware_bandwidth_utilization_pct=None,
            threshold=result.get("threshold_comparison"),
            score_comparison=result.get("score_comparison"),
            selection=result["selection_comparison"],
            workers=result["runtime_workers"],
            post_size=result.get("batch_get_post_size"),
            tuning=result.get("tuning"),
            timing=result["timing"],
            timing_spread=timing_spread(result["timing"]),
            production_source=result.get("production_source"),
            native_diagnostics=result.get("native_diagnostics"),
            seed_ub_eligibility=result.get("seed_ub_eligibility"),
        )
    effects = {}
    for transport in (value for value in ("urma", "ubmem") if any(key.startswith(value + "-") for key in table)):
        best = transport + "-serial"
        if study in ("seed-ub", "local-victims", "register-kth"):
            effects[transport] = summarize_execution_study(table, transport, study)
            continue
        if study in ("skip", "upstream", "threshold-only"):
            prefix = (
                "recent"
                if study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth")
                else "fused"
            )
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
            if study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth"):
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
            if study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth"):
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
    for transport, effect in effects.items():
        effect["timing_comparisons"] = timing_comparisons(table, transport, study)
    execution_order = list(results)
    if study == "register-kth":
        table = {key: table[key] for key in ("urma-upstream-fused", "urma-recent-local", "urma-recent-register")}
    return dict(
        status="pass",
        study=study,
        diagnostics_commit=source_commit(),
        geometry=first["geometry"],
        snapshot_sha256=first["snapshot_sha256"],
        table=table,
        execution_order=execution_order,
        display_order=list(table),
        effects=effects,
        scope=first["scope"],
        timing_boundary=first["timing_boundary"],
        timing_reporting="N/min/p50/mean/max include every sample; gains report both p50 and mean. Tail flags are heuristic (mean/p50>=1.10 or max/p50>=1.50), not significance tests or exclusions.",
        bandwidth_accounting=dict(
            graph_equivalent="Mean transferred payload bytes / mean complete-graph microseconds / 1000; decimal GB/s, not isolated Fetch bandwidth",
            useful_payload="Transferred payload minus unused speculative prefetch; not unique cold demand",
            payload_efficiency="Useful transferred bytes / total transferred bytes, not hardware bandwidth utilization",
            hardware_bandwidth_utilization="N/A: no measured hardware reference or isolated transfer interval",
        ),
    )


def print_summary(summary):
    print(
        "case                          N     min_us     p50_us    mean_us     max_us  cold_miss%  prefetched  unused_prefetch  waste%  read_MiB",
        flush=True,
    )
    for name, row in summary["table"].items():
        r = row["reads"]
        waste = "n/a" if r["prefetch_waste_pct"] is None else f"{r['prefetch_waste_pct']:.2f}"
        spread = row["timing_spread"]
        n = str(spread["count"]) if spread["count"] is not None else "n/a"
        low = f"{spread['min_us']:.3f}" if spread["min_us"] is not None else "n/a"
        high = f"{spread['max_us']:.3f}" if spread["max_us"] is not None else "n/a"
        print(
            f"{row['display_name']:28} {n:>3} {low:>10} {row['p50_us']:10.3f} {row['mean_us']:10.3f} {high:>10} "
            f"{100 * r['cold_miss_rate']:10.4f} {r['prefetch']:11.1f} {r['unused_prefetch']:16.1f} "
            f"{waste:>7} {r['bytes'] / (1 << 20):9.3f}",
            flush=True,
        )
        if spread["flagged"]:
            ratio = f"{spread['max_to_p50']:.3f}" if spread["max_to_p50"] is not None else "n/a"
            print(
                f"  TIMING_NOTE tail_spread mean/p50={spread['mean_to_p50']:.3f} max/p50={ratio}; "
                "all samples kept, not a significance test",
                flush=True,
            )
        efficiency = row["payload_efficiency_pct"]
        efficiency_text = "n/a" if efficiency is None else f"{efficiency:.2f}%"
        print(
            f"  GRAPH_BW total={row['graph_total_GBps']:.3f}GB/s useful={row['graph_useful_GBps']:.3f}GB/s "
            f"payload_efficiency={efficiency_text} hardware_utilization=N/A",
            flush=True,
        )
    if summary.get("study") == "register-kth":
        print(
            "EXECUTION_ORDER " + " -> ".join(display_name(key, summary["study"]) for key in summary["execution_order"]),
            flush=True,
        )
    print(
        "BANDWIDTH_SCOPE complete-graph equivalent payload rates; not isolated Fetch bandwidth or link utilization.",
        flush=True,
    )
    for transport, effect in summary["effects"].items():
        if summary.get("study") in ("seed-ub", "local-victims", "register-kth"):
            candidate = {"seed-ub": "seed", "local-victims": "local", "register-kth": "register"}[summary["study"]]
            if prediction := effect.get("seed_ub_eligibility"):
                print(
                    f"SEED_UB_ELIGIBILITY predicted={prediction['predicted_eligible_queries']}/{prediction['total_queries']} "
                    f"fallback={prediction['predicted_fallback_queries']} not_runtime_observation",
                    flush=True,
                )
            main = summary["table"][f"{transport}-upstream-fused"]["threshold"]
            recent = summary["table"][f"{transport}-recent-{candidate}"]["threshold"]
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
            print_timing_gains(transport, effect)
            traffic = effect[f"traffic_main_to_{candidate}"]
            print(
                f"TRAFFIC {transport} main_to_{candidate} "
                + " ".join(
                    f"{key}={value:.2f}%" if value is not None else f"{key}=n/a" for key, value in traffic.items()
                ),
                flush=True,
            )
            print(
                "BASELINE exact main fusion; feature variants share one arena, no serial rerun or instrumented replay.",
                flush=True,
            )
            continue
        if summary.get("study") in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth"):
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
            print_timing_gains(transport, effect)
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
            print_timing_gains(transport, effect)
            continue
        if summary.get("study") == "pipeline":
            print(
                f"CHECK {transport} initial_threshold=unchanged_bit_exact final_scores=bit_exact "
                "final_topk=unchanged_ordered KV_Index_outputs=verified",
                flush=True,
            )
            print_timing_gains(transport, effect)
            continue
        a, b = (summary["table"][f"{transport}-fused-{p}"]["threshold"] for p in ("resident", "recent"))
        print(
            f"THRESHOLD {transport} gap_mean={a['mean_gap']:.7g}->{b['mean_gap']:.7g} "
            f"equal_final={100 * a['equal_fraction']:.2f}%->{100 * b['equal_fraction']:.2f}% "
            "native_threshold=bit_exact final_scores=bit_exact final_selection=unchanged_ordered",
            flush=True,
        )
        print_timing_gains(transport, effect)
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
            (("fused", True), ("fused", False))
            if args.study in ("seed-ub", "local-victims", "register-kth")
            else (("serial", True), ("fused", True), ("fused", False))
            if args.study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth")
            else (("serial", False), ("fused", False))
        )
    ]
    for transport, mode, reference in cases:
        prefix = (
            "upstream-"
            if reference
            else "recent-"
            if args.study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth")
            else ""
        )
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
                    "seed-ub": " current-kth/release-kth/seed-ub share DDR",
                    "local-victims": " release-kth/local-victims share DDR",
                    "register-kth": " register-kth then local-victims share DDR",
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
            if args.study == "register-kth" and (
                report.get("execution_order") != ["register", "local"] or list(report["cases"]) != ["register", "local"]
            ):
                raise ValueError("register-kth must execute candidate before local-victims in the shared arena")
            case_prefix = (
                f"{transport}-recent"
                if args.study in ("upstream", "threshold-only", "seed-ub", "local-victims", "register-kth")
                else name
            )
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
