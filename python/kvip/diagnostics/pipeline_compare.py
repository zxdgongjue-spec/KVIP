# SPDX-License-Identifier: Apache-2.0
"""Compare exact main, the recent-index PR and opt-in post-threshold variants.

Main serial, main fusion and PR fusion each have a fresh process. All candidate
variants share one replay process/arena in the explicitly requested order.
"""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

from kvip.diagnostics.archive import file_digest, source_commit
from kvip.diagnostics.compare import read_means, reduction, timing_spread
from kvip.diagnostics.pipeline_options import (
    DEFAULT_VARIANTS, OPTIONS, OPTIONAL_OPTIONS, NUMERIC_OPTIONS, VARIANTS,
    admission_for, early_threshold_skipped, is_auto_variant, options_for, q_workers_for,
)
from kvip.diagnostics.upstream import validate_checkout


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("main", "pr", "candidate"):
        parser.add_argument(f"--{role}-root", type=Path, required=True)
        parser.add_argument(f"--{role}-sha", required=True, help="Exact full 40-character production commit")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True, help="Shared current KVAF build-npu")
    parser.add_argument("--kvaf-server", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc", default="Ascend950DT_9582")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--diagnostic-replays", type=int, default=1, help="Separate fused diagnostic replays after headline timing"
    )
    parser.add_argument("--pipeline-variants", nargs="+", choices=tuple(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--topk-probe", action="store_true", help="Independent pruning-counter replay after all headline timings")
    parser.add_argument("--brief", action="store_true", help="Print one timing/traffic line per case; keep full JSON")
    parser.add_argument("--matched-before", action="store_true", help="Also run the PR checkout with matching pre-existing candidate configurations")
    parser.add_argument("--check-mode-switch", action="store_true",
                        help="Verify one light-low graph with quotas 0/512/0; restore snapshot outside timing")
    parser.add_argument("--check-auto-route", action="store_true", help="Verify device feedback and request lifecycle on auto variants")
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.diagnostic_replays < 0:
        parser.error("Require nonnegative device/warmup and positive iterations")
    if len(set(args.pipeline_variants)) != len(args.pipeline_variants):
        parser.error("pipeline-variants must be unique; order is preserved")
    if args.check_mode_switch and "light-low" not in args.pipeline_variants:
        parser.error("check-mode-switch requires the light-low variant")
    auto = any(is_auto_variant(name) for name in args.pipeline_variants)
    if args.check_auto_route and not auto:
        parser.error("check-auto-route requires an automatic route variant")
    if auto and args.topk_probe:
        parser.error("auto excludes topk-probe")
    if args.result_dir.exists():
        parser.error("result-dir already exists; choose a new directory")
    for role in ("main", "pr", "candidate"):
        try:
            root = validate_checkout(getattr(args, role + "_root"), getattr(args, role + "_sha"))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        setattr(args, role + "_root", root)
    args.snapshot, args.build_dir = args.snapshot.resolve(), args.build_dir.resolve()
    if not (args.snapshot / "manifest.json").is_file():
        parser.error(f"Missing snapshot manifest: {args.snapshot}")
    for name in ("kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o"):
        if not (args.build_dir / "bin" / name).is_file():
            parser.error(f"Missing explicitly requested KVAF kernel: {args.build_dir / 'bin' / name}")
    return args


def before_variants(args):
    if not getattr(args, "matched_before", False):
        return []
    return [name for name in ("ready-batch4", "workers4", "metadata") if name in args.pipeline_variants]


def jobs(args):
    plan = [
        dict(name="main-serial", role="main", mode="serial", policy="resident"),
        dict(name="main-resident", role="main", mode="fused", policy="resident"),
        dict(name="pr-recent", role="pr", mode="fused", policy="recent"),
    ]
    if variants := before_variants(args):
        plan.append(dict(name="before", role="pr", mode="fused", policy="recent", variants=variants))
    plan.append(dict(name="candidate", role="candidate", mode="fused", policy="recent"))
    return plan


def command(args, job):
    """Use only supported replay arguments; serial post/inflight are fixed there."""
    result = [
        sys.executable,
        "-B",
        "-m",
        "kvip.diagnostics.replay",
        "--snapshot",
        str(args.snapshot),
        "--build-dir",
        str(args.build_dir),
        "--transport",
        "urma",
        "--mode",
        job["mode"],
        "--threshold-policy",
        job["policy"],
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
        "--upstream-root",
        str(getattr(args, job["role"] + "_root")),
        "--upstream-revision",
        getattr(args, job["role"] + "_sha"),
        "--output",
        str(args.result_dir / (job["name"] + ".json")),
        "--study",
        "post-threshold" if job["role"] == "candidate" or job["name"] == "before" else "threshold",
    ]
    if job["mode"] == "serial":
        result += ["--serial-workers", "32"]
    else:
        result += ["--diagnostic-replays", str(args.diagnostic_replays)]
    if job["role"] == "candidate" or job["name"] == "before":
        result += ["--pipeline-variants", *job.get("variants", args.pipeline_variants)]
        if job["role"] == "candidate" and getattr(args, "topk_probe", False):
            result += ["--topk-probe"]
        if job["role"] == "candidate" and getattr(args, "check_mode_switch", False):
            result += ["--check-mode-switch"]
        if job["role"] == "candidate" and getattr(args, "check_auto_route", False):
            result += ["--check-auto-route"]
    return result


def unpack_report(job, report, variants):
    if report.get("status") != "pass":
        raise ValueError(f"{job['name']} did not produce a passing replay result")
    if job["role"] != "candidate" and job["name"] != "before":
        if "cases" in report:
            raise ValueError(f"{job['name']} must contain exactly one production case")
        return {job["name"]: report}
    if report.get("study") != "post-threshold" or report.get("execution_order") != variants:
        raise ValueError("Candidate study/order differs from the requested variants")
    if list(report.get("cases", {})) != variants:
        raise ValueError("Candidate case keys/order differ from the requested variants")
    prefix = "before-" if job["name"] == "before" else "candidate-"
    return {prefix + name: report["cases"][name] for name in variants}


def _validate_case(args, name, result, manifest_digest):
    role = "main" if name.startswith("main-") else "pr" if name == "pr-recent" or name.startswith("before-") else "candidate"
    mode = "serial" if name == "main-serial" else "fused"
    expected = dict(
        transport="urma",
        mode=mode,
        snapshot_sha256=manifest_digest,
        device=args.device,
        soc=args.soc,
        warmup=args.warmup,
    )
    if result.get("status") != "pass" or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name}: result status, input or invocation metadata differs")
    source = result.get("production_source") or {}
    if (
        source.get("kind") != "exact_upstream_checkout"
        or source.get("commit") != getattr(args, role + "_sha")
        or Path(source.get("root", "")).resolve() != getattr(args, role + "_root").resolve()
    ):
        raise ValueError(f"{name}: production source is not the requested exact checkout")
    if Path(result.get("kvaf_build", "")).resolve() != args.build_dir.resolve():
        raise ValueError(f"{name}: KVAF build differs from the shared requested build")
    timing = result["timing"]
    samples = timing.get("samples_us", [])
    if (
        timing.get("count") != args.iterations
        or len(samples) != args.iterations
        or len(result.get("reads_samples", [])) != args.iterations
        or any(not math.isfinite(value) or value <= 0 for value in samples)
        or not all(math.isfinite(timing[key]) and timing[key] > 0 for key in ("mean_us", "p50_us"))
    ):
        raise ValueError(f"{name}: incomplete or invalid headline timing/read samples")
    ordered = sorted(samples)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    if not math.isclose(timing["mean_us"], sum(samples) / n, rel_tol=1e-9) or not math.isclose(
        timing["p50_us"], median, rel_tol=1e-9
    ):
        raise ValueError(f"{name}: headline timing disagrees with retained samples")
    if mode == "serial":
        serial = dict(runtime_workers=32, batch_get_post_size=128, batch_get_max_inflight_total=2048, jettys_per_core=2)
        if any(result.get(key) != value for key, value in serial.items()):
            raise ValueError("Main serial must use 32 workers, post=128, inflight=2048 and two jettys/core")
    else:
        requested_workers = q_workers_for(name.split("-", 1)[1]) if role == "candidate" or name.startswith("before-") else 2
        if result.get("tuning", {}).get("q_workers") != requested_workers:
            raise ValueError(f"{name}: fused q_workers differs from the requested variant")
        if result.get("score_comparison", {}).get("status") != "bit_exact":
            raise ValueError(f"{name}: fused captured LI scores were not verified bit-exact")
        threshold = result.get("threshold_comparison") or {}
        policy = "resident" if name == "main-resident" else "recent"
        skipped = early_threshold_skipped(result.get("tuning", {}))
        status = "not_computed" if skipped else "bit_exact"
        if threshold.get("status") != status or threshold.get("policy") != policy:
            raise ValueError(f"{name}: initial threshold policy was not verified bit-exact")
        automatic = bool(result.get("tuning", {}).get("auto_route"))
        if skipped and (threshold.get("initial_keys") is not None or
                        (not automatic and any(row["prefetch"] != 0 for row in result["reads_samples"]))):
            raise ValueError(f"{name}: low mode must skip early threshold and issue zero prefetch reads")
        if automatic:
            route = result.get("auto_route", {})
            samples = route.get("samples", [])
            if (route.get("status") != "pass" or len(samples) != args.iterations
                    or route.get("capture_and_warmup_learning") is not False
                    or route.get("decision_and_feedback_in_timing") is not True
                    or any(row["mode"] == "low" and row["observed_prefetch"] != 0 for row in samples)):
                raise ValueError(f"{name}: incomplete automatic route validation")
            if getattr(args, "check_auto_route", False) and route.get("lifecycle", {}).get("status") != "pass":
                raise ValueError(f"{name}: missing auto lifecycle validation")
        if name == "candidate-light-low" and getattr(args, "check_mode_switch", False):
            switch = result.get("mode_switch_check", {})
            rows = switch.get("sequence", [])
            if (switch.get("status") != "pass" or switch.get("graph_captures") != 1
                    or switch.get("captured_addresses_unchanged") is not True
                    or switch.get("snapshot_restored_before_each") is not True
                    or switch.get("included_in_timing") is not False
                    or [row.get("quota") for row in rows] != [0, 512, 0]
                    or [row.get("threshold") for row in rows] != ["not_computed", "bit_exact", "not_computed"]
                    or any(row.get("prefetch") != 0 for row in rows if row.get("quota") == 0)
                    or any(row.get("threshold_sentinel_unchanged") is not True for row in rows if row.get("quota") == 0)
                    or any(row.get("ordered_output_sha256") != result["selection_comparison"]["ordered_output_sha256"] for row in rows)):
                raise ValueError(f"{name}: incomplete same-graph mode-switch validation")
        if args.diagnostic_replays:
            profile = result.get("native_diagnostics") or {}
            if (
                profile.get("included_in_timing") is not False
                or profile.get("units") != "raw_device_cycles"
                or profile.get("count") != args.diagnostic_replays
                or len(profile.get("samples", [])) != args.diagnostic_replays
            ):
                raise ValueError(f"{name}: missing separate uninstrumented-headline diagnostic contract")


def profile_summary(result):
    """Pool only raw samples of the same role; never sum overlapping stages."""
    profile = result.get("native_diagnostics")
    if not profile:
        return None
    fields = {
        "init_total": "init_total",
        "init_route": "init_route",
        "init_route_publish": "init_route_publish",
        "init_route_probe": "init_route_probe",
        "init_fill": "init_fill",
        "init_tail_setup": "init_tail_setup",
        "init_finish": "init_finish",
        "ready_wait": "fetch_query_ready_wait",
        "get_post": "post_active",
        "get_drain": "cq_drain",
        "get_native": "fetch_native_get",
        "get_pack": "fetch_pack",
        "publication_pack": "publication_pack",
        "producer": "producer_span",
        "index": "index_consumer_span",
        "index_total": "index_role_span",
        "index_metadata": "index_metadata",
        "dependency_wait": "fetch_dependency_wait",
        "prefetch_join": "final_prefetch_join",
        "early_prefetch_drain": "early_prefetch_drain",
    }
    values = {}
    for label, key in fields.items():
        samples, statuses = [], []
        for replay in profile["samples"]:
            metric = replay.get("metrics", {}).get(key, {})
            statuses.append(metric.get("status", "unavailable"))
            if metric.get("status") == "available":
                samples.extend(metric["samples_cycles"])
        status = "available" if samples else (
            "not_entered" if statuses and all(value == "not_entered" for value in statuses) else "unavailable"
        )
        values[label] = dict(status=status, count=len(samples), p50_cycles=float(median(samples)) if samples else None)
    return dict(count=profile["count"], units="raw_device_cycles", included_in_timing=False, metrics=values,
                compact_plan_counts=[item["compact_plan_counts"] for item in profile["samples"]
                                     if item.get("compact_plan_counts") is not None],
                direct_index_counts=[item["direct_index_counts"] for item in profile["samples"]
                                     if item.get("direct_index_counts") is not None],
                ready_batch_counts=[item["ready_batch_counts"] for item in profile["samples"] if item.get("ready_batch_counts")],
                ready_batch_by_query=[dict(replay=i + 1, queries=item["ready_batch_by_query"])
                                      for i, item in enumerate(profile["samples"]) if item.get("ready_batch_by_query")])


def summarize(args, results, manifest_digest):
    expected_order = ["main-serial", "main-resident", "pr-recent",
                      *("before-" + v for v in before_variants(args)),
                      *("candidate-" + v for v in args.pipeline_variants)]
    if list(results) != expected_order:
        raise ValueError("Comparison is missing cases or has an unexpected execution order")
    first = results["main-serial"]
    for name, result in results.items():
        _validate_case(args, name, result, manifest_digest)
        if any(result.get(key) != first.get(key) for key in ("geometry", "scope", "timing_boundary")):
            raise ValueError(f"{name}: geometry or complete-graph timing scope differs")
    pr = results["pr-recent"]
    for name, result in results.items():
        if name == "main-serial":
            continue  # Serial's validated BF16 boundary selection may differ.
        if any(
            result["selection_comparison"][key] != pr["selection_comparison"][key]
            for key in ("selected_set_sha256", "ordered_output_sha256")
        ):
            raise ValueError(f"{name}: fused final ordered TopK differs from PR")
        if result["threshold_comparison"]["final_keys"] != pr["threshold_comparison"]["final_keys"]:
            raise ValueError(f"{name}: fused final threshold differs from PR")
        if name.startswith(("candidate-", "before-")):
            if (not early_threshold_skipped(result.get("tuning", {}))
                    and result["threshold_comparison"]["initial_keys"] != pr["threshold_comparison"]["initial_keys"]):
                raise ValueError(f"{name}: post-threshold optimization changed the early threshold")
            variant = name.split("-", 1)[1]
            expected_options = options_for(variant)
            tuning = result.get("tuning") or {}
            if any(tuning.get(key) != value for key, value in expected_options.items()):
                raise ValueError(f"{name}: candidate build options differ")
            admission = admission_for(variant)
            if tuning.get("admission_controls") != (list(admission) if admission else None):
                raise ValueError(f"{name}: candidate admission controls differ")
            rank = expected_options.get("prefetch_rank", 0)
            gate = result["threshold_comparison"].get("prefetch_gate", {})
            if rank and (gate.get("status") != "bit_exact" or gate.get("rank") != rank):
                raise ValueError(f"{name}: speculative rank gate was not verified")
            excluded = (*OPTIONS, *OPTIONAL_OPTIONS, *NUMERIC_OPTIONS, "q_workers", "admission_controls", "auto_low_active")
            if {key: value for key, value in tuning.items() if key not in excluded} != {
                key: value for key, value in pr.get("tuning", {}).items() if key != "q_workers"
            }:
                raise ValueError(f"{name}: candidate changed another tuning option or prefetch budget")
    table = {}
    control_name = "candidate-control" if "candidate-control" in results else "candidate-ready-batch4"
    control = results.get(control_name)
    for name, result in results.items():
        reads = read_means(result)
        timing = result["timing"]
        table[name] = dict(
            mean_us=timing["mean_us"],
            p50_us=timing["p50_us"],
            timing=timing,
            timing_spread=timing_spread(timing),
            reads=reads,
            whole_graph_GBps=reads["bytes"] / timing["mean_us"] / 1000,
            gains={
                label: {
                    "mean_reduction_pct": reduction(reference["timing"]["mean_us"], timing["mean_us"]),
                    "p50_reduction_pct": reduction(reference["timing"]["p50_us"], timing["p50_us"]),
                }
                for label, reference in (
                    ("vs_pr", pr),
                    ("vs_serial", first),
                    ("vs_main_fused", results["main-resident"]),
                )
            },
            production_source=result["production_source"],
            tuning=result.get("tuning"),
            selection_comparison=result["selection_comparison"],
            threshold_comparison=result.get("threshold_comparison"),
            mode_switch_check=result.get("mode_switch_check"),
            auto_route=result.get("auto_route"),
            diagnostic_summary=profile_summary(result),
            capacity_skipped_row_windows=(sum(
                item["capacity_skipped_row_windows"] for item in result["reads_samples"]
            ) / len(result["reads_samples"]) if all(
                "capacity_skipped_row_windows" in item for item in result["reads_samples"]
            ) else None),
        )
        table[name]["gains"]["vs_control"] = (
            {
                "mean_reduction_pct": reduction(control["timing"]["mean_us"], timing["mean_us"]),
                "p50_reduction_pct": reduction(control["timing"]["p50_us"], timing["p50_us"]),
            }
            if control is not None and name.startswith("candidate-")
            else None
        )
        matched = "before-" + name.removeprefix("candidate-")
        if matched not in results:
            matched = "before-ready-batch4"
        reference = results.get(matched) if name.startswith("candidate-") else None
        table[name]["gains"]["vs_matched_before"] = (
            dict(reference=matched,
                 mean_reduction_pct=reduction(reference["timing"]["mean_us"], timing["mean_us"]),
                 p50_reduction_pct=reduction(reference["timing"]["p50_us"], timing["p50_us"]))
            if reference else None
        )
    control_end = results.get("candidate-control-end" if control_name == "candidate-control" else "candidate-ready-batch4-end")
    drift = None
    if control is not None and control_end is not None:
        drift = {
            metric: dict(
                start_us=control["timing"][metric + "_us"],
                end_us=control_end["timing"][metric + "_us"],
                delta_us=control_end["timing"][metric + "_us"] - control["timing"][metric + "_us"],
                drift_pct=100 * (control_end["timing"][metric + "_us"] / control["timing"][metric + "_us"] - 1),
            )
            for metric in ("mean", "p50")
        }
    end_pair = None
    metadata_end = results.get("candidate-metadata-end")
    if metadata_end is not None and control_end is not None:
        end_pair = {
            metric: dict(
                control_us=control_end["timing"][metric + "_us"],
                metadata_us=metadata_end["timing"][metric + "_us"],
                reduction_pct=reduction(control_end["timing"][metric + "_us"], metadata_end["timing"][metric + "_us"]),
            )
            for metric in ("mean", "p50")
        }
    mapping_pairs = {}
    for experiment, baseline in (("inverse-map", "ready-batch4"),
                                 ("inverse-map-skip", "saturated-skip"),
                                 ("capacity-inverse-map", "capacity")):
        candidate_row = table.get("candidate-" + experiment)
        baseline_row = table.get("candidate-" + baseline)
        if candidate_row is not None and baseline_row is not None:
            mapping_pairs[experiment] = dict(
                reference=baseline,
                **{metric + "_reduction_pct": reduction(
                    baseline_row[metric + "_us"], candidate_row[metric + "_us"]
                ) for metric in ("mean", "p50")},
            )
    rank_pairs = {}
    for experiment, baseline in (("rank2048", "ready-batch4"),
                                 ("rank1792", "rank2048"),
                                 ("rank1536", "rank2048"),
                                 ("capacity-rank2048", "capacity"),
                                 ("capacity-rank1792", "capacity-rank2048")):
        candidate_row = table.get("candidate-" + experiment)
        baseline_row = table.get("candidate-" + baseline)
        if candidate_row is not None and baseline_row is not None:
            rank_pairs[experiment] = dict(
                reference=baseline,
                **{metric + "_reduction_pct": reduction(
                    baseline_row[metric + "_us"], candidate_row[metric + "_us"]
                ) for metric in ("mean", "p50")},
            )
    light_pairs = {}
    for experiment, baseline in (("light-low", "quota-zero"), ("light-high", "ready-batch4")):
        candidate_row, baseline_row = table.get("candidate-" + experiment), table.get("candidate-" + baseline)
        if candidate_row is not None and baseline_row is not None:
            light_pairs[experiment] = dict(reference=baseline, **{
                metric + "_reduction_pct": reduction(baseline_row[metric + "_us"], candidate_row[metric + "_us"])
                for metric in ("mean", "p50")})
    light_drift = None
    if "candidate-light-high" in table and "candidate-light-high-end" in table:
        light_drift = {metric: dict(
            start_us=table["candidate-light-high"][metric + "_us"],
            end_us=table["candidate-light-high-end"][metric + "_us"],
            drift_pct=100 * (table["candidate-light-high-end"][metric + "_us"] /
                             table["candidate-light-high"][metric + "_us"] - 1))
            for metric in ("mean", "p50")}
    return dict(
        schema_version=1,
        status="pass",
        study="post-threshold-production-comparison",
        diagnostics_commit=source_commit(),
        snapshot=str(args.snapshot),
        snapshot_sha256=manifest_digest,
        geometry=first["geometry"],
        scope=first["scope"],
        timing_boundary=first["timing_boundary"],
        execution_order=list(results),
        candidate_shared_arena_order=args.pipeline_variants,
        primary_candidate_comparison="vs_control" if control is not None else None,
        control_name=control_name if control is not None else None,
        control_end_drift=drift,
        inverse_mapping_pairs=mapping_pairs,
        prefetch_rank_pairs=rank_pairs,
        light_mode_pairs=light_pairs,
        light_high_end_drift=light_drift,
        metadata_end_vs_control_end=end_pair,
        diagnostic_replays_per_fused_case=args.diagnostic_replays,
        serial_configuration=dict(workers=32, post_size=128, max_inflight=2048, jettys_per_core=2),
        serial_fused_selected_sets_identical=(
            first["selection_comparison"]["selected_set_sha256"] == pr["selection_comparison"]["selected_set_sha256"]
        ),
        table=table,
        limitations=[
            "Device complete-graph microseconds, not service TPOT or isolated Fetch latency.",
            "whole_graph_GBps is mean transferred bytes divided by mean complete-graph time; it is not link bandwidth or utilization.",
            "Main serial, main fused and PR fused use separate processes/arenas. Only candidate variants share one arena.",
            "Optional before-* cases use the PR checkout and matching pre-existing options; their arena is separate from candidate. This controls source/configuration, not physical page placement.",
            "Execution order is retained; no randomized order, confidence interval or cross-process physical-placement equivalence is claimed.",
            "All timing/read samples are retained. Mean/p50 tail heuristics are not significance tests.",
            "Control-end drift is reported without correction. Instrumented role cycles are separate from headline timings and cannot be added into graph latency.",
            "Serial and fused TopK may differ only as accepted by the existing replay oracle; their measured cold/transfer counts remain separate.",
            "Admission controls are fixed before graph capture. Headline times include device gate/quota work, not a per-step CPU policy update or its H2D transfer.",
            "Optional mode-switch verification reuses one graph with quotas 0/512/0, restoring the snapshot before each untimed replay. It is not a learned selector or a proof of consecutive service-state evolution.",
        ],
    )


def print_summary(summary, brief=False):
    for name, pair in summary.get("light_mode_pairs", {}).items():
        print(f"LIGHT_PAIR {name}_vs_{pair['reference']} mean={pair['mean_reduction_pct']:+.2f}% "
              f"p50={pair['p50_reduction_pct']:+.2f}% same_arena_no_drift_correction", flush=True)
    if drift := summary.get("light_high_end_drift"):
        print(f"LIGHT_HIGH_DRIFT mean={drift['mean']['drift_pct']:+.2f}% "
              f"p50={drift['p50']['drift_pct']:+.2f}% no_correction", flush=True)
    skipped = [name for name, row in summary["table"].items()
               if (row.get("threshold_comparison") or {}).get("status") == "not_computed"
               and not row.get("auto_route")]
    if skipped:
        print("EARLY_THRESHOLD_NOT_COMPUTED " + ",".join(skipped) + " prefetch=0; final_scores_topk_kv_index=verified", flush=True)
    for row in summary["table"].values():
        if row.get("mode_switch_check"):
            print("MODE_SWITCH_PASS one_graph quotas=0/512/0 captured_addresses=unchanged "
                  "low_threshold_sentinel=unchanged "
                  "snapshot_restore_each=true untimed=true continuous_service_state=not_tested", flush=True)
        if route := row.get("auto_route"):
            print(f"AUTO_ROUTE_PASS low_steps={route['low_steps']} high_steps={route['high_steps']} "
                  "feedback=device_previous_success capture_warmup_learning=false device_policy_in_timing=true", flush=True)
            if route.get("lifecycle"):
                print("AUTO_LIFECYCLE_PASS one_graph feedback_reset=false packed_row_reorder=true "
                      "prediction_fault_injection=true rejection=identity_not_transport_fault snapshot_restore_each=true", flush=True)
    if brief:
        print("case                   mean_us    p50_us  prefetch    final   unused  vsSerial_mean/p50%  vsBefore_mean/p50%", flush=True)
        for name, row in summary["table"].items():
            reads, gains = row["reads"], row["gains"]["vs_serial"]
            matched = row["gains"].get("vs_matched_before")
            comparison = "n/a" if matched is None else f"{matched['mean_reduction_pct']:+7.2f}/{matched['p50_reduction_pct']:+7.2f}"
            print(f"{name:22} {row['mean_us']:9.3f} {row['p50_us']:9.3f} "
                  f"{reads['prefetch']:9.1f} {reads['final']:8.1f} {reads['unused_prefetch']:8.1f} "
                  f"{gains['mean_reduction_pct']:+7.2f}/{gains['p50_reduction_pct']:+7.2f} {comparison:>19}", flush=True)
            if row.get("capacity_skipped_row_windows") is not None:
                print(f"  SKIPPED_ROW_WINDOWS={row['capacity_skipped_row_windows']:.1f} suppression_counts=scanned_only", flush=True)
            gate = (row.get("threshold_comparison") or {}).get("prefetch_gate")
            if gate:
                prefetch = reads["prefetch"]
                waste = f"{100 * reads['unused_prefetch'] / prefetch:.2f}%" if prefetch else "n/a"
                print(f"  GATE rank={gate['rank']} verified=bit_exact "
                      f"above_final={gate['above_final_queries']}/{gate['queries']} "
                      f"mean_raise_steps={gate['mean_raise_key_steps']:.3f} waste_of_prefetch={waste}", flush=True)
            if name.startswith("candidate-") and row.get("diagnostic_summary"):
                metrics = row["diagnostic_summary"]["metrics"]
                fields = []
                for key in ("producer", "index_total", "index", "dependency_wait", "get_native", "prefetch_join", "early_prefetch_drain"):
                    metric = metrics.get(key, {})
                    value = metric.get("p50_cycles")
                    absent = "not_entered" if metric.get("status") == "not_entered" else "na"
                    fields.append(f"{key}={value:.0f}" if value is not None else f"{key}={absent}")
                print("  ROLE_P50_CYCLES " + " ".join(fields), flush=True)
                for sample in row["diagnostic_summary"].get("ready_batch_by_query", []):
                    print(f"  GET_BY_Q replay={sample['replay']} " + " ".join(
                        f"Q{q['query']}(requests={q['requests']},calls={q['calls']},"
                        f"native_max_cycles={q['native_max_cycles']},active={q['active_workers']}/{q['worker_count']})"
                        for q in sample["queries"]), flush=True)
        for name, pair in summary.get("inverse_mapping_pairs", {}).items():
            print(f"PAIR {name}_vs_{pair['reference']} mean={pair['mean_reduction_pct']:+.2f}% "
                  f"p50={pair['p50_reduction_pct']:+.2f}% same_arena_no_drift_correction", flush=True)
        for name, pair in summary.get("prefetch_rank_pairs", {}).items():
            print(f"RANK_PAIR {name}_vs_{pair['reference']} mean={pair['mean_reduction_pct']:+.2f}% "
                  f"p50={pair['p50_reduction_pct']:+.2f}% same_arena_no_drift_correction", flush=True)
        drift = summary["control_end_drift"]
        if drift:
            print(f"CONTROL_DRIFT {summary['control_name']} mean={drift['mean']['drift_pct']:+.2f}% "
                  f"p50={drift['p50']['drift_pct']:+.2f}% no_correction", flush=True)
        print("CHECK scores=bit_exact fused_ordered_topk=unchanged computed_safe_thresholds=unchanged full_graph_us=true", flush=True)
        if any(row.get("diagnostic_summary") for row in summary["table"].values()):
            print("PROFILE_SCOPE separate_replay; index_total=role_entry_to_exit; "
                  "index=InitShadow_to_commit_excluding_early_drain_and_status; cycles_are_not_additive", flush=True)
        return
    print(
        "case                  mean_us    p50_us  vsControl_p50/mean%  vsPR_p50/mean%  vsSerial_p50/mean%", flush=True
    )
    for name, row in summary["table"].items():
        pr, serial = row["gains"]["vs_pr"], row["gains"]["vs_serial"]
        control = row["gains"]["vs_control"]
        control_text = (
            "n/a" if control is None else f"{control['p50_reduction_pct']:+7.2f}/{control['mean_reduction_pct']:+7.2f}"
        )
        waste = row["reads"]["prefetch_waste_pct"]
        waste_text = "n/a" if waste is None else f"{waste:.2f}"
        print(
            f"{name:21} {row['mean_us']:9.3f} {row['p50_us']:9.3f} "
            f"{control_text:>19} "
            f"{pr['p50_reduction_pct']:+7.2f}/{pr['mean_reduction_pct']:+7.2f} "
            f"{serial['p50_reduction_pct']:+8.2f}/{serial['mean_reduction_pct']:+8.2f}",
            flush=True,
        )
        profile = row["diagnostic_summary"]
        if profile is not None:
            fields = [
                f"{key}=" + (("not_entered" if value.get("status") == "not_entered" else "n/a")
                             if value["p50_cycles"] is None else f"{value['p50_cycles']:.0f}")
                for key, value in profile["metrics"].items()
            ]
            print("  PROFILE p50_cycles " + " ".join(fields), flush=True)
            if profile["ready_batch_counts"]:
                counts = profile["ready_batch_counts"]
                print("  GET_BATCH " + " ".join(
                    f"calls={item['native_calls']} requests={item['completed_requests']} workers={item['worker_count']}"
                    for item in counts), flush=True)
        reads = row["reads"]
        print(
            f"  TRAFFIC cold_miss%={100 * reads['cold_miss_rate']:.4f} "
            f"read_MiB={reads['bytes'] / 1048576:.4f} "
            f"unused_bytes={reads['unused_prefetch_bytes']:.1f} "
            f"unused_MiB={reads['unused_prefetch_bytes'] / 1048576:.4f} "
            f"waste%={waste_text} whole_graph_GBps={row['whole_graph_GBps']:.3f}",
            flush=True,
        )
        if row["timing_spread"]["flagged"]:
            print(
                f"  TIMING_NOTE {name}: {';'.join(row['timing_spread']['reasons'])}; all samples retained", flush=True
            )
    print("CHECK fused_scores=bit_exact fused_ordered_topk=unchanged computed_candidate_thresholds=PR_bit_exact", flush=True)
    print("ORDER " + " -> ".join(summary["execution_order"]), flush=True)
    if summary["control_end_drift"] is not None:
        drift = summary["control_end_drift"]
        print(
            f"CONTROL_DRIFT mean={drift['mean']['delta_us']:+.3f}us/{drift['mean']['drift_pct']:+.2f}% "
            f"p50={drift['p50']['delta_us']:+.3f}us/{drift['p50']['drift_pct']:+.2f}% no_correction",
            flush=True,
        )
    if summary["metadata_end_vs_control_end"] is not None:
        pair = summary["metadata_end_vs_control_end"]
        print(
            f"END_PAIR metadata-end_vs_control-end mean={pair['mean']['reduction_pct']:+.2f}% "
            f"p50={pair['p50']['reduction_pct']:+.2f}% no_drift_correction",
            flush=True,
        )
    if summary["diagnostic_replays_per_fused_case"]:
        print(
            "PROFILE_SCOPE separate instrumented replays; raw role cycles overlap: do not sum or convert to graph us.",
            flush=True,
        )
    print(
        "SCOPE candidate gains primarily vs same-arena control; baseline arenas are separate. "
        "whole_graph_GBps is not link bandwidth/utilization; gains are not significance tests.",
        flush=True,
    )


def run(args):
    args.result_dir.mkdir(parents=True, exist_ok=False)
    manifest_digest = file_digest(args.snapshot / "manifest.json")
    plan = jobs(args)
    provenance = dict(
        status="running",
        snapshot=str(args.snapshot),
        snapshot_sha256=manifest_digest,
        diagnostics_commit=source_commit(),
        production_sources={
            role: dict(root=str(getattr(args, role + "_root")), commit=getattr(args, role + "_sha"))
            for role in ("main", "pr", "candidate")
        },
        jobs=[dict(job, command=command(args, job)) for job in plan],
        physical_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        logical_device=args.device,
    )
    provenance_path = args.result_dir / "invocation.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    # Keep this harness's diagnostics first; replay.activate then isolates every
    # production module to the requested exact checkout in the child process.
    env = os.environ.copy()
    python_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results = {}
    probe_report = None
    for job in plan:
        name = job["name"]
        output, log = (args.result_dir / (name + suffix) for suffix in (".json", ".log"))
        print(
            f"RUN {name}" + (f" variants={','.join(args.pipeline_variants)}" if name == "candidate" else ""), flush=True
        )
        try:
            with log.open("w") as stream:
                process = subprocess.run(
                    command(args, job), env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if process.returncode:
                raise RuntimeError(f"child exit={process.returncode}")
            report = json.loads(output.read_text())
            cases = unpack_report(job, report, job.get("variants", args.pipeline_variants))
            for case_name, case in cases.items():
                _validate_case(args, case_name, case, manifest_digest)
            results.update(cases)
            if job["role"] == "candidate" and getattr(args, "topk_probe", False):
                probe_report = report["topk_probe_report"]
                reference = next(iter(cases.values()))
                if "timing" in probe_report or probe_report.get("status") != "pass":
                    raise ValueError("TopK probe must pass without contributing a latency sample")
                for key in ("production_source", "snapshot_sha256", "score_comparison", "selection_comparison", "threshold_comparison"):
                    if probe_report.get(key) != reference.get(key):
                        raise ValueError(f"TopK probe differs from its checked production reference: {key}")
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            provenance.update(
                status="failed",
                failed_job=name,
                error=str(error),
                completed_cases=list(results),
                result=str(output),
                log=str(log),
            )
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
            print(f"PIPELINE_COMPARE_FAILED case={name} error={error}\nRESULT={output}\nLOG={log}", flush=True)
            if log.is_file():
                with log.open(errors="replace") as stream:
                    print("".join(deque(stream, maxlen=30)), end="", flush=True)
            raise SystemExit(1) from error
        log.unlink()  # Successful native/build chatter is not retained.
    try:
        summary = summarize(args, results, manifest_digest)
    except (ValueError, KeyError, TypeError) as error:
        provenance.update(
            status="failed", failed_job="summary_validation", error=str(error), completed_cases=list(results)
        )
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(
            f"PIPELINE_COMPARE_FAILED summary_validation={error}; per-case JSON retained in {args.result_dir}",
            flush=True,
        )
        raise SystemExit(1) from error
    summary["result_files"] = {job["name"]: str(args.result_dir / (job["name"] + ".json")) for job in plan}
    if probe_report is not None:
        summary["topk_probe"] = probe_report["topk_probe"]
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance.update(status="pass", completed_cases=list(results))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print_summary(summary, brief=getattr(args, "brief", False))
    if probe_report is not None:
        from kvip.diagnostics.topk_probe import print_summary as print_probe

        print_probe(probe_report["topk_probe"])
    print(f"PIPELINE_COMPARE_PASS RESULT={args.result_dir / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    run(arguments(argv))


if __name__ == "__main__":
    main()
