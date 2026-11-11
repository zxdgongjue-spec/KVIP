# SPDX-License-Identifier: Apache-2.0
"""Four production paths on one real snapshot, one transport/process at a time."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from kvip.diagnostics.demand import DEMAND_ACCOUNTING
from kvip.diagnostics.console import compact_report, emit, failure_tail, print_admission, report_only
from kvip.diagnostics.prefetch_admission import ADMISSION_ACCOUNTING, mean_admission

CASES = (("ubmem", "serial"), ("urma", "serial"), ("ubmem", "fused"), ("urma", "fused"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--ubmem-build-dir", type=Path)
    parser.add_argument("--urma-build-dir", type=Path)
    parser.add_argument("--kvaf-server", help="Dedicated local URMA CPU service")
    parser.add_argument(
        "--quiet", action="store_true", help="Only case progress and compact results; native logs appear on failure"
    )
    parser.add_argument(
        "--brief", action="store_true", help="One row per path: latency, measured cold misses and transferred payload"
    )
    parser.add_argument("--report-only", type=Path, help="Summarize an existing result JSON without device work")
    parser.add_argument(
        "--from-results",
        type=Path,
        nargs=4,
        metavar=("UBMEM_SERIAL", "URMA_SERIAL", "UBMEM_FUSED", "URMA_FUSED"),
        help="Compare saved results without device work; choose the lowest validated mean from each fused ablation",
    )
    parser.add_argument("--output", type=Path, help="New summary JSON, only with --from-results")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--stage-iterations", type=int, default=3)
    parser.add_argument(
        "--split-index-stages",
        action="store_true",
        help="Split UBmem fused completion/plan, output adaptation and Index apply diagnostics",
    )
    parser.add_argument("--ubmem-serial-workers", type=int, default=32)
    parser.add_argument(
        "--urma-serial-post-size",
        type=int,
        choices=(32, 128),
        default=128,
        help="URMA serial WQEs per post; 32 requires the optional post32 kernel",
    )
    parser.add_argument(
        "--urma-serial-max-inflight",
        type=int,
        help="Forward only to URMA serial: 0 submits whole core slices; omitted keeps native default 2048",
    )
    serial = parser.add_mutually_exclusive_group()
    serial.add_argument("--urma-serial-workers", type=int, help="URMA serial Get workers (default: 2)")
    serial.add_argument(
        "--urma-serial-workers-candidates",
        type=int,
        nargs=2,
        metavar=("BASE", "CANDIDATE"),
        help="Measure both URMA serial worker counts and use the fastest validated device mean as baseline",
    )
    parser.add_argument("--ubmem-fused-tuning", help="JSON options passed only to UBmem fused replay")
    parser.add_argument("--urma-fused-tuning", help="JSON options passed only to URMA fused replay")
    parser.add_argument("--result-dir", type=Path, help="New directory; all cases and one summary JSON")
    args = parser.parse_args(argv)
    if args.report_only is not None or args.from_results is not None:
        if any(
            (
                args.snapshot,
                args.ubmem_build_dir,
                args.urma_build_dir,
                args.kvaf_server,
                args.soc,
                args.result_dir,
                args.ubmem_fused_tuning,
                args.urma_fused_tuning,
            )
        ):
            parser.error("Saved-result modes cannot be combined with capture/build/run options")
        if args.report_only is not None and (args.from_results is not None or args.output is not None):
            parser.error("--report-only cannot be combined with --from-results/--output")
        if args.from_results is not None and (args.output is None or args.output.exists()):
            parser.error("--from-results requires a new --output path")
        return args
    if args.output is not None:
        parser.error("--output requires --from-results")
    if any(
        value is None
        for value in (
            args.snapshot,
            args.ubmem_build_dir,
            args.urma_build_dir,
            args.kvaf_server,
            args.soc,
            args.result_dir,
        )
    ):
        parser.error("comparison requires --snapshot, both build dirs, --kvaf-server, --soc and --result-dir")
    if args.urma_serial_workers is None:
        args.urma_serial_workers = 2
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.stage_iterations < 0:
        parser.error("Invalid device/warmup/iterations/stage-iterations")
    if any(not 1 <= n <= 48 for n in (args.ubmem_serial_workers, args.urma_serial_workers)):
        parser.error("Serial worker counts must be 1..48")
    candidates = args.urma_serial_workers_candidates
    if candidates is not None and (len(set(candidates)) != 2 or any(not 1 <= n <= 48 for n in candidates)):
        parser.error("URMA serial candidates must be two distinct worker counts in 1..48")
    from kvip.diagnostics.replay_runtime import validate_urma_serial_max_inflight

    try:
        for workers in candidates or (args.urma_serial_workers,):
            validate_urma_serial_max_inflight(args.urma_serial_max_inflight, workers)
    except ValueError as error:
        parser.error(str(error))
    from kvip.diagnostics.fused_replay import parse_fused_tuning

    for transport in ("ubmem", "urma"):
        name = transport + "_fused_tuning"
        value = getattr(args, name)
        if value is not None:
            try:
                setattr(args, name, parse_fused_tuning(value, transport))
            except ValueError as error:
                parser.error(str(error))
    if args.result_dir.exists():
        parser.error("--result-dir already exists; choose a new directory")
    if args.ubmem_build_dir.resolve() == args.urma_build_dir.resolve():
        parser.error("URMA and UBmem require separate KVAF builds")
    return args


def case_plan(args):
    """Keep the default four cases; a two-point URMA sweep adds just one run."""
    for transport, mode in CASES:
        candidates = getattr(args, "urma_serial_workers_candidates", None)
        if (transport, mode) == ("urma", "serial") and candidates:
            for workers in candidates:
                yield transport, mode, workers
        else:
            yield transport, mode, None


def command(args, transport, mode, serial_workers=None):
    build = getattr(args, transport + "_build_dir")
    name = f"{transport}-{mode}"
    if serial_workers is not None:
        if (transport, mode) != ("urma", "serial"):
            raise ValueError("Worker candidates apply only to URMA serial replay")
        name += f"-workers{serial_workers}"
    result = args.result_dir / f"{name}.json"
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
        workers = serial_workers if serial_workers is not None else getattr(args, transport + "_serial_workers")
        values += ["--serial-workers", str(workers)]
        inflight = getattr(args, "urma_serial_max_inflight", None)
        if transport == "urma" and inflight is not None:
            values += ["--urma-serial-max-inflight", str(inflight)]
        if transport == "urma":
            values += ["--urma-serial-post-size", str(getattr(args, "urma_serial_post_size", 128))]
    else:
        if getattr(args, "split_index_stages", False):
            values.append("--split-index-stages")
        tuning = getattr(args, transport + "_fused_tuning", None)
        if tuning is not None:
            values += ["--fused-tuning", json.dumps(tuning, sort_keys=True)]
    return values, result


def summarize(results):
    required = {"ubmem-serial", "ubmem-fused", "urma-fused"}
    serial_cases = [
        name for name, result in results.items() if (result.get("transport"), result.get("mode")) == ("urma", "serial")
    ]
    if not required.issubset(results) or not serial_cases or set(results) != required | set(serial_cases):
        raise ValueError("All four cases are required")
    first = results["ubmem-serial"]
    for name, result in results.items():
        if result.get("status") != "pass":
            raise ValueError(f"Case did not pass validation: {name}")
        for key in ("snapshot_sha256", "geometry", "scope", "soc", "device", "kvip_commit"):
            if result.get(key) != first.get(key) or key not in result:
                raise ValueError(f"Incomparable {key}: {name}")
        if result.get("content", "full") != first.get("content", "full"):
            raise ValueError(f"Incomparable content: {name}")
        identity = f"{result['transport']}-{result['mode']}"
        candidate = f"urma-serial-workers{result['runtime_workers']}"
        if name != identity and not (identity == "urma-serial" and name == candidate):
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
            tuning=result.get("tuning"),
            serial_get_config={
                key: result[key]
                for key in (
                    "batch_get_max_inflight_total",
                    "batch_get_max_inflight_total_source",
                    "jettys_per_core",
                    "batch_get_post_size",
                )
                if key in result
            },
            get_kernel=result.get("get_kernel"),
            prefetch_admission=mean_admission(result["reads_samples"]),
            selection=result.get("selection_comparison"),
            reads_per_round={
                key: float(np.mean([row[key] for row in result["reads_samples"]]))
                for key in (
                    "prefetch",
                    "final",
                    "transfers",
                    "unused_prefetch",
                    "useful_transfers",
                    "cold_demand",
                    "candidate_hot_refetch",
                    "prefetch_cold",
                    "prefetch_candidate_hot",
                    "prefetch_other_hot",
                    "prefetch_abc",
                    "prefetch_unclassified",
                    "unused_prefetch_cold",
                    "unused_prefetch_candidate_hot",
                    "unused_prefetch_other_hot",
                    "unused_prefetch_abc",
                )
                if key in result["reads_samples"][0]
            },
            demand_per_round={
                key: float(np.mean([row[key] for row in result["reads_samples"]]))
                for key in (
                    "selected_unique",
                    "cold_demand",
                    "candidate_hot_refetch",
                    "hot_hit",
                    "abc_hit",
                    "selected_read_demand",
                )
                if key in result["reads_samples"][0]
            },
        )
        demand = table[name]["demand_per_round"]
        if "selected_unique" in demand:
            demand["cold_miss_rate"] = (
                demand["cold_demand"] / demand["selected_unique"] if demand["selected_unique"] else 0.0
            )
            demand["cold_payload_bytes"] = demand["cold_demand"] * result["geometry"]["token_bytes"]
        if "useful_transfer_bytes" in result["reads_samples"][0]:
            table[name]["mean_useful_transfer_bytes"] = float(
                np.mean([row["useful_transfer_bytes"] for row in result["reads_samples"]])
            )
        normalized = dict(actual_transfers=table[name]["mean_read_bytes"] / (timing["mean_us"] * 1000))
        if "mean_useful_transfer_bytes" in table[name]:
            normalized["useful_transfers"] = table[name]["mean_useful_transfer_bytes"] / (timing["mean_us"] * 1000)
        if "cold_payload_bytes" in demand:
            normalized["cold_demand"] = demand["cold_payload_bytes"] / (timing["mean_us"] * 1000)
        table[name]["graph_normalized_payload_GBps"] = normalized
    ratios = {}
    demand_deltas = {}
    selected_urma = min(serial_cases, key=lambda name: table[name]["mean_us"])
    for transport in ("ubmem", "urma"):
        serial_case = "ubmem-serial" if transport == "ubmem" else selected_urma
        serial, fused = table[serial_case]["mean_us"], table[f"{transport}-fused"]["mean_us"]
        target = min(serial * 0.65, serial / 1.5)
        ratios[transport] = dict(
            serial_case=serial_case,
            serial_over_fused=serial / fused,
            latency_reduction_percent=(1 - fused / serial) * 100,
            p50_serial_over_fused=table[serial_case]["p50_us"] / table[f"{transport}-fused"]["p50_us"],
            p50_latency_reduction_percent=(1 - table[f"{transport}-fused"]["p50_us"] / table[serial_case]["p50_us"])
            * 100,
            target_fused_us=target,
            target_met=fused <= serial * 0.65 and fused <= serial / 1.5,
        )
        serial_demand, fused_demand = (
            table[serial_case]["demand_per_round"],
            table[f"{transport}-fused"]["demand_per_round"],
        )
        if "selected_unique" in serial_demand and "selected_unique" in fused_demand:
            demand_deltas[transport] = {
                key: fused_demand[key] - serial_demand[key]
                for key in ("cold_demand", "candidate_hot_refetch", "selected_unique", "cold_miss_rate")
            }
    selection_pairs = {}
    for mode in ("serial", "fused"):
        names = ("ubmem-serial", selected_urma) if mode == "serial" else ("ubmem-fused", "urma-fused")
        selections = [table[name]["selection"] for name in names]
        if all(value is not None for value in selections):
            selection_pairs[mode] = selections[0]["selected_set_sha256"] == selections[1]["selected_set_sha256"]
    return dict(
        status="pass",
        snapshot_sha256=first["snapshot_sha256"],
        content=first.get("content", "full"),
        fidelity=first.get("fidelity", first.get("representation")),
        input_validation=first.get("input_validation"),
        geometry=first["geometry"],
        scope=first["scope"],
        table=table,
        demand_accounting=DEMAND_ACCOUNTING,
        prefetch_admission_accounting=ADMISSION_ACCOUNTING,
        demand_delta_fused_minus_serial=demand_deltas,
        payload_rate_accounting="Decimal GB/s = payload bytes / full graph mean_us / 1000; graph-normalized throughput, not Get-stage or hardware bandwidth.",
        fusion_vs_serial=ratios,
        performance_target=dict(
            metric="timing.mean_us",
            minimum_latency_reduction_percent=35.0,
            minimum_speedup=1.5,
            target_met=all(value["target_met"] for value in ratios.values()),
        ),
        urma_serial_selection=dict(
            candidates=serial_cases,
            selected=selected_urma,
            criterion="lowest validated device graph timing.mean_us among measured candidates",
            improvement_over_first_percent=(1 - table[selected_urma]["mean_us"] / table[serial_cases[0]]["mean_us"])
            * 100,
        ),
        same_selection_across_transports=selection_pairs,
        score_validation={
            name: result["score_comparison"] for name, result in results.items() if "score_comparison" in result
        },
        stage_diagnostics={name: result.get("stage_diagnostics") for name, result in results.items()},
        native_get_profiles={
            name: result["native_get_profile"] for name, result in results.items() if "native_get_profile" in result
        },
        notes=[
            "Full-path latency uses one uninstrumented production graph per case.",
            "Stage diagnostics add graph/event boundaries: use to localize cost, not as an exact decomposition.",
            "Serial uses native CANN LI plus production LIDU/Get; fused uses its own LI and Index update.",
            "TopK ID sets may differ only at a verified BF16 boundary tie; read bytes reflect each path's actual selection.",
            "The snapshot fixes inputs and pre-state, not speculative prefetch scheduling or physical pages.",
            "Correctness PASS is separate from performance target_met; candidate_call and stage timings never enter the target.",
        ],
    )


def summarize_saved(paths):
    """Reuse measured evidence; validate every candidate before selecting by mean."""
    if len(paths) != 4:
        raise ValueError("Expected UBmem serial, URMA serial, UBmem fused, URMA fused results")
    identities = ("ubmem-serial", "urma-serial", "ubmem-fused", "urma-fused")
    selected, candidates = {}, {}
    for identity, path in zip(identities, paths):
        report = json.loads(Path(path).read_text())
        if report.get("status") != "pass":
            raise ValueError(f"Saved run failed: {path}")
        cases = report.get("cases", [report])
        if not cases or ("cases" in report and not identity.endswith("-fused")):
            raise ValueError("Only fused results may contain an ablation")
        for case in cases:
            if f"{case.get('transport')}-{case.get('mode')}" != identity:
                raise ValueError(f"Saved case identity differs from {identity}")
            # summarize() checks successful validation, identity, positive means,
            # snapshot hash, geometry, scope, revision, device and SoC together.
            probe = dict(selected, **{identity: case})
            reference = next(iter(probe.values()))
            for key in ("snapshot_sha256", "geometry", "scope", "soc", "device", "kvip_commit"):
                if key not in case or case[key] != reference.get(key):
                    raise ValueError(f"Incomparable {key}: {identity}")
            if case.get("status") != "pass" or any(
                not np.isfinite(case["timing"][key]) or case["timing"][key] <= 0 for key in ("mean_us", "p50_us")
            ):
                raise ValueError(f"Invalid saved timing: {identity}")
        winner = min(cases, key=lambda case: case["timing"]["mean_us"])
        selected[identity] = winner
        candidates[identity] = dict(
            path=str(Path(path).resolve()),
            selected=winner.get("name", identity),
            cases=[dict(name=case.get("name", identity), mean_us=case["timing"]["mean_us"]) for case in cases],
        )
    result = summarize(selected)
    result["saved_result_selection"] = candidates
    result["notes"].append(
        "Fused winners are selected from this finite measured sweep, not independently remeasured; "
        "the full candidate list is retained. Serial references are separate same-revision runs."
    )
    return result


def brief_report(summary, *, path=None):
    """Keep measured per-path demand visible without printing per-window counters."""
    if summary.get("status") != "pass" or "table" not in summary:
        raise ValueError("Brief comparison requires a successful four-path summary")
    geometry = summary["geometry"]
    emit(
        f"COMPARE_PASS batch={geometry['batch']} token_bytes={geometry['token_bytes']} "
        f"content={summary.get('content', 'full')}; device graph us, payload/graph decimal GB/s"
    )
    emit("case          mean_us    p50_us    cold/selected    miss%    read_MiB    payload_GB/s")
    for name, row in summary["table"].items():
        demand = row.get("demand_per_round", {})
        if "cold_miss_rate" in demand:
            counts = f"{demand['cold_demand']:.0f}/{demand['selected_unique']:.0f}"
            rate = f"{100 * demand['cold_miss_rate']:.4f}"
        else:
            counts = rate = "--"
        emit(
            f"{name:13s} {row['mean_us']:9.3f} {row['p50_us']:9.3f} {counts:>16s} {rate:>8s} "
            f"{row['mean_read_bytes'] / (1 << 20):11.3f} "
            f"{row['graph_normalized_payload_GBps']['actual_transfers']:15.3f}"
        )
    for transport, value in summary["fusion_vs_serial"].items():
        emit(
            f"FUSION {transport}: speedup={value['serial_over_fused']:.3f}x "
            f"latency_reduction={value['latency_reduction_percent']:.2f}%"
        )
    if checks := summary.get("score_validation"):
        emit("LI_SCORE_CHECK " + " ".join(f"{name}={value['status']}" for name, value in checks.items()))
    if pairs := summary.get("same_selection_across_transports"):
        emit("SAME_SELECTION " + " ".join(f"{name}={str(value).lower()}" for name, value in pairs.items()))
    if path is not None:
        emit(f"Results: {Path(path).resolve()}")


def display(summary):
    print("\ncase          mean_us     p50_us   read_MiB/round   runtime_workers")
    for name, row in summary["table"].items():
        print(
            f"{name:13s} {row['mean_us']:10.3f} {row['p50_us']:10.3f} "
            f"{row['mean_read_bytes'] / (1 << 20):16.3f} {row['runtime_workers']:13d}"
        )
        print(
            "  actual transfers/round: "
            + ", ".join(
                f"{key}={value:.1f}"
                for key, value in row["reads_per_round"].items()
                if key not in ("cold_demand", "candidate_hot_refetch")
            )
        )
        print_admission(row.get("prefetch_admission"))
        demand = row["demand_per_round"]
        if demand:
            print(
                "  initial-state demand/round: "
                + ", ".join(
                    f"{key}={value:.1f}"
                    for key, value in demand.items()
                    if key not in ("cold_miss_rate", "cold_payload_bytes")
                )
            )
        if "cold_miss_rate" in demand:
            print(
                f"  cold miss rate: {demand['cold_miss_rate']:.4%} = cold_demand / selected_unique "
                "(per-request Q4 union; initial hot/ABC excluded from cold)"
            )
        if "mean_useful_transfer_bytes" in row:
            print(
                f"  useful transferred payload: {row['mean_useful_transfer_bytes'] / (1 << 20):.3f} MiB/round (includes hot refetch)"
            )
        print(
            "  graph-normalized payload GB/s: "
            + ", ".join(f"{key}={value:.3f}" for key, value in row["graph_normalized_payload_GBps"].items())
        )
        if row["tuning"] is not None:
            print("  effective tuning: " + json.dumps(row["tuning"], sort_keys=True))
        if row["serial_get_config"]:
            print("  serial Get config: " + json.dumps(row["serial_get_config"], sort_keys=True))
        if row["get_kernel"] is not None:
            print("  Get object: " + json.dumps(row["get_kernel"], sort_keys=True))
        selection = row["selection"]
        if selection is not None:
            print(
                f"  selection: {selection['status']}; changed_queries={selection['changed_queries']}; "
                f"replacements={selection['total_replacements']}; "
                f"min_overlap={selection['min_overlap_fraction']:.6f}"
            )
    selection = summary["urma_serial_selection"]
    print(
        f"URMA serial baseline: {selection['selected']}; "
        f"improvement over first measured candidate={selection['improvement_over_first_percent']:.2f}%"
    )
    for transport, values in summary["fusion_vs_serial"].items():
        print(
            f"{transport}: serial/fused={values['serial_over_fused']:.3f}x; "
            f"latency reduction={values['latency_reduction_percent']:.2f}%; "
            f"target <= {values['target_fused_us']:.3f} us: {'MET' if values['target_met'] else 'NOT MET'}"
        )
    print("Targets use device graph mean only; require both >=35% reduction and >=1.5x speedup.")
    print(
        "Candidate-hot demand is required in serial and potential in fusion; useful transfers are not pure cold misses."
    )
    for transport, values in summary["demand_delta_fused_minus_serial"].items():
        print(
            f"{transport} fused - serial demand: "
            + ", ".join(
                f"{key}={value:+.4f}" if key == "cold_miss_rate" else f"{key}={value:+.1f}"
                for key, value in values.items()
            )
        )
    print(summary["payload_rate_accounting"])
    for name, profile in summary["native_get_profiles"].items():
        totals, maxima = profile["totals"], profile["maxima"]
        print(
            f"{name} extra native profile (cycles, outside headline): active_cores={profile['active_cores']}; "
            f"requests={totals['active_requests']}; posts={totals['total_posts']}; "
            f"post_active sum/max={totals['post_active_cycles']}/{maxima['post_active_cycles']}; "
            f"drain_wait sum/max={totals['drain_wait_cycles']}/{maxima['drain_wait_cycles']}"
        )
    for mode, same in summary["same_selection_across_transports"].items():
        print(f"{mode}: UBmem/URMA selected ID sets identical={same}")
        if not same:
            print("  Different tied selections: the latency difference cannot be attributed only to transport.")
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
    if args.report_only is not None:
        if args.brief:
            try:
                brief_report(json.loads(args.report_only.read_text()), path=args.report_only)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise SystemExit(f"FAILED reading comparison: {error}") from error
        else:
            report_only(args.report_only)
        return
    if getattr(args, "from_results", None) is not None:
        summary = summarize_saved(args.from_results)
        with args.output.open("x") as output:
            output.write(json.dumps(summary, indent=2) + "\n")
        for name, selected in summary["saved_result_selection"].items():
            print(f"SELECTED {name}={selected['selected']}", flush=True)
        (brief_report if args.brief else compact_report)(summary, path=args.output)
        return
    args.result_dir.mkdir(parents=True)
    print(
        f"CONFIG device={args.device} warmup={args.warmup} iterations={args.iterations} stages={args.stage_iterations}",
        flush=True,
    )
    results = {}
    for transport, mode, workers in case_plan(args):
        argv, result_file = command(args, transport, mode, workers)
        name = result_file.stem
        print(f"CASE_BEGIN {name}: restore real DDR/pre-state, then measure", flush=True)
        # Keep successful native setup chatter out of the terminal. The temporary
        # spool is removed immediately, not retained as another benchmark log.
        with tempfile.NamedTemporaryFile(mode="w+", prefix="kvip-compare-", suffix=".log", delete=False) as output:
            spool = Path(output.name)
            try:
                result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=False)
            except BaseException as error:
                output.flush()
                failure_tail(spool, message=f"{name}: {type(error).__name__}: {error}")
                raise
            if result.returncode:
                output.flush()
                failure_tail(spool, message=f"{name} exited {result.returncode}; result: {result_file}")
                raise SystemExit(1)
            try:
                case_result = json.loads(result_file.read_text())
                if case_result.get("status") != "pass" or "timing" not in case_result:
                    raise ValueError("Child returned no validated timing result")
            except Exception as error:
                output.flush()
                failure_tail(spool, message=f"{name}: {type(error).__name__}: {error}")
                raise
        spool.unlink()
        results[name] = case_result
        print(f"CASE_PASS {name}: mean={results[name]['timing']['mean_us']:.3f} us", flush=True)
    summary = summarize(results)
    if summary.get("content") == "synthetic_token_replay":
        emit("CONTENT synthetic_token_replay: generated scores/KV bytes; not real-model timing")
    elif summary.get("content") == "real_li_synthetic_kv":
        emit("CONTENT real_li_synthetic_kv: real Q/Index; generated KV bytes; single-device operator timing")
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.brief:
        brief_report(summary, path=args.result_dir / "summary.json")
    elif args.quiet:
        compact_report(summary, path=args.result_dir / "summary.json")
    else:
        display(summary)
        print(f"\nResults: {args.result_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
