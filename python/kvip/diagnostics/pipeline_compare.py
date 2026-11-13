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
import subprocess
import sys

from kvip.diagnostics.archive import file_digest, source_commit
from kvip.diagnostics.compare import read_means, reduction, timing_spread
from kvip.diagnostics.pipeline_options import OPTIONS, VARIANTS, options_for
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
    parser.add_argument("--pipeline-variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.iterations < 1:
        parser.error("Require nonnegative device/warmup and positive iterations")
    if len(set(args.pipeline_variants)) != len(args.pipeline_variants):
        parser.error("pipeline-variants must be unique; order is preserved")
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


def jobs(args):
    return [
        dict(name="main-serial", role="main", mode="serial", policy="resident"),
        dict(name="main-resident", role="main", mode="fused", policy="resident"),
        dict(name="pr-recent", role="pr", mode="fused", policy="recent"),
        dict(name="candidate", role="candidate", mode="fused", policy="recent"),
    ]


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
        "post-threshold" if job["role"] == "candidate" else "threshold",
    ]
    if job["mode"] == "serial":
        result += ["--serial-workers", "32"]
    if job["role"] == "candidate":
        result += ["--pipeline-variants", *args.pipeline_variants]
    return result


def unpack_report(job, report, variants):
    if report.get("status") != "pass":
        raise ValueError(f"{job['name']} did not produce a passing replay result")
    if job["role"] != "candidate":
        if "cases" in report:
            raise ValueError(f"{job['name']} must contain exactly one production case")
        return {job["name"]: report}
    if report.get("study") != "post-threshold" or report.get("execution_order") != variants:
        raise ValueError("Candidate study/order differs from the requested variants")
    if list(report.get("cases", {})) != variants:
        raise ValueError("Candidate case keys/order differ from the requested variants")
    return {"candidate-" + name: report["cases"][name] for name in variants}


def _validate_case(args, name, result, manifest_digest):
    role = "main" if name.startswith("main-") else "pr" if name == "pr-recent" else "candidate"
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
        if result.get("score_comparison", {}).get("status") != "bit_exact":
            raise ValueError(f"{name}: fused captured LI scores were not verified bit-exact")
        threshold = result.get("threshold_comparison") or {}
        policy = "resident" if name == "main-resident" else "recent"
        if threshold.get("status") != "bit_exact" or threshold.get("policy") != policy:
            raise ValueError(f"{name}: initial threshold policy was not verified bit-exact")


def summarize(args, results, manifest_digest):
    expected_order = ["main-serial", "main-resident", "pr-recent", *("candidate-" + v for v in args.pipeline_variants)]
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
        if name.startswith("candidate-"):
            if result["threshold_comparison"]["initial_keys"] != pr["threshold_comparison"]["initial_keys"]:
                raise ValueError(f"{name}: post-threshold optimization changed the early threshold")
            expected_options = options_for(name.removeprefix("candidate-"))
            tuning = result.get("tuning") or {}
            if {key: tuning.get(key) for key in OPTIONS} != expected_options:
                raise ValueError(f"{name}: candidate build options differ")
            if {key: value for key, value in tuning.items() if key not in OPTIONS} != pr.get("tuning"):
                raise ValueError(f"{name}: candidate changed another tuning option or prefetch budget")
    table = {}
    control = results.get("candidate-control")
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
        )
        table[name]["gains"]["vs_control"] = (
            {
                "mean_reduction_pct": reduction(control["timing"]["mean_us"], timing["mean_us"]),
                "p50_reduction_pct": reduction(control["timing"]["p50_us"], timing["p50_us"]),
            }
            if control is not None and name.startswith("candidate-")
            else None
        )
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
        serial_configuration=dict(workers=32, post_size=128, max_inflight=2048, jettys_per_core=2),
        serial_fused_selected_sets_identical=(
            first["selection_comparison"]["selected_set_sha256"] == pr["selection_comparison"]["selected_set_sha256"]
        ),
        table=table,
        limitations=[
            "Device complete-graph microseconds, not service TPOT or isolated Fetch latency.",
            "whole_graph_GBps is mean transferred bytes divided by mean complete-graph time; it is not link bandwidth or utilization.",
            "Main serial, main fused and PR fused use separate processes/arenas. Only candidate variants share one arena.",
            "Execution order is retained; no randomized order, confidence interval or cross-process physical-placement equivalence is claimed.",
            "All timing/read samples are retained. Mean/p50 tail heuristics are not significance tests.",
            "Serial and fused TopK may differ only as accepted by the existing replay oracle; their measured cold/transfer counts remain separate.",
        ],
    )


def print_summary(summary):
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
    print("CHECK fused_scores=bit_exact fused_ordered_topk=unchanged candidate_thresholds=PR_bit_exact", flush=True)
    print("ORDER " + " -> ".join(summary["execution_order"]), flush=True)
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
            cases = unpack_report(job, report, args.pipeline_variants)
            for case_name, case in cases.items():
                _validate_case(args, case_name, case, manifest_digest)
            results.update(cases)
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
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance.update(status="pass", completed_cases=list(results))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print_summary(summary)
    print(f"PIPELINE_COMPARE_PASS RESULT={args.result_dir / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    run(arguments(argv))


if __name__ == "__main__":
    main()
