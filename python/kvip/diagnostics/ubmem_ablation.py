# SPDX-License-Identifier: Apache-2.0
"""Same-transport fused variants sharing one runtime and imported DDR arena."""

from copy import copy
import gc
import json

import numpy as np

from kvip.diagnostics import replay_measure
from kvip.diagnostics.console import emit
from kvip.operators.lifetch.geometry import prefetch_budget


def parse_variants(value, transport="ubmem"):
    from kvip.diagnostics.fused_replay import validate_fused_tuning

    values = json.loads(value)
    if not isinstance(values, list) or not 2 <= len(values) <= 4:
        raise ValueError("fused-variants requires two to four named tuning objects")
    result, names = [], set()
    for item in values:
        if not isinstance(item, dict) or set(item) != {"name", "tuning"}:
            raise ValueError("Each fused variant requires name and tuning")
        name = item["name"]
        if not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name in names:
            raise ValueError("Variant names must be unique ASCII identifiers")
        names.add(name)
        result.append((name, validate_fused_tuning(item["tuning"], transport)))
    return result


def plan_cases(geometry):
    """Keep captured geometry fixed; change only kernel switches/read budgets."""
    total, window = prefetch_budget(geometry["prefetch_limit"], geometry["window_limit"], geometry["requested_windows"])
    return [
        ("baseline", dict(parallel_init=False, balanced_index=False, prefetch_limit=total, window_limit=window)),
        ("optimized", dict(parallel_init=True, balanced_index=True, prefetch_limit=total, window_limit=window)),
        (
            "optimized_half_prefetch",
            dict(parallel_init=True, balanced_index=True, prefetch_limit=total // 2, window_limit=max(1, window // 2)),
        ),
    ]


def service_cases(archive, active_batches):
    """Keep B-capacity fixed and retain the captured queries/owners for active rows."""
    from kvip.diagnostics.active_rows import mask_active_rows

    geometry = archive.manifest["geometry"]
    batch = geometry["batch"]
    counts = active_batches if active_batches is not None else [1, batch]
    if not counts or any(type(n) is not int or not 1 <= n <= batch for n in counts):
        raise ValueError(f"Active request counts must be integers in 1..{batch}")
    total, window = prefetch_budget(geometry["prefetch_limit"], geometry["window_limit"], geometry["requested_windows"])
    common = dict(parallel_init=False, balanced_index=True, prefetch_limit=total // 2, window_limit=max(1, window // 2))
    plans = []
    for count in dict.fromkeys(counts):
        view = mask_active_rows(archive, count)
        variants = [("dense_baseline", False, False)]
        if count < batch:
            variants.append(("skip_only", True, False))
        variants.append(("compact_skip", True, True))
        for label, skip, compact in variants:
            name = f"b{count}_{label}"
            tuning = dict(common, skip_inactive=skip, compact_reserved=compact)
            plans.append((name, tuning, view))
    return plans


def summarize_case(case):
    stages = case.get("stage_diagnostics", {}).get("stages", {})
    includes_completion = any(name.startswith("complete_index_") for name in stages)
    index_stages = [
        value["samples_us"] for name, value in stages.items() if name.startswith(("index_", "complete_index_"))
    ]
    if index_stages and len({len(values) for values in index_stages}) != 1:
        raise ValueError("Index stage samples must represent the same diagnostic iterations")
    reads = case["reads_samples"]
    index_time = float(np.median(np.sum(np.asarray(index_stages, dtype=np.float64), axis=0))) if index_stages else None
    return dict(
        name=case["name"],
        tuning=case["tuning"],
        full_graph_mean_us=case["timing"]["mean_us"],
        full_graph_p50_us=case["timing"]["p50_us"],
        init_diagnostic_p50_us=stages.get("init", {}).get("p50_us"),
        index_diagnostic_p50_us=None if includes_completion else index_time,
        completion_index_diagnostic_p50_us=index_time if includes_completion else None,
        **{
            key: float(np.median([sample[key] for sample in reads]))
            for key in (
                "prefetch",
                "final",
                "unused_prefetch",
                "bytes",
                "useful_transfers",
                "useful_transfer_bytes",
                "selected_unique",
                "cold_demand",
                "candidate_hot_refetch",
                "hot_hit",
                "abc_hit",
                "selected_read_demand",
                "cold_miss_rate",
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
            if key in reads[0]
        },
    )


def print_summary(report, output=None):
    def time(value):
        return "-" if value is None else f"{value:.3f}"

    print(
        "case  device_mean_us  device_p50_us  init_diag_us  index_diag_us  "
        "completion_index_diag_us  prefetch  final  unused  read_bytes"
    )
    for row in report["summary"]:
        print(
            f"{row['name']}  {time(row['full_graph_mean_us'])}  {time(row['full_graph_p50_us'])}  "
            f"{time(row['init_diagnostic_p50_us'])}  "
            f"{time(row['index_diagnostic_p50_us'])}  {time(row.get('completion_index_diagnostic_p50_us'))}  "
            f"{row['prefetch']:g}  {row['final']:g}  "
            f"{row['unused_prefetch']:g}  {row['bytes']:g}"
        )
        if "cold_miss_rate" in row:
            print(
                f"  initial-state demand: cold={row['cold_demand']:g}, "
                f"candidate_hot={row['candidate_hot_refetch']:g}, selected_union={row['selected_unique']:g}; "
                f"cold/selected_union={row['cold_miss_rate']:.4%}; "
                f"useful_transfers={row['useful_transfers']:g} (includes hot refetch, not pure cold)"
            )
        if "prefetch_cold" in row:
            print(
                f"  prefetch initial residency: cold={row['prefetch_cold']:g}, "
                f"candidate_hot={row['prefetch_candidate_hot']:g}, other_hot={row['prefetch_other_hot']:g}, "
                f"abc={row['prefetch_abc']:g}, unclassified={row['prefetch_unclassified']:g}; "
                f"unused: cold={row['unused_prefetch_cold']:g}, "
                f"candidate_hot={row['unused_prefetch_candidate_hot']:g}, "
                f"other_hot={row['unused_prefetch_other_hot']:g}, abc={row['unused_prefetch_abc']:g}"
            )
    for case in report.get("cases", []):
        stages = case.get("stage_diagnostics", {}).get("stages", {})
        details = [f"{name}={time(value['p50_us'])}" for name, value in stages.items()]
        if details:
            print(f"{case['name']} diagnostic p50 us: " + ", ".join(details))
    if len(report["summary"]) > 1:
        baseline = report["summary"][0]
        for row in report["summary"][1:]:
            old, new = baseline["full_graph_mean_us"], row["full_graph_mean_us"]
            print(
                f"{row['name']} vs {baseline['name']}: speedup={old / new:.3f}x; "
                f"latency reduction={100 * (1 - new / old):.2f}% (same-runtime fused comparison)"
            )
    print("Init/index are separate instrumented diagnostics; they are not additive components of full-graph latency.")
    print("Merged completion/index time includes completion bookkeeping; the index-only column is unavailable for it.")
    calls = [(case["name"], case["candidate_call"]) for case in report.get("cases", []) if "candidate_call" in case]
    if calls:
        print("case  group_call_mean_us  group_call_p50_us  cpu_prepare_p50_us  h2d_diag_p50_us  h2d_bytes")
        for name, call in calls:
            payload = call["payload"]
            size = payload["candidate_h2d_bytes"] + payload["dense_protection_h2d_bytes"]
            print(
                f"{name}  {time(call['wall']['mean_us'])}  {time(call['wall']['p50_us'])}  "
                f"{time(call['cpu_prepare']['p50_us'])}  "
                f"{time(call.get('h2d_diagnostic', {}).get('p50_us'))}  {size}"
            )
        print(
            "Group call starts with captured candidates ready; excludes full-step/shared preparation. Do not multiply by 21."
        )
    if output is not None:
        print(f"Result JSON: {output.resolve()}")


def run_ablation(args, archive, retained):
    """Failure keeps the active case and shared Runtime alive in _RETAINED_RUNS."""
    service = getattr(args, "service_ablation", False)
    variants = getattr(args, "fused_variants", None)
    if args.mode != "fused" or args.transport not in ("ubmem", "urma"):
        raise ValueError("Fused ablation requires --mode fused and a supported transport")
    if args.transport != "ubmem" and (variants is None or service or getattr(args, "ubmem_ablation", False)):
        raise ValueError("URMA ablation requires explicit --fused-variants")
    if variants is not None:
        plans = [(name, tuning, archive) for name, tuning in variants]
    elif service:
        plans = service_cases(archive, getattr(args, "active_batches", None))
    else:
        plans = [(name, tuning, archive) for name, tuning in plan_cases(archive.manifest["geometry"])]
    context = replay_measure.allocate_context(args, archive, retained)
    cases = []
    for name, tuning, case_archive in plans:
        emit(f"CASE_BEGIN {args.transport}-{name}: {tuning}")
        case_args = copy(args)
        case_args.fused_tuning = tuning
        result = replay_measure.measure_case(case_args, case_archive, context, retained)
        cases.append(dict(name=name, **result))
        emit(
            f"CASE_PASS {args.transport}-{name}: mean={result['timing']['mean_us']:.3f} us "
            f"p50={result['timing']['p50_us']:.3f} us",
            flush=True,
        )
        # measure_case reset its graphs, closed its lease, then dropped tensor
        # references. Reclaim cycles before allocating the next shape-identical
        # case; shared Runtime/table/DDR remain live and are never reimported.
        gc.collect()
    replay_measure.release_context(context)
    result = dict(
        schema_version=2,
        status="pass",
        transport=args.transport,
        mode="fused",
        ablation=(
            f"{args.transport}_explicit_variants"
            if variants is not None
            else "ubmem_active_rows_compact_metadata"
            if service
            else "ubmem_init_index_prefetch"
        ),
        snapshot=str(archive.directory),
        geometry=archive.manifest["geometry"],
        cases=cases,
        summary=[summarize_case(case) for case in cases],
        interpretation=(
            "Sequential cases reuse one Runtime, imported DDR arena, access table and transport connections. "
            "Full-graph latency is isolated group latency, not service TPOT; stage diagnostics add instrumentation."
        ),
    )
    if service:
        result["interpretation"] += (
            " Small batches are explicitly derived padding views of captured requests, not newly captured service waves. "
            "Both variants disable the regressing parallel init and retain balanced Index + half prefetch. "
            "Device inverse-map construction is timed in the graph. "
            "Optional candidate_call separately measures one group's candidate staging/inverse map/H2D to completion; "
            "it does not reconstruct the full service step."
        )
    if not getattr(args, "quiet", False):
        print_summary(result)
    return result
