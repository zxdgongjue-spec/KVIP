"""Print only comparable graph timings and actual transport counters."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from kvip.diagnostics.scan_options import DEFAULT_VARIANTS, ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS, ECONOMY_VARIANTS, SEED_VARIANTS, FAST_VARIANTS, SPARSE_VARIANTS, BOUNDARY_VARIANTS, INCREMENTAL_VARIANTS, options_for


def load_report(path, *, require_rolling=False, require_batched=False, require_windowed=False, require_lean=False, require_safe=False, require_economy=False, require_seed=False, require_fast=False, require_sparse=False, require_boundary=False, require_incremental=False, expected_native=None, snapshot=None):
    data = json.loads(path.read_text())
    order = tuple(data.get("execution_order", ()))
    if data.get("status") != "pass" or order not in (DEFAULT_VARIANTS, ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS, ECONOMY_VARIANTS, SEED_VARIANTS, FAST_VARIANTS, SPARSE_VARIANTS, BOUNDARY_VARIANTS, INCREMENTAL_VARIANTS):
        raise ValueError("Missing or incomplete controlled scan experiment")
    if require_rolling and order not in (ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS, ECONOMY_VARIANTS, SEED_VARIANTS, FAST_VARIANTS, SPARSE_VARIANTS, BOUNDARY_VARIANTS, INCREMENTAL_VARIANTS):
        raise ValueError("Old scan/budget experiment: rolling/refilter cases were not measured")
    if require_batched and order != BATCHED_VARIANTS:
        raise ValueError("Batched histogram variants were not measured")
    if require_windowed and order != WINDOWED_VARIANTS:
        raise ValueError("Window accumulation variants were not measured")
    if require_lean and order != LEAN_VARIANTS:
        raise ValueError("Compact and first-window controls were not measured")
    if require_safe and order != SAFE_VARIANTS:
        raise ValueError("Safe/refill controls were not measured")
    if require_economy and order != ECONOMY_VARIANTS:
        raise ValueError("Economy controls were not measured")
    if require_incremental and order != INCREMENTAL_VARIANTS:
        raise ValueError("Incremental producer comparison was not measured")
    if require_boundary and order != BOUNDARY_VARIANTS:
        raise ValueError("Boundary readback and refilter controls were not measured")
    if require_sparse and order != SPARSE_VARIANTS:
        raise ValueError("Sparse histogram and refilter controls were not measured")
    if require_fast and order != FAST_VARIANTS:
        raise ValueError("Packed histogram controls were not measured")
    if require_seed and order != SEED_VARIANTS:
        raise ValueError("Seed reuse controls were not measured")
    if require_incremental:
        candidate = data["cases"]["reverse-incremental"]
        if candidate.get("incremental_check", {}).get("status") != "exact_prefix_ranks_and_pending":
            raise ValueError("Incremental device prefix/pending oracle has not passed")
    digest = None if snapshot is None else hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name in order:
        case = data["cases"][name]
        if (require_fast or require_sparse or require_boundary or require_incremental) and case.get("geometry", {}).get("rounds") != 9:
            raise ValueError(f"{name}: expected resident plus eight cold windows")
        if expected_native and case.get("production_source", {}).get("commit") != expected_native:
            raise ValueError(f"{name}: native revision does not match {expected_native}")
        if digest and case.get("snapshot_sha256") != digest:
            raise ValueError(f"{name}: snapshot manifest does not match")
        if (require_rolling or require_batched or require_windowed or require_lean or require_safe or require_economy or require_seed or require_fast or require_sparse or require_boundary) and options_for(name)["rolling_threshold"]:
            if case.get("rolling_threshold_check", {}).get("status") != "bit_exact":
                raise ValueError(f"{name}: rolling threshold oracle has not passed")
    return data


def traffic(case):
    samples = case["reads_samples"]
    if not samples:
        raise ValueError("Missing verified transfer counters")
    return dict(
        waste=mean(row["unused_prefetch_bytes"] for row in samples),
        repair=case["geometry"]["token_bytes"] * mean(row["final"] for row in samples),
        total=mean(row["bytes"] for row in samples),
        useful=mean(row["useful_transfer_bytes"] for row in samples),
    )


def reduction(before, after):
    return f"{100 * (1 - after / before):+.2f}%" if before else "n/a"


def report(path, *, require_rolling=False, require_batched=False, require_windowed=False, require_lean=False, require_safe=False, require_economy=False, require_seed=False, require_fast=False, require_sparse=False, require_boundary=False, require_incremental=False, expected_native=None, snapshot=None, check_only=False):
    data = load_report(path, require_rolling=require_rolling, require_batched=require_batched, require_windowed=require_windowed, require_lean=require_lean, require_safe=require_safe, require_economy=require_economy, require_seed=require_seed, require_fast=require_fast, require_sparse=require_sparse, require_boundary=require_boundary, require_incremental=require_incremental, expected_native=expected_native, snapshot=snapshot)
    if check_only:
        return
    order = data["execution_order"]
    cases = data["cases"]
    natives = sorted({case.get("production_source", {}).get("commit", "unknown") for case in cases.values()})
    readers = sorted({case.get("diagnostics_commit", "unknown") for case in cases.values()})
    print(f"SOURCE native={','.join(natives)} reader={','.join(readers)}")
    print(f"RESULT_FILE={path}")
    if tuple(order) in (ECONOMY_VARIANTS, SEED_VARIANTS, FAST_VARIANTS, SPARSE_VARIANTS, BOUNDARY_VARIANTS, INCREMENTAL_VARIANTS):
        from kvip.diagnostics.scan_economy_report import report as economy_report
        economy_report(cases, order)
        return
    transfers = {name: traffic(cases[name]) for name in order}
    baseline = cases["forward"]["timing"]
    print("variant            mean_us   p50_us  mean_gain%  waste_MiB repair_MiB total_MiB useful_GB/s")
    for name in order:
        case = cases[name]
        timing = case["timing"]
        waste, repair, total, useful = (transfers[name][key] for key in ("waste", "repair", "total", "useful"))
        gain = 100 * (1 - timing["mean_us"] / baseline["mean_us"])
        print(f"{name:17} {timing['mean_us']:9.3f} {timing['p50_us']:8.3f} {gain:11.2f} "
              f"{waste/2**20:10.3f} {repair/2**20:10.3f} {total/2**20:9.3f} "
              f"{useful/timing['mean_us']/1000:11.3f}")
    reads = cases["forward"]["reads_samples"]
    cold = mean(row["cold_demand"] for row in reads)
    selected = mean(row["selected_unique"] for row in reads)
    drift = 100 * (cases["forward-end"]["timing"]["mean_us"] / baseline["mean_us"] - 1)
    median_drift = 100 * (cases["forward-end"]["timing"]["p50_us"] / baseline["p50_us"] - 1)
    print(f"CHECK ordered_TopK=identical initial/final_thresholds=identical cold_miss={100*cold/selected:.4f}% "
          f"baseline_mean_drift={drift:+.2f}% baseline_p50_drift={median_drift:+.2f}%")
    pairs = (("forward", "reverse"), ("forward-wide", "reverse-wide"), ("reverse", "reverse-wide"))
    if tuple(order) == ROLLING_VARIANTS:
        pairs = (("forward", "reverse"), ("reverse", "reverse-roll"),
                 ("reverse-roll", "reverse-refilter"), ("forward", "forward-refilter"),
                 ("forward-refilter", "reverse-refilter"))
    if tuple(order) == BATCHED_VARIANTS:
        pairs = (("reverse", "reverse-batch"), ("reverse-batch", "reverse-batch-ref"),
                 ("reverse-refilter", "reverse-batch-ref"))
    if tuple(order) == WINDOWED_VARIANTS:
        pairs = (("reverse-refilter", "reverse-window-ref"), ("reverse-batch-ref", "reverse-window-ref"),
                 ("reverse-window", "reverse-window-ref"), ("forward", "forward-window-ref"))
    if tuple(order) == LEAN_VARIANTS:
        pairs = (("forward", "reverse"), ("reverse", "reverse-window-ref"),
                 ("reverse-window-ref", "reverse-compact-ref"), ("reverse-compact-ref", "reverse-first-ref"))
    if tuple(order) == SAFE_VARIANTS:
        pairs = (("forward", "reverse"), ("reverse", "reverse-window-ref"),
                 ("reverse-window-ref", "reverse-safe-ref"), ("reverse-safe-ref", "reverse-refill-ref"))
    for left, right in pairs:
        a, b = cases[left]["timing"], cases[right]["timing"]
        timing_change = (f"CHANGE {left}->{right} mean_gain={100*(1-b['mean_us']/a['mean_us']):+.2f}% "
                         f"p50_gain={100*(1-b['p50_us']/a['p50_us']):+.2f}%")
        a, b = transfers[left], transfers[right]
        if tuple(order) in (BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS):
            print(f"{timing_change} waste_save_MiB={(a['waste']-b['waste'])/2**20:+.4f} "
                  f"repair_delta_MiB={(b['repair']-a['repair'])/2**20:+.4f} "
                  f"total_save_MiB={(a['total']-b['total'])/2**20:+.4f}")
            continue
        print(timing_change)
        print(f"  BYTES waste_save_MiB={(a['waste']-b['waste'])/2**20:+.4f} "
              f"waste_save={reduction(a['waste'], b['waste'])} "
              f"repair_delta_MiB={(b['repair']-a['repair'])/2**20:+.4f} "
              f"total_save_MiB={(a['total']-b['total'])/2**20:+.4f} "
              f"total_save={reduction(a['total'], b['total'])}")
    checks = {name: cases[name]["rolling_threshold_check"] for name in order
              if cases[name].get("rolling_threshold_check")}
    if tuple(order) in (BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS):
        print("ROLLING_CHECK " + " ".join(
            f"{name}={check['status']}/capped:{check['capped_queries']}" for name, check in checks.items()))
    for name in (() if tuple(order) in (BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS) else order):
        check = cases[name].get("rolling_threshold_check")
        if check:
            print(f"THRESHOLD {name} bounded_CDF={check['status']} "
                  f"tightened={check['tightened_queries']}/{check['queries']} "
                  f"capped={check['capped_queries']} mean_key_gap_to_final={check['mean_final_gap_keys']:.3f}")
    if tuple(order) in (WINDOWED_VARIANTS, LEAN_VARIANTS, SAFE_VARIANTS):
        candidates = ("reverse", "reverse-compact-ref", "reverse-first-ref") if tuple(order) == LEAN_VARIANTS else None
        if tuple(order) == SAFE_VARIANTS:
            candidates = ("reverse", "reverse-safe-ref", "reverse-refill-ref")
        check_acceptance(cases, candidates)
    if tuple(order) in (LEAN_VARIANTS, SAFE_VARIANTS):
        for name in order:
            check = cases[name].get("rolling_threshold_check")
            if check:
                print(f"THRESHOLD {name} scope={check.get('scope', 'all_scanned_windows')} "
                      f"mean_key_gap_to_final={check['mean_final_gap_keys']:.3f}")
            profile = cases[name].get("native_diagnostics")
            if profile:
                # Keep only existing production role markers. No invented
                # seed timing from slots that this upstream does not expose.
                for sample in profile["samples"]:
                    fields = []
                    for label, key in (("producer", "producer_span"), ("join", "final_prefetch_join"),
                                       ("post", "post_active"), ("drain", "cq_drain")):
                        metric = sample["metrics"].get(key, {})
                        if metric.get("status") == "available":
                            fields.append(f"{label}={metric['median_cycles']:.0f}/{metric['max_cycles']}")
                    print(f"PROFILE {name} cycles_p50/max " + " ".join(fields))
        print("PROFILE_SCOPE separate instrumented replay; overlapping role cycles, not graph microseconds")
    if tuple(order) == SAFE_VARIANTS:
        from kvip.diagnostics.prefetch_admission import mean_admission
        for name in order:
            samples = cases[name]["reads_samples"]
            useful_prefetch = mean(row["prefetch"]-row["unused_prefetch"] for row in samples)
            cold_useful = mean(row["prefetch_cold"]-row["unused_prefetch_cold"] for row in samples)
            cold_demand = mean(row["cold_demand"] for row in samples)
            admission = mean_admission(samples)
            if admission is None:
                raise ValueError(f"{name}: missing native admission counters")
            print(f"ADMISSION {name} useful_prefetch={useful_prefetch:.1f} "
                  f"cold_prefetch_pct={100*cold_useful/cold_demand if cold_demand else 0:.2f}% "
                  f"unused_budget={admission['remaining_admissible_budget']:.1f} "
                  "posts_by_window=" + ",".join(f"{v:.1f}" for v in admission["posted_by_epoch"]))
        print("ADMISSION_SCOPE cold_prefetch_pct=all-prefetch/final-cold-demand; not before-last or completion overlap")
    print("NOTE saves>0=less bytes; repair_delta>0=more final repair; drift>0=slower baseline repeat")
    print("NOTE useful_GB/s=useful_bytes/full_graph_time; not link utilization or coldEarly/lead")


def check_acceptance(cases, candidates=None):
    """Conservative screening, not statistical significance or auto-deployment."""
    first, last = (cases[name]["timing"] for name in ("forward", "forward-end"))
    margins = {key: max(1.0, 100*abs(last[key]/first[key]-1)) for key in ("mean_us", "p50_us")}
    accepted = []
    for name in candidates or ("reverse-window", "reverse-window-ref", "forward-window-ref"):
        timing = cases[name]["timing"]
        controls = [first, last]
        if name in ("reverse-compact-ref", "reverse-first-ref", "reverse-safe-ref", "reverse-refill-ref"):
            controls.append(cases["reverse"]["timing"])
        gain = {key: 100*(1-timing[key]/min(control[key] for control in controls)) for key in margins}
        ok = all(gain[key] > margins[key] for key in margins)
        if ok:
            accepted.append(name)
        print(f"GATE {name} {'CANDIDATE' if ok else 'KEEP_BASELINE'} "
              f"mean_gain={gain['mean_us']:+.2f}% p50_gain={gain['p50_us']:+.2f}% "
              f"required_mean>{margins['mean_us']:.2f}% required_p50>{margins['p50_us']:.2f}%")
    print("DECISION " + ("retest_candidates="+",".join(accepted) if accepted else "keep_original_fused"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--require-rolling", action="store_true")
    parser.add_argument("--require-batched", action="store_true")
    parser.add_argument("--require-windowed", action="store_true")
    parser.add_argument("--require-lean", action="store_true")
    parser.add_argument("--require-safe", action="store_true")
    parser.add_argument("--require-economy", action="store_true")
    parser.add_argument("--require-seed", action="store_true")
    parser.add_argument("--require-fast", action="store_true")
    parser.add_argument("--require-sparse", action="store_true")
    parser.add_argument("--require-boundary", action="store_true")
    parser.add_argument("--require-incremental", action="store_true")
    parser.add_argument("--expected-native")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report(args.result, require_rolling=args.require_rolling, require_batched=args.require_batched, require_windowed=args.require_windowed, require_lean=args.require_lean, require_safe=args.require_safe, require_economy=args.require_economy, require_seed=args.require_seed, require_fast=args.require_fast, require_sparse=args.require_sparse, require_boundary=args.require_boundary, require_incremental=args.require_incremental, expected_native=args.expected_native,
               snapshot=args.snapshot, check_only=args.check_only)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"REPORT_REJECTED: {error}\n")


if __name__ == "__main__":
    main()
