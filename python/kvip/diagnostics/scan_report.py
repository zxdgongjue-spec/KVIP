"""Print only comparable graph timings and actual transport counters."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from kvip.diagnostics.scan_options import DEFAULT_VARIANTS, ROLLING_VARIANTS


def load_report(path, *, require_rolling=False, expected_native=None, snapshot=None):
    data = json.loads(path.read_text())
    order = tuple(data.get("execution_order", ()))
    if data.get("status") != "pass" or order not in (DEFAULT_VARIANTS, ROLLING_VARIANTS):
        raise ValueError("Missing or incomplete controlled scan experiment")
    if require_rolling and order != ROLLING_VARIANTS:
        raise ValueError("Old scan/budget experiment: rolling/refilter cases were not measured")
    digest = None if snapshot is None else hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name in order:
        case = data["cases"][name]
        if expected_native and case.get("production_source", {}).get("commit") != expected_native:
            raise ValueError(f"{name}: native revision does not match {expected_native}")
        if digest and case.get("snapshot_sha256") != digest:
            raise ValueError(f"{name}: snapshot manifest does not match")
        if require_rolling and name in ("reverse-roll", "reverse-refilter", "forward-refilter"):
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


def report(path, *, require_rolling=False, expected_native=None, snapshot=None, check_only=False):
    data = load_report(path, require_rolling=require_rolling, expected_native=expected_native, snapshot=snapshot)
    if check_only:
        return
    order = data["execution_order"]
    cases = data["cases"]
    natives = sorted({case.get("production_source", {}).get("commit", "unknown") for case in cases.values()})
    readers = sorted({case.get("diagnostics_commit", "unknown") for case in cases.values()})
    print(f"SOURCE native={','.join(natives)} reader={','.join(readers)}")
    print(f"RESULT_FILE={path}")
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
    print(f"CHECK ordered_TopK=identical thresholds=identical cold_miss={100*cold/selected:.4f}% "
          f"baseline_mean_drift={drift:+.2f}% baseline_p50_drift={median_drift:+.2f}%")
    pairs = (("forward", "reverse"), ("forward-wide", "reverse-wide"), ("reverse", "reverse-wide"))
    if tuple(order) == ROLLING_VARIANTS:
        pairs = (("forward", "reverse"), ("reverse", "reverse-roll"),
                 ("reverse-roll", "reverse-refilter"), ("forward", "forward-refilter"),
                 ("forward-refilter", "reverse-refilter"))
    for left, right in pairs:
        a, b = cases[left]["timing"], cases[right]["timing"]
        print(f"CHANGE {left}->{right} mean_gain={100*(1-b['mean_us']/a['mean_us']):+.2f}% "
              f"p50_gain={100*(1-b['p50_us']/a['p50_us']):+.2f}%")
        a, b = transfers[left], transfers[right]
        print(f"  BYTES waste_save_MiB={(a['waste']-b['waste'])/2**20:+.4f} "
              f"waste_save={reduction(a['waste'], b['waste'])} "
              f"repair_delta_MiB={(b['repair']-a['repair'])/2**20:+.4f} "
              f"total_save_MiB={(a['total']-b['total'])/2**20:+.4f} "
              f"total_save={reduction(a['total'], b['total'])}")
    for name in order:
        check = cases[name].get("rolling_threshold_check")
        if check:
            print(f"THRESHOLD {name} bounded_CDF={check['status']} "
                  f"tightened={check['tightened_queries']}/{check['queries']} "
                  f"capped={check['capped_queries']} mean_key_gap_to_final={check['mean_final_gap_keys']:.3f}")
    print("NOTE saves>0=less bytes; repair_delta>0=more final repair; drift>0=slower baseline repeat")
    print("NOTE useful_GB/s=useful_bytes/full_graph_time; not link utilization or coldEarly/lead")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--require-rolling", action="store_true")
    parser.add_argument("--expected-native")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report(args.result, require_rolling=args.require_rolling, expected_native=args.expected_native,
               snapshot=args.snapshot, check_only=args.check_only)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"REPORT_REJECTED: {error}\n")


if __name__ == "__main__":
    main()
