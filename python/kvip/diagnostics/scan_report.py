"""Print only comparable graph timings and actual transport counters."""

import argparse
import json
from pathlib import Path
from statistics import mean

from kvip.diagnostics.scan_options import DEFAULT_VARIANTS, ROLLING_VARIANTS


def report(path):
    data = json.loads(path.read_text())
    order = tuple(data.get("execution_order", ()))
    if data.get("status") != "pass" or order not in (DEFAULT_VARIANTS, ROLLING_VARIANTS):
        raise ValueError("Missing or incomplete controlled scan experiment")
    cases = data["cases"]
    baseline = cases["forward"]["timing"]
    print("variant            mean_us   p50_us  mean_gain%  waste_MiB repair_MiB total_MiB useful_GB/s")
    for name in order:
        case = cases[name]
        samples = case["reads_samples"]
        if not samples:
            raise ValueError("Missing verified transfer counters")
        timing = case["timing"]
        width = case["geometry"]["token_bytes"]
        waste = mean(row["unused_prefetch_bytes"] for row in samples)
        repair = width * mean(row["final"] for row in samples)
        total = mean(row["bytes"] for row in samples)
        useful = mean(row["useful_transfer_bytes"] for row in samples)
        gain = 100 * (1 - timing["mean_us"] / baseline["mean_us"])
        print(f"{name:17} {timing['mean_us']:9.3f} {timing['p50_us']:8.3f} {gain:11.2f} "
              f"{waste/2**20:10.3f} {repair/2**20:10.3f} {total/2**20:9.3f} "
              f"{useful/timing['mean_us']/1000:11.3f}")
    reads = cases["forward"]["reads_samples"]
    cold = mean(row["cold_demand"] for row in reads)
    selected = mean(row["selected_unique"] for row in reads)
    drift = 100 * (cases["forward-end"]["timing"]["mean_us"] / baseline["mean_us"] - 1)
    print(f"CHECK ordered_TopK=identical thresholds=identical cold_miss={100*cold/selected:.4f}% "
          f"baseline_repeat_drift={drift:+.2f}%")
    pairs = (("forward", "reverse"), ("forward-wide", "reverse-wide"), ("reverse", "reverse-wide"))
    if order == ROLLING_VARIANTS:
        pairs = (("forward", "reverse"), ("reverse", "reverse-roll"),
                 ("reverse-roll", "reverse-refilter"), ("forward", "forward-refilter"),
                 ("forward-refilter", "reverse-refilter"))
    for left, right in pairs:
        a, b = cases[left]["timing"], cases[right]["timing"]
        print(f"CHANGE {left}->{right} mean_gain={100*(1-b['mean_us']/a['mean_us']):+.2f}% "
              f"p50_gain={100*(1-b['p50_us']/a['p50_us']):+.2f}%")
    for name in order:
        check = cases[name].get("rolling_threshold_check")
        if check:
            print(f"THRESHOLD {name} bounded_CDF={check['status']} "
                  f"tightened={check['tightened_queries']}/{check['queries']} "
                  f"capped={check['capped_queries']} mean_key_gap_to_final={check['mean_final_gap_keys']:.3f}")
    print("NOTE useful_GB/s=useful_bytes/full_graph_time; not link utilization or coldEarly/lead")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    report(args.result)


if __name__ == "__main__":
    main()
