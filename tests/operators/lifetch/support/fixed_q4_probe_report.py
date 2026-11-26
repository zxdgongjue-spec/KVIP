# SPDX-License-Identifier: Apache-2.0
"""Report cold-pipeline probes and paired, same-list terminal policy replays."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


SLOW_DONORS = (1, 3, 4, 8, 12, 14)
FAST_DONORS = tuple(i for i in range(16) if i not in SLOW_DONORS)
METRICS = ("tail", "graph")
EQUIVALENCE_US = 2.0


def load_probe(path):
    record = json.loads(path.read_text())
    if record["identity"]["physical_npu"] != 3:
        raise ValueError(f"{path}: expected B24 physical NPU3")
    return record


def samples(record):
    ident, pairs = record["identity"], record["result"]["pairs"]
    if (record["probe"], ident["group"], ident["batch"]) != ("post-pair", 17, 1) or len(pairs) != 10:
        raise ValueError("Expected ten G17 BS1 pairs")
    values = []
    order = []
    for p in pairs:
        order.append(tuple(p["order"]))
        values.append([[p[policy][key] for key in METRICS] for policy in ("four", "one")])
    times = np.asarray(values, dtype=np.float64)
    if not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("Invalid event or kernel timing in pair")
    if order.count(("four", "one")) != 5 or order.count(("one", "four")) != 5:
        raise ValueError(f"Expected five pairs per order, found {order}")
    return times, np.array([o[0] == "one" for o in order])


def report_rows(records, label):
    """Give every archived donor equal weight, retaining its five AB and five BA pairs."""
    by_donor = [samples(r) for r in records]
    all_times = np.concatenate([v for v, _ in by_donor], axis=0)
    delta = all_times[:, 0] - all_times[:, 1]
    rng = np.random.default_rng(241103)
    replicas = np.zeros((10000, len(METRICS)), dtype=np.float64)
    for times, mask in by_donor:
        d = times[:, 0] - times[:, 1]
        for m in (mask, ~mask):
            strata = d[m]
            draw = rng.integers(len(strata), size=(len(replicas), len(strata)))
            replicas += strata[draw].mean(axis=1) / (2 * len(records))
    ci = np.quantile(replicas, [.05, .95], axis=0)
    for j, metric in enumerate(METRICS):
        four, one = [all_times[:, i, j] for i in range(2)]
        lo, hi = ci[:, j]
        if lo >= -EQUIVALENCE_US and hi <= EQUIVALENCE_US:
            verdict = "WITHIN_2US"
        elif lo > EQUIVALENCE_US:
            verdict = "FASTER_OVER_2US"
        elif hi < -EQUIVALENCE_US:
            verdict = "SLOWER_OVER_2US"
        else:
            verdict = "INCONCLUSIVE"
        print(f"G17 BS1 {label} {metric} four_mean/P50_us={mean(four):.2f}/{median(four):.2f} "
              f"one_mean/P50_us={mean(one):.2f}/{median(one):.2f} "
              f"paired_saved_mean/P50_us={mean(delta[:, j]):+.2f}/{median(delta[:, j]):+.2f} "
              f"mean_saved_90pct_interval_us={lo:+.2f}/{hi:+.2f} {verdict}")


def case(path):
    record = load_probe(path)
    ident = record["identity"]
    if record["probe"] == "cold":
        from fixed_q4_cold_profile import format_report

        if ident["batch"] != 32 or ident["group"] not in (2, 10):
            raise ValueError("Expected G2/G10 BS32")
        for line in format_report(ident["group"], record["result"]):
            print(line)
    else:
        donor, = ident["donor_rows"]
        work = record["result"]["worklist"]
        receipts = record["result"]["receipts"]
        counts = {}
        for policy in ("four", "one"):
            actual = [r for r in receipts if r["policy"] == policy]
            if (len(actual) != 10 or {r["pair"] for r in actual} != set(range(10))
                    or any(r["sha256"] != work["sha256"] or r["per_q_tokens"] != work["per_q_tokens"]
                           for r in actual)):
                raise ValueError(f"R{donor} missing or unmatched {policy} GET receipts")
            cqes = {tuple(r["per_q_cqes"]) for r in actual}
            if len(cqes) != 1:
                raise ValueError(f"R{donor} unexpected change in {policy} CQ count: {cqes}")
            counts[policy] = next(iter(cqes))
        print(f"G17 BS1 R{donor} matched_GET_lists=PASS paired_calls=10 "
              f"per_Q_tokens={work['per_q_tokens']} CQEs_four={list(counts['four'])} "
              f"CQEs_one={list(counts['one'])}")
        report_rows([record], f"R{donor}")


def summary(root):
    print("G17 saved_us=four_minus_one interval=90pct_within_donor_and_order_bootstrap "
          "equivalence_band_us=-2..+2")
    for donors, label in ((SLOW_DONORS, "six_slow"), (FAST_DONORS, "ten_other")):
        records = []
        for donor in donors:
            r = load_probe(root / f"g17/bs1/cohort{donor}/fused/PROBE_RESULT.json")
            if r["identity"]["donor_rows"] != [donor]:
                raise ValueError(f"Incorrect donor in cohort {donor}")
            records.append(r)
        report_rows(records, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("case", "summary"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "case":
        case(args.path)
    else:
        summary(args.path)
