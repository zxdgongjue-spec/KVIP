# SPDX-License-Identifier: Apache-2.0
"""Summarize the two exact 10k fixtures measured on before/candidate/before-end."""

import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median
import sys


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def read_arm(path, layouts, unit):
    cases = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("INDEX_MODEL_START "):
            header = fields(line)
            layout = header["layout"]
            if layout not in layouts or layout in cases or (header["B"], header["cores"]) != ("16", "64"):
                raise ValueError(f"{path}: unexpected fixture {layout}")
            current = cases[layout] = dict(events=int(header["events"]), moves=int(header["moves"]))
        elif line.startswith("DEVICE_INDEX_SAMPLES_US "):
            samples = [float(v) for v in line.split()[1:]]
            if len(samples) != 6 or not all(isfinite(v) and v > 0 for v in samples):
                raise ValueError(f"{path}: need six positive uninstrumented samples")
            current.update(samples_us=samples, mean_us=mean(samples), p50_us=median(samples))
        elif line.startswith("DEVICE_INDEX_US "):
            row = fields(line)
            if (
                row["diagnostics_excluded"] != "true"
                or row["samples"] != "6"
                or row["warmup"] != "1"
                or not isclose(current["mean_us"], float(row["mean"]), abs_tol=0.000501)
                or not isclose(current["p50_us"], float(row["p50"]), abs_tol=0.000501)
            ):
                raise ValueError(f"{path}: sample/summary mismatch")
            current["verified_timing"] = True
        elif line.startswith("MODEL_EXACT_PASS "):
            row = fields(line)
            if (int(row["moved"]), int(row["events"])) != (current["moves"], current["events"]):
                raise ValueError(f"{path}: incomplete work")
            current["verified_payload"] = True
    if set(cases) != set(layouts) or any(
        case[unit] != 10000 or not case.get("verified_timing") or not case.get("verified_payload")
        for case in cases.values()
    ):
        raise ValueError(f"{path}: missing exact measured and verified 10k fixture")
    return cases


def summarize(directory, unit, old, candidate):
    layouts = ("even", "skew") if unit == "moves" else ("even-events", "skew-events")
    arms = ("before", "after", "before-end")
    root = Path(directory)
    src = {arm: json.loads((root / arm / "model-source.json").read_text()) for arm in ("before", "after")}
    if (
        src["before"]["revision"] != old
        or src["after"]["revision"] != candidate
        or old == candidate
        or any(src["before"][k] != src["after"][k] for k in ("harness_sha256", "mode", "soc", "include"))
        or src["before"]["mode"] != "npu"
    ):
        raise ValueError("Build provenance differs between pinned control and candidate")
    results = {arm: read_arm(root / (arm + ".log"), layouts, unit) for arm in arms}
    findings = {}
    for lay in layouts:
        first, new, last = [results[a][lay] for a in arms]
        if len({(r["events"], r["moves"]) for r in (first, new, last)}) != 1:
            raise ValueError(f"{lay}: workload changed across arms")

        def fmt(row):
            return f"{row['mean_us']:.3f}/{row['p50_us']:.3f}"

        def delta(l, r):
            return {k: l[k] - r[k] for k in ("mean_us", "p50_us")}

        drift = delta(last, first)
        gain_first, gain_last = delta(first, new), delta(last, new)
        target = {k: new[k] <= 50.0 for k in ("mean_us", "p50_us")}
        findings[lay] = dict(
            before_end_minus_before_us=drift,
            saved_vs_first_us=gain_first,
            saved_vs_last_us=gain_last,
            at_most_50us=target,
        )
        print(
            f"INDEX_10K_RESULT layout={lay} events={new['events']} moves={new['moves']} mean/p50_us"
            f" before={fmt(first)} after={fmt(new)} before_end={fmt(last)}"
            f" drift={fmt(drift)} saved_first={fmt(gain_first)} saved_last={fmt(gain_last)}"
            f" at_most_50us={'PASS' if target['mean_us'] else 'MISS'}/{'PASS' if target['p50_us'] else 'MISS'}"
            " payload=PASS"
        )
    report = dict(unit=unit, sources=src, runs=results, comparisons=findings)
    with (root / "summary.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(f"INDEX_10K_SUMMARY={root / 'summary.json'}")
    return report


if __name__ == "__main__":
    summarize(*sys.argv[1:])
