# SPDX-License-Identifier: Apache-2.0
"""Compare complete automatic graphs before/after removing LOW publication work."""

import argparse
import json
from pathlib import Path
from statistics import mean

from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.post_index_order_report import _low_traffic, _tuning, _validate
from kvip.diagnostics.read_through_report import change, load_case, pair, require, timing


ROLES = ("serial", "before", "after", "after-end", "before-end", "serial-end")


def summarize(directory, group):
    root = Path(directory)
    cases = {role: load_case(str(root / f"g{group}-{role}.json"), role) for role in ROLES}
    reference = cases["before"]
    for role, case in cases.items():
        for field in ("snapshot_sha256", "kvaf_build", "device", "soc", "warmup", "geometry"):
            require(case.get(field) is not None and case[field] == reference[field], f"{role}: differing {field}")
        require(len(case["_samples"]) == len(reference["_samples"]), f"{role}: differing sample count")
        if role.startswith("serial"):
            continue
        _validate(case, role)  # Normal Post -> Index -> CQ, not wait-first.
        require(_tuning(case) == _tuning(reference), f"{role}: differing tuning")
        for field in ("ordered_output_sha256", "selected_set_sha256"):
            value = case.get("selection_comparison", {}).get(field)
            require(value is not None and value == reference["selection_comparison"].get(field),
                    f"{role}: changed TopK/output")
        require([row["mode"] for row in case["auto_route"]["samples"]]
                == [row["mode"] for row in reference["auto_route"]["samples"]],
                f"{role}: changed route decisions")
        require(_low_traffic(case) == _low_traffic(reference), f"{role}: changed LOW Get traffic")
    for label in ("serial", "before", "after"):
        start, end = cases[label], cases[label + "-end"]
        require(start["kvip_commit"] == end["kvip_commit"] and _tuning(start) == _tuning(end),
                f"{label}: first/end source or policy differs")
        require(start.get("library_sha256") == end.get("library_sha256"), f"{label}: first/end binary differs")
        if label != "serial":
            require(start.get("library_sha256"), f"{label}: missing binary digest")

    # Required before/after probes are separate from the headline samples.
    # Validate them before printing any performance or PASS claims.
    for role in ("before", "after"):
        diagnostics = cases[role].get("native_diagnostics", {})
        require(diagnostics.get("included_in_timing") is False
                and len(diagnostics.get("samples", [])) == 2, f"{role}: two separate diagnostics required")
        clock = diagnostics.get("counter_calibration", {})
        require(clock.get("status") == "measured" and clock.get("included_in_timing") is False,
                f"{role}: counter calibration required")
        checked = calibration_summary(clock.get("samples", []))
        require(abs(checked["cycles_per_us"] - clock.get("cycles_per_us", 0))
                <= checked["cycles_per_us"] * 1e-9, f"{role}: invalid counter calibration")

    print(f"RESULT_BEGIN G{group} units=us complete_graph=true mean/p50_separate=true")
    for role, case in cases.items():
        all_time, steady = timing(case, False), timing(case, True)
        traffic = case["reads_samples"][1:]
        route = case.get("auto_route", {})
        modes = f"{route['low_steps']}/{route['high_steps']}" if route else "-"
        print(f"{role:11} ALL_mean/p50={all_time[0]:.3f}/{all_time[1]:.3f} "
              f"STEADY_mean/p50={steady[0]:.3f}/{steady[1]:.3f} "
              f"Get={mean(row['transfers'] for row in traffic):.1f} "
              f"unused={mean(row['unused_prefetch'] for row in traffic):.1f} low/high={modes}")
    for candidate in ("after", "after-end"):
        gains = " ".join(f"vs_{control}={pair(change(timing(cases[control], True), timing(cases[candidate], True)))}"
                         for control in ("before", "before-end", "serial", "serial-end"))
        print(f"GAIN {candidate} STEADY_mean/p50 {gains}")
    for label in ("serial", "before", "after"):
        drift = tuple(-value for value in change(timing(cases[label], True), timing(cases[label + '-end'], True)))
        print(f"DRIFT {label} end/first STEADY_mean/p50={pair(drift)} unadjusted=true")
    for role in ("before", "after"):
        diagnostics = cases[role].get("native_diagnostics", {})
        clock = diagnostics["counter_calibration"]
        print(f"CLOCK {role} cycles_per_us={clock['cycles_per_us']:.3f} "
              "approximate=true headline_excluded=true")
        for number, sample in enumerate(diagnostics.get("samples", []), 1):
            phase = sample.get("reads", {}).get("post_index", {})
            if not phase.get("active"):
                print(f"PHASE_END {role} sample={number} LOW_not_entered=true")
                continue
            require(diagnostics.get("included_in_timing") is False, f"{role}: diagnostics entered headline")
            ends = phase.get("phase_end_from_first_begin_cycles", {})
            fields = " ".join(f"{name}={ends.get(name)}" for name in ("plan", "pack", "post_begin", "post", "index", "cq"))
            work = phase.get("index_work", {})
            intervals = work.get("per_core_cycles", {})
            detail = " ".join(f"{name}_p50/max={intervals[name]['p50']:g}/{intervals[name]['max']}"
                              for name in ("index_work", "index_barrier", "cq_wait") if name in intervals)
            print(f"PHASE_END {role} sample={number} cumulative_cycles {fields}")
            print(f"INDEX_CQ {role} sample={number} cycles {detail}")
    candidate_times = [timing(cases[role], True) for role in ("after", "after-end")]
    serial_times = [timing(cases[role], True) for role in ("serial", "serial-end")]
    parity = all(all(c <= s for c, s in zip(candidate, serial))
                 for candidate in candidate_times for serial in serial_times)
    goal = all(all(c <= .65 * s for c, s in zip(candidate, serial))
               for candidate in candidate_times for serial in serial_times)
    print("CHECK correctness=PASS lifecycle=PASS prediction_recovery=PASS identical_low_traffic=PASS")
    print(f"OBSERVED all_brackets_mean_and_p50_parity={str(parity).lower()} reduction_ge35pct={str(goal).lower()}")
    print("SCOPE snapshot_restored_each_step=true device_learning_in_timing=true phase_ends_not_additive=true")
    print(f"RESULT_END G{group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    try:
        summarize(args.directory, args.group)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"SUMMARY_FAILED G{args.group}: {error}") from None


if __name__ == "__main__":
    main()
