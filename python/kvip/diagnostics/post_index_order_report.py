# SPDX-License-Identifier: Apache-2.0
"""Compare complete Post/CQ/Index and Post/Index/CQ graphs from one binary."""

import argparse
from pathlib import Path

from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.pipeline_options import options_for
from kvip.diagnostics.read_through_report import change, load_case, pair, require, timing


ROLES = ("wait-first", "overlap", "overlap-end", "wait-first-end")


def _validate(case, role):
    expected = role.startswith("wait-first")
    require(case.get("post_index_wait_first") is expected, f"{role}: missing/mismatched execution order")
    tuning = case["tuning"]
    require(all(tuning.get(k) == v for k, v in options_for("auto-post-index").items()),
            f"{role}: not the complete Post-Index implementation")
    require(not any(tuning.get(k) for k in ("low_miss_read_through", "low_miss_compact_plan", "auto_route_simt")),
            f"{role}: incompatible scratch/compact experiments")
    low = []
    for observed, reads in zip(case["auto_route"]["samples"], case["reads_samples"]):
        detail = reads.get("post_index", {})
        if observed["mode"] == "high":
            require(detail.get("status") == "not_entered" and detail.get("active") is False,
                    f"{role}: stale low workspace interpreted on high route")
            continue
        require(detail.get("status") == "pass" and detail.get("active") is True
                and detail.get("wait_first") is expected
                and detail.get("submitted_requests") == detail.get("completed_requests") == reads["final"],
                f"{role}: missing checked low execution/counts")
        require(type(detail.get("fallback")) is bool, f"{role}: missing fallback status")
        low.append(detail)
    return low


def _print_diagnostics(case, role):
    diagnostics = case.get("native_diagnostics", {})
    clock = diagnostics.get("counter_calibration", {})
    require(clock.get("status") == "measured" and clock.get("included_in_timing") is False,
            f"{role}: --calibrate-counter was not completed")
    checked = calibration_summary(clock.get("samples", []))
    rate = checked["cycles_per_us"]
    require(abs(rate - clock.get("cycles_per_us", 0)) <= rate * 1e-9,
            f"{role}: invalid clock summary")
    print(f"CLOCK {role} measured_cycles_per_us={rate:.3f} "
          f"range={checked['rate_min']:.3f}-{checked['rate_max']:.3f} headline_excluded=true")
    rows = diagnostics.get("samples", [])
    require(rows and diagnostics.get("included_in_timing") is False, f"{role}: missing separate diagnostics")
    for number, row in enumerate(rows, 1):
        phase = row.get("reads", {}).get("post_index", {})
        if not phase.get("active"):
            print(f"INDEX_DIAG {role} sample={number} low_not_entered")
            continue
        work = phase.get("index_work", {})
        require(work.get("status") == "available" and work.get("abi") == "PIX2"
                and work.get("index_implementation") == "headline_same_VF", f"{role}: PIX2 same-VF markers required")
        require(phase.get("wait_first") == role.startswith("wait-first"), f"{role}: wrong diagnostic order")
        assigned, moved = work["assigned_events"], work["moved_tuples"]
        intervals = work["per_core_cycles"]
        fields = " ".join(f"{name}_p50/max_cy={intervals[name]['p50']:g}/{intervals[name]['max']}"
                          for name in ("index_work", "index_barrier", "cq_wait", "cq_barrier") if name in intervals)
        approximate = " ".join(f"{name}_p50/max_us_approx={intervals[name]['p50']/rate:.3f}/{intervals[name]['max']/rate:.3f}"
                               for name in ("index_work", "cq_wait") if name in intervals)
        spans = work["global_span_cycles"]
        print(f"INDEX_DIAG {role} sample={number} assigned/moved={assigned['total']}/{moved['total']} "
              f"moved_min/p50/max={moved['min']}/{moved['p50']:g}/{moved['max']} "
              f"index_span_us_approx={spans['index_work']/rate:.3f} "
              f"post_join_to_finish_us_approx={spans['post_barrier_to_completion']/rate:.3f} {fields}")
        print(f"CALIBRATED_DIAG {role} sample={number} "
              f"index_scope={'current_impl_isolated' if role.startswith('wait-first') else 'concurrent_with_Get'} "
              f"{approximate}")
        print(f"ORDER_DIAG {role} sample={number} policy={phase['phase_policy']} "
              f"blocking_fallback={str(phase['fallback']).lower()} "
              f"logical_index_rw_bytes={work['logical_index_rw_bytes']}")


def summarize(args):
    root = Path(args.directory)
    cases = {role: load_case(str(root / f"g{args.group}-{role}.json"), role) for role in ROLES}
    first = cases["wait-first"]
    lows = {}
    for role, case in cases.items():
        lows[role] = _validate(case, role)
        for key in ("snapshot_sha256", "kvaf_build", "device", "soc", "warmup", "geometry", "kvip_commit", "library_sha256"):
            require(case.get(key) is not None and case[key] == first.get(key), f"{role}: differing {key}")
        require(case["library_sha256"], f"{role}: binary digest is required")
        require(len(case["_samples"]) == len(first["_samples"]), f"{role}: differing sample count")
        for key in ("ordered_output_sha256", "selected_set_sha256"):
            require(case["selection_comparison"].get(key) is not None
                    and case["selection_comparison"][key] == first["selection_comparison"].get(key),
                    f"{role}: different exact TopK selection")
        require([row["mode"] for row in case["auto_route"]["samples"]]
                == [row["mode"] for row in first["auto_route"]["samples"]], f"{role}: route decisions changed")
        require([[row[key] for key in ("prefetch", "final", "transfers", "bytes")]
                 for row in case["reads_samples"]]
                == [[row[key] for key in ("prefetch", "final", "transfers", "bytes")]
                    for row in first["reads_samples"]], f"{role}: Get traffic changed")
    print(f"RESULT_BEGIN G{args.group} same_binary=true all_Index_and_Get_work=retained units=us")
    for role, case in cases.items():
        all_mean, all_p50 = timing(case, False)
        steady_mean, steady_p50 = timing(case, True)
        route = case["auto_route"]
        print(f"{role:16} ALL_mean/p50={all_mean:.3f}/{all_p50:.3f} "
              f"STEADY_mean/p50={steady_mean:.3f}/{steady_p50:.3f} "
              f"low/high={route['low_steps']}/{route['high_steps']} "
              f"fallback={sum(row['fallback'] for row in lows[role])}/{len(lows[role])}")
    for candidate in ("overlap", "overlap-end"):
        for control in ("wait-first", "wait-first-end"):
            for steady in (False, True):
                a, b = timing(cases[control], steady), timing(cases[candidate], steady)
                delta = tuple(x-y for x, y in zip(a, b))
                print(f"ORDER_GAIN {candidate}_vs_{control} {'STEADY' if steady else 'ALL'} "
                      f"mean/p50={pair(change(a, b))} saved_us={delta[0]:.3f}/{delta[1]:.3f}")
    for role in ("wait-first", "overlap"):
        drift = tuple(-x for x in change(timing(cases[role], True), timing(cases[role+'-end'], True)))
        print(f"DRIFT {role} end/first STEADY_mean/p50={pair(drift)} unadjusted=true")
    references = {}
    for label in ("serial", "serial-end", "before"):
        path = getattr(args, label.replace("-", "_"), None)
        if path is None:
            continue
        reference = references[label] = load_case(str(path), label)
        for key in ("snapshot_sha256", "kvaf_build", "device", "soc", "warmup", "geometry"):
            require(reference.get(key) == first.get(key), f"{label}: differing {key}")
        print(f"REFERENCE {label} ALL_mean/p50={timing(reference, False)[0]:.3f}/{timing(reference, False)[1]:.3f} "
              f"STEADY_mean/p50={timing(reference, True)[0]:.3f}/{timing(reference, True)[1]:.3f}")
        for role in ("overlap", "overlap-end"):
            print(f"GAIN_vs_{label} {role} STEADY_mean/p50={pair(change(timing(reference, True), timing(cases[role], True)))}")
    if "serial" in references and "serial-end" in references:
        start, end = references["serial"], references["serial-end"]
        require(start["kvip_commit"] == end["kvip_commit"], "Serial first/end source differs")
        drift = tuple(-x for x in change(timing(start, True), timing(end, True)))
        print(f"DRIFT serial end/first STEADY_mean/p50={pair(drift)} unadjusted=true")
    if "before" in references:
        print("REFERENCE_SCOPE before=single_point_not_bracketed")
    for role, case in cases.items():
        _print_diagnostics(case, role)
    fallback = (any(row["fallback"] for rows in lows.values() for row in rows)
                or any(row.get("reads", {}).get("post_index", {}).get("fallback", False)
                       for case in cases.values() for row in case.get("native_diagnostics", {}).get("samples", [])))
    print(f"CHECK correctness=PASS lifecycle=PASS prediction_recovery=PASS identical_traffic=PASS "
          f"async_order_exercised={str(bool(lows['wait-first']) and not fallback).lower()}")
    print("解释：完整图的同次序对照差值才是净掩盖收益；校准后的Index跨度含核间启动差异，不能当作Index理论极限。")
    print("PIX2与headline使用相同Index VF；GetSystemCycle边界仍有诊断扰动。逻辑字节不是HBM总线流量。")
    print(f"RESULT_END G{args.group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--serial", type=Path)
    parser.add_argument("--serial-end", type=Path)
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    try:
        summarize(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"SUMMARY_FAILED G{args.group}: {error}") from None


if __name__ == "__main__":
    main()
