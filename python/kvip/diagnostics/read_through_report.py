# SPDX-License-Identifier: Apache-2.0
"""Validate and print one same-run serial/control/read-through comparison.

Inputs are completed kvip.diagnostics.replay reports named gN-serial.json,
gN-before.json, gN-read-through.json, gN-before-end.json and gN-serial-end.json
in the same result directory. Each fused report must contain one case.
No result is accepted merely because its filename contains PASS or a case name.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


ROLES = ("serial", "before", "read-through", "before-end", "serial-end")
LIFECYCLE = (
    "learning-disabled", "duplicate", "backward", "identity-rejected", "resume",
    "new-generation", "old-generation", "new-generation-resume", "packed-row-reorder",
    "restore-row-order", "prediction_fault_injection_low", "prediction_fault_recovery_low",
    "prediction_fault_injection_high", "prediction_fault_recovery_high",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-6)


def load_case(value, role):
    path, sep, name = value.partition("::")
    report = json.loads(Path(path).read_text())
    require(report.get("status") == "pass", f"{role}: report did not pass")
    if "cases" in report:
        cases = report["cases"]
        require(isinstance(cases, dict) and bool(cases), f"{role}: empty case report")
        if not sep:
            require(len(cases) == 1, f"{role}: use FILE::case-name for a multi-case report")
            name = next(iter(cases))
        require(name in cases, f"{role}: missing case {name}")
        case = cases[name]
    else:
        require(not sep, f"{role}: case selector supplied for a flat report")
        case = report
    require(case.get("status") == "pass" and case.get("transport") == "urma", f"{role}: not a passing URMA case")
    expected_mode = "serial" if role.startswith("serial") else "fused"
    require(case.get("mode") == expected_mode, f"{role}: unexpected execution mode")
    source = case.get("production_source", {})
    require(source.get("kind") == "exact_upstream_checkout" and source.get("commit") == case.get("kvip_commit")
            and isinstance(source.get("commit"), str) and len(source["commit"]) == 40,
            f"{role}: missing pinned production-source provenance")
    timing = case.get("timing", {})
    samples = timing.get("samples_us", [])
    require(len(samples) >= 2 and timing.get("count") == len(samples), f"{role}: incomplete timing samples")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples), f"{role}: invalid timing samples")
    require(close(mean(samples), timing["mean_us"]) and close(median(samples), timing["p50_us"]),
            f"{role}: timing summary differs from raw samples")
    reads = case.get("reads_samples", [])
    require(len(reads) == len(samples), f"{role}: missing per-iteration traffic")
    width = case.get("geometry", {}).get("token_bytes")
    require(type(width) is int and width > 0, f"{role}: invalid token width")
    for row in reads:
        require(all(type(row.get(key)) in (int, float) and row[key] >= 0
                    for key in ("prefetch", "final", "transfers", "bytes", "unused_prefetch", "useful_transfers")),
                f"{role}: invalid traffic counters")
        require(close(row["prefetch"] + row["final"], row["transfers"])
                and close(row["transfers"] * width, row["bytes"])
                and close(row["transfers"] - row["unused_prefetch"], row["useful_transfers"]),
                f"{role}: inconsistent traffic counters")
    if expected_mode == "serial":
        expected = dict(runtime_workers=32, batch_get_post_size=128,
                        batch_get_max_inflight_total=2048, jettys_per_core=2)
        require(all(case.get(key) == val for key, val in expected.items()), f"{role}: not the configured strong serial baseline")
    else:
        validate_auto(case, role)
    case["_samples"], case["_path"] = samples, str(Path(path))
    return case


def validate_auto(case, role):
    tuning, route = case.get("tuning", {}), case.get("auto_route", {})
    require(tuning.get("auto_route") is True, f"{role}: automatic routing disabled")
    require(route.get("status") == "pass" and route.get("capture_and_warmup_learning") is False
            and route.get("decision_and_feedback_in_timing") is True
            and route.get("snapshot_restored_each_step") is True, f"{role}: wrong automatic timing contract")
    samples, reads = route.get("samples", []), case["reads_samples"]
    require(len(samples) == len(reads), f"{role}: missing route observations")
    require(route.get("low_steps") == sum(row.get("mode") == "low" for row in samples)
            and route.get("high_steps") == sum(row.get("mode") == "high" for row in samples)
            and route["low_steps"] + route["high_steps"] == len(samples), f"{role}: route counts disagree")
    for observed, read in zip(samples, reads):
        require(observed.get("observed_prefetch") == read["prefetch"]
                and observed.get("observed_final") == read["final"]
                and observed.get("observed_useful_get") == read["useful_transfers"]
                and observed.get("necessary_get") == read["useful_transfers"], f"{role}: route/Get accounting differs")
        require(observed["mode"] != "low" or read["prefetch"] == 0, f"{role}: LOW issued speculative reads")
    life = route.get("lifecycle", {})
    require(life.get("status") == "pass" and life.get("graph_captures") == 1
            and life.get("captured_addresses_unchanged") is True
            and life.get("feedback_reset") is False
            and life.get("real_packed_row_reorder") is True
            and life.get("prediction_fault_injection") is True, f"{role}: --check-auto-route was not completed")
    rows = life.get("sequence", [])
    require([row.get("name") for row in rows] == list(LIFECYCLE), f"{role}: incomplete lifecycle sequence")
    by_name = {row["name"]: row for row in rows}
    for name in ("learning-disabled", "duplicate", "backward", "identity-rejected", "old-generation"):
        require(by_name[name].get("learned_owners") == [], f"{role}: rejected step learned feedback")
    require(by_name["prediction_fault_injection_low"].get("mode") == "low"
            and by_name["prediction_fault_injection_high"].get("mode") == "high", f"{role}: prediction injection did not exercise both paths")
    require(case.get("score_comparison", {}).get("status") == "bit_exact", f"{role}: LI scores not verified")


def validate_candidate(case):
    tuning, meta = case["tuning"], case.get("read_through", {})
    require(tuning.get("low_miss_read_through") is True, "candidate: low_miss_read_through is not enabled")
    require(not any(tuning.get(key) for key in ("auto_route_simt", "low_miss_compact_plan", "low_miss_direct_index")),
            "candidate: incompatible promotion/SIMT experiments enabled")
    original, actual = meta.get("original_hot_abc_stride"), meta.get("output_stride")
    require(type(original) is int and type(actual) is int and actual - original == 8192
            and meta.get("scratch_slots_per_owner") == 8192, "candidate: temporary arena/ABC contract missing")
    geometry = case["geometry"]
    require(original == geometry["output_stride"]
            and meta.get("scratch_bytes") == geometry["owners"] * 8192 * geometry["token_bytes"],
            "candidate: scratch byte count differs from owner/stride geometry")
    frozen = meta.get("low_persistent_check", {}) or {}
    require(frozen.get("status") == "pass" and frozen.get("persistent_index_unchanged") is True
            and frozen.get("hot_abc_bytes_unchanged") is True and frozen.get("scratch_slots") == 8192,
            "candidate: LOW persistent state was not verified unchanged")
    carry = case["auto_route"]["lifecycle"].get("persistent_carry", {})
    require(carry.get("status") == "pass" and carry.get("graph_captures") == 1
            and carry.get("snapshot_restores") == 1 and carry.get("persistent_reset_between_steps") is False
            and carry.get("captured_addresses_unchanged") is True, "candidate: non-restored carry check failed/missing")
    rows = carry.get("sequence", [])
    require([row.get("name") for row in rows] == ["low", "low-repacked", "high-after-low"]
            and [row.get("mode") for row in rows] == ["low", "low", "high"]
            and all(row.get("persistent_state_verified") is True for row in rows),
            "candidate: incomplete low/repacked-low/high carry validation")


def timing(case, steady):
    values = case["_samples"][1:] if steady else case["_samples"]
    return mean(values), median(values)


def change(base, candidate):
    return tuple(100 * (a - b) / a for a, b in zip(base, candidate))


def pair(values):
    return "/".join(f"{value:+.2f}%" for value in values)


def summarize(args):
    directory = Path(args.directory)
    cases = {role: load_case(str(directory / f"g{args.group}-{role}.json"), role) for role in ROLES}
    candidate, first = cases["read-through"], cases["serial"]
    validate_candidate(candidate)
    for role, case in cases.items():
        for field in ("snapshot_sha256", "kvaf_build", "device", "soc", "warmup"):
            require(case.get(field) is not None and case.get(field) == first.get(field), f"{role}: differing {field}")
        require(case["geometry"] == first["geometry"] and len(case["_samples"]) == len(first["_samples"]),
                f"{role}: differing load or sample count")
    for start, end in (("serial", "serial-end"), ("before", "before-end")):
        require(cases[start].get("kvip_commit") == cases[end].get("kvip_commit"), f"{start}: first/end source differs")
        # This field is the last observed device decision, not build policy.
        tunings = [{key: value for key, value in cases[role].get("tuning", {}).items()
                    if key != "auto_low_active"} for role in (start, end)]
        require(tunings[0] == tunings[1], f"{start}: first/end tuning differs")
    for role in ("before", "before-end"):
        require(not cases[role]["tuning"].get("low_miss_read_through"), f"{role}: control unexpectedly uses read-through")
        for field in ("selected_set_sha256", "ordered_output_sha256"):
            require(candidate.get("selection_comparison", {}).get(field)
                    == cases[role].get("selection_comparison", {}).get(field)
                    and candidate.get("selection_comparison", {}).get(field) is not None,
                    f"{role}: fused output selection differs")
    print(f"RESULT_BEGIN G{args.group} URMA BS={first['geometry']['batch']} token_bytes={first['geometry']['token_bytes']}")
    print("单位us；每列mean/p50。STEADY统一剔除各case首个计时样本；ALL保留冷启动。")
    print("case             ALL_mean/p50       STEADY_mean/p50   steady_Get  steady_MiB  low/high")
    for role, case in cases.items():
        all_mean, all_p50 = timing(case, False)
        steady_mean, steady_p50 = timing(case, True)
        reads = case["reads_samples"][1:]
        route = case.get("auto_route", {})
        modes = f"{route['low_steps']}/{route['high_steps']}" if route else "-"
        print(f"{role:16} {all_mean:8.3f}/{all_p50:8.3f} {steady_mean:8.3f}/{steady_p50:8.3f} "
              f"{mean(row['transfers'] for row in reads):10.1f} {mean(row['bytes'] for row in reads)/2**20:10.3f} {modes}")
    for steady in (False, True):
        label = "STEADY" if steady else "ALL"
        gains = [f"vs_{role}={pair(change(timing(cases[role], steady), timing(candidate, steady)))}"
                 for role in ("serial", "serial-end", "before", "before-end")]
        print(f"GAIN_{label} mean/p50 " + " ".join(gains))
    for start, end in (("serial", "serial-end"), ("before", "before-end")):
        all_drift = tuple(-x for x in change(timing(cases[start], False), timing(cases[end], False)))
        steady_drift = tuple(-x for x in change(timing(cases[start], True), timing(cases[end], True)))
        print(f"DRIFT {start} end/first ALL_mean/p50={pair(all_drift)} STEADY_mean/p50={pair(steady_drift)} 未校正")
    rt = candidate["read_through"]
    print(f"CHECK correctness=PASS lifecycle=PASS prediction_recovery=PASS low_persistent=UNCHANGED "
          f"one_graph_carry=PASS scratch_slots/owner=8192 scratch_MiB={rt['scratch_bytes']/2**20:.3f}")
    low_parity = all(all(c <= b for c, b in zip(timing(candidate, True), timing(cases[role], True)))
                     for role in ("serial", "serial-end"))
    goal = all(all(c <= .65*b for c, b in zip(timing(candidate, True), timing(cases[role], True)))
               for role in ("serial", "serial-end"))
    print(f"OBSERVED_vs_both_serial_controls steady_mean_and_p50_parity={str(low_parity).lower()} "
          f"latency_reduction_ge35pct={str(goal).lower()}")
    print("边界：冻结同一输入；LOW不晋升KV。carry仅验证状态正确，未证明后续100轮缓存/性能收益。")
    print(f"RESULT_END G{args.group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    try:
        summarize(args)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"SUMMARY_FAILED G{args.group}: {error}") from None


if __name__ == "__main__":
    main()
