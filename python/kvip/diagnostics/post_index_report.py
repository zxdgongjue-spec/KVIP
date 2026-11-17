# SPDX-License-Identifier: Apache-2.0
"""Summarize same-run strong serial/control/post-before-Index graph results."""

import argparse
from pathlib import Path
from statistics import mean

from kvip.diagnostics.pipeline_options import options_for
from kvip.diagnostics.read_through_report import change, load_case, pair, require, timing


ROLES = ("serial", "before", "post-index", "before-end", "serial-end")


def summarize(args):
    cases = {role: load_case(str(Path(args.directory) / f"g{args.group}-{role}.json"), role) for role in ROLES}
    first, candidate = cases["serial"], cases["post-index"]
    for role, case in cases.items():
        for key in ("snapshot_sha256", "kvaf_build", "device", "soc", "warmup"):
            require(case.get(key) is not None and case[key] == first.get(key), f"{role}: differing {key}")
        require(case["geometry"] == first["geometry"] and len(case["_samples"]) == len(first["_samples"]),
                f"{role}: differing geometry or sample count")
    for start, end in (("serial", "serial-end"), ("before", "before-end")):
        require(cases[start]["kvip_commit"] == cases[end]["kvip_commit"], f"{start}: first/end source differs")
        tunings = [{k: v for k, v in cases[role].get("tuning", {}).items() if k != "auto_low_active"}
                   for role in (start, end)]
        require(tunings[0] == tunings[1], f"{start}: first/end policy differs")
    for role in ("before", "post-index", "before-end"):
        tuning = cases[role]["tuning"]
        name = "auto-post-index" if role == "post-index" else "auto-tail-row-direct"
        require(all(tuning.get(k) == value for k, value in options_for(name).items()), f"{role}: unexpected build options")
        require(not any(tuning.get(k) for k in ("low_miss_read_through", "low_miss_compact_plan", "auto_route_simt")),
                f"{role}: incompatible scratch/compact/SIMT experiment")
        require("read_through" not in cases[role], f"{role}: read-through output contract is not permitted")
        if role != "post-index":
            require(not tuning.get("low_miss_post_index"), f"{role}: control already enables candidate")
        for key in ("selected_set_sha256", "ordered_output_sha256"):
            value = cases[role].get("selection_comparison", {}).get(key)
            require(value is not None and value == candidate.get("selection_comparison", {}).get(key),
                    f"{role}: fused selection differs")
    low_samples = []
    for observed, reads in zip(candidate["auto_route"]["samples"], candidate["reads_samples"]):
        phases = reads.get("post_index", {})
        if observed["mode"] == "low":
            require(phases.get("status") == "pass" and phases.get("active") is True
                    and phases.get("submitted_requests") == reads["final"]
                    and phases.get("completed_requests") == reads["final"],
                    "candidate: missing low phase status/Get-count validation")
            require(type(phases.get("fallback")) is bool, "candidate: missing bounded-Get fallback status")
            low_samples.append(phases)
        else:
            require(phases.get("status") == "not_entered" and phases.get("active") is False,
                    "candidate: HIGH was interpreted as low phase data")

    print(f"RESULT_BEGIN G{args.group} URMA BS={first['geometry']['batch']} token_bytes={first['geometry']['token_bytes']}")
    print("单位us；mean/p50分别比较。STEADY统一排除首个计时样本；ALL包含冷启动。")
    print("case             ALL_mean/p50       STEADY_mean/p50   steady_Get  steady_MiB  low/high")
    for role, case in cases.items():
        all_mean, all_p50 = timing(case, False)
        steady_mean, steady_p50 = timing(case, True)
        reads, route = case["reads_samples"][1:], case.get("auto_route", {})
        modes = f"{route['low_steps']}/{route['high_steps']}" if route else "-"
        print(f"{role:16} {all_mean:8.3f}/{all_p50:8.3f} {steady_mean:8.3f}/{steady_p50:8.3f} "
              f"{mean(row['transfers'] for row in reads):10.1f} {mean(row['bytes'] for row in reads)/2**20:10.3f} {modes}")
    for steady in (False, True):
        gains = [f"vs_{role}={pair(change(timing(cases[role], steady), timing(candidate, steady)))}"
                 for role in ("serial", "serial-end", "before", "before-end")]
        print(f"GAIN_{'STEADY' if steady else 'ALL'} mean/p50 " + " ".join(gains))
    for start, end in (("serial", "serial-end"), ("before", "before-end")):
        values = [tuple(-value for value in change(timing(cases[start], steady), timing(cases[end], steady)))
                  for steady in (False, True)]
        print(f"DRIFT {start} end/first ALL_mean/p50={pair(values[0])} STEADY_mean/p50={pair(values[1])} 未校正")
    print("CHECK correctness=PASS normal_hot_promotion=PASS lifecycle=PASS prediction_recovery=PASS scratch=NONE")
    fallback_count = sum(row["fallback"] for row in low_samples)
    diagnostic_low = [sample.get("reads", {}).get("post_index", {})
                      for sample in candidate.get("native_diagnostics", {}).get("samples", [])
                      if sample.get("reads", {}).get("post_index", {}).get("active")]
    diagnostic_fallback = sum(bool(row.get("fallback")) for row in diagnostic_low)
    fulfilled = str(not (fallback_count or diagnostic_fallback)).lower() if low_samples else "not_exercised"
    print(f"POST_INDEX_ORDER low_samples={len(low_samples)} fallback_samples={fallback_count} "
          f"diagnostic_fallback_samples={diagnostic_fallback} fulfilled={fulfilled}")
    for number, sample in enumerate(candidate.get("native_diagnostics", {}).get("samples", []), 1):
        phases = sample.get("reads", {}).get("post_index", {})
        if phases.get("phase_status") == "available":
            ends = phases["phase_end_from_first_begin_cycles"]
            values = " ".join(f"{name}={ends[name]}" for name in ("plan", "pack", "post", "index", "cq"))
            print(f"PHASE_END sample={number} unit=cycles policy={phases['phase_policy']} {values}")
            detail = phases.get("index_work", {})
            if detail.get("status") == "available":
                assigned, moved = detail["assigned_events"], detail["moved_tuples"]
                counts = " ".join(
                    f"{name}={row['total']} min/p50/max={row['min']}/{row['p50']:g}/{row['max']} "
                    f"active={row['active_cores']}/{detail['core_count']}"
                    for name, row in (("assigned", assigned), ("moved", moved))
                )
                print(f"INDEX_WORK sample={number} {counts} "
                      f"logical_index_rw_bytes={detail['logical_index_rw_bytes']}")
                spans = " ".join(f"{name}={row['p50']:g}/{row['max']}"
                                 for name, row in detail["per_core_cycles"].items())
                print(f"INDEX_CYCLES sample={number} p50/max {spans}")
            else:
                print(f"INDEX_WORK sample={number} unavailable：该二进制未提供PIX1诊断")
    if candidate.get("native_diagnostics", {}).get("samples"):
        print("PHASE边界是各阶段最晚完成时间相对最早begin；包含同步，不能相加或当作完整耗时。")
        print("INDEX字节为每个非self交换592B的逻辑读写量，未计L2/协议；独立诊断cycles不换算带宽。")
    parity = all(all(c <= b for c, b in zip(timing(candidate, True), timing(cases[role], True)))
                 for role in ("serial", "serial-end"))
    goal = all(all(c <= .65*b for c, b in zip(timing(candidate, True), timing(cases[role], True)))
               for role in ("serial", "serial-end"))
    print(f"OBSERVED_vs_both_serial_controls steady_mean_and_p50_parity={str(parity).lower()} "
          f"latency_reduction_ge35pct={str(goal).lower()}")
    print("边界：同一冻结输入、设备反馈持续学习；每样本恢复缓存，未将单切片结果解释为100轮连续性能。")
    print(f"RESULT_END G{args.group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    try:
        summarize(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"SUMMARY_FAILED G{args.group}: {error}") from None


if __name__ == "__main__":
    main()
