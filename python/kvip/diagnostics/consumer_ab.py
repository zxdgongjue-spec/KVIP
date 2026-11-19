# SPDX-License-Identifier: Apache-2.0
"""Host20: bracket real Update on the original eight-window graph with the same transport and snapshots."""

import argparse
from collections import deque
from copy import copy
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.high_index_speed import one_to_one, two
from kvip.diagnostics.high_index_study import GROUPS, replay_command
from kvip.diagnostics.original_native_window import print_original_clock
from kvip.diagnostics.read_through_report import load_case, require, timing


FIXED_FUSED_REVISION = "4cb3917d1861776a396e0f1a402e6b940fcabed4"
ARMS = ("before", "after", "after-end", "before-end")
OVERLAP_ARMS = ("original", *ARMS, "original-end")
THREE_SOURCE_ARMS = ("original", "before", "middle", "after", "after-end", "middle-end", "before-end", "original-end")


def overlap_sources(baseline, candidate, intermediate):
    return ({"before": baseline, "middle": intermediate, "after": candidate}
            if intermediate else {"before": baseline, "after": candidate})


def command(args, group, arm, *, profile=False):
    run = copy(args)
    if not arm.startswith("after"):
        run.revision = {"original": FIXED_FUSED_REVISION, "before": args.baseline,
                        "middle": getattr(args, "intermediate", None)}[arm.split("-")[0]]
        run.source = args.cache / "adaptive-sources" / f"auto-{run.revision[:8]}"
    cmd = replay_command(run, group, "original")
    cmd[cmd.index("--output") + 1] = str(args.result_dir / f"g{group}-{arm}.json")
    cmd.append("--verify-consumer-work")
    if profile:
        if not args.high_overlap or arm not in ("original", "after") or group != 7:
            raise ValueError("Kernel accounting requires G7 original and candidate HIGH")
        if arm == "after":
            cmd.append("--high-post-index")
        cmd[cmd.index("--iterations") + 1] = "2"
        cmd[cmd.index("--output") + 1] = str(args.result_dir / f"g7-{arm}-level0.json")
        cmd.extend(("--diagnostic-replays", "0", "--profile-graph-event", "--profile-metrics", "none",
                    "--profile-dir", str(args.result_dir / f"g7-{arm}-level0")))
        return cmd
    if getattr(args, "high_overlap", False):
        # Every headline uses the common Post/Wait completion ABI. Stage counters
        # belong to the separate replay and are decoded by their on-device marker.
        if not arm.startswith("original"):
            cmd.append("--high-post-index")
        if arm in ("before", "middle", "after"):
            cmd.extend(("--diagnostic-replays", "1", "--calibrate-counter"))
    elif arm in ("before", "after"):
        cmd.extend(("--diagnostic-replays", "2", "--calibrate-counter"))
    if arm == "original" and args.high_overlap:
        cmd.extend(("--diagnostic-replays", "1", "--calibrate-counter"))
    return cmd


def work_key(read):
    work = read["consumer_work"]
    return (work["event_sha256"], work["descriptor_sha256"], work["event_count"],
            read["prefetch"], read["final"], read["unused_prefetch"])


def size_near_pairs(cases, control):
    """Match on four counts and the steady step, never on duration or by reusing samples."""
    def counts(sample):
        return (sample["prefetch"], sample["final"], sample["unused_prefetch"], sample["consumer_work"]["event_count"])

    deltas = []
    for suffix in ("", "-end"):
        left, right = cases[control + suffix], cases["after" + suffix]
        for i in range(1, len(left["_samples"])):
            a = counts(left["reads_samples"][i])
            b = counts(right["reads_samples"][i])
            if all(abs(x-y) <= max(1, min(x, y)*0.02) for x, y in zip(a, b)):
                deltas.append(left["_samples"][i] - right["_samples"][i])
    return deltas


def diagnostic_wait(case):
    diag = case["native_diagnostics"]
    clock = diag["counter_calibration"]
    require(clock["status"] == "measured" and not clock["included_in_timing"] and not diag["included_in_timing"]
            and len(diag["samples"]) == 2, "Missing separate calibrated role telemetry")
    rate = calibration_summary(clock["samples"])["cycles_per_us"]
    max_wait = []
    apply = []
    ends = []
    for row in diag["samples"]:
        require(row["auto_route"]["mode"] == "high", "Expected naturally selected HIGH")
        raw = row["raw"]["fetch"]
        # Row finalizer polls flag 6 after draining its prefetch CQ. This may overlap
        # independent final Get work: the wait is a local dependency, not global exposed latency.
        waits = [b[105] - b[104] for b in raw]
        require(len(raw) == 16 and all(b[104] > 0 for b in raw) and min(waits) >= 0,
                "Unavailable per-row finalizer wait timestamps")
        max_wait.append(max(waits) / rate)
        raw_roles = row["raw"]
        workers = raw_roles["workers_per_query"]
        consumer_ends = [raw_roles["cores"][b * raw_roles["resolver_cores"] + 1][1] for b in range(len(raw))]
        # ReadyBatchIo::Finish publishes the last actual blocking Get return at Q offset 4.
        # A worker without a Get writes zero. The owner's prefetch CQ join is at offset 100.
        get_ends = [b[q * 24 + 4] for b in raw[:workers] for q in range(4)]
        prefetch_joins = [b[100] for b in raw]
        require(min(consumer_ends) > 0 and max(get_ends) > 0 and min(prefetch_joins) > 0,
                "Missing Index or transport endpoint")
        ends.append((max(consumer_ends) - max(max(get_ends), max(prefetch_joins))) / rate)
        item = row["metrics"]["index_apply_total"]
        if item["status"] == "available":
            apply.append(item["max_cycles"] / rate)
    return max_wait, apply, ends


def schedule_format(source):
    kernel = source / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    if "high_post_index::RunRow(" not in (kernel/"streaming.cpp").read_text():
        return "PIH1"
    impl = (kernel/"high_index_async.h").read_text()
    for tag, marker in (("0x52514936", "RQI6"), ("0x52514935", "RQI5"), ("0x52514934", "RQI4"), ("0x52514933", "RQI3"), ("0x52514932", "RQI2"), ("0x52514931", "RQI1")):
        if tag in impl:
            return marker
    raise ValueError(f"Unrecognized row schedule diagnostic ABI at {source}")


def summarize(group, cases, baseline, candidate, high_overlap=False, schedule_formats=("PIH1", "RQI1"), intermediate=None):
    arms = THREE_SOURCE_ARMS if intermediate else (OVERLAP_ARMS if high_overlap else ARMS)
    sources = overlap_sources(baseline, candidate, intermediate)
    require(set(cases) == set(arms), "Missing or unexpected run in the measurement bracket")
    require(not high_overlap or len(schedule_formats) == len(sources), "Missing source schedule ABI")
    arm_sources = {"original": FIXED_FUSED_REVISION, **sources} if high_overlap else sources
    hashes = {}
    base = cases["before"]
    geometry = base["geometry"]
    require((geometry["requested_windows"], geometry["batch"], geometry["token_count"], geometry["token_bytes"])
            == (8, 16, 64, 656), "Expected BS16 Q4, one 656-byte layer and eight windows")
    for arm, c in cases.items():
        for key in ("geometry", "snapshot_sha256", "kvaf_build", "soc", "device", "diagnostics_commit",
                    "warmup", "tuning", "graph_backend"):
            require(base.get(key) is not None and c[key] == base[key], f"{arm}: unmatched {key}")
        require(c["device"] == 0 and c["graph_backend"] == "kvaf.AclGraph" and len(c["_samples"]) == len(c["reads_samples"]) == 10,
                f"{arm}: invalid graph or samples")
        source = arm_sources[arm.split("-")[0]]
        require(c["kvip_commit"] == source and bool(c["library_sha256"])
                and hashes.setdefault(source, c["library_sha256"]) == c["library_sha256"]
                and c["high_post_index"] == (high_overlap and not arm.startswith("original")) and c["post_index_wait_first"] is False,
                f"{arm}: wrong source, binary or production scheduling")
        require([s["mode"] for s in c["auto_route"]["samples"]] == ["high"] * 10
                and c["tuning"]["low_miss_post_index"] and c["tuning"]["auto_route"],
                f"{arm}: expected natural HIGH with the fixed-graph LOW path available")
        for key in ("selected_set_sha256", "ordered_output_sha256"):
            require(c["selection_comparison"][key] and c["selection_comparison"][key] == base["selection_comparison"][key],
                    f"{arm}: different final selection or output")
        verified = c["verified_consumer_work"]
        require(verified["ordered_index_verified"] is True and verified["event_count"] > 0
                and all(verified[k] == c["reads_samples"][-1]["consumer_work"][k]
                        for k in ("event_sha256", "descriptor_sha256", "event_count")),
                f"{arm}: final graph did not verify the ordered persistent Index update")
    require(len(set(hashes.values())) == len(hashes),
            "Comparison arms require distinct binaries")

    times = {arm: timing(cases[arm], True) for arm in arms}
    def diff(a, b):
        return [x - y for x, y in zip(times[a], times[b])]

    saved = [diff(b, a) for b in ("before", "before-end") for a in ("after", "after-end")]
    print(f"G{group} GRAPH_us_mean/p50 " + " ".join(f"{arm}={two(times[arm])}" for arm in arms)
          + f" saved_min={two(map(min, zip(*saved)))} saved_max={two(map(max, zip(*saved)))}"
          + f" drift_old={two(diff('before-end','before'))} drift_new={two(diff('after-end','after'))}")
    records = {}
    for arm in arms:
        c = cases[arm]
        records[arm] = [(work_key(r), t) for r, t in zip(c["reads_samples"][1:], c["_samples"][1:])]
    pairs = []
    for old, new in (("before", "after"), ("before-end", "after-end")):
        pairs.extend(one_to_one(records[old], records[new]))
    delta = [a - b for a, b in pairs]
    reads = ["{:.1f}/{:.1f}".format(*(mean(r[k] for r in cases[a]["reads_samples"][1:]) for k in ("prefetch", "final")))
             for a in arms]
    print(f"G{group} GET_prefetch/final " + " ".join(f"{a}={r}" for a, r in zip(arms, reads))
          + f" identical_work_pairs={len(pairs)}/18 paired_saved_us_mean/p50="
          + (two((mean(delta), median(delta))) if pairs else "unavailable"))
    if high_overlap:
        gains = [diff(o, a) for o in ("original", "original-end") for a in ("after", "after-end")]
        clean_pairs = []
        for o, a in (("original", "after"), ("original-end", "after-end")):
            clean_pairs.extend(one_to_one(records[o], records[a]))
        clean = [a - b for a, b in clean_pairs]
        print(f"G{group} ORIGINAL_per_row_vs_candidate_us saved_min={two(map(min, zip(*gains)))}"
              f" saved_max={two(map(max, zip(*gains)))} drift_original={two(diff('original-end','original'))}"
              f" identical_work_pairs={len(clean)}/18 paired_saved_us_mean/p50="
              + (two((mean(clean), median(clean))) if clean else "unavailable"))
        # Use same-side graphs for paired *brackets*. Individual samples
        # may have different speculative work; exact matches are above.
        reductions = [tuple(100 * (b-c) / b for b, c in zip(times[ref], times[candidate_arm]))
                      for ref in ("original", "original-end") for candidate_arm in ("after", "after-end")]
        first, last = reductions[0], reductions[-1]
        print(f"G{group} FIXED_FUSED source={FIXED_FUSED_REVISION[:8]} policy=original_per_row flag_high_post_index=0"
              f" ref_first/end_mean/p50_us={two(times['original'])},{two(times['original-end'])}"
              f" candidate_first/end_mean/p50_us={two(times['after'])},{two(times['after-end'])}"
              f" reduction_pct_first/end_mean/p50={two(first)},{two(last)}"
              f" all_brackets_reduction_pct_min/max_mean/p50={two(map(min, zip(*reductions)))},{two(map(max, zip(*reductions)))}"
              f" slower_pct_first/end_mean/p50={two(0-v for v in first)},{two(0-v for v in last)}")
        if intermediate:
            middle_gain = [diff(b, m) for b in ("before", "before-end") for m in ("middle", "middle-end")]
            middle = [diff(b, a) for b in ("middle", "middle-end") for a in ("after", "after-end")]
            exact = []
            for suffix in ("", "-end"):
                exact.extend(one_to_one(records["middle" + suffix], records["after" + suffix]))
            delta = [a-b for a, b in exact]
            print(f"G{group} VS_middle_us saved_min={two(map(min, zip(*middle)))} saved_max={two(map(max, zip(*middle)))}"
                  f" middle_vs_before_min={two(map(min, zip(*middle_gain)))} middle_vs_before_max={two(map(max, zip(*middle_gain)))}"
                  f" drift_middle={two(diff('middle-end','middle'))} identical_work_pairs={len(exact)}/18"
                  " paired_saved_us_mean/p50=" + (two((mean(delta), median(delta))) if delta else "unavailable"))
        near = []
        for control in ("original", *tuple(sources)[:-1]):
            delta = size_near_pairs(cases, control)
            value = two((mean(delta), median(delta))) if delta else "unavailable"
            near.append(f"{control}={len(delta)}/18,{value}")
        print(f"G{group} SIZE_ONLY_same_step kept,saved_us_mean/p50 " + " ".join(near)
              + " tolerance_per_count=2pct_or_1 descriptor_identity_unchecked=true")
        for arm in arms:
            if arm.startswith("original"):
                continue
            for row in cases[arm]["reads_samples"]:
                phase = row["high_post_index"]
                require(phase["core_count"] == 64 and 0 <= phase["fallback_cores"] <= 64 and
                        sum(phase["final_get_by_core"]) == row["final"], "HIGH transport count mismatch")
        print(f"G{group} VERIFIED_last_changed_Index_positions " + " ".join(
            f"{a}={cases[a]['verified_consumer_work']['index_positions_changed']}"
            for a in ("original", *sources)) + " transport64_max_fallback_cores="
            + "/".join(str(max(s["high_post_index"]["fallback_cores"] for s in cases[a]["reads_samples"]))
                       for a in sources))
        for arm, marker in zip(sources, schedule_formats):
            diag = cases[arm]["native_diagnostics"]
            clock = diag["counter_calibration"]
            require(not diag["included_in_timing"] and not clock["included_in_timing"]
                    and len(diag["samples"]) == 1 and clock["status"] == "measured",
                    f"{arm}: missing separate HIGH replay and independent clock")
            sample = diag["samples"][0]
            metrics = sample["reads"]["high_post_index"]["schedule_timing"]
            require(sample["auto_route"]["mode"] == "high" and metrics["format"] == marker
                    and sample["reads"]["consumer_work"]["ordered_index_verified"] is True
                    and sample["reads"]["high_post_index"]["persistent_moves_checked"],
                    f"{arm}: wrong stage instrumentation, route or persistent Index")
            rate = calibration_summary(clock["samples"])["cycles_per_us"]
            keys = ("first_post_begin", "last_post_begin", "last_post_return", "last_final_wait_return",
                    "longest_lane_index", "index_completion_spread", "last_index_minus_last_final_wait")
            outer = metrics.get("outer_entry")
            outer_times = (two(outer[key] / rate for key in ("first_post", "last_wait", "last_role_exit"))
                           if outer else "unavailable")
            print(f"G{group} TRACE_us {arm} marker={marker} " + " ".join(f"{k}={metrics[k]/rate:.2f}" for k in keys)
                  + f" final_Get={sample['reads']['final']} nonempty_lanes={metrics['nonempty_lanes']}"
                  + f" calls={metrics['final_api_calls']} blocking_lanes={metrics['blocking_final_lanes']}"
                  + f" last_wait_lane/get/max_lane/max_row={metrics['last_wait_lane']}/{metrics['last_wait_lane_get']}/"
                    f"{metrics['max_lane_get']}/{metrics['max_row_get']}"
                  + f" same_lane_last_post_return_us={metrics['last_wait_lane_post_return']/rate:.2f}"
                  + f" from_outer_entry_us_firstPost/lastWait/lastExit={outer_times}")
            if marker.startswith("RQI"):
                high = sample["reads"]["high_post_index"]
                roles = "/".join(f"{v/rate:.2f}" for v in metrics["validation_by_role_p50"])
                reassign = metrics.get("get_reassignment")
                split_load = (f" helpers/Get_moved/max_Q_before={reassign['helpers']}/{reassign['transferred']}/"
                              f"{reassign['max_canonical_lane']}" if reassign else "")
                role_stage = "descriptor_checked" if marker in ("RQI4", "RQI5", "RQI6") else "to_valid"
                spread = "descriptor_check" if marker in ("RQI4", "RQI5", "RQI6") else "valid"
                print(f"G{group} LOAD {arm} role0/1/2/3_{role_stage}_p50_us={roles}"
                      f" max_row_{spread}_spread_us={metrics['validation_spread_max']/rate:.2f}"
                      f" lane_Index_us_p50/max={metrics['lane_index_p50']/rate:.2f}/{metrics['longest_lane_index']/rate:.2f}"
                      f" last_retire_to_row_exit_us_p50/max={metrics['row_retire_to_exit_p50']/rate:.2f}/{metrics['row_retire_to_exit_max']/rate:.2f}"
                      f" moves_total/max_row/max_lane={metrics['moved_total']}/{metrics['max_row_moves']}/{metrics['max_moves_per_lane']}"
                      f" rows_single_owner/gather={high['single_owner_rows']}/{high['row_gathers']}" + split_load)
                if marker in ("RQI4", "RQI5", "RQI6"):
                    early = metrics["early_post"]
                    def times_us(*keys):
                        return "/".join(f"{early[k]/rate:.2f}" for k in keys)
                    print(f"G{group} PIPE {arm} from_outer_entry_us"
                          f" first/last_Q_ready={times_us('first_query_ready','last_query_ready')}"
                          f" first/last_Q_seen={times_us('first_query_observed','last_query_observed')}"
                          f" last_plan/TND={times_us('last_plan','last_tnd')}"
                          f" lanes_firstPost_before_rowJoin={early['lanes_post_before_row_join']}/{early['nonempty_lanes']}"
                          f" lanes_firstPost_before_row_plan_or_TND_done="
                          f"{early['lanes_post_before_row_plan_or_tnd']}/{early['nonempty_lanes']}")
                if marker in ("RQI5", "RQI6"):
                    cq = metrics["retire"]
                    poll = metrics.get("poll_work")
                    counts = (f" final_probes_total/empty={poll['probes_total']}/{poll['empty_probes']}"
                              f" polled/paced_lanes={poll['polled_lanes']}/{poll['paced_lanes']}"
                              f" probes_per_polled_lane_p50/max={poll['probes_p50']:g}/{poll['probes_max']}"
                              f" paced_lane_us_sum/max={poll['paced_ticks_total']/rate:.2f}/{poll['paced_ticks_max']/rate:.2f}"
                              f" last_lane_probes/paced_us={poll['latest_lane_probes']}/{poll['latest_lane_paced_ticks']/rate:.2f}"
                              if poll else "")
                    print(f"G{group} POST_CQ {arm} software_wall_us"
                          f" prefetch_owner_p50/max={cq['prefetch_p50']/rate:.2f}/{cq['prefetch_max']/rate:.2f}"
                          f" final_nonempty_p50/max={cq['final_p50']/rate:.2f}/{cq['final_max']/rate:.2f}"
                          f" last_lane={cq['last_lane']} last_lane_final={cq['last_lane_final']/rate:.2f}" + counts)
        print(f"G{group} CHECK real_Index_order/TopK/KV/lifecycle/submitted_completed=PASS", flush=True)
        return
    stages = []
    for arm in ("before", "after"):
        waits, apply, ends = diagnostic_wait(cases[arm])
        app = two(apply) if len(apply) == 2 else "unavailable"
        stages.append(f"{arm}_max_row_index_wait={two(waits)} {arm}_max_row_apply={app}"
                      f" {arm}_last_index_minus_last_get_or_prefetch_join={two(ends)}")
    print(f"G{group} DIAGNOSTIC_two_replays_us " + " ".join(stages))
    print(f"G{group} CHECK real_Index_order/TopK/KV/lifecycle=PASS", flush=True)


def run_replay(args, group, arm, *, profile=False):
    cmd = command(args, group, arm, profile=profile)
    path = Path(cmd[cmd.index("--output") + 1])
    log = path.with_name("." + path.stem + ".log")
    with log.open("x") as out:
        status = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    if status:
        with log.open(errors="replace") as out:
            print("".join(deque(out, maxlen=8)), file=sys.stderr, end="")
        raise SystemExit(f"G{group}/{arm} failed rc={status} log={log}")
    return load_case(str(path), arm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--intermediate", help="Additional full-AIV control between baseline and candidate")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--kernel-account", action="store_true", help="Two independent HIGH Level0 replays, only for G7")
    p.add_argument("--server", required=True)
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--groups", type=int, nargs="+", choices=GROUPS, required=True)
    p.add_argument("--high-overlap", action="store_true", help="Bracket both full-AIV HIGH arms with the original per-row schedule")
    args = p.parse_args()
    if args.kernel_account and (not args.high_overlap or 7 not in args.groups):
        p.error("--kernel-account requires --high-overlap and group 7")
    if args.intermediate and not args.high_overlap:
        p.error("--intermediate requires --high-overlap")
    arms = THREE_SOURCE_ARMS if args.intermediate else (OVERLAP_ARMS if args.high_overlap else ARMS)
    formats = ()
    if args.high_overlap:
        sources = overlap_sources(args.baseline, args.revision, args.intermediate)
        formats = tuple(schedule_format(args.source if arm == "after" else args.cache/"adaptive-sources"/f"auto-{sha[:8]}")
                        for arm, sha in sources.items())
    for group in args.groups:
        cases = {arm: run_replay(args, group, arm) for arm in arms}
        summarize(group, cases, args.baseline, args.revision, args.high_overlap, formats, args.intermediate)
        if args.high_overlap:
            print_original_clock(group, cases["original"])
        if group == 7 and args.kernel_account:
            from kvip.diagnostics.high_graph_account import print_account
            for arm in ("original", "after"):
                print_account(arm, run_replay(args, group, arm, profile=True), cases[arm])


if __name__ == "__main__":
    main()
