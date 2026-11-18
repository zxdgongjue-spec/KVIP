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
from kvip.diagnostics.read_through_report import load_case, require, timing


ARMS = ("before", "after", "after-end", "before-end")
OVERLAP_ARMS = ("original", *ARMS, "original-end")


def command(args, group, arm):
    run = copy(args)
    if arm.startswith(("before", "original")):
        run.revision = args.baseline
        run.source = args.cache / "adaptive-sources" / f"auto-{run.revision[:8]}"
    cmd = replay_command(run, group, "original")
    cmd[cmd.index("--output") + 1] = str(args.result_dir / f"g{group}-{arm}.json")
    cmd.append("--verify-consumer-work")
    if getattr(args, "high_overlap", False):
        # Every headline uses the common Post/Wait completion ABI. Stage counters
        # belong to the separate replay and are decoded by their on-device marker.
        if not arm.startswith("original"):
            cmd.append("--high-post-index")
        if arm in ("before", "after"):
            cmd.extend(("--diagnostic-replays", "1", "--calibrate-counter"))
    elif arm in ("before", "after"):
        cmd.extend(("--diagnostic-replays", "2", "--calibrate-counter"))
    return cmd


def work_key(read):
    work = read["consumer_work"]
    return (work["event_sha256"], work["descriptor_sha256"], work["event_count"],
            read["prefetch"], read["final"], read["unused_prefetch"])


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
    for tag, marker in (("0x52514932", "RQI2"), ("0x52514931", "RQI1")):
        if tag in impl:
            return marker
    raise ValueError(f"Unrecognized row schedule diagnostic ABI at {source}")


def summarize(group, cases, baseline, candidate, high_overlap=False, schedule_formats=("PIH1", "RQI1")):
    arms = OVERLAP_ARMS if high_overlap else ARMS
    base = cases["before"]
    geometry = base["geometry"]
    require((geometry["requested_windows"], geometry["batch"], geometry["token_count"], geometry["token_bytes"])
            == (8, 16, 64, 656), "Expected BS16 Q4, one 656-byte layer and eight windows")
    for arm, c in cases.items():
        for key in ("geometry", "snapshot_sha256", "kvaf_build", "soc", "device", "diagnostics_commit",
                    "warmup", "tuning", "graph_backend"):
            require(base.get(key) is not None and c[key] == base[key], f"{arm}: unmatched {key}")
        require(c["device"] == 0 and c["graph_backend"] == "kvaf.AclGraph" and len(c["_samples"]) == 10,
                f"{arm}: invalid graph or samples")
        ref = "before" if arm.startswith(("before", "original")) else "after"
        require(c["kvip_commit"] == (baseline if ref == "before" else candidate)
                and c["library_sha256"] == cases[ref]["library_sha256"] and bool(c["library_sha256"])
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
    require(cases["before"]["library_sha256"] != cases["after"]["library_sha256"], "Same binary in both arms")

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
        for arm in arms:
            if arm.startswith("original"):
                continue
            for row in cases[arm]["reads_samples"]:
                phase = row["high_post_index"]
                require(phase["core_count"] == 64 and 0 <= phase["fallback_cores"] <= 64 and
                        sum(phase["final_get_by_core"]) == row["final"], "HIGH transport count mismatch")
        print(f"G{group} VERIFIED_last_changed_Index_positions " + " ".join(
            f"{a}={cases[a]['verified_consumer_work']['index_positions_changed']}"
            for a in ("original", "before", "after")) + " transport64_max_fallback_cores="
            + "/".join(str(max(s["high_post_index"]["fallback_cores"] for s in cases[a]["reads_samples"]))
                       for a in ("before", "after")))
        for arm, marker in zip(("before", "after"), schedule_formats):
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
            print(f"G{group} TRACE_us {arm} marker={marker} " + " ".join(f"{k}={metrics[k]/rate:.2f}" for k in keys)
                  + f" final_Get={sample['reads']['final']} nonempty_lanes={metrics['nonempty_lanes']}"
                  + f" calls={metrics['final_api_calls']} blocking_lanes={metrics['blocking_final_lanes']}")
            if marker.startswith("RQI"):
                high = sample["reads"]["high_post_index"]
                roles = "/".join(f"{v/rate:.2f}" for v in metrics["validation_by_role_p50"])
                print(f"G{group} LOAD {arm} role0/1/2/3_to_valid_p50_us={roles}"
                      f" max_row_valid_spread_us={metrics['validation_spread_max']/rate:.2f}"
                      f" lane_Index_us_p50/max={metrics['lane_index_p50']/rate:.2f}/{metrics['longest_lane_index']/rate:.2f}"
                      f" moves_total/max_row/max_lane={metrics['moved_total']}/{metrics['max_row_moves']}/{metrics['max_moves_per_lane']}"
                      f" rows_single_owner/gather={high['single_owner_rows']}/{high['row_gathers']}")
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--server", required=True)
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--groups", type=int, nargs="+", choices=GROUPS, required=True)
    p.add_argument("--high-overlap", action="store_true", help="Bracket both full-AIV HIGH arms with the original per-row schedule")
    args = p.parse_args()
    arms = OVERLAP_ARMS if args.high_overlap else ARMS
    formats = ()
    if args.high_overlap:
        baseline_source = args.cache/"adaptive-sources"/f"auto-{args.baseline[:8]}"
        formats = (schedule_format(baseline_source), schedule_format(args.source))
    for group in args.groups:
        cases = {}
        for arm in arms:
            cmd = command(args, group, arm)
            log = args.result_dir / f".g{group}-{arm}.log"
            with log.open("x") as out:
                status = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
            if status:
                with log.open(errors="replace") as out:
                    print("".join(deque(out, maxlen=8)), file=sys.stderr, end="")
                raise SystemExit(f"G{group}/{arm} failed rc={status} log={log}")
            cases[arm] = load_case(cmd[cmd.index("--output") + 1], arm)
        summarize(group, cases, args.baseline, args.revision, args.high_overlap, formats)


if __name__ == "__main__":
    main()
