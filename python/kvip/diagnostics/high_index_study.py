# SPDX-License-Identifier: Apache-2.0
"""Compare original HIGH, full-AIV HIGH and its same-binary wait-first control."""

import argparse
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
import subprocess
import sys

from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.read_through_report import load_case, require, timing


GROUPS = (7, 10, 17, 4)
ROLES = ("original", "wait-first", "overlap", "overlap-end", "wait-first-end", "original-end")


def replay_command(args, group, role):
    command = [
        sys.executable, "-B", "-m", "kvip.diagnostics.replay",
        "--snapshot", str(args.cache / "skip-results.po80Nw" / f"g{group}-input"),
        "--build-dir", str(args.cache / "KVAF/build-npu"), "--transport", "urma", "--mode", "fused",
        "--study", "post-threshold", "--threshold-policy", "recent", "--pipeline-variants", "auto-post-index",
        "--upstream-root", str(args.source), "--upstream-revision", args.revision,
        "--kvaf-server", args.server, "--device", "0", "--soc", "Ascend950DT_9582",
        "--check-auto-route", "--warmup", "2", "--iterations", "10", "--quiet",
        "--output", str(args.result_dir / f"g{group}-{role}.json"),
    ]
    if not role.startswith("original"):
        command.append("--high-post-index")
    if role.startswith("wait-first"):
        command.append("--post-index-wait-first")
    if role in ("overlap", "wait-first"):
        command.extend(("--diagnostic-replays", "2", "--calibrate-counter"))
    return command


def work_key(read):
    phase = read.get("high_post_index")
    if not phase or phase["fallback"]:
        return None
    return (phase["event_sha256"], phase["descriptor_sha256"], tuple(phase["final_get_by_core"]),
            read["prefetch"], read["final"], read["unused_prefetch"])


def pair_string(values):
    return "/".join(f"{value:.3f}" for value in values)


def order_samples(case, expected_count=2):
    diagnostics = case["native_diagnostics"]
    clock = diagnostics["counter_calibration"]
    require(clock["status"] == "measured" and clock["included_in_timing"] is False
            and diagnostics["included_in_timing"] is False and len(diagnostics["samples"]) == expected_count,
            "Missing independent HIGH diagnostics or counter calibration")
    rate = calibration_summary(clock["samples"])["cycles_per_us"]
    samples = []
    for sample in diagnostics["samples"]:
        reads = sample["reads"]
        phase = reads.get("high_post_index")
        if phase is None:
            require(sample["auto_route"]["mode"] == "low", "HIGH telemetry missing")
            continue
        require(sample["auto_route"]["mode"] == "high" and phase["core_count"] == 64
                and phase["wait_first"] == case["post_index_wait_first"] and phase["persistent_moves_checked"],
                "Incorrect diagnostic path, grid or execution order")
        samples.append((work_key(reads), phase, rate, reads))
    return samples


def equivalent_savings(cases):
    """Weight each exact work key by its smaller sample count. Never manufacture matched pairs."""
    sides = []
    for label in ("wait-first", "overlap"):
        values = defaultdict(list)
        for role in (label, label + "-end"):
            case = cases[role]
            for sample, read in zip(case["_samples"][1:], case["reads_samples"][1:]):
                key = work_key(read)
                if key is not None:
                    values[key].append(sample)
        sides.append(values)
    deltas = []
    for key in sides[0].keys() & sides[1].keys():
        a, b = (values[key] for values in sides)
        deltas.extend([mean(a) - mean(b)] * min(len(a), len(b)))
    return len(deltas), mean(deltas) if deltas else None


def summarize(group, cases):
    base = cases["original"]
    require(base["geometry"]["requested_windows"] == 8 and base["geometry"]["batch"] == 16
            and base["geometry"]["token_bytes"] == 656 and base["geometry"]["token_count"] == 64,
            "Expected the BS16 Q4 single-layer eight-window fixture")
    modes = [s["mode"] for s in base["auto_route"]["samples"]]
    for role in ROLES:
        case = cases[role]
        for key in ("geometry", "snapshot_sha256", "library_sha256", "kvip_commit", "kvaf_build", "warmup", "soc", "device"):
            require(case.get(key) is not None and case[key] == base[key], f"{role}: unmatched {key}")
        require(case["library_sha256"] and case["high_post_index"] == (not role.startswith("original"))
                and case["post_index_wait_first"] == role.startswith("wait-first"), f"{role}: incorrect treatment")
        for key in ("ordered_output_sha256", "selected_set_sha256"):
            require(case["selection_comparison"][key] and case["selection_comparison"][key] == base["selection_comparison"][key],
                    f"{role}: changed verified selection/output")
        route = [s["mode"] for s in case["auto_route"]["samples"]]
        require(len(route) == len(modes) and len(route) == 10
                and case["tuning"]["admission_controls"] == base["tuning"]["admission_controls"],
                f"{role}: different input length, replay count or admission policy")
        for mode, read in zip(route, case["reads_samples"]):
            phase = read.get("high_post_index")
            expect = mode == "high" and case["high_post_index"]
            require(bool(phase) == expect, f"{role}: full-index readback disagrees with natural route")
            if expect:
                require(phase["core_count"] == 64 and phase["wait_first"] == case["post_index_wait_first"]
                        and sum(phase["final_get_by_core"]) == read["final"], f"{role}: incorrect HIGH work")
    print(f"G{group} STEADY units=us mean/p50 first,end (first timed replay excluded)")
    for role in ("original", "wait-first", "overlap"):
        first, end = (cases[k] for k in (role, role+"-end"))
        a, b = timing(first, True), timing(end, True)
        reads = first["reads_samples"][1:] + end["reads_samples"][1:]
        phases = [read["high_post_index"] for read in reads if "high_post_index" in read]
        high = sum(item["mode"] == "high" for run in (first, end) for item in run["auto_route"]["samples"][1:])
        print(f"  {role:10} {pair_string(a)}  {pair_string(b)}  drift_us={pair_string([y-x for x,y in zip(a,b)])}"
              f"  high={high}/{len(reads)} prefetch/final={mean(r['prefetch'] for r in reads):.1f}/{mean(r['final'] for r in reads):.1f}"
              f" blocking_fallback={sum(p['fallback'] for p in phases) if phases else '-'}")
    profiles = {}
    for role in ("wait-first", "overlap"):
        profiles[role] = order_samples(cases[role])
        for i, (_, phase, rate, read) in enumerate(profiles[role], 1):
            joined = f"{phase['joined_span']/rate:.3f}" if "joined_span" in phase else "unavailable"
            split = (" plan+join/gather+join/scatter_us="
                     + pair_string(phase[name]/rate for name in ("plan_join", "gather_join", "scatter_end"))
                     if "plan_join" in phase else "")
            print(f"  INDEX {role}#{i} moved_AIV={phase['active_aiv']}/{phase['core_count']}"
                  f" moves={sum(phase['moved'])} events={phase['event_count']} work={phase['event_sha256'][:10]}"
                  f" prefetch/final={read['prefetch']}/{read['final']} fallback={int(phase['fallback'])}"
                  f" index_span_us={phase['index_span']/rate:.3f} cq_span_us={phase['cq_span']/rate:.3f}"
                  f" index/CQ_core_p50_us={phase['index_core_p50']/rate:.3f}/{phase['cq_core_p50']/rate:.3f}"
                  f" post_join_to_finish_us={joined}{split}")
    count, saving = equivalent_savings(cases)
    matched = []
    for key, wait, rate, _ in profiles["wait-first"]:
        if key is None:
            continue
        for other_key, overlap, other_rate, _ in profiles["overlap"]:
            if (key == other_key and wait["moved"] == overlap["moved"]
                    and "joined_span" in wait and "joined_span" in overlap):
                isolated = wait["index_span"]/rate
                saved = wait["joined_span"]/rate - overlap["joined_span"]/other_rate
                matched.append(isolated - saved)
    bounds = [tuple(x-y for x, y in zip(timing(cases[ctrl], True), timing(cases[treat], True)))
              for ctrl in ("wait-first", "wait-first-end") for treat in ("overlap", "overlap-end")]
    means, medians = zip(*bounds)
    matched_saving = f"{saving:.3f}" if count else "unavailable"
    print(f"  ORDER raw_graph_saved_mean_us={min(means):.3f}..{max(means):.3f}"
          f" raw_graph_saved_p50_us={min(medians):.3f}..{max(medians):.3f}"
          f" matched_steady_work_n={count} matched_graph_saved_mean_us={matched_saving}")
    print(f"  EXPOSURE matched_diag_pairs={len(matched)} estimate_us="
          + (f"{min(matched):.3f}..{max(matched):.3f}" if matched else "unavailable")
          + " (isolated_Index_span minus saved_post_join_span; signed, approximate, excludes prefetch CQ)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--result-dir", required=True, type=Path)
    args = p.parse_args(argv)
    for group in GROUPS:
        manifest = args.cache / "skip-results.po80Nw" / f"g{group}-input/manifest.json"
        if not manifest.is_file():
            p.error(f"snapshot missing: {manifest}")
    for group in GROUPS:
        cases = {}
        print(f"BEGIN G{group} auto HIGH full-AIV vs original and wait-first", flush=True)
        for role in ROLES:
            path = args.result_dir / f"g{group}-{role}.json"
            log = args.result_dir / f".g{group}-{role}.log"
            with log.open("w") as f:
                rc = subprocess.run(replay_command(args, group, role), stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                with log.open(errors="replace") as f:
                    print("".join(deque(f, maxlen=8)), file=sys.stderr, end="")
                raise SystemExit(f"G{group} {role} failed rc={rc} log={log}")
            cases[role] = load_case(str(path), role)
        summarize(group, cases)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
