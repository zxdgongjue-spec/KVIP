# SPDX-License-Identifier: Apache-2.0
"""Measure a HIGH Index candidate on host20 against the former full-AIV and original fused paths."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from copy import copy
from pathlib import Path
from statistics import mean, median

from kvip.diagnostics.high_index_study import GROUPS, order_samples, replay_command, work_key
from kvip.diagnostics.read_through_report import load_case, require, timing


BASE = "dc7904e8e874956de6df8352265bb6ac64d1d644"
# The serial-CQ counterfactual and the candidate have exactly the same compiled binary.
# Bracket all three treatments by the same previous full-AIV binary.
ARMS = ("old", "original", "candidate", "wait", "old-end")
DIAGNOSTICS = 4
PHASES = ("plan_join", "gather_join", "scatter_end", "index_span", "cq_span", "joined_span")


def command(args, group, arm):
    opts = copy(args)
    if arm in ("old", "original", "old-end"):
        opts.revision = BASE
        opts.source = args.cache / "adaptive-sources" / f"auto-{BASE[:8]}"
    role = {"old": "overlap", "original": "original", "candidate": "overlap",
            "wait": "wait-first", "old-end": "overlap-end"}[arm]
    cmd = replay_command(opts, group, role)
    cmd[cmd.index("--output")+1] = str(args.result_dir / f"g{group}-{arm}.json")
    if "--diagnostic-replays" in cmd:
        cmd[cmd.index("--diagnostic-replays")+1] = str(DIAGNOSTICS)
    return cmd


def pair(values):
    return f"{mean(values):.3f}/{median(values):.3f}"


def two(values):
    return "/".join(f"{v:.3f}" for v in values)


def one_to_one(left, right):
    """Pair observations once per exact work key; never cross-match duplicate counts."""
    grouped = defaultdict(deque)
    for key, value in left:
        if key is not None:
            grouped[key].append(value)
    return [(grouped[key].popleft(), value) for key, value in right if grouped.get(key)]


def index_rows(case):
    rows = []
    for key, phase, rate, _ in order_samples(case, expected_count=DIAGNOSTICS):
        require(all(k in phase for k in PHASES), "Missing calibrated full-AIV phase boundaries")
        times = {k: phase[k] / rate for k in PHASES}
        rows.append(dict(key=key, events=phase["event_count"], moves=sum(phase["moved"]),
                         lane_moves=phase["moved"], active=phase["active_aiv"],
                         row_load=phase.get("row_load"), last_end_aiv=phase.get("last_end_aiv"),
                         **times))
    require(rows, "No naturally selected HIGH in diagnostic samples")
    return rows


def timed_tasks(case):
    return [(work_key(r), t) for r, t in zip(case["reads_samples"][1:], case["_samples"][1:])]


def profile_tasks(rows):
    # The complete event stream determines the resulting persistent permutation.
    # Different candidates may assign the same physical moves to different AIV lanes.
    return [((r["key"], r["moves"]) if r["key"] is not None else None, r) for r in rows]


def qualified_targets(group, rows):
    for unit in ("events", "moves"):
        exact = [r for r in rows if r[unit] == 10000]
        nearest = min(rows, key=lambda x: abs(x[unit]-10000))
        large = [r for r in rows if r[unit] >= 10000]
        print(f"G{group} TARGET_10K_{unit.upper()} exact_n={len(exact)}"
              f" exact_index_us={pair([r['index_span'] for r in exact]) if exact else 'unmeasured'}"
              f" nearest_{unit}={nearest[unit]} nearest_index_us={nearest['index_span']:.3f}"
              f" ge10k_n={len(large)} ge10k_under50_n={sum(r['index_span'] < 50 for r in large)}")


def summarize(group, cases, revision):
    base = cases["old"]
    require(base["device"] == 0, "Expected the host20 physical-NPU4 logical-device-zero mapping")
    geometry = base["geometry"]
    require((geometry["batch"], geometry["token_count"], geometry["token_bytes"], geometry["requested_windows"])
            == (16, 64, 656, 8), "Expected the BS16 Q4 656B eight-window fixture")
    for arm, case in cases.items():
        for key in ("geometry", "snapshot_sha256", "kvaf_build", "warmup", "soc", "device", "diagnostics_commit"):
            require(base.get(key) is not None and case.get(key) == base[key],
                    f"{arm}: incomparable {key}")
        require(case["kvip_commit"] == (revision if arm in ("candidate", "wait") else BASE)
                and case["high_post_index"] == (arm != "original")
                and case["post_index_wait_first"] == (arm == "wait"),
                f"{arm}: wrong source, Index policy or CQ order")
        for field in ("selected_set_sha256", "ordered_output_sha256"):
            require(case["selection_comparison"][field] == base["selection_comparison"][field],
                    f"{arm}: final selection changed")
        modes = [s["mode"] for s in case["auto_route"]["samples"]]
        require(modes == ["high"] * 10, f"{arm}: natural HIGH case switched route: {modes}")
        for read in case["reads_samples"]:
            high = read.get("high_post_index")
            require(bool(high) == (arm != "original"), f"{arm}: unexpected path")
            if high:
                require(high["core_count"] == 64 and sum(high["final_get_by_core"]) == read["final"]
                        and high["wait_first"] == (arm == "wait"),
                        f"{arm}: incorrect worker count/order/Get counts")
    libs = {k: v["library_sha256"] for k, v in cases.items()}
    require(libs["old"] == libs["original"] == libs["old-end"] and libs["candidate"] == libs["wait"]
            and libs["old"] and libs["candidate"] and libs["candidate"] != libs["old"],
            "Expected distinct baseline/candidate binaries and same-binary schedule controls")
    ts = {arm: timing(case, True) for arm, case in cases.items()}
    print(f"G{group} GRAPH_steady_us_mean/p50 " + " ".join(f"{k}={two(v)}" for k, v in ts.items()))
    drift = [b - a for a, b in zip(ts["old"], ts["old-end"])]
    first_last = [[v - c for v, c in zip(ts[arm], ts["candidate"])] for arm in ("old", "old-end")]
    print(f"G{group} OLD_END_MINUS_START_us={two(drift)} candidate_saved_vs_old_start/end_us="
          f"{two(first_last[0])},{two(first_last[1])} saved_vs_original_us="
          + two([a-b for a, b in zip(ts["original"], ts["candidate"])]))
    profiles = {arm: index_rows(cases[arm]) for arm in ("old", "candidate", "wait")}
    for arm, rows in profiles.items():
        print(f"G{group} PHASE {arm} n={len(rows)} mean/p50"
              f" events={pair([r['events'] for r in rows])} moves={pair([r['moves'] for r in rows])}"
              f" AIV_used={min(r['active'] for r in rows)}..{max(r['active'] for r in rows)}/64"
              + " " + " ".join(f"{k}_us={pair([r[k] for r in rows])}" for k in PHASES)
              + f" fallback={sum(r['key'] is None for r in rows)}")
        if arm in ("old", "candidate") and rows[0]["row_load"]:
            # Report the critical shape on the actual slowest instrumented sample.
            worst = max(rows, key=lambda r: r["index_span"])
            load = worst["row_load"]
            fields = ("max_row_events", "max_row_moves", "dense_eligible_rows", "simple_swap_rows")
            share = 100 * load["two_cycle_moves"] / worst["moves"] if worst["moves"] else 0
            print(f"G{group} ROWS {arm} slowest_index_us={worst['index_span']:.3f} "
                  + " ".join(f"{f}={load[f]}" for f in fields)
                  + f" verified_two_cycle_moves={load['two_cycle_moves']} ({share:.1f}%)"
                  + " last_end_aiv=" + json.dumps(worst["last_end_aiv"], separators=(",", ":")))
    old_vs_new = one_to_one(profile_tasks(profiles["old"]), profile_tasks(profiles["candidate"]))
    wait_vs_new = one_to_one(profile_tasks(profiles["wait"]), profile_tasks(profiles["candidate"]))
    graph = one_to_one(timed_tasks(cases["wait"]), timed_tasks(cases["candidate"]))
    print(f"G{group} MATCHED old/candidate_n={len(old_vs_new)} index_saved_us="
          + (pair([a["index_span"]-b["index_span"] for a,b in old_vs_new]) if old_vs_new else "unavailable")
          + f" wait/candidate_n={len(wait_vs_new)} approx_exposed_index_us="
          + (pair([a["index_span"]-(a["joined_span"]-b["joined_span"]) for a,b in wait_vs_new])
             if wait_vs_new else "unavailable")
          + f" uninstrumented_wait/candidate_n={len(graph)} graph_saved_us="
          + (pair([a-b for a,b in graph]) if graph else "unavailable"))
    qualified_targets(group, profiles["candidate"])
    print(f"G{group} CHECK exact_scores_topk/persistent_Index/Get_KV/lifecycle=PASS", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    args = p.parse_args(argv)
    for group in GROUPS:
        cases = {}
        for arm in ARMS:
            cmd = command(args, group, arm)
            output = cmd[cmd.index("--output")+1]
            log = args.result_dir / f".g{group}-{arm}.log"
            with log.open("w") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode:
                with log.open(errors="replace") as f:
                    print("".join(deque(f, maxlen=8)), file=sys.stderr, end="")
                raise SystemExit(f"G{group}/{arm} failed rc={proc.returncode} log={log}")
            cases[arm] = load_case(output, arm)
        summarize(group, cases, args.revision)


if __name__ == "__main__":
    main()
