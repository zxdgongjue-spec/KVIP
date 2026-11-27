# SPDX-License-Identifier: Apache-2.0
"""Read completed local physical-NPU4 natural two-row graph receipts, without launching a worker."""
import argparse
from collections import defaultdict
import json
from math import isclose, isfinite
import os
from pathlib import Path
from statistics import mean, median

from large_batch_placement import field, placement

NATIVE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
CANDIDATE = "two-row-natural"
GROUPS = (7, 2, 10, 4, 17)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def local_identity():
    def marker(path):
        p = Path(path)
        return p.read_text().strip() if p.is_file() else ""
    return os.uname().nodename, marker("/etc/machine-id"), marker("/proc/sys/kernel/random/boot_id")


def local(root, identity):
    host, machine, boot = identity
    if field(root, "host") != host or field(root, "physical_npu") != "4":
        return False
    ref_machine = field(root, "machine_id")
    if machine and ref_machine:
        return machine == ref_machine
    return bool(boot and boot == field(root, "boot_id"))


def evaluate(record, control):
    require(record["identity"] == control["identity"], "Input, host, runtime or graph differs")
    for item in (record, control):
        require(item["correctness"] == "PASS" and item["get_workers"] == 32 and
                item["full_KV_Index_checked_calls"] == item["calls"] == len(item["traffic"]) == len(item["samples"]) == 10
                and (item["checked_queries"], item["checked_rows"]) == (1280, 320), "Missing ten completed states")
        require(all(isfinite(x[k]) and x[k] > 0 for x in item["samples"] for k in ("tail", "LI")),
                "Invalid device stage times")
        require(isclose(mean(x["tail"] for x in item["samples"]),item["mean_us"]["tail"],abs_tol=1e-6)
                and isclose(item["graph_mean_us"]*10,item["graph_event_us"],abs_tol=1e-5),
                "Recorded samples disagree with the graph event or tail summary")
    ident = record["identity"]
    require(ident["physical_npu"] == 4 and ident["batch"] == 32 and ident["native"] == NATIVE and
            ident["protocol"] == "ten_independent_states_one_graph_one_event_v1" and
            record["arm"] == CANDIDATE and control["arm"] == "fused", "Unknown candidate/control")
    image = record["terminal_build"]
    require(record["cold_build_fingerprint"] == control["cold_build_fingerprint"] == image["cold_build"] and
            record["measured_cold_symbol"] == control["measured_cold_symbol"] == "v9_li_large_four_ranks" and
            record["measured_tail_symbol"] == image["measured_tail_symbol"] == "v9_test_two_row_terminal" and
            image["tail_text"] and image["original_tail_text"] and image["tail_text"] != image["original_tail_text"],
            "Missing actual cold and terminal binary identity")
    decisions = record["terminal_decisions"]
    require(len(decisions) == len(record["parked_rows_per_call"]) == 10, "Incomplete scheduling receipts")

    def bucket_samples(arm):
        groups = defaultdict(list)
        for traffic, timing in zip(arm["traffic"], arm["samples"]):
            rows = traffic["row_query_final"]
            windows = traffic["row_window_prefetch"]
            require(len(rows) == len(windows) == 32 and all(len(r) == 4 for r in [*rows, *windows])
                    and all(isinstance(v, int) and v >= 0 for r in [*rows,*windows] for v in r)
                    and sum(map(sum,rows)) == traffic["final"] and sum(map(sum,windows)) == traffic["prefetch"],
                    "Malformed per-row traffic counts")
            signature = tuple(tuple(row) for row in [*rows,*windows])
            groups[signature].append(timing["tail"])
        return groups
    current, old = bucket_samples(record), bucket_samples(control)
    matched = sum(min(len(v), len(old[k])) for k, v in current.items())
    differences = [(mean(v) - mean(old[k]), min(len(v),len(old[k]))) for k,v in current.items() if old[k]]
    eligible, eligible_first, parked, split, single, empty = [], [], [], [], [], []
    categories = defaultdict(list)
    for i, (traffic, timing, decision) in enumerate(zip(record["traffic"], record["samples"], decisions)):
        rows = traffic["row_query_final"]
        # Identical arithmetic to PlanTerminalBatches / the two-row all-four eligibility vote.
        candidates = {r for r, counts in enumerate(rows) if sum(counts) >= 256 and min(counts) >= 2}
        passed = decision["ordered_pairs"]
        routes = decision["team_rows"]
        require(decision["accepted"] and len(routes) == 16 and all(routes) and
                sorted(r for route in routes for r in route) == list(range(32)) and
                decision["normal_handoffs"] + decision["early_handoffs"] == 16 and
                decision["early_handoffs"] == decision["parked_rows"] == len(passed) == record["parked_rows_per_call"][i]
                and all(parent in candidates and any((parent,child) == step for route in routes
                    for step in zip(route,route[1:])) for parent,child in passed),
                "An early assignment disagrees with ownership, eligibility or the device receipt")
        eligible.append(len(candidates))
        eligible_first.append(sum(route[0] in candidates for route in routes))
        parked.append(len(passed))
        split.append(sum(own>1 and sum(row)>=256 for row in rows for own in row))
        single.append(sum(own>0 for row in rows for own in row)-split[-1])
        empty.append(128-single[-1]-split[-1])
        label = "paired" if passed else ("eligible_unpaired" if candidates else "no_eligible_row")
        categories[label].append(timing["tail"])
    require(sum(bool(n) for n in parked)==record["calls_with_row_pair"],"Trigger count disagrees with receipts")
    return dict(graph_delta=record["graph_mean_us"]-control["graph_mean_us"],
                li_delta=record["mean_us"]["LI"]-control["mean_us"]["LI"],
                tail_delta=record["mean_us"]["tail"]-control["mean_us"]["tail"],
                eligible=eligible, eligible_in_first_wave=eligible_first, parked=parked,
                final_Q_zero_single_multi_per_call=[mean(empty),mean(single),mean(split)],
                tail_by_activation={label:[len(v),mean(v),median(v)] for label,v in categories.items()},
                matched_counts_calls=matched,
                matched_counts_candidate_minus_control_tail=sum(x*n for x,n in differences)/matched if matched else None,
                control_tail_text=image["original_tail_text"], candidate_tail_text=image["tail_text"])


def audit(work, identity):
    area = work / "peer-baseline"
    if not area.is_dir():
        print(f"NO_LOCAL_ARCHIVE physical_NPU=4 path={area}")
        return
    roots = sorted(area.glob("fixed-q4-audit-two-row-natural.*"), key=lambda r:r.stat().st_mtime_ns, reverse=True)
    for group in GROUPS:
        for root in roots:
            if not local(root, identity):
                continue
            reference = Path(field(root, "audit_reference"))
            cell = f"g{group}/bs32/block1"
            candidate = root / cell / CANDIDATE
            control = reference / cell / "fused"
            result_file = candidate / "two-row-natural.json"
            if not result_file.is_file() or not (candidate/"placement_verified").is_file():
                continue
            require(reference.resolve().parent == area.resolve() and local(reference, identity)
                    and all(field(root,k) and field(root,k) == field(reference,k) for k in ("native","kvaf","kvaf_tree"))
                    and field(root,"audit_reader") and (control/"placement_verified").is_file()
                    and placement(candidate/"thp.json")==placement(control/"thp.json"),
                    f"Local reference or placement is unverifiable for {result_file}")
            result = evaluate(read(result_file),read(control/"packed_result.json"))
            print(f"G{group} BS32 graph/LI/tail_candidate_minus_control_us={result['graph_delta']:+.2f}/"
                  f"{result['li_delta']:+.2f}/{result['tail_delta']:+.2f} result={result_file}")
            print(f"G{group} AUDIT "+json.dumps(result,separators=(",",":")))
            break
        else:
            print(f"G{group} BS32 NO_LOCAL_COMPLETED_TWO_ROW_RESULT physical_NPU=4")
    print("CQ_CLOCK_SOURCE=unavailable_in_natural_receipts traffic_matching=counts_only")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    audit(p.parse_args().work.resolve(),local_identity())
