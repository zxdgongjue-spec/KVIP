# SPDX-License-Identifier: Apache-2.0
"""Read LI phases, without launching an operator. Only combine clocks within one invocation."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

CARRY = "983734aa8a94310f8090055d916d9a49ab9b4645"
ORIGINAL = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"
CONTROL = "10442f1ffa314606c23f54b73e49ecfc372f1f9a"
GROUPS = (2, 10, 4, 17, 7)
BATCHES = (20, 24, 28, 32)


def require(test, reason):
    if not test:
        raise ValueError(reason)


def field(root, name):
    return (root / (name + ".txt")).read_text().strip()


def read(path):
    return json.loads(path.read_text())


def timing(root, group, batch, count):
    parts, ids = [], []
    for block in range(1, count + 1):
        directory = root / f"g{group}/bs{batch}/block{block}/fused"
        record = read(directory / "packed_result.json")
        ident = record["identity"]
        require((directory / "placement_verified").exists() and record["correctness"] == "PASS"
                and record["arm"] == "fused" and record["calls"] == len(record["samples"]) == 10
                and record["full_KV_Index_checked_calls"] == 10 and record["checked_queries"] == 40 * batch
                and record["checked_rows"] == 10 * batch and ident["native"] == field(root, "native")
                and (ident["group"], ident["batch"], ident["physical_npu"]) == (group, batch, 3)
                and ident["host"] == field(root, "host")
                and ident["protocol"] == "ten_independent_states_one_graph_one_event_v1", str(directory))
        for s in record["samples"]:
            require(all(isfinite(s[k]) and s[k] > 0 for k in ("LI1", "LI23", "LI"))
                    and isclose(s["LI1"] + s["LI23"], s["LI"], abs_tol=.001), str(directory))
        identity = {k: v for k, v in ident.items() if k not in ("native", "protocol")}
        ids.append(identity)
        parts.append(record)
    require(all(i == ids[0] for i in ids), "Block input or runtime mismatch")
    return ids[0], parts


def fmt(values):
    return "/".join(f"{v:.2f}" for v in values)


def packed_values(parts, key):
    values = [s[key] for part in parts for s in part["samples"]]
    return [mean(values), median(values)]


def li_diagnostic(raw):
    li = raw["separate_LI_diagnostic"]
    clocks, mhz = li["per_query_cycle_stamps"], li["clock_mhz"]
    batch = len(clocks)
    events = raw["separate_window_event_u64"]
    require(batch in BATCHES and len(events) == 4 and all(len(e) == batch for e in events)
            and isfinite(mhz) and mhz > 0, "Unusable historical LI geometry")
    rows = []
    for row, q in enumerate(clocks):
        require(len(q) == 4 and all(len(t) == 4 and 0 < t[0] < t[1] <= t[2] < t[3] for t in q)
                and all(t[2:] == q[0][2:] for t in q), f"Invalid LI clocks in row {row}")
        enter, leave = q[0][2:]
        first_end = events[0][row][1]
        rows_done = [e[row] for e in events]
        require(enter <= first_end <= leave, "Window zero timing")
        for epoch, e in enumerate(rows_done):
            require(len(e) == 16 and e[2] == ((epoch << 32) | row) and e[3] == 1, "Incomplete window")
        last = first_end
        sums = dict(first=first_end-enter, cold_advance=0, prior_check=0, rank_join=0, row_service=0)
        for e in rows_done[1:]:
            arrived, checked, joined, released = e[4], e[13], e[14], e[1]
            require(last <= arrived <= checked <= joined <= e[0] <= released <= leave, "Window clocks")
            sums["cold_advance"] += arrived-last
            sums["prior_check"] += checked-arrived
            sums["rank_join"] += joined-checked
            sums["row_service"] += released-joined
            last = released
        sums["retire"] = leave-last
        require(sum(sums.values()) == leave-enter, "Unaccounted LI3 critical-path interval")
        start2, end2 = min(t[0] for t in q), max(t[1] for t in q)
        rows.append(dict(row=row, li3_end=leave, li2_end=end2, li3_start=enter,
                         gap_from_threshold_us=(enter-end2)/mhz,
                         li2_span_us=(end2-start2)/mhz, phases_us={k:v/mhz for k,v in sums.items()}))
    critical = max(rows, key=lambda r:r["li3_end"])
    out = dict(row=critical["row"], phases_us=critical["phases_us"], union_us=li["spans"],
               last_row_li2_us=critical["li2_span_us"], last_row_threshold_to_li3_us=critical["gap_from_threshold_us"],
               last_minus_median_row_exit_us=(critical["li3_end"]-median(r["li3_end"] for r in rows))/mhz)
    if batch == 32:
        # At BS32 each row owns two AIV lanes. Q0/Q2 are cold-start threshold jobs, Q1/Q3 run second.
        # Only swap the second jobs, holding each lane's actual start and inter-job gap fixed.
        # The second counterfactual removes all within-row imbalance, launch skew and inter-job gaps,
        # allowing continuous division of the observed work. Both freeze the suffix of every row.
        old_end = max(r["li3_end"] for r in rows)
        ends_swap, ends_ideal = [], []
        for r,q in zip(rows, clocks):
            require(q[0][1] <= q[1][0] and q[2][1] <= q[3][0], "Threshold lane mapping differs")
            durations = [x[1]-x[0] for x in q]
            new_after_swap = max(q[1][0]+durations[3], q[3][0]+durations[1])
            credit_swap = max(0, r["li2_end"]-new_after_swap)
            ideal_finish = min(x[0] for x in q) + sum(durations)/2.0
            halves = max(0, r["li2_end"] - ideal_finish)
            ends_swap.append(r["li3_end"]-credit_swap)
            ends_ideal.append(r["li3_end"]-halves)
        out.update(swap_second_jobs_optimistic_us=(old_end-max(ends_swap))/mhz,
                   perfect_within_row_balance_ceiling_us=(old_end-max(ends_ideal))/mhz)
    return out


def report(root):
    old, control = Path(field(root, "packed_source")), Path(field(root, "reference"))
    require(field(root, "native") == CARRY and field(old, "native") == ORIGINAL
            and field(control, "native") == CONTROL and field(root, "physical_npu") == "3"
            and field(control, "packed_source") == str(old), "Wrong archive chain")
    for key in ("host", "physical_npu", "kvaf", "kvaf_tree"):
        require(field(root,key) and field(root,key) == field(old,key) == field(control,key), "Mismatch: "+key)
    identity_key = "machine_id" if all((p/"machine_id.txt").is_file() and field(p,"machine_id")
                                       for p in (root,old,control)) else "boot_id"
    require(field(root,identity_key) and field(root,identity_key) == field(old,identity_key)
            == field(control,identity_key), "Different machines")
    print("Current four-window carry, same-card original ten-call control, archived original instrumented call.")
    cells = []
    for batch in BATCHES:
        for group in GROUPS:
            ident, parts = timing(root,group,batch,1)
            baseline_ident, baseline = timing(control,group,batch,2)
            require(ident == baseline_ident, "Packed inputs differ")
            require(parts[0].get("large_window_policy") == "four-window-carry", "Wrong window policy")
            single = read(old / f"g{group}/bs{batch}/fused/ARM_RESULT.json")
            require(single["arm"] == "fused" and single["identity"]["native"] == ORIGINAL
                    and single["identity"]["protocol"] == "one_restored_graph_per_device_event_with_profiler"
                    and all(single["identity"].get(k) == v for k,v in ident.items()), "Diagnostic input mismatch")
            current = {k:packed_values(parts,k) for k in ("LI1","LI23")}
            previous = {k:packed_values(baseline,k) for k in current}
            loc = f"G{group} BS{batch}"
            print(f"{loc} CURRENT LI1_mean/P50_us={fmt(current['LI1'])} LI23_mean/P50_us={fmt(current['LI23'])}"
                  f" control_minus_current_LI1/LI23_mean_us={fmt([previous[k][0]-current[k][0] for k in current])}")
            try:
                diag = li_diagnostic(single)
            except (ValueError, KeyError) as error:
                diag = {"unavailable": str(error)}
                print(f"{loc} ORIGINAL_DIAG unavailable={error}")
            else:
                u, s = diag["union_us"], diag["phases_us"]
                print(f"{loc} ORIGINAL_DIAG row={diag['row']} LI2/LI3/overlap_us="
                  f"{fmt([u['LI2_active_us'],u['LI3_active_us'],u['overlap_us']])}"
                  f" first/cold_advance/prior_check/rank_join/row_service/retire_us={fmt(s.values())}")
            density = parts[0].get("window_density")
            if density is not None:
                print(f"{loc} CURRENT_DIAG LI3_span_us={density['li3_span_us']:.2f}"
                      f" mean_prior_check_by_window_us={fmt(w['mean_check_to_rank_us'] for w in density['epochs'])}"
                      f" mean_rank_through_service_by_window_us={fmt(w['mean_rank_and_service_us'] for w in density['epochs'])}")
            cells.append(dict(group=group,batch=batch,current=current,control=previous,
                              original_diagnostic=diag,current_density=density))
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--json",type=Path)
    args = parser.parse_args()
    cells = report(args.root.resolve())
    if args.json:
        args.json.write_text(json.dumps(cells,indent=2)+"\n")
