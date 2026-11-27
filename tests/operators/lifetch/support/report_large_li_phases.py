# SPDX-License-Identifier: Apache-2.0
"""Read LI phases, without launching an operator. Only combine clocks within one invocation."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

from large_batch_placement import placement

CARRY = "983734aa8a94310f8090055d916d9a49ab9b4645"
ORIGINAL = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"
CONTROL = "10442f1ffa314606c23f54b73e49ecfc372f1f9a"
GROUPS = (2, 10, 4, 17, 7)
BATCHES = (20, 24, 28, 32)


def require(test, reason):
    if not test:
        raise ValueError(reason)


def field(root, name):
    path = root / (name + ".txt")
    return path.read_text().strip() if path.is_file() else ""


def read(path):
    return json.loads(path.read_text())


def timing(root, group, batch, revision):
    parts, ids = [], []
    for block in (1, 2):
        directory = root / f"g{group}/bs{batch}/block{block}/fused"
        if not (directory / "placement_verified").is_file() or not (directory / "packed_result.json").is_file():
            continue
        record = read(directory / "packed_result.json")
        ident = record["identity"]
        require((directory / "placement_verified").exists() and record["correctness"] == "PASS"
                and record["arm"] == "fused" and record["calls"] == len(record["samples"]) == 10
                and record["full_KV_Index_checked_calls"] == 10 and record["checked_queries"] == 40 * batch
                and record["checked_rows"] == 10 * batch and ident["native"] == revision
                and (ident["group"], ident["batch"]) == (group, batch)
                and type(ident["physical_npu"]) == int and ident["physical_npu"] >= 0 and ident["host"]
                and (not field(root, "physical_npu") or str(ident["physical_npu"]) == field(root, "physical_npu"))
                and (not field(root, "host") or ident["host"] == field(root, "host"))
                and ident["protocol"] == "ten_independent_states_one_graph_one_event_v1", str(directory))
        for s in record["samples"]:
            require(all(isfinite(s[k]) and s[k] > 0 for k in ("LI1", "LI23", "LI"))
                    and isclose(s["LI1"] + s["LI23"], s["LI"], abs_tol=.001), str(directory))
        identity = {k: v for k, v in ident.items() if k not in ("native", "protocol")}
        ids.append(identity)
        parts.append(record)
    if not parts:
        raise FileNotFoundError(f"No completed packed block at {root}/g{group}/bs{batch}")
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
    require(all(isfinite(li["spans"][k]) for k in ("LI2_active_us", "LI3_active_us", "overlap_us")),
            "Unusable diagnostic span values")
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



def score_exposure(record, diagnostic):
    """Split each owner cold interval; sibling work is a separate, potentially overlapping series."""
    row = diagnostic["row"]
    clock = record["separate_LI_diagnostic"]["clock_mhz"]
    prior_window, *cold = [epoch[row] for epoch in record["separate_window_event_u64"]]
    owner = record["separate_score_span_cycles"]["post_owner"][row]
    sibling = record["separate_score_span_cycles"]["sibling"][row]
    sibling_checks = record["separate_sibling_predecessor_cycles"][row]
    require(len(owner) == len(sibling) == len(sibling_checks) == len(cold), "Wrong number of score intervals")
    parts = []
    for event, o, s, waited in zip(cold, owner, sibling, sibling_checks):
        left, right = prior_window[1], event[4]
        require(0 < o[0] <= o[1] <= right and 0 < s[0] <= s[1] <= waited[0] <= waited[1]
                and waited[1] >= left and event[1] >= waited[1], "Invalid paired score timestamps")
        in_cold = max(0, min(o[1], right) - max(o[0], left))
        before = max(0, min(right, o[0]) - left)
        after = max(0, right - max(left, o[1]))
        require(before + in_cold + after == right - left, "Cold split does not close")
        parts.append(dict(pre_score=before/clock, owner_score_call=in_cold/clock, post_score_to_rank=after/clock,
                          sibling_score_call=(s[1]-s[0])/clock, sibling_exposed_predecessor=max(0,left-waited[0])/clock,
                          sibling_score_overlap_previous_rank_service=max(0,min(s[1],left)-max(s[0],prior_window[13]))/clock))
        prior_window = event
    require(isclose(sum(p["pre_score"]+p["owner_score_call"]+p["post_score_to_rank"] for p in parts),
                    diagnostic["phases_us"]["cold_advance"],abs_tol=1e-6), "LI3 critical row does not close")
    return parts


def same_machine(left, right):
    for name in ("host", "physical_npu", "kvaf", "kvaf_tree"):
        if not field(left, name) or field(left, name) != field(right, name):
            return False
    for name in ("machine_id", "boot_id"):
        a, b = field(left, name), field(right, name)
        if a and b:
            return a == b
    return False


def candidate_roots(root, reference):
    # Existing links take precedence. An archive moved as a whole can retain its
    # original absolute paths, so try the same basename under the current parent.
    choices = []
    path = Path(reference)
    choices.extend((path if path.is_absolute() else root.parent / path, root.parent / path.name))
    return list(dict.fromkeys(p.resolve() for p in choices if p.is_dir() and p.resolve() != root))


def compatible_roots(root, explicit, link_key, pattern, revision, extra=()):
    def verified(paths):
        return [p for p in paths if p.is_dir() and field(p, "native") == revision and same_machine(root, p)]

    if explicit is not None:
        return verified([explicit.resolve()])
    for link in (field(root, link_key), *extra):
        if link:
            matches = verified(candidate_roots(root, link))
            if matches:
                return matches
    return verified(sorted(root.parent.glob(pattern)))



def packed_placement(root, group, batch):
    cell = root / f"g{group}/bs{batch}"
    values = {placement(folder / "thp.json") for folder in cell.glob("block[12]/fused")
              if (folder / "placement_verified").is_file() and (folder / "packed_result.json").is_file()}
    require(len(values) == 1, f"Missing or inconsistent registered memory placement in {cell}")
    return values.pop()


def supported_control(roots, group, batch, current_root, current_id):
    matches = []
    for root in roots:
        try:
            ident, parts = timing(root, group, batch, CONTROL)
            if packed_placement(root, group, batch) != packed_placement(current_root, group, batch):
                continue
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"G{group} BS{batch} CONTROL_SKIPPED root={root} reason={error}")
            continue
        if ident == current_id:
            matches.append((root, parts))
    if len(matches) > 1:
        raise ValueError(f"multiple matching packed controls; pass --control ({[str(p) for p, _ in matches]})")
    if not matches:
        raise FileNotFoundError("no matching same-card packed control")
    return matches[0]


def original_diagnostic(roots, group, batch, ident):
    matches = []
    for root in roots:
        path = root / f"g{group}/bs{batch}/fused/ARM_RESULT.json"
        if not path.is_file():
            continue
        record = read(path)
        if (record.get("arm") == "fused" and record["identity"].get("native") == ORIGINAL
                and record["identity"].get("protocol") == "one_restored_graph_per_device_event_with_profiler"
                and all(record["identity"].get(k) == v for k, v in ident.items())):
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(f"multiple matching diagnostic files; pass --source ({list(map(str, matches))})")
    if not matches:
        raise FileNotFoundError("no matching same-card original instrumented call")
    return li_diagnostic(read(matches[0]))


def show_critical(loc, label, record):
    spans = record["union_us"]
    print(f"{loc} {label} row={record['row']} LI2/LI3/overlap_us="
          f"{fmt(spans[k] for k in ('LI2_active_us', 'LI3_active_us', 'overlap_us'))} "
          f"first/cold_advance/prior_check/rank_join/row_service/retire_us={fmt(record['phases_us'].values())}")


def report(root, rank_revision=None, batches=BATCHES, groups=GROUPS, source=None, control=None):
    policy = "four-ranks" if rank_revision else "four-window-carry"
    revision = rank_revision or CARRY
    if not root.is_dir():
        print(f"ARCHIVE_UNAVAILABLE root={root} reason=directory_missing")
        return []
    if field(root, "native"):
        require(field(root, "native") == revision, f"Unexpected candidate revision in {root}")
    controls = compatible_roots(root, control, "reference", "fixed-q4-packed.*", CONTROL)
    sources = compatible_roots(root, source, "packed_source", "fixed-q4.*", ORIGINAL,
                               [field(p, "packed_source") for p in controls])
    print(f"CURRENT_ARCHIVE={root} policy={policy} physical_NPU={field(root, 'physical_npu') or 'from_cell'}")
    # Validating one input record never requires any optional comparison root.
    cells = []
    for batch in batches:
        for group in groups:
            loc = f"G{group} BS{batch}"
            try:
                ident, parts = timing(root, group, batch, revision)
            except FileNotFoundError as error:
                print(f"{loc} CURRENT unavailable={error}")
                continue
            require(all(p.get("large_window_policy") == policy for p in parts), f"Wrong window policy at {loc}")
            current = {k: packed_values(parts, k) for k in ("LI1", "LI23")}
            print(f"{loc} CURRENT calls={10 * len(parts)} LI1_mean/P50_us={fmt(current['LI1'])} "
                  f"LI23_mean/P50_us={fmt(current['LI23'])}")
            previous = diag = density = critical = None
            control_root = None
            try:
                control_root, reference = supported_control(controls, group, batch, root, ident)
            except (FileNotFoundError, ValueError) as error:
                print(f"{loc} CONTROL unavailable={error}")
            else:
                previous = {k: packed_values(reference, k) for k in current}
                print(f"{loc} CONTROL calls={10 * len(reference)} control_minus_current_LI1/LI23_mean_us="
                      f"{fmt(previous[k][0]-current[k][0] for k in current)}")

            # The serial oracle is stored per cell even in copied archives which
            # have lost packed_source.txt. Only trust its ancestral gN/bsN name.
            oracle = parts[0].get("serial_oracle")
            source_roots = sources
            if source is None and oracle:
                path = Path(oracle)
                if path.parts[-3:] == (f"g{group}", f"bs{batch}", "serial"):
                    linked = compatible_roots(root, None, "packed_source", "fixed-q4.*", ORIGINAL, [path.parents[2]])
                    source_roots = linked or sources
            try:
                diag = original_diagnostic(source_roots, group, batch, ident)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                print(f"{loc} ORIGINAL_DIAG unavailable={error}")
            else:
                show_critical(loc, "ORIGINAL_DIAG", diag)

            raw = next((p for p in parts if "window_density" in p or "separate_LI_diagnostic" in p), parts[0])
            density = raw.get("window_density")
            if density is not None:
                try:
                    message = (f"{loc} CURRENT_DIAG LI3_span_us={density['li3_span_us']:.2f} "
                               f"mean_prior_check_by_window_us={fmt(w['mean_check_to_rank_us'] for w in density['epochs'])} "
                               f"mean_rank_through_service_by_window_us={fmt(w['mean_rank_and_service_us'] for w in density['epochs'])}")
                except (ValueError, KeyError, TypeError) as error:
                    density = None
                    print(f"{loc} CURRENT_DIAG unavailable={error}")
                else:
                    print(message)
            else:
                print(f"{loc} CURRENT_DIAG unavailable=missing_window_density")
            if rank_revision:
                try:
                    require(raw["measured_cold_symbol"] == "v9_li_large_four_ranks"
                            and raw["diagnostic_cold_symbol"] == "v9_li_large_four_ranks_trace"
                            and raw["cold_build_fingerprint"] != raw["diagnostic_cold_build_fingerprint"]
                            and raw["diagnostic_ranks"]["checked_queries"] == 4 * batch
                            and raw["window_grants_per_row"] == [256]*4, "Missing separate four-rank clock proof")
                    critical = li_diagnostic(raw)
                    show_critical(loc, "CURRENT_LAST_ROW", critical)
                    exposures = score_exposure(raw, critical)
                    critical["by_cold_window"] = exposures
                    print(f"{loc} CURRENT_LAST_ROW by_cold_window "
                          + " ".join(f"{key}_us={fmt(w[key] for w in exposures)}" for key in exposures[0]))
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    print(f"{loc} CURRENT_LAST_ROW detail_unavailable={error}")
            cells.append(dict(group=group, batch=batch, current=current, current_calls=10 * len(parts), control=previous,
                              control_root=str(control_root) if control_root else None, original_diagnostic=diag,
                              current_density=density, current_critical=critical))
    print(f"PHASE_REPORT_CELLS={len(cells)}/{len(batches)*len(groups)}")
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--rank-revision", help="Expected exact four-rank candidate revision")
    parser.add_argument("--source", type=Path, help="Same-card original single-call diagnostic archive")
    parser.add_argument("--control", type=Path, help="Same-card previous fused packed archive")
    parser.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=GROUPS)
    parser.add_argument("--batches", nargs="+", type=int, choices=BATCHES, default=BATCHES)
    args = parser.parse_args()
    cells = report(args.root.resolve(), args.rank_revision, args.batches, args.groups, args.source, args.control)
    if args.json:
        args.json.write_text(json.dumps(cells, indent=2)+"\n")
