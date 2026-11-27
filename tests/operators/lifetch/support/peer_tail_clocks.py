# SPDX-License-Identifier: Apache-2.0
"""Read existing original-tail clocks. A consumed CQE may already have been visible earlier."""
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from statistics import median

from large_batch_placement import field, placement
from portable_two_row_audit import local, read, require, NATIVE


def pressure(record):
    tail = record["tail"]
    mhz = tail["clock_mhz"]
    rows = tail["decoded"]["qs"]
    batch = record["identity"]["batch"]
    require(isfinite(mhz) and mhz > 0 and len(rows) == 4 * batch and
            len({(q["row"], q["query"]) for q in rows}) == len(rows), "Incomplete row/Q clocks")
    waits = [[], []]
    post_to_retire = [[], []]
    tokens = [0, 0]
    changes = []
    for q in rows:
        prev = 0
        last = 0
        for i, block in enumerate(q["batches"]):
            kind = int(i > 0)
            begin, posted = block["post_begin"], block["post_end"]
            draining, retired = block["drain_begin"], block["drain_end"]
            count = block["prefix_end"] - prev
            require(0 < count and last <= begin <= posted <= draining <= retired,
                    "Inconsistent ordinary four-batch tail clock order")
            waits[kind].append((retired - draining) / mhz)
            post_to_retire[kind].append((retired - posted) / mhz)
            tokens[kind] += count
            if retired > posted:
                changes.extend([(posted, 1), (retired, -1)])
            prev, last = block["prefix_end"], retired
        require(prev == q["own"], "Batch counters differ from this Q's final GETs")
    require(sum(tokens) == record["traffic"]["final"], "Final GET count differs from clock trace")
    pending = peak = 0
    for _, delta in sorted(changes):
        pending += delta
        peak = max(peak, pending)
    require(pending == 0, "Missing CQ retirement")
    return dict(peak_unretired=peak, batches=list(map(len, waits)), tokens=tokens,
                post_to_retire_p50_us=[median(x) if x else None for x in post_to_retire],
                drain_p50_us=[median(times) if times else None for times in waits])


def simultaneous_drains(events, owners):
    """Intersect row waits, taking the union of overlapping private Q waits within each row."""
    changes = defaultdict(Counter)
    for row, begin, end in events:
        if end > begin:
            changes[begin][row] += 1
            changes[end][row] -= 1
    active = Counter()
    start = None
    spans = []
    for instant, delta in sorted(changes.items()):
        active.update(delta)
        full = sum(count > 0 for count in active.values()) == owners
        if full and start is None:
            start = instant
        elif not full and start is not None:
            spans.append((start, instant))
            start = None
    require(start is None and not any(active.values()), "Unbalanced private CQ waits")
    return spans


def first_wave(record):
    tail = record["tail"]
    mhz = tail["clock_mhz"]
    decoded = tail["decoded"]
    rows = defaultdict(list)
    for q in decoded["qs"]:
        rows[q["row"]].append(q)
    require(len(rows) > 16 and all(len(qs) == 4 for qs in rows.values()), "Expected 16 active Q4 teams plus successors")
    start = {r:min(q["start_us"] for q in qs) for r,qs in rows.items()}
    end = {r:max(q["until_exit_us"] for q in qs) for r,qs in rows.items()}
    order = sorted(rows, key=start.get)
    initial = order[:16]
    first_free = min(end[r] for r in initial)
    second_start = start[order[16]]
    require(max(start[r] for r in initial) < first_free <= second_start,
            "Cannot identify the original single-row teams from this trace")
    events = [(r,i,b["drain_begin"],b["drain_end"]) for r in initial
              for q in rows[r] for i,b in enumerate(q["batches"])]
    def intervals(first_only):
        spans = simultaneous_drains(((r,a,z) for r,i,a,z in events if not first_only or i == 0), len(initial))
        # All sixteen original rows are present in each span, so no successor could yet have started.
        origin = decoded["origin"]
        return [((a-origin)/mhz,(z-origin)/mhz) for a,z in spans]

    first, any_batch = intervals(True), intervals(False)
    return dict(first_team_free_us=first_free, first_successor_start_us=second_start,
                first_batch_spans_us=first, all_teams_first_batch_us=sum(z-a for a,z in first),
                all_teams_any_batch_us=sum(z-a for a,z in any_batch), terminal_span_us=decoded["terminal_span_us"])


def report(area, current):
    for batch in (28, 32):
        roots = sorted(area.glob("four-rank-diagnostic.*"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
        for root in roots:
            arm = root / f"g7/bs{batch}/fused"
            if not local(root,current) or not (arm/"placement_verified").is_file() or not (arm/"single_diagnostic.json").is_file():
                continue
            d = read(arm/"single_diagnostic.json")
            archive = Path(d["input_archive"])
            if not archive.is_file():
                continue
            ref = archive.parents[4]
            if (ref.parent.resolve() != area.resolve() or not local(ref,current) or field(ref,"native") != NATIVE
                or any(not field(root,k) or field(root,k)!=field(ref,k) for k in ("native","kvaf","kvaf_tree"))
                or archive != ref / f"g7/bs{batch}/block1/fused/packed_result.json"
                or not (archive.parent/"placement_verified").is_file()):
                continue
            base = read(archive)
            if (d["correctness"]!="PASS" or d["measured_calls"]!=0 or d["diagnostic_calls"]!=1
                    or d["identity"]!=base["identity"] or placement(arm/"thp.json")!=placement(archive.parent/"thp.json")):
                continue
            result = pressure(d)
            nums = lambda items: "/".join("NA" if v is None else f"{v:.2f}" for v in items)
            print(f"G7 BS{batch} ORIGINAL_CQ diagnostic_calls=1 peak_Post_to_CQ_retired_Qs={result['peak_unretired']} "
                  f"first/later_drain_P50_us={nums(result['drain_p50_us'])} "
                  f"first/later_tokens={result['tokens'][0]}/{result['tokens'][1]} "
                  f"matched_traffic_count_calls={d['archive_matches_window_and_final_counts']}/10")
            wave = first_wave(d)
            print(f"G7 BS{batch} FIRST_WAVE first_Post_end_to_retirement_P50_us={result['post_to_retire_p50_us'][0]:.2f} "
                  f"all_16_teams_in_first_drain_us={wave['all_teams_first_batch_us']:.2f} "
                  f"all_16_teams_in_any_drain_us={wave['all_teams_any_batch_us']:.2f} "
                  f"earliest_team_done_us={wave['first_team_free_us']:.2f} "
                  f"next_row_start_us={wave['first_successor_start_us']:.2f} terminal_span_us={wave['terminal_span_us']:.2f}")
            break
        else:
            print(f"G7 BS{batch} ORIGINAL_CQ=UNAVAILABLE missing_local_diagnostic")
