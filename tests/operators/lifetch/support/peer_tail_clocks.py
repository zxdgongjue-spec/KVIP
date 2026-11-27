# SPDX-License-Identifier: Apache-2.0
"""Read existing original-tail clocks. A consumed CQE may already have been visible earlier."""
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
                drain_p50_us=[median(times) if times else None for times in waits])


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
            break
        else:
            print(f"G7 BS{batch} ORIGINAL_CQ=UNAVAILABLE missing_local_diagnostic")
