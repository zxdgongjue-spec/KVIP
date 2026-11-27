# SPDX-License-Identifier: Apache-2.0
"""Audit a baseline and a two-credit arm on the same physical card."""
import argparse
import json
import tarfile
from pathlib import Path

from large_batch_placement import placement, same_card
from replay_private_prefix_credit import exact_query_assignments


def check(condition, message):
    if not condition:
        raise ValueError(message)


def compare_rows(before, after):
    keys = ("group","batch","sequence_sha256","physical_npu","host","donor_rows","protocol","calls",
            "warmup","hardware","torch","torch_npu","cann","kvaf_get_kernel_sha256",
            "token_bytes","registered_pages","physical_history")
    check(all(before["identity"][k] == after["identity"][k] for k in keys), "Mismatched inputs or device")
    check(before["identity"]["native"] != after["identity"]["native"], "Need two executable revisions")


def compare_samples(before, after):
    compare_rows(before, after)
    check(all(s["correctness"] == "PASS" and s["diagnostic_calls"] == 1 and
              s["archive_matches_window_and_final_counts"] == 10 and
              s["tail"]["ordered_Index_rows"] == s["identity"]["batch"] for s in (before,after)),
          "Incomplete original or candidate diagnostic")
    a,s1 = exact_query_assignments(before)
    b,s2 = exact_query_assignments(after)
    check(before["traffic"]["row_window_prefetch"] == after["traffic"]["row_window_prefetch"],
          "Window admission counts differ")
    check([[tuple(x for x,_ in q) for q in row] for row in a] ==
          [[tuple(x for x,_ in q) for q in row] for row in b],
          "Ordered final token lists differ for a private query")
    q0 = before["tail"]["decoded"]["qs"]
    q1 = after["tail"]["decoded"]["qs"]
    check(len(q0) == len(q1) == 4 * len(a), "Missing queries")
    for old,new in zip(q0,q1):
        check((old["row"],old["query"]) == (new["row"],new["query"]), "Different query order")
        check([z["prefix_end"] for z in old["batches"]] == [z["prefix_end"] for z in new["batches"]],
              "Different private prefix boundaries")
        if len(old["batches"]) > 1:
            check(old.get("private_credit",1) == 1 and new.get("private_credit") == 2,
                  "Expected original and two-credit executables")
    return dict(ordered_IDs_match=True, ordered_ID_and_slot_hashes=[s1,s2],
                exact_slot_assignment_match=(s1==s2))


def collect(root):
    rows = {}
    for path in root.glob("g*/bs*/block1/fused/packed_result.json"):
        info = json.loads(path.read_text())
        check((path.parent/"placement_verified").is_file(), f"Missing placement at {path}")
        check(info["full_KV_Index_checked_calls"] == info["calls"] == 10 and info["correctness"]=="PASS" and
              info["large_window_policy"]=="four-ranks" and info["diagnostic_ranks"]["ranks"]==[1024,1568,1792,2048],
              f"Incomplete verification at {path}")
        key = info["identity"]["group"], info["identity"]["batch"]
        rows[key] = info
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_run",type=Path)
    parser.add_argument("new_run",type=Path)
    parser.add_argument("old_clock_archive",type=Path)
    parser.add_argument("new_clock_archive",type=Path)
    args=parser.parse_args()
    check(same_card(args.old_run,args.new_run), "Archive roots differ in physical card or machine")
    before,after=collect(args.old_run), collect(args.new_run)
    expected={(g,b) for g in (2,4,7,10,17) for b in (20,24,28,32)}
    check(set(before)==set(after)==expected,"Expected the 20 four-rank cells in both arms")
    result={}
    with tarfile.open(args.old_clock_archive) as old, tarfile.open(args.new_clock_archive) as new:
        for key in sorted(expected):
            x,y=before[key],after[key]
            compare_rows(x,y)
            rel = Path(f"g{key[0]}/bs{key[1]}/block1/fused/thp.json")
            check(placement(args.old_run/rel)==placement(args.new_run/rel), f"{key}: different NUMA/UB")
            check(x["measured_cold_symbol"] == y["measured_cold_symbol"] and
                  x["cold_build_fingerprint"].split("/")[-1] == y["cold_build_fingerprint"].split("/")[-1],
                  f"{key}: changed experimental cold LI build inputs")
            for p,q in zip(x["traffic"],y["traffic"],strict=True):
                for field in ("row_window_prefetch","row_query_final"):
                    check(p[field]==q[field],f"{key}: {field} differs")
            old_us,new_us=(s["graph_mean_us"] for s in (x,y))
            record=dict(graph_us=[old_us,new_us],graph_reduction_pct=100*(old_us-new_us)/old_us,
                        tail_us=[s["mean_us"]["tail"] for s in (x,y)])
            g,b=key
            if g==7 and b in (28,32):
                name=f"g7/bs{b}/single_diagnostic.json"
                samples=[json.load(a.extractfile(name)) for a in (old,new)]
                check(all(s["identity"]==t["identity"] for s,t in zip(samples,(x,y))),"Wrong clock archive")
                record.update(compare_samples(*samples))
            result[f"G{g}_BS{b}"]=record
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
