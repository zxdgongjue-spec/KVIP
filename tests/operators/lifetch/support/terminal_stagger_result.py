# SPDX-License-Identifier: Apache-2.0
"""Validate two identically compiled private terminal arms before comparing their times."""
from pathlib import Path
import json
import sys
from four_rank_final_map import compare
from large_batch_placement import placement


def report(pair):
    a, b = [json.loads((pair / f"gate{i}/stagger.json").read_bytes()) for i in (0, 1)]
    if placement(pair / "gate0/thp.json") != placement(pair / "gate1/thp.json"):
        raise ValueError("Paired CPU placement differs")
    for k in ("identity", "frozen_journal_sha256", "expected_window_sha256", "expected_final_sha256",
              "tail_text", "guard_sha256", "cold_build", "cold_build_fingerprint", "compiler_sha256"):
        if not a.get(k) or a[k] != b.get(k):
            raise ValueError(f"Mismatched gate controls in {k}")
    for i, d in enumerate((a, b)):
        if (not (pair / f"gate{i}/placement_verified").is_file() or d["gate_word"] != i or d["correctness"] != "PASS"
                or d["arm"] != "stagger-pair" or d["calls"] != 10 or d["full_KV_Index_checked_calls"] != 10
                or len(d["traffic"]) != 10 or len(d["timed_journal_checks"]) != 10):
            raise ValueError("Incomplete frozen control or candidate")
    bs = a["identity"]["batch"]
    check = compare(a, b, [None] * bs)
    ca, cb = a["timed_journal_checks"], b["timed_journal_checks"]
    same_live = sum(all(x[k] == y[k] for k in ("actual_window_sha256", "actual_final_sha256"))
                    for x, y in zip(ca, cb, strict=True))
    original = [sum(c["ordered_match"] for c in checks) for checks in (ca, cb)]
    print(f"G{a['identity']['group']} BS{bs} live_GET_vs_live={same_live}/10 "
          f"off_vs_original={original[0]}/10 on_vs_original={original[1]}/10 "
          f"identical_Index_rows={check['checked_map_rows']-check['map_mismatch_rows']}/{10*bs}")
    guard = [sum(x["native_final_guard_failures"] for x in checks) for checks in (ca, cb)]
    print(f"native_final_guard_failed_queries_off/on={guard[0]}/{guard[1]}")
    if any(guard) or not check["guarded_pass"] or same_live != 10 or original != [10,10]:
        for arm, checks in (("off", ca), ("on", cb)):
            bad = [(i,x.get("final_source_order_bad"),x.get("final_destinations_bad"),x.get("first_mismatch"))
                   for i,x in enumerate(checks) if not x["ordered_match"]]
            if bad:
                print(f"{arm} first_failure_call/final_source_order_bad/private_destinations_bad/entry={bad[0]}")
        print("CONTROLLED_SAVINGS=UNAVAILABLE")
        return
    group = a["identity"]["group"]
    print(f"G{group} BS{bs} CHECK TopK/KV/ordered_Index=PASS full_frozen_GET=10/10 same_LI_and_tail_binary=PASS")
    four = 0
    for call in b["traffic"]:
        for row in call["row_query_final"]:
            if sum(row) >= 256:
                four += sum(len({(own*k+15)//16 for k in (1, 6, 11, 16)}) == 4 for own in row)
    print(f"G{group} BS{bs} four_batch_queries={four}/{40*bs} on_both_arms")
    for name in ("graph", "LI", "tail"):
        def value(d):
            return d["graph_mean_us"] if name == "graph" else d["mean_us"][name]
        print(f"G{group} BS{bs} {name}_mean_us original/gate={value(a):.2f}/{value(b):.2f} "
              f"saved_us={check['controlled_savings_us'][name]:+.2f}")
        if name != "graph":
            print(f"G{group} BS{bs} {name}_P50_us original/gate={a['p50_us'][name]:.2f}/{b['p50_us'][name]:.2f}")


if __name__ == "__main__":
    report(Path(sys.argv[1]))
