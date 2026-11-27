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
    if not check["guarded_pass"]:
        raise ValueError(f"Cross-arm private GET lists, counts or ordered Index differed: {check}")
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
