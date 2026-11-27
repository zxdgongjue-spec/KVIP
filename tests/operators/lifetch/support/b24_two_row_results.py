# SPDX-License-Identifier: Apache-2.0
"""Check the recorded two-row experiment and compare it with the matching control."""
import argparse
import hashlib
from pathlib import Path

from export_large_rank_evidence import observation
from four_rank_accept import matched, read
from four_rank_followup import serial_reference
from large_batch_placement import field, placement


def find(directory, ref, group, batch, mode):
    rel = Path(f"g{group}/bs{batch}/block1") / mode
    for root in sorted(ref.parent.glob(f"fixed-q4-audit-{mode}.*"), reverse=True):
        if (field(root, "audit_reader") != field(directory, "audit_reader")
                or field(root, "audit_reference") != str(ref) or not matched(root, ref)):
            continue
        arm = root / rel
        if (arm / "placement_verified").is_file() and (arm / f"{mode}.json").is_file():
            return arm
    return None


def metrics(record):
    return [record["graph_mean_us"], record["mean_us"]["LI"], record["mean_us"]["tail"]]


def joined(values, signed=False):
    return "/".join(format(n, "+.2f" if signed else ".2f") for n in values)


def show(directory, ref, group, batch, mode):
    root = directory.parents[3]
    record = read(directory / f"{mode}.json")
    block = ref / f"g{group}/bs{batch}/block1"
    base = read(block / "fused/packed_result.json")
    _, identity = observation(ref, block, group, batch, "fused")
    if (batch != 32 or not matched(root, ref) or not field(root, "audit_reader")
            or field(root, "audit_reference") != str(ref) or field(root, "native") != field(ref, "native")
            or not (directory / "placement_verified").is_file() or placement(directory / "thp.json") != identity[1]
            or record["correctness"] != "PASS" or record["arm"] != mode
            or record["identity"] != base["identity"] or record["calls"] != 10
            or record["full_KV_Index_checked_calls"] != 10
            or (record["checked_queries"], record["checked_rows"]) != (40 * batch, 10 * batch)):
        raise ValueError(f"Tail result differs in input, placement, timing or correctness: {directory}")
    tag = f"G{group} BS{batch} {mode}"
    if mode == "two-row-frozen":
        orig = Path(record["original_control"])
        if (orig.parent.parent.parent.name != f"g{group}" or orig.parent.parent.name != f"bs{batch}"
                or not matched(orig.parents[3], ref) or not (orig / "placement_verified").is_file()
                or placement(orig / "thp.json") != identity[1]):
            raise ValueError("Missing matching fixed-list control")
        with (orig / "frozen.json").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["original_control_sha256"]:
                raise ValueError("The recorded fixed-list control was modified")
        base = read(orig / "frozen.json")
        fields = ("guard_sha256", "frozen_journal_sha256", "expected_window_sha256",
                  "expected_final_sha256", "compiler_sha256")
        if any(base[k] != record[k] for k in fields):
            raise ValueError("Original and new tail used different frozen GET inputs or toolchain")
        if not record.get("guarded_pass") or not base.get("guarded_pass"):
            print(f"{tag} CONTROLLED_SAVINGS=UNAVAILABLE "
                  f"ordered_calls={record.get('paired_ordered_calls')} "
                  f"unknown_map_rows={record.get('map_unknown_rows')} "
                  f"mismatch_map_rows={record.get('map_mismatch_rows')} result={directory}")
            return
        for arm in (record, base):
            checks = arm["timed_journal_checks"]
            if len(checks) != 10 or not all(
                c["ordered_match"] and c["actual_window_sha256"] == arm["expected_window_sha256"]
                and c["actual_final_sha256"] == arm["expected_final_sha256"] for c in checks
            ):
                raise ValueError("Ordered window or final GET differed in the controlled graph")
        print(f"{tag} ORDERED_GET=10/10 control=10/10")
    fresh, old = metrics(record), metrics(base)
    print(f"{tag} CHECK TopK_score/KV/Index=PASS calls=10/10")
    print(f"{tag} graph/LI/tail_mean_us={joined(fresh)} "
          f"vs_original_graph/LI/tail_saved_us={joined([x-y for x, y in zip(old, fresh)], True)}")
    print(f"{tag} LI/tail_P50_us={record['p50_us']['LI']:.2f}/{record['p50_us']['tail']:.2f}")
    if mode == "two-row-natural":
        print(f"{tag} row_pair_triggered_calls={record['calls_with_row_pair']}/10 "
              f"parked_rows_each_call={','.join(map(str, record['parked_rows_per_call']))}")
        decisions = record["terminal_decisions"]
        if (len(decisions) != 10 or
                any(x["early_handoffs"] != x["parked_rows"] or
                    x["normal_handoffs"] + x["early_handoffs"] != batch - 16 for x in decisions)):
            raise ValueError("Missing or inconsistent successor receipt in measured graph")
        sample = next((i for i, decision in enumerate(decisions) if decision["early_handoffs"]), 0)
        chosen = decisions[sample]
        routes = " ".join(f"T{i}=" + ">".join(map(str, rows)) for i, rows in enumerate(chosen["team_rows"]))
        print(f"{tag} TEAM_ROUTE call={sample+1} finish_then_claim={chosen['normal_handoffs']} "
              f"early_claim={chosen['early_handoffs']} {routes}")
        serial = serial_reference(ref, identity, group, batch)
        if serial is None:
            print(f"{tag} vs_same_card_serial=UNAVAILABLE")
        else:
            gain = 100 * (1 - fresh[0] / serial["graph"])
            print(f"{tag} vs_same_card_serial saved_graph_us={serial['graph']-fresh[0]:+.2f} "
                  f"gain_pct={gain:+.2f} source={serial['source']}")
    print(f"{tag} result={directory}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("show", "cached"))
    p.add_argument("directory", type=Path)
    p.add_argument("ref", type=Path)
    p.add_argument("group", type=int)
    p.add_argument("batch", type=int)
    p.add_argument("mode", choices=("two-row-natural", "two-row-frozen"))
    a = p.parse_args()
    if a.action == "show":
        show(a.directory, a.ref, a.group, a.batch, a.mode)
    else:
        arm = find(a.directory, a.ref, a.group, a.batch, a.mode)
        if arm:
            print(arm)
