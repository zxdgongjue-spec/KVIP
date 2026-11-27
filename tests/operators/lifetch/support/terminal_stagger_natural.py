# SPDX-License-Identifier: Apache-2.0
"""Confirm a controlled tail gain in the ordinary ten-call inference graph."""
import argparse
import json
from pathlib import Path

from large_batch_placement import field, placement
from four_rank_accept import matched

BASE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"


def archived_pair(out, ref, group, batch):
    rel = Path(f"g{group}/bs{batch}/block1")
    reader = field(out, "audit_reader")
    for arm in sorted(out.parent.glob(f"fixed-q4-audit-first-receipt-pair.*/{rel}/gate1/stagger.json"),
                      key=lambda p: p.stat().st_mtime_ns, reverse=True):
        pair = arm.parent.parent
        root = pair.parents[2]
        if (root == out or not matched(out, root) or field(root, "audit_reader") != reader
                or field(root, "native") != BASE or field(root, "audit_reference") != str(ref)):
            continue
        if all((pair / name / "placement_verified").is_file() and (pair / name / "stagger.json").is_file()
               for name in ("gate0", "gate1")):
            return pair
    return None


def archived_natural(out, ref, group, batch, pair):
    rel = Path(f"g{group}/bs{batch}/block1/fused")
    for marker in sorted(out.parent.glob(f"fixed-q4-audit-first-receipt-pair.*/{rel}/placement_verified"),
                         key=lambda p: p.stat().st_mtime_ns, reverse=True):
        root = marker.parents[4]
        if (root != out and matched(out, root) and field(root, "native") == BASE
                and field(out, "audit_reader") == field(root, "audit_reader")
                and field(root, "audit_reference") == str(ref)
                and placement(marker.parent/"thp.json") == placement(pair/"gate1/thp.json")):
            return root
    return None


def eligible(pair, group, batch):
    from terminal_stagger_result import report
    gate = json.loads((pair / "gate0/stagger.json").read_bytes())
    if (gate["identity"]["group"], gate["identity"]["batch"]) != (group, batch):
        raise ValueError("Wrong cell in the fixed-list test")
    saved = report(pair)
    return bool(saved and saved["graph"] > 0 and saved["tail"] > 0)


def measure(a):
    import acl
    import large_batch_pair
    import large_two_cold
    import fixed_q4_packed_worker as packed
    from b24_two_row_attach import install_profiler
    from terminal_stagger_attach import identity, install

    if a.batch not in (20, 24, 28, 32) or a.mode != "stagger-natural":
        raise ValueError("Unsupported natural graph")
    install(large_two_cold, a.timed, a.sdk / "build-npu", acl.get_soc_name(), True, natural=True)
    install_profiler("v9_terminal_first_cq")
    ordinary = packed.measure_packed

    def checked(ex, proof, serial, torch, states, *args, **kwargs):
        if not kwargs["fused"] or kwargs.get("window_density") or len(states) != 10:
            raise ValueError("Expected ten ordinary, independent fused states")
        if any(hasattr(state, "captured_final_plan") or hasattr(state, "frozen_binding") for state in states):
            raise ValueError("Natural performance may not use a forced GET journal")
        result = ordinary(ex, proof, serial, torch, states, *args, **kwargs)
        images = [identity(state, True) for state in states]
        for state, traffic in zip(states, proof["traffic"], strict=True):
            expected = [[int(sum(row) >= 256)]*2 for row in traffic["row_query_final"]]
            if state.first_cq_permits[:, :, 0].cpu().tolist() != expected:
                raise ValueError("Natural GET counts and device permission flags differ")
        if len({state.forward.terminal.workspace.data_ptr() for state in states}) != 10 or any(
                image != images[0] for image in images):
            raise ValueError("Natural graph has inconsistent terminal bindings")
        proof["stagger_natural_image"] = images[0]
        return result

    packed.measure_packed = checked
    try:
        large_batch_pair.run(argparse.Namespace(trace=a.trace, group=a.group, batch=a.batch, round=1,
            cohort=0, prepared=a.bank, arm="fused", kvaf_root=a.sdk, kvaf_server=a.server, output=a.out,
            packed_ten=True, captured_score_oracle=True, large_window_policy="four-ranks",
            compare_packed_candidate=True, carry_credit=False, window_density=False, serial_result=None,
            reuse_serial_archive=None, compare_archived_pair=False, keep_trace=False))
    finally:
        packed.measure_packed = ordinary


def summary(out, ref, group, batch, pair):
    from export_large_rank_evidence import observation
    from four_rank_followup import serial_reference
    rel = Path(f"g{group}/bs{batch}/block1")
    new, ident = observation(out, out/rel, group, batch, "fused")
    rec = json.loads((out/rel/"fused/packed_result.json").read_bytes())
    paired = json.loads((pair/"gate1/stagger.json").read_bytes())
    manifest = json.loads((Path(paired["original_journal"]) / "journal.json").read_bytes())
    image = rec["stagger_natural_image"]
    if (field(out, "native") != BASE or rec["get_workers"] != 32 or image["gate_word"] != 1
            or image["tail_text"] == paired["tail_text"]
            or rec["measured_cold_symbol"] != "v9_li_large_four_ranks"
            or manifest["journal_sha256"] != paired["frozen_journal_sha256"]
            or rec["cold_build_fingerprint"] != manifest["cold_build"]
            or image["cold_build"] != manifest["cold_build"]
            or any(rec["identity"].get(k) != v for k,v in paired["identity"].items())
            or ident[1] != placement(pair/"gate1/thp.json")):
        raise ValueError("Natural input, LI, environment or image differs from the checked pair")
    tag = f"G{group} BS{batch}"
    print(f"{tag} NATURAL CHECK TopK_score/KV/Index_state/private_CQ=PASS calls=10/10 "
          f"queries={40*batch} rows={10*batch}")
    print(f"{tag} natural graph_mean_us={new['graph']:.2f} "
          f"LI_mean/P50_us={new['LI']:.2f}/{rec['p50_us']['LI']:.2f} "
          f"tail_mean/P50_us={new['tail']:.2f}/{rec['p50_us']['tail']:.2f}")
    if (ref/rel/"fused/packed_result.json").is_file():
        old, previous = observation(ref, ref/rel, group, batch, "fused")
        if previous != ident or not matched(out, ref):
            raise ValueError("Four-window reference differs from natural measurement")
        print(tag, "vs_original_natural", *(f"{k}_saved_us={old[k]-new[k]:+.2f}" for k in ("graph","LI","tail")))
    else:
        print(f"{tag} vs_original_natural UNAVAILABLE same-card archive missing")
    serial = serial_reference(out, ident, group, batch)
    if serial:
        gain = 100*(1-new["graph"]/serial["graph"])
        print(f"{tag} vs_same_card_serial graph_saved_us={serial['graph']-new['graph']:+.2f} "
              f"gain_pct={gain:+.2f} gap_to_15pct_us={new['graph']-.85*serial['graph']:+.2f} source={serial['source']}")
    else:
        print(f"{tag} vs_same_card_serial UNAVAILABLE same-card archive missing")


if __name__ == "__main__":
    import sys
    action, out, ref, group, batch, *extra = sys.argv[1:]
    out, ref, group, batch = Path(out), Path(ref), int(group), int(batch)
    if action == "find":
        print(archived_pair(out, ref, group, batch) or "")
    elif action == "find-natural":
        print(archived_natural(out, ref, group, batch, Path(extra[0])) or "")
    elif action == "eligible":
        print("PROMOTE" if eligible(Path(extra[0]), group, batch) else "SKIP")
    elif action == "summary":
        summary(out, ref, group, batch, Path(extra[0]))
    else:
        raise ValueError(action)
