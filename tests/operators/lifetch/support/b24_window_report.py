# SPDX-License-Identifier: Apache-2.0
"""Compare checked natural windows to the archived same-card fused and serial graphs."""
import argparse
from pathlib import Path
from statistics import mean

from b24_window_policy import MODES, policy
from export_large_rank_evidence import observation
from four_rank_accept import matched, read
from four_rank_followup import serial_reference
from large_batch_placement import field, placement

GROUPS = (7, 2, 10, 4, 17)
BATCHES = (32, 28, 24, 20)
CONTROL = "window-x3-early"
# At X8 both rank curves returned identical admissions in the CPU replay.
MEASURE = ("window-x2-early", "window-x3-linear", "window-x4-early", "window-x8-early",
           "window-x2-linear", "window-x4-linear")


def cached(reader, ref, group, batch, mode):
    rel = Path(f"g{group}/bs{batch}/block1/{mode}")
    for root in sorted(ref.parent.glob(f"fixed-q4-audit-{mode}.*"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        arm = root / rel
        if (field(root, "audit_reader") == reader and field(root, "audit_reference") == str(ref)
                and matched(root, ref) and field(root, "native") == field(ref, "native")
                and (arm / "placement_verified").is_file() and (arm / "window-study.json").is_file()):
            return arm


def stats(d):
    return [d["graph_mean_us"], *(d["mean_us"][k] for k in ("LI", "LI1", "LI23", "tail"))]


def values(path, ref, group, batch, mode):
    base = ref / f"g{group}/bs{batch}/block1"
    original = read(base / "fused/packed_result.json")
    _, identity = observation(ref, base, group, batch, "fused")
    if mode == CONTROL:
        if original["diagnostic_ranks"]["ranks"] != list(policy(CONTROL)[0]):
            raise ValueError("Unexpected archived control ranks")
        return original, original, identity
    r = read(path / "packed_result.json")
    receipt = read(path / "window-study.json")
    if (receipt != {"mode": mode, "reference": str(ref)} or not matched(path.parents[3], ref)
            or not (path / "placement_verified").is_file() or placement(path / "thp.json") != identity[1]
            or r["identity"] != original["identity"] or r["actual_ranks"] != list(policy(mode)[0])
            or r["window_grants_per_row"] != list(policy(mode)[1]) or r["correctness"] != "PASS"
            or r["full_KV_Index_checked_calls"] != 10 or r["checked_queries"] != batch*40
            or r["checked_rows"] != batch*10 or r["checked_rank_queries"] != batch*4):
        raise ValueError(f"Policy, board, NUMA, input or graph checks differ at {path}")
    return r, original, identity


def show(path, ref, group, batch, mode):
    r, orig, identity = values(path, ref, group, batch, mode)
    t = stats(r)
    tag = f"G{group} BS{batch} {mode}"
    fmt = lambda seq: "/".join(f"{v:.2f}" for v in seq)
    print(f"{tag} CHECK TopK_score/KV/Index/private_CQ=PASS calls=10/10 "
          f"graph/LI/LI1/LI23/tail_mean_us={fmt(t)} vs_four_window_saved_us={fmt(a-b for a,b in zip(stats(orig),t))}")
    serial = serial_reference(ref, identity, group, batch)
    if serial:
        g = serial["graph"]
        print(f"{tag} vs_same_card_serial_gain_pct={100*(1-t[0]/g):+.2f} "
              f"remaining_to_15/20_us={fmt((t[0]-.85*g,t[0]-.8*g))}")
    else:
        print(f"{tag} same_card_strong_serial=UNAVAILABLE")
    d = r["window_density"]
    if mode == "window-x4-earned":
        print(f"{tag} cumulative_row_credit=256/448/640/832/1024 ceilings=256/320/384/448/192")
    print(f"{tag} actual_ranks={r['actual_ranks']} grants={r['window_grants_per_row']} "
          f"diagnostic_posted={[p['posted_tokens'] for p in d['epochs']]} "
          f"diagnostic_used_by_final={[p['used_by_final_topk'] for p in d['epochs']]} "
          f"sibling_wait_for_predecessor_us={fmt(p['mean_sibling_exposed_previous_us'] for p in d['epochs'][1:])} "
          f"measured_final_GET_mean={mean(v['final'] for v in r['traffic']):.1f}")


def summary(reader, ref):
    for batch in BATCHES:
        by_policy = {}
        for mode in (CONTROL, *MEASURE):
            rows = []
            for group in GROUPS:
                arm = None if mode == CONTROL else cached(reader, ref, group, batch, mode)
                if mode != CONTROL and not arm:
                    break
                r, _, identity = values(arm, ref, group, batch, mode)
                s = serial_reference(ref, identity, group, batch)
                rows.append((*stats(r)[:2], 100*(1-r["graph_mean_us"]/s["graph"]) if s else None))
            if len(rows) == len(GROUPS):
                minimum = min(v[2] for v in rows) if all(v[2] is not None for v in rows) else None
                by_policy[mode] = (mean(v[0] for v in rows), mean(v[1] for v in rows), minimum)
        for mode, (graph, li, minimum) in by_policy.items():
            gain = "NA" if minimum is None else f"{minimum:.2f}"
            print(f"BS{batch} {mode} five_trace_mean_graph/LI_us={graph:.2f}/{li:.2f} worst_trace_gain_pct={gain}")
        if len(by_policy) == 1 + len(MEASURE):
            print(f"BS{batch} lowest_five_trace_mean_graph={min(by_policy,key=lambda m: by_policy[m][0])} "
                  f"lowest_five_trace_mean_LI={min(by_policy,key=lambda m: by_policy[m][1])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("cached", "show", "summary"))
    p.add_argument("directory")
    p.add_argument("ref", type=Path)
    p.add_argument("group", type=int, nargs="?")
    p.add_argument("batch", type=int, nargs="?")
    p.add_argument("mode", choices=MODES, nargs="?")
    a = p.parse_args()
    if a.action == "summary":
        summary(a.directory, a.ref)
    elif a.action == "show":
        show(Path(a.directory), a.ref, a.group, a.batch, a.mode)
    else:
        reader = a.directory if len(a.directory) == 40 and all(c in "0123456789abcdef" for c in a.directory) else field(Path(a.directory), "audit_reader")
        if reader:
            hit = cached(reader, a.ref, a.group, a.batch, a.mode)
            if hit:
                print(hit)
