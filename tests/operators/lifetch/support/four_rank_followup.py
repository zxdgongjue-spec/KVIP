# SPDX-License-Identifier: Apache-2.0
"""Compare one new ten-state graph with the explicitly selected same-card four-window archive."""
import argparse
import json
from pathlib import Path
from statistics import mean

from export_large_rank_evidence import observation
from fixed_q4_candidate_only import cell as verified_cell
from four_rank_tail_clocks import expected_final_prefixes
from fixed_q4_packed_report import REFERENCE
from large_batch_placement import field, same_card
from reporting import atomic_json


STAGES = ("LI", "LI1", "LI23", "tail")
CELLS = [(g, b) for g in (7, 2, 10, 4, 17) for b in (32, 28, 24, 20)]
FOCUS = [(7, 32), (7, 28), (2, 32), (10, 32)]
RANKS = [1024, 1568, 1792, 2048]


def verify(root, ref):
    if (not same_card(root, ref) or field(ref, "physical_npu") != "4" or
            field(ref, "native") != "d0fc5963881bd38fa97eff0f6ef40c757a6e8713" or
            any(not field(root, name) or field(root, name) != field(ref, name) for name in ("kvaf", "kvaf_tree"))):
        raise ValueError(f"Four-window origin, host, selected physical card or KVAF differs: {ref}")
    recorded = json.loads((ref / "packed_summary.json").read_text())
    cells = [(v["group"], v["batch"]) for v in recorded["cells"]]
    if (sorted(cells) != sorted(CELLS) or recorded["blocks"] != 1 or
            recorded["protocol"] != "ten_independent_calls_one_device_event" or recorded["measured_arms"] != ["fused"]):
        raise ValueError(f"Incomplete four-window archive: {ref}")
    (root / "followup_reference.txt").write_text(str(ref) + "\n")


def final_lists(traffic, batch):
    by_row = [[] for _ in range(batch)]
    predicted = []
    for call in traffic:
        rows = call["row_query_final"]
        if (len(rows) != batch or any(len(qs) != 4 or any(type(n) is not int or n < 0 for n in qs) for qs in rows)
                or sum(map(sum, rows)) != call["final"]):
            raise ValueError("Invalid measured per-row final GET counts")
        predicted.append(sum(len(expected_final_prefixes(own, sum(qs))) for qs in rows for own in qs))
        for r, qs in enumerate(rows):
            by_row[r].append(qs)
    return dict(row_ge256_calls=[sum(sum(qs) >= 256 for qs in calls) for calls in by_row],
                mean_final_by_row_q=[[mean(q) for q in zip(*calls)] for calls in by_row],
                predicted_final_cq_batches_per_call=predicted)


def values(root, g, b):
    block = root / f"g{g}/bs{b}/block1"
    rec = json.loads((block / "fused/packed_result.json").read_text())
    marks = verified_cell(root, g, b)
    times, identity = observation(root, block, g, b, "fused")
    if marks["ranks"]["ranks"] != RANKS:
        raise ValueError(f"Window ranks changed at {block}")
    stages = marks["mean_us"]["fused"]
    measured = rec["cold_build_fingerprint"]
    diagnostic = rec["diagnostic_cold_build_fingerprint"]
    if (len(rec["traffic"]) != 10 or measured == diagnostic or
            measured.split("/")[0] != diagnostic.split("/")[0] or
            rec["measured_cold_symbol"] != "v9_li_large_four_ranks" or
            rec["diagnostic_cold_symbol"] != "v9_li_large_four_ranks_trace" or
            rec["selection_oracle"] != "captured_logical_scores" or rec["get_workers"] != 32):
        raise ValueError(f"Expected the verified four-rank candidate at {block}")
    lists = final_lists(rec["traffic"], b) if g == 7 and b in (28, 32) else None
    return dict(graph=times["graph"], means={k:stages[k] for k in STAGES}, final_lists=lists,
                p50={k:rec["p50_us"][k] for k in STAGES}, posted=[
                    mean(s["posted_per_window"][i] for s in rec["traffic"]) for i in range(4)],
                final=mean(s["final"] for s in rec["traffic"]), prefetch=mean(s["prefetch"] for s in rec["traffic"]),
                revision=field(root, "native"), cold_build=measured, diagnostic_cold_build=diagnostic), identity


def serial_reference(root, identity, group, batch):
    # Never substitute a result from a different board, placement or strong-serial implementation.
    sources = sorted(root.parent.glob(f"*/g{group}/bs{batch}/block[12]/serial/packed_result.json"))
    matches = {}
    for path in sources:
        base = path.parents[4]
        if (field(base, "native") != REFERENCE or not same_card(root, base) or any(
                field(base, k) != field(root, k) for k in ("kvaf", "kvaf_tree"))):
            continue
        try:
            times, old_id = observation(base, path.parents[1], group, batch, "serial")
        except (OSError, ValueError, TypeError, KeyError):
            continue
        if old_id == identity:
            matches.setdefault(base, []).append(times)
    if not matches:
        return None
    # Prefer a complete two-block archive, then the latest matching archive, never mix blocks from separate runs.
    src = max(matches, key=lambda p:(len(matches[p]), p.stat().st_mtime_ns, p.name))
    return dict(source=str(src), blocks=len(matches[src]), **{k:mean(v[k] for v in matches[src]) for k in ("graph", "LI", "tail")})


def compare(root, ref, g, b):
    new, identity = values(root,g,b)
    old, old_id = values(ref,g,b)
    if field(root, "followup_new_device_build") == "1" and any(
        new[key] == old[key] for key in ("cold_build", "diagnostic_cold_build")
    ):
        raise ValueError(f"G{g} BS{b}: changed device source replayed an archived cold image")
    if identity != old_id:
        raise ValueError(f"G{g} BS{b}: candidate differs in input, device, runtime, or NUMA/UB from {ref}")
    tag = f"G{g} BS{b}"
    print(f"{tag} CHECK TopK_score/KV/Index/private_CQ=PASS calls=10/10 queries={40*b}/{40*b} rows={10*b}/{10*b}")
    for arm, data in (("archived", old), ("candidate", new)):
        parts = " ".join(f"{k}_mean/P50_us={data['means'][k]:.2f}/{data['p50'][k]:.2f}" for k in STAGES)
        posted = "/".join(f"{n:.1f}" for n in data["posted"])
        print(f"{tag} {arm} graph_mean_us={data['graph']:.2f} {parts} "
              f"window_GET_mean={posted} prefetch/final_GET_mean={data['prefetch']:.1f}/{data['final']:.1f}")
        if data["final_lists"] is not None:
            lists = data["final_lists"]
            active = ",".join(f"{r}({n}/10)" for r, n in enumerate(lists["row_ge256_calls"]) if n)
            include_model = arm == "archived" or field(root, "followup_same_pacing") == "1"
            batches = lists["predicted_final_cq_batches_per_call"]
            model = f"{mean(batches):.1f}" if include_model else "UNAVAILABLE_changed_terminal"
            if not include_model:
                lists.pop("predicted_final_cq_batches_per_call")
            print(f"{tag} {arm} row_ids_zero_based_ge256(calls)=[{active}] "
                  f"measured_final_GET_mean={data['final']:.1f} rule_predicted_final_CQ_batches_mean={model}")
            watched = (13, 29) if b == 32 else (13, 18)
            rows = " ".join(f"row{r}=" + "/".join(f"{n:.1f}" for n in lists["mean_final_by_row_q"][r]) for r in watched)
            print(f"{tag} {arm} measured_Q1_to_Q4_final_GET_mean {rows}")
    graph_saved = old["graph"] - new["graph"]
    saved = {k:old["means"][k]-new["means"][k] for k in STAGES}
    parts = " ".join(f"{k}_mean/P50_us={saved[k]:+.2f}/{old['p50'][k]-new['p50'][k]:+.2f}" for k in STAGES)
    print(f"{tag} saved_vs_archived graph_mean_us={graph_saved:+.2f} "
          f"gain_pct={100*graph_saved/old['graph']:+.2f} {parts}")
    serial = serial_reference(root, identity, g, b)
    if serial:
        gain = 100 * (1 - new["graph"]/serial["graph"])
        print(f"{tag} vs_same_card_serial graph_saved_us={serial['graph']-new['graph']:+.2f} "
              f"gain_pct={gain:+.2f} LI_saved_us={serial['LI']-new['means']['LI']:+.2f} "
              f"tail_saved_us={serial['tail']-new['means']['tail']:+.2f} blocks={serial['blocks']} source={serial['source']}")
    else:
        print(f"{tag} vs_same_card_serial=UNAVAILABLE")
    path = root / f"g{g}/bs{b}/comparison.json"
    atomic_json(path,dict(group=g, batch=b, archived=old, candidate=new, saved_mean_us=dict(graph=graph_saved,**saved),
                          serial=serial, input_sha256=identity[0]["sequence_sha256"], placement=identity[1]))


def summarize(root, ref, scope):
    wanted = FOCUS if scope == "focus" else CELLS
    records = [json.loads((root / f"g{g}/bs{b}/comparison.json").read_text()) for g,b in wanted]
    if [(r["group"],r["batch"]) for r in records] != wanted:
        raise ValueError("Incomplete follow-up measurements")
    atomic_json(root / "followup_summary.json", dict(reference=str(ref), candidates=records, scope=scope))
    print(f"SUMMARY={root/'followup_summary.json'} cells={len(records)}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("verify","cell","summary"):
        s=sub.add_parser(name)
        s.add_argument("root", type=Path)
        s.add_argument("ref", type=Path)
        if name == "cell":
            s.add_argument("group",type=int)
            s.add_argument("batch",type=int)
        if name == "summary":
            s.add_argument("scope",choices=("focus","all"))
    a=p.parse_args()
    if a.cmd=="cell": compare(a.root,a.ref,a.group,a.batch)
    elif a.cmd=="verify": verify(a.root,a.ref)
    else: summarize(a.root,a.ref,a.scope)
