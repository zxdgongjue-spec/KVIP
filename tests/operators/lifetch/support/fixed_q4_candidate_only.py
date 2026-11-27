# SPDX-License-Identifier: Apache-2.0
"""Report a timed four-rank candidate when this host has no same-card reference."""
import argparse
import json
from pathlib import Path

from fixed_q4_window_ranks import read_blocks
from large_batch_placement import field
from reporting import atomic_json


def cell(root, group, batch):
    if field(root, "selection_oracle") != "captured_logical_scores" or field(root, "candidate_blocks") != "1":
        raise ValueError(f"Unexpected candidate-only source at {root}")
    observed, (identity, placement), ranks = read_blocks(
        root, group, batch, "four-ranks", "v9_li_large_four_ranks", 1
    )
    path = root / f"g{group}/bs{batch}/block1/fused/packed_result.json"
    record = json.loads(path.read_text())
    if record.get("selection_oracle") != "captured_logical_scores" or record.get("serial_oracle") is not None:
        raise ValueError(f"Unexpected correctness oracle at {path}")
    expected_donors = list(range(16)) + [i * 16 // (batch - 16) for i in range(batch - 16)]
    if identity["donor_rows"] != expected_donors or identity["protocol"] != "ten_independent_states_one_graph_one_event_v1":
        raise ValueError(f"Unexpected donor requests or protocol at {path}")
    graph = observed["graph"]
    means = {stage: observed[stage] for stage in ("graph", "LI1", "LI23", "tail")}
    means["LI"] = record["mean_us"]["LI"]
    return dict(group=group, batch=batch, cohorts=1, blocks=1, calls_per_graph=10, calls_by_arm={"fused": 10},
                mean_us={"fused": means}, stage_p50_us={"fused": record["p50_us"]}, graph_event_us=graph * 10,
                source_sha256=identity["sequence_sha256"], oracle=record["selection_oracle"],
                measured_NUMA_UB=placement, ranks=ranks, correctness=record["correctness"],
                tokens_per_call={"fused": {key: observed[key] for key in ("prefetch", "final")}},
                reference_status="unavailable_same_card", graph_gain_pct=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_subparsers(dest="mode", required=True)
    single = modes.add_parser("cell")
    single.add_argument("root", type=Path)
    single.add_argument("group", type=int)
    single.add_argument("batch", type=int, choices=(20, 24, 28, 32))
    summary = modes.add_parser("summary")
    summary.add_argument("root", type=Path)
    summary.add_argument("--groups", required=True, type=int, nargs="+")
    summary.add_argument("--batches", required=True, type=int, nargs="+", choices=(20, 24, 28, 32))
    args = ap.parse_args()
    if args.mode == "cell":
        row = cell(args.root, args.group, args.batch)
        tag = f"G{args.group} BS{args.batch}"
        times, p50 = row["mean_us"]["fused"], row["stage_p50_us"]["fused"]
        print(f"{tag} CHECK TopK_score_vs_captured_scores/KV/Index/private_CQ=PASS "
              f"calls=10/10 queries={args.batch * 40}/{args.batch * 40}")
        print(f"{tag} fused graph_mean_us={times['graph']:.2f} " + " ".join(
            f"{k}_mean/P50_us={times[k]:.2f}/{p50[k]:.2f}" for k in ("LI", "LI1", "LI23", "tail")))
        print(f"{tag} RANK actual={row['ranks']['ranks']} vs_serial/previous_fused=UNAVAILABLE no_same_card_archive")
        return
    rows = [cell(args.root, group, batch) for group in args.groups for batch in args.batches]
    if len({tuple(r["measured_NUMA_UB"]) for r in rows}) != 1:
        raise ValueError("NUMA or adapter changed between measured cells")
    atomic_json(args.root / "packed_summary.json", dict(protocol="ten_independent_calls_one_device_event",
                blocks=1, measured_arms=["fused"], cells=rows, reference_status="unavailable_same_card"))
    print(f"SUMMARY={args.root / 'packed_summary.json'} cells={len(rows)} vs_serial/previous_fused=UNAVAILABLE")


if __name__ == "__main__":
    main()
