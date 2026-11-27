# SPDX-License-Identifier: Apache-2.0
"""Verify the ranks actually consumed and compare ten-call windows on the selected card."""
import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean

import numpy as np

from fixed_q4_packed_report import read_arm
from large_batch_placement import field, same_card
from rank_replay_archives import CONTROL_REV, TWO_COLD_REV, newest


def decode_ranks(records, query_ends, expected):
    # The old, short-history encoding packs ready/covered/rank at bits 0/18/36.
    # The replay accepts only the same fixed Q4 128K geometry as the experiment.
    words = np.asarray(records).view(np.uint64)
    ends = np.asarray(query_ends).astype(np.int64)
    if ends.ndim != 2 or ends.shape[1] != 4 or words.shape != (*ends.shape, 32) or len(expected) != 4:
        raise ValueError("Invalid four-window rank trace geometry")
    scheduled = words[:, :, :4]
    if not scheduled.all() or (scheduled >> 63).any() or (ends <= 0).any() or (ends >= 1 << 18).any():
        raise ValueError("Incomplete or long-history rank trace")
    ranks = scheduled >> 36
    if not np.all(ranks == np.asarray(expected)):
        raise ValueError(f"Measured ranks differ from {expected}: {np.unique(ranks.reshape(-1, 4), axis=0)}")
    covered = ((scheduled >> 18) & ((1 << 18) - 1)).astype(np.int64)
    if (np.diff(covered, axis=2) < 0).any() or (covered > ends[:, :, None]).any():
        raise ValueError("Invalid coverage counters")
    initial = covered[:, :, :1]
    remaining = ends[:, :, None] - initial
    if (remaining <= 0).any():
        raise ValueError("No cold region in a 128K rank comparison")
    # Reconstruct the archived coverage-based rule on this diagnostic's actual coverage.
    old = 1024 + ((covered - initial) * 1024 + remaining // 2) // remaining
    return dict(ranks=list(expected), checked_queries=int(ends.size),
                coverage_rule_min=old.min(axis=(0, 1)).tolist(),
                coverage_rule_max=old.max(axis=(0, 1)).tolist())


STAGES = ("graph", "LI1", "LI23", "tail", "prefetch", "final")
ARCHIVES = (
    ("1+3_fixed", "fixed-q4-packed.*", CONTROL_REV, None, None, 2),
    ("1+2_fixed", "fixed-q4-aiv-2cold.*", TWO_COLD_REV, "two-cold", "v9_li_large_two_cold", 2),
)


def read_blocks(root, group, batch, policy, symbol, blocks):
    observations = []
    identity = None
    measured_ranks = None
    for block in range(1, blocks + 1):
        pair = root / f"g{group}/bs{batch}/block{block}"
        row = json.loads((pair / "fused/packed_result.json").read_text())
        obs, ident = read_arm(pair, group, batch, 0, root, "fused", traffic=True)
        if (len(row["samples"]) != 10 or row["full_KV_Index_checked_calls"] != 10
                or row.get("large_window_policy") != policy or row.get("measured_cold_symbol") != symbol):
            raise ValueError(f"Unexpected kernel, protocol or coverage at {pair}")
        for stage in ("LI1", "LI23"):
            obs[stage] = mean(s[stage] for s in row["samples"])
            if not isclose(obs[stage], row["mean_us"][stage], abs_tol=1e-6):
                raise ValueError(f"Invalid {stage} timing at {pair}")
        observations.append(obs)
        if identity and ident != identity:
            raise ValueError(f"Input or placement changed at {pair}")
        identity = ident
        if policy == "four-ranks" and block == 1:
            measured_ranks = row["diagnostic_ranks"]
            if measured_ranks["checked_queries"] != batch * 4 or row["window_grants_per_row"] != [256] * 4:
                raise ValueError(f"Missing rank or per-window budget proof at {pair}")
    return {s: mean(o[s] for o in observations) for s in STAGES}, identity, measured_ranks


def compare(root, group, batch):
    count = int(field(root, "candidate_blocks"))
    if count not in (1, 2) or field(root, "physical_npu") not in ("3", "4"):
        raise ValueError("Expected a completed rank experiment on the selected physical NPU")
    current, identity, ranks = read_blocks(root, group, batch, "four-ranks", "v9_li_large_four_ranks", count)
    print(f"G{group} BS{batch} RANK actual={ranks['ranks']} queries={ranks['checked_queries']} "
          f"old_rule_on_same_coverage_min/max={ranks['coverage_rule_min']}/{ranks['coverage_rule_max']}")
    for label, pattern, revision, policy, symbol, blocks in ARCHIVES:
        linked = field(root, "reference") if label == "1+3_fixed" else field(root, "two_cold_reference")
        choices = [Path(linked)] if linked else newest(root.parent, pattern, revision, root)
        match = None
        for old in choices:
            if (field(old, "native") != revision or not same_card(old, root)
                    or any(not field(root, key) or field(root, key) != field(old, key)
                           for key in ("kvaf", "kvaf_tree"))):
                continue
            try:
                values, old_identity, _ = read_blocks(old, group, batch, policy, symbol, blocks)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as err:
                print(f"G{group} BS{batch} {label} skipped_archive={old} reason={err}")
                continue
            if old_identity == identity:
                match = values
                break
        if match is None:
            print(f"G{group} BS{batch} vs_{label} UNAVAILABLE no matching completed same-card comparison")
            continue
        values = match
        savings = "/".join(f"{values[s] - current[s]:+.2f}" for s in STAGES[:4])
        print(f"G{group} BS{batch} vs_{label} archived_minus_candidate_graph/LI1/LI23/tail_us={savings} "
              f"prefetch_old/new={values['prefetch']:.1f}/{current['prefetch']:.1f} "
              f"final_old/new={values['final']:.1f}/{current['final']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("group", type=int, choices=[2, 10, 4, 17, 7])
    parser.add_argument("batch", type=int, choices=[20, 24, 28, 32])
    args = parser.parse_args()
    compare(args.root.resolve(), args.group, args.batch)
