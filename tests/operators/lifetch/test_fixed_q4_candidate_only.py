# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fixed_q4_candidate_only import cell
from large_batch_pair import captured_score_oracle, check_scores
from workloads import ReplayRound


def test_captured_score_oracle_selects_all_strict_winners_and_allows_cutoff_ties():
    history = 2055
    scores = np.zeros((1, 4, history), dtype=np.uint16)
    scores[:, :, 1:2050] = 50
    scores[:, :, 11] = 1000
    scores[:, :, 2050:] = 2000
    valid = np.ones_like(scores, dtype=bool)
    valid[:, :, 2050:] = False
    valid[:, :, 1] = False
    inp = SimpleNamespace(geometry=dict(batch=1, history=history), query_ends=np.full((1, 4), 2050),
                          logical_score_keys=scores, logical_score_valid=valid)
    inp.validate_topk = lambda ids: ReplayRound.validate_topk(inp, ids)
    selected = captured_score_oracle(inp)
    assert selected.shape == (10, 1, 4, 2048)
    assert not selected.flags.writeable
    assert np.all(selected[0] == np.arange(2, 2050))
    assert check_scores(inp, selected[0], selected[3]) == 4
    valid[:, :, 1] = True
    candidate = np.array(selected[0])
    candidate[:, :, -1] = 1
    inp.validate_topk(candidate)
    assert check_scores(inp, np.sort(candidate), selected[0]) == 0
    candidate[:, :, 9] = 0  # omit the strict winner
    with pytest.raises(AssertionError, match="ranking differs"):
        inp.validate_topk(candidate)


def test_fused_only_report_requires_ten_validated_calls_and_local_mapping(tmp_path):
    root = tmp_path
    for k, v in dict(selection_oracle="captured_logical_scores", candidate_blocks=1,
                     native="rank", physical_npu=4, host="b20").items():
        (root / f"{k}.txt").write_text(str(v))
    for batch in (20, 24, 28, 32):
        for group in (2, 4, 7, 10, 17):
            out = root / f"g{group}/bs{batch}/block1/fused"
            out.mkdir(parents=True)
            (out / "placement_verified").touch()
            (out / "thp.json").write_text(json.dumps(dict(status="measured", page_mode="thp",
                numa_pages_overlapping_vmas={"3": 80}, ub_device="ub4", coverage_min_pct=99.8)))
            extra = batch - 16
            identity = dict(group=group, batch=batch, physical_npu=4, host="b20", native="rank",
                            protocol="ten_independent_states_one_graph_one_event_v1",
                            donor_rows=list(range(16)) + [16*i//extra for i in range(extra)], sequence_sha256="f"*64)
            row = dict(arm="fused", identity=identity, correctness="PASS", checked_queries=batch*40, checked_rows=batch*10,
                       selection_oracle="captured_logical_scores", serial_oracle=None, calls=10,
                       full_KV_Index_checked_calls=10, large_window_policy="four-ranks",
                       diagnostic_ranks=dict(checked_queries=batch*4, ranks=[1024,1568,1792,2048]),
                       measured_cold_symbol="v9_li_large_four_ranks", window_grants_per_row=[256]*4,
                       p50_us=dict(LI=260, LI1=30, LI23=230, tail=80), mean_us=dict(LI=260, LI1=30, LI23=230, tail=80),
                       samples=[dict(LI1=30, LI23=230)] * 10, traffic=[dict(prefetch=21, final=7)]*10,
                       graph_mean_us=345, graph_event_us=3450)
            path = out / "packed_result.json"
            path.write_text(json.dumps(row))
            actual = cell(root, group, batch)
            assert actual["mean_us"]["fused"]["graph"] == 345 and actual["graph_gain_pct"] is None
    row["identity"]["physical_npu"] = 3
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="mismatched packed result"):
        cell(root, group, batch)
