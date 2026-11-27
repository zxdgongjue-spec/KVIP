# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fixed_q4_window_ranks import decode_ranks


def test_device_ranks_checked_for_every_query_and_compared_with_actual_coverage():
    ranks = (1024, 1568, 1792, 2048)
    words = np.zeros((2, 4, 32), dtype=np.uint64)
    for epoch, (count, rank) in enumerate(zip((1000, 1100, 1200, 1300), ranks)):
        words[:, :, epoch] = (rank << 36) | (count << 18) | count
    ends = np.full((2, 4), 1300)
    result = decode_ranks(words, ends, ranks)
    assert result["checked_queries"] == 8
    assert result["coverage_rule_min"] == result["coverage_rule_max"] == [1024, 1365, 1707, 2048]
    words[1, 3, 2] ^= np.uint64(1 << 36)
    with pytest.raises(ValueError, match="Measured ranks differ"):
        decode_ranks(words, ends, ranks)
    words[1, 3, 2] = 0
    with pytest.raises(ValueError, match="Incomplete"):
        decode_ranks(words, ends, ranks)


def test_compare_reuses_both_completed_ten_call_controls(tmp_path, capsys):
    import json
    from fixed_q4_window_ranks import ARCHIVES, compare

    def make(root, revision, policy, symbol, blocks, graph):
        root.mkdir()
        for k, v in {"host": "b24", "physical_npu": "3", "machine_id": "host-machine",
                     "native": revision, "packed_source": "frozen", "kvaf": "sdk",
                     "kvaf_tree": "tree", "candidate_blocks": str(blocks)}.items():
            (root / f"{k}.txt").write_text(v)
        for block in range(1, blocks + 1):
            path = root / f"g7/bs32/block{block}/fused"
            path.mkdir(parents=True)
            (path / "placement_verified").touch()
            (path / "thp.json").write_text(json.dumps(dict(status="measured", page_mode="thp",
                numa_pages_overlapping_vmas={"3": 40}, ub_device="nic", coverage_min_pct=100)))
            ranks = dict(ranks=[1024, 1568, 1792, 2048], coverage_rule_min=[1024, 1365, 1706, 2048],
                         coverage_rule_max=[1024, 1366, 1707, 2048], checked_queries=128)
            record = dict(identity=dict(group=7, batch=32, host="b24", physical_npu=3, native=revision),
                          arm="fused", checked_queries=1280, checked_rows=320, calls=10, correctness="PASS",
                          samples=[{"LI1": 10, "LI23": 80}] * 10, mean_us=dict(LI=90, LI1=10, LI23=80, tail=30),
                          graph_mean_us=graph, graph_event_us=10 * graph, large_window_policy=policy,
                          measured_cold_symbol=symbol, full_KV_Index_checked_calls=10,
                          traffic=[dict(prefetch=5, final=2)] * 10,
                          window_grants_per_row=[256] * 4, diagnostic_ranks=ranks)
            (path / "packed_result.json").write_text(json.dumps(record))

    new = tmp_path / "new"
    make(new, "candidate", "four-ranks", "v9_li_large_four_ranks", 1, 170)
    for _, name, revision, policy, symbol, blocks in ARCHIVES:
        make(tmp_path / name, revision, policy, symbol, blocks, 180)
    compare(new, 7, 32)
    assert capsys.readouterr().out.count("archived_minus_candidate_graph/LI1/LI23/tail_us=+10.00") == 2
    (tmp_path / ARCHIVES[1][1] / "machine_id.txt").write_text("different")
    with pytest.raises(ValueError, match="Wrong reference"):
        compare(new, 7, 32)
