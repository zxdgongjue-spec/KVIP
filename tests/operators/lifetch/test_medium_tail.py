# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from correctness import FETCH_AB_MAGIC, validate_terminal_receipts
from medium_tail_report import plan, traffic


def test_medium_boundaries_and_private_receipts():
    rows = [[64,63,63,1], [63,64,65,0], [0,127,128,0], [253,1,2,0], [0,0,0,0]]
    expected = [
        [(64,), (63,), (63,), (1,)],
        [(63,), (32,32), (33,32), ()],
        [(), (64,63), (64,64), ()],
        [(16,79,79,79), (1,), (1,1), ()],
        [(), (), (), ()],
    ]
    batches = [[plan(n,sum(row),True) for n in row] for row in rows]
    assert batches == expected
    counts = np.array(rows)
    views = list(range(20))[::-1]
    statuses = np.zeros((20,4),dtype=np.uint64)
    statuses[views,0] = FETCH_AB_MAGIC
    statuses[views,1] = counts.ravel()
    statuses[views,3] = [len(b) for row in expected for b in row]
    validate_terminal_receipts(counts,statuses,views,medium_rows=True)
    with pytest.raises(RuntimeError,match="row=1 Q=2"):
        validate_terminal_receipts(counts,statuses,views)
    statuses[views[5],3] = 1
    with pytest.raises(RuntimeError,match="row=1 Q=2"):
        validate_terminal_receipts(counts,statuses,views,medium_rows=True)


def test_report_requires_actual_medium_receipts():
    data = {"identity":{"batch":1}, "traffic":[{"row_query_final":[[193,0,0,0]],"final":193}]*10,
            "measured_private_final_CQEs":[[[2,0,0,0]]]*10}
    assert traffic(data, True) == [193,97,97,2]
    assert traffic(data, False) == [193,193,193,1]
    data["measured_private_final_CQEs"] = [[[1,0,0,0]]]*10
    with pytest.raises(ValueError,match="disagree"):
        traffic(data, True)


def test_comparison_of_archives_and_missing_reference(monkeypatch,capsys):
    import medium_tail_report as report
    def sample(medium):
        return dict(identity={"batch":1}, mean_us={"LI":4,"LI1":1,"LI23":3,"tail":6},
                    graph_mean_us=10 if medium else 12, original_library="base", correctness="PASS",
                    original_tail_text="old", tail_text="new" if medium else "old",
                    medium_tail=medium, cold_build_fingerprint="earned", traffic=[{
                        "row_query_final":[[193,0,0,0]],"row_window_prefetch":[[1,1,1,1,1]], "final":193}]*10,
                    measured_private_final_CQEs=[[[2,0,0,0]]]*10)
    monkeypatch.setattr(report,"reference",lambda area,batch,reader,mode: None if mode==report.MODE
                        else (sample(mode==report.MEDIUM), "node3"))
    report.compare(None,32,"ref")
    text=capsys.readouterr().out
    assert "saved_mean_us=+2.00" in text or "saved_mean_us=2.00" in text
    assert "matched_row_Q_counts=10/10 identical_LI_build=1" in text
    assert "vs_window-x4-early UNAVAILABLE" in text
