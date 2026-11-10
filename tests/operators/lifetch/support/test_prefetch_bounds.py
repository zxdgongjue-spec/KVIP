import numpy as np

from prefetch_audit import score_rows, selected_bounds


def test_selected_bounds_excludes_padding_and_reports_positions():
    selected = np.array([[[0, 131071, -1, 131072, -2]]], dtype=np.int32)
    result = selected_bounds(selected, 131072)
    assert result["invalid_count"] == 2
    assert result["examples"] == [
        {"index": (0, 0, 3), "value": 131072},
        {"index": (0, 0, 4), "value": 4294967294},
    ]


def test_score_rows_distinguishes_zero_and_all_ones():
    raw = np.array([[0, 1], [65535, 2], [3, 4], [0, 0]], dtype='<u2').view(np.uint8)
    rows = score_rows(raw, 2)
    assert rows[0]["zero"] == 1 and rows[0]["maximum"] == 1
    assert rows[1]["all_ones"] == 1 and rows[1]["minimum"] == 2
    assert rows[3]["nonzero"] == 0
