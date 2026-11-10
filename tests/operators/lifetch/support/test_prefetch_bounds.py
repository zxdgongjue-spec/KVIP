import numpy as np

from prefetch_audit import selected_bounds


def test_selected_bounds_excludes_padding_and_reports_positions():
    selected = np.array([[[0, 131071, -1, 131072, -2]]], dtype=np.int32)
    result = selected_bounds(selected, 131072)
    assert result["invalid_count"] == 2
    assert result["examples"] == [
        {"index": (0, 0, 3), "value": 131072},
        {"index": (0, 0, 4), "value": 4294967294},
    ]
