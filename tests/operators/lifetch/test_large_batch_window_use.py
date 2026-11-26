# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from large_batch_window_use import window_use


def test_midpoint_reports_unique_wanted_prefetch_excluding_terminal_reads():
    records = np.zeros((5, 2, 16), dtype=np.uint64)
    records[:, :, 2] = (np.arange(5, dtype=np.uint64) << 32)[:, None]
    records[:, :, 3] = 1
    records[:, :, 10] = [[0, 0], [1, 0], [2, 1], [1, 0], [0, 0]]
    records[2, :, 12] = [180, 240]
    records[3, :, 4] = [200, 300]
    src = np.array([[7, 10, 11, 99, 22], [5, 44, -1, -1, -1]])
    winners = np.tile(np.arange(1024, 3072), (2, 4, 1))
    winners[0, :, :4] = [7, 10, 11, 22]
    winners[1, :, :2] = [5, 44]
    report = window_use(records, src, winners, [4, 1], 2)
    assert report["posted_by_epoch"] == [0, 1, 3, 1, 0]
    assert report["useful_by_epoch"] == [0, 1, 3, 0, 0]
    assert report["useful_mid_lead_to_next_ready_us"] == pytest.approx((20 * 2 + 60) / 3 / 2)
    assert window_use(records, src, winners + 10000, [4, 1], 2)["useful_mid_lead_to_next_ready_us"] is None

    with pytest.raises(ValueError, match="prefix"):
        window_use(records, src, winners, [5, 1], 2)
    records[2, 0, 12] = 0
    with pytest.raises(ValueError, match="doorbell"):
        window_use(records, src, winners, [4, 1], 2)


@pytest.mark.parametrize("middle", [(129, 0), (128, 129), (0, 257)])
def test_middle_windows_share_the_original_request_quota(middle):
    records = np.zeros((5, 1, 16), dtype=np.uint64)
    records[:, 0, 2] = np.arange(5, dtype=np.uint64) << 32
    records[:, 0, 3] = 1
    records[2:4, 0, 10] = middle
    count = sum(middle)
    with pytest.raises(ValueError, match="quota"):
        window_use(records, np.arange(count)[None, :], np.zeros((1, 4, 2048)), [count], 1)
