import numpy as np
import pytest

from kvip.diagnostics.ranked_audit import decode


def fixture():
    times = np.zeros(16*256, dtype=np.uint64)
    li = times[16*64:16*128].reshape(64, 16)
    events = np.zeros((8, 8, 16), dtype=np.uint64)
    for q in range(64):
        li[q, [12, 13, 14, 15, 0, 1, 7]] = np.array([100, 120, 300, 310, 330, 335, 340])+q
        e = events[q//8, q%8]
        e[2] = q//2
        e[[4, 13, 14, 1]] = np.array([150, 150, 200, 200])+q
    return times, events


def test_phase_sum_and_latest_query_do_not_add_cross_core_maxima():
    times, events = fixture()
    result = decode(times, events, 16, 2, ranked=True)
    assert result["latest_query"]["query"] == 63
    for row in result["records"]:
        assert sum(row["parts_us"].values()) == row["total_us"] == 120
        assert row["parts_us"]["own_fetch"] == 25
        assert row["parts_us"]["final_rank_wait"] == 5
        assert row["parts_us"]["scan_excluding_own_fetch"] == 65
    baseline = decode(times, events, 16, 2, ranked=False)
    assert baseline["phases"]["final_rank_wait"]["max"] == 0
    assert baseline["phases"]["final_rank_work"]["max"] == 15


def test_missing_native_markers_and_wrong_core_are_not_filled_with_zero():
    times, events = fixture()
    times.reshape(-1)[16*64:16*128].reshape(64, 16)[:, 12:16] = 0
    assert decode(times, events, 16, 2, ranked=True)["status"] == "unavailable"
    times, events = fixture()
    events[0, 2, 2] = 0
    with pytest.raises(ValueError, match="duplicate"):
        decode(times, events, 16, 2, ranked=True)


def test_fetch_must_belong_to_own_cold_scan():
    times, events = fixture()
    events[0, 0, 1] = 301
    with pytest.raises(ValueError, match="outside"):
        decode(times, events, 16, 2, ranked=True)
