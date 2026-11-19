import numpy as np
import pytest

from kvip.diagnostics.cold_window_compare import decode, edges


def test_g7_physical_coverage_and_partial_last_pair():
    original = np.diff(edges(131328, 16384, reference=False))
    reference = np.diff(edges(131328, 16384, reference=True))
    assert list(original) == [112, 112, 112, 113] * 2
    assert list(reference) == [116] * 7 + [86]
    assert sum(reference) == sum(original) == 898


def test_own_fetch_exclusion_including_the_last_window():
    t = np.zeros((64, 8, 8), np.int64)
    cuts = edges(131328, 16384, reference=False)
    for c, e in np.ndindex(64, 8):
        t[c, e, :2] = [1000 + 100*e, 1040 + 100*e]
        t[c, e, 4:6] = cuts[e:e+2]
        if e == c//8:
            t[c, e, 2:4] = [1042 + 100*e, 1050 + 100*e]
    out = decode(t, 2, 1)
    assert out["scan_us"][0] == [20] * 8
    assert out["outside_us"][0] == (740-8)/2
    assert out["outside_us"][-1] == (750-8)/2
    t[63, 7, 1] = 0
    with pytest.raises(ValueError):
        decode(t, 2, 1)


def test_swapped_or_omitted_shard_is_rejected():
    t = np.zeros((64, 8, 8), np.int64)
    with pytest.raises(ValueError, match="tile bounds"):
        decode(t, 1, 3)
