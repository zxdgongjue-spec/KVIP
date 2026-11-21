"""Device receipt checks stay outside graph timing."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "support"))
from query_owner_report import check_offset_plan


def plan():
    # Prefetch occupies 5, including when no selected query needs it. Q1
    # protects 7. Q2 shares token 11 with Q1. Only Q3 needs an overflow slot.
    lru = np.array([[5, 7, 3, 4]])
    ids = np.array([[[11, 7], [11, 12], [13, 7], [11, 13]]])
    loc = np.array([[[3, 7], [3, 4], [0, 7], [3, 0]]])
    src = np.zeros((1, 2051), dtype=int)
    dst = src.copy()
    src[0, 0], dst[0, 0] = 44, 5
    src[0, 2048:], dst[0, 2048:] = [11, 12, 13], [3, 4, 0]
    counts = np.array([[1, 1, 1, 1, 0]])
    posts = np.zeros((1, 4, 32), dtype=int)
    posts[:, :, 0] = posts[:, :, 5] = counts[:, 1:]
    posts[:, :, 12] = 1
    posts[:, 2:, 11] = 1
    inv = np.full((1, 8), -1)
    inv[0, lru[0]] = np.arange(4)
    return src, dst, counts, posts, lru, ids, loc, inv


def test_first_owner_prefix_and_unused_lru():
    assert check_offset_plan(*plan()) == dict(extended_slots=1, independent_rows=1)


@pytest.mark.parametrize("fault", ["reuse", "double", "order", "protected", "owner", "index", "shared_location"])
def test_reject_bad_plan(fault):
    src, dst, counts, posts, lru, ids, loc, inv = plan()
    if fault == "reuse":
        dst[0, -1] = 5
    elif fault == "double":
        dst[0, -1] = 4
    elif fault == "order":
        dst[0, -1] = 1
    elif fault == "protected":
        dst[0, -1] = 7
    elif fault == "owner":
        src[0, -3:] = [12, 11, 13]
    elif fault == "shared_location":
        loc[0, 1, 0] = 6
    else:
        posts[0, 1, 12] = 0
    with pytest.raises(ValueError):
        check_offset_plan(src, dst, counts, posts, lru, ids, loc, inv)


def test_full_cache_scan_replaces_exhausted_candidates():
    # All four reserve-list slots were consumed by prefetch. Three useful
    # misses must go to ordinary hot-cache slots with their original owners.
    src, dst, counts, posts, lru, ids, loc, inv = plan()
    counts[0, 0] = 4
    dst[0, :4] = lru[0]
    dst[0, 2048:] = [0, 1, 2]
    loc[0] = [[0, 7], [0, 1], [2, 7], [0, 2]]
    posts[0, :, 11] = 1
    assert check_offset_plan(src, dst, counts, posts, lru, ids, loc, inv)["extended_slots"] == 3


def test_full_size_q4_allocations_preserve_every_prefetch():
    resident, top, used = 16384, 2048, 1024
    # Only the consumed prefix is populated. All 8192 final IDs are distinct.
    lru = np.full((1, top), -1)
    inv = np.full((1, resident), -1)
    lru[0, :used] = np.arange(0, 2 * used, 2)
    inv[0, lru[0, :used]] = np.arange(used)
    chosen = np.flatnonzero(inv[0] < 0)[:4 * top]
    ids = np.arange(resident, resident + 4 * top).reshape(1, 4, top)
    loc = chosen.reshape(ids.shape)
    src = np.zeros((1, 5 * top), dtype=int)
    dst = src.copy()
    src[0, :used] = np.arange(resident + 4 * top, resident + 4 * top + used)
    dst[0, :used] = lru[0, :used]
    src[0, top:], dst[0, top:] = ids.ravel(), chosen
    counts = np.array([[used, top, top, top, top]])
    posts = np.zeros((1, 4, 32), dtype=int)
    posts[:, :, 0] = posts[:, :, 5] = top
    posts[:, :, 11:13] = 1
    assert check_offset_plan(src, dst, counts, posts, lru, ids, loc, inv)["extended_slots"] == 8192
