"""Cpu-only checks of the diagnostic clock and sparse tile ABI."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from fixed_q4_cold_profile import _read_pass, _wall_buckets


def test_exclusive_wall_counts_overlap_once_and_clips_to_window():
    buckets = _wall_buckets([(10, 20), (30, 38)], [(0, 11, 17), (1, 5, 13), (1, 12, 16), (2, 32, 40)])
    assert {k: v for k, v in buckets.items() if v} == {
        "cold_read": 1, "peer_wait+cold_read": 5, "peer_wait": 1, "other": 5, "score": 6
    }


def _sample(stride, residue):
    ranges = [(2, 4), (4, 6), (6, 8)]
    cap = 9
    words = np.zeros(16 + (cap + 3) * 3 * 16, dtype=np.int64)
    words[:4] = [14, cap, stride, residue]
    records = words[16:].reshape(3, cap + 3, 16)
    for lane in range(3):
        for ep, (start, end) in enumerate(ranges):
            footer = records[lane, cap + ep]
            tick = end * 100
            footer[:8] = np.arange(8) + tick
            footer[8:12] = [start, end, tick + 8, tick + 9]
            for tile in range(start, end):
                if tile % stride != residue:
                    continue
                origin = tile * 100
                ticks = [1, 2, 3, 5, 6, 8, 9, 10] if not lane else [1, 11, 1, 2, 12, 15, 16, 18]
                records[lane, tile, :8] = origin + np.array(ticks)
                if tile % 2:
                    records[lane, tile, 2:4] = 0  # K/scale banks still resident
    return words, ranges


def test_rotating_sparse_matches_full_tile_census():
    bounds = [(200, 400), (410, 600), (602, 810)]
    data, ranges = _sample(1, 0)
    dense, wall = _read_pass(data, bounds, ranges, 10., dense=True)
    assert sum(wall.values()) == pytest.approx(59.8)
    sparse = [_read_pass(_sample(2, r)[0], bounds, ranges, 10., dense=False)[0] for r in (0, 1)]
    for lane in dense:
        assert dense[lane]["count"].tolist() == [6, 3, 6, 6]
        for key in ("count", "raw_us", "bracket_us"):
            np.testing.assert_allclose(sparse[0][lane][key] + sparse[1][lane][key], dense[lane][key])
    # Reject a phase that was only issued or copied halfway, rather than calling it a short read.
    data[16 + 2 * 16 + 3] = 0
    with pytest.raises(ValueError, match="Missing tile timestamps"):
        _read_pass(data, bounds, ranges, 10., dense=True)
