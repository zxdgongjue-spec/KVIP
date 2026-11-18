# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.fetch_window_topk import FetchRank, RequestWindows, check_staggered, publication
from kvip.diagnostics.incremental_topk import select


def test_fetch_reference_tracks_all_ties_and_hot_winners():
    rng = np.random.default_rng(946)
    for levels in (1, 2, 3, 7, 64, 256):
        scores = rng.integers(levels, size=160, dtype=np.uint16)
        ids = rng.permutation(len(scores))
        for eligible in (ids % 3 == 0, ids < 0, ids >= 0):
            keeper = FetchRank(scores[:30], ids[:30], eligible, k=20)
            values, ranked, kth = select(scores[:30], ids[:30], 20)
            for start in range(30, len(ids), 7):
                new_scores, new_ids = scores[start:start + 7], ids[start:start + 7]
                keeper.append(new_scores, new_ids, eligible)
                values, ranked, kth = select(np.r_[values, new_scores], np.r_[ranked, new_ids], 20)
                assert keeper.kth == kth
                assert keeper.eligible_ids() == ranked[eligible[ranked]].tolist()


def test_common_version_and_post_budget():
    request = RequestWindows(2)
    request.offer(1, 0, [0, 8])
    request.offer(1, 1, [8, 4])
    assert request.advance() is None
    with pytest.raises(ValueError):
        request.offer(0, 1, [])
    request.offer(0, 0, [0, 1])
    prefix = request.advance()
    posted = set()
    assert publication(prefix, posted, 2, 3) == [0, 1]
    request.offer(0, 1, [1, 2])
    assert publication(request.advance(), posted, 2, 3) == [2]
    assert request.advance() is None
    snapshots = [[[i, q + 8] for q in range(4)] for i in range(9)]
    check_staggered(snapshots, snapshots, 2, 6)


def test_bitmap_evidence_maps_reverse_logical_tiles_to_physical_scores():
    from kvip.diagnostics.fetch_rank_check import bitmap_evidence
    from kvip.diagnostics.incremental_topk import scan_windows

    resident, history = 2048, 20480
    ends = np.full(4, history)
    scores = np.zeros((4, history), dtype=np.uint16)
    logical = scan_windows(history, resident, ends, False)[0]
    physical = scan_windows(history, resident, ends, True)[0]
    words = np.zeros(history // 16, dtype=np.uint32)
    scores[0, physical[[1, 18, 128]]] = 99
    # One wrong bit in a retained tile and one entire missing tile.
    words[logical[1] // 16] = 1 << (2 * (logical[1] % 16))
    msg = bitmap_evidence(words, scores, ends, (0, 0), resident, 0, 1, 98)
    assert "missing_bits=2 missing_nonempty_tiles=1" in msg
    assert f"first_physical_tiles={[int(physical[128]) // 128]}" in msg


def test_window_collector_flushes_without_losing_candidates_or_tie_order():
    from kvip.diagnostics.fetch_window_topk import collect_window

    rng = np.random.default_rng(411)
    seed_ids = np.arange(2048)
    seed = rng.integers(1, 5, 2048, dtype=np.uint16)
    ids = np.arange(2048, 20480).reshape(-1, 2048)[::-1].reshape(-1)
    scores = rng.integers(1, 256, len(ids), dtype=np.uint16)
    state = select(seed, seed_ids)
    actual, counts = collect_window(state, scores, ids)
    expected = select(np.r_[seed, scores], np.r_[seed_ids, ids])
    assert counts[2] > 0
    assert actual[2] == expected[2]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
