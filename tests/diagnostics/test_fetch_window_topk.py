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
