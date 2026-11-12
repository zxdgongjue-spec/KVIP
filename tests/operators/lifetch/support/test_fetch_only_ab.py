import numpy as np
import pytest

from fetch_only_ab import TOPK, build_group_plan


def test_plan_matches_per_query_worker_order_and_conflicts():
    batch = 4
    posts = np.zeros((batch, 4, 32), dtype=np.int32)
    for b in range(batch):
        for q in range(4):
            posts[b, q, 4] = b * 8192 + q * TOPK
    for b in range(batch):
        posts[b, 0, 0], posts[b, 0, 5] = 2, 1
    posts[1, 1, 0] = posts[1, 1, 5] = 1
    blocks = np.arange(batch * 8192, dtype=np.int32)
    offsets = blocks + 3
    destinations = blocks + 7
    scratch, counts, queries, all_dest = build_group_plan(posts, blocks, offsets, destinations, batch)
    assert counts.tolist() == [[2, 2], [2, 2], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    assert scratch[0, :4].tolist() == [0, 16384, 1, 16385]
    assert scratch[1, :4].tolist() == [8192, 24576, 8193, 24577]
    stride = batch * TOPK
    assert scratch[0, stride:stride+4].tolist() == [3, 16387, 4, 16388]
    assert len(queries[0]) == 8 and len(queries[1]) == 1 and len(all_dest) == 9
    posts[0, 0, 5] = 3
    with pytest.raises(AssertionError, match='invalid descriptor'):
        build_group_plan(posts, blocks, offsets, destinations, batch)
