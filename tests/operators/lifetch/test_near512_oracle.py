# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from near512_oracle import _certify_sets, _sharded_reference


def test_native_shards_include_short_fragments_and_prove_ties():
    # Prefix lengths include a row ending before the last shard, a row with only
    # 1..4 visible positions in its last shard, and a full last shard.
    ends = np.array([[122,123,124,125], [257,258,259,260], [360,361,362,363]], dtype=np.int32)
    k, history, batch = 8, 384, len(ends)
    rng = np.random.default_rng(3)
    by_query = rng.integers(-3, 4, (batch, 4, history))
    logical = np.broadcast_to(np.arange(history)[None,:,None], (batch, history, 1))
    inputs = SimpleNamespace(query_ends=ends, logical_keys=logical, logical_scales=np.ones((batch,history)))
    calls = []

    def native(keys, scales, bounds, mode):
        calls.append((keys.shape[1], mode))
        assert keys.shape[1] % 128 == 0
        out = np.empty((batch,4,k), np.int32)
        for row,q in np.ndindex(batch,4):
            ids = keys[row,:bounds[row,q],0]
            assert len(ids) >= k
            out[row,q] = np.argsort(-by_query[row,q,ids], kind='stable')[:k]
        return out

    expected = native(logical, inputs.logical_scales, ends, 3)
    winner = _sharded_reference(inputs, native, k, 128)
    np.testing.assert_array_equal(expected, winner)
    alternate = np.empty_like(expected)
    for row, q in np.ndindex(batch,4):
        ids = np.arange(ends[row,q] - 1, -1, -1)
        alternate[row,q] = ids[np.argsort(-by_query[row,q,ids], kind='stable')[:k]]
    cache = {}
    _certify_sets(inputs, alternate, winner, native, cache)
    before = len(calls)
    _certify_sets(inputs, alternate, winner, native, cache)
    assert before == len(calls) and cache
    invalid = alternate.copy()
    invalid[0,0,0] = by_query[0,0,:ends[0,0]].argmin()
    with pytest.raises(AssertionError, match='equivalence not proved'):
        _certify_sets(inputs, invalid, winner, native, cache)
    invalid[0,0,0] = ends[0,0]
    with pytest.raises(AssertionError, match='domains'):
        _certify_sets(inputs, invalid, winner, native, cache)
