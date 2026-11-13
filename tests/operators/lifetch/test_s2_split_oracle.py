# SPDX-License-Identifier: Apache-2.0
"""CPU admission only, no A5 kernel or CANN TopK validation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from s2_split_oracle import (  # noqa: E402
    plan_static_s2, rank_tokens, merge_shard_topk,
)


@pytest.mark.parametrize("k,cores", [
    (8192, 1), (32768, 1), (32768, 2), (32768, 4),
    (65536, 2), (65536, 4), (98304, 2), (98304, 4),
    (131072, 2), (131072, 4),
])
def test_shards_cover_actual_k_and_resident_prefix(k, cores):
    shards = plan_static_s2(k, cores)
    assert len(shards) == cores
    assert shards[0].token_start == 0
    assert shards[0].token_end >= 8192
    assert shards[-1].token_end == k
    assert all(s.token_count > 0 and s.token_count % 128 == 0 for s in shards)
    assert sum(s.token_count for s in shards) == k
    assert all(a.token_end == b.token_start for a, b in zip(shards, shards[1:]))
    assert max(s.token_count for s in shards) - min(s.token_count for s in shards) <= 128


@pytest.mark.parametrize("k,cores", [(8192, 2), (8192, 4), (16384, 4)])
def test_reject_partition_breaking_resident_owner(k, cores):
    with pytest.raises(ValueError, match="Resident prefix"):
        plan_static_s2(k, cores)


@pytest.mark.parametrize("k,cores", [(32768, 2), (32768, 4), (131072, 2), (131072, 4)])
def test_global_topk_equals_union_of_local_topk_with_tail_winners(k, cores):
    rng = np.random.default_rng(20261113)
    scores = rng.random(k, dtype=np.float32)
    scores[0] = 2.0
    scores[8191] = 2.5
    scores[k - 128] = 3.0
    scores[k - 1] = 4.0
    shards = plan_static_s2(k, cores)
    actual = merge_shard_topk(scores, shards)
    expected = rank_tokens(scores, 0, k)
    assert actual == expected
    assert k - 1 in actual and k - 128 in actual


def test_cpu_tie_order_is_explicit_but_not_claimed_to_match_native():
    scores = np.full(32768, 0.5, dtype=np.float32)
    scores[-1] = 10.0
    shards = plan_static_s2(32768, 4)
    actual = merge_shard_topk(scores, shards)
    assert actual == rank_tokens(scores, 0, 32768)
    assert actual[0] == 32767


@pytest.mark.parametrize("k,cores,resident", [
    (131071, 2, 8192), (131072, 3, 8192),
    (131072, 4, 8193), (4096, 2, 8192), (131072, 0, 8192),
])
def test_invalid_geometry_fails_closed(k, cores, resident):
    with pytest.raises(ValueError):
        plan_static_s2(k, cores, resident=resident)
