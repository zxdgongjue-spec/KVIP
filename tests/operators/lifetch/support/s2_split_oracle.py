# SPDX-License-Identifier: Apache-2.0
"""P0-C5: CPU-only reference for static S2 score partitions.

NOT a V9 scheduler, device kernel or CANN TopK oracle. A single Query
is partitioned over 1/2/4 AIC groups at 128-token S2 tile boundaries.
Any final score consumer must see a complete logical score row first.
Ordering for this CPU-only oracle is score descending, Token ID ascending.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class S2Shard:
    core: int
    token_start: int
    token_end: int

    @property
    def token_count(self):
        return self.token_end - self.token_start


def plan_static_s2(actual_k, cores, *, resident=8192, tile=128):
    """Return disjoint, full-coverage, tile-aligned S2 ranges.

    First shard must own the entire Resident prefix. Fail closed
    rather than assuming current V9 rotation/prefix semantics can
    safely span multiple physical cores.
    """
    if cores not in (1, 2, 4):
        raise ValueError("only 1/2/4 AIC groups allowed")
    if not isinstance(actual_k, int) or not isinstance(resident, int):
        raise ValueError("actual K and resident must be integers")
    if tile != 128 or actual_k < resident or resident < 2048:
        raise ValueError("requires tile=128, K>=resident>=2048")
    if actual_k % tile or resident % tile:
        raise ValueError("K and resident must be 128-aligned")
    tiles = actual_k // tile
    if tiles < cores:
        raise ValueError("insufficient S2 tiles")
    quotient, remainder = divmod(tiles, cores)
    lens = [quotient + (i < remainder) for i in range(cores)]
    pos = 0
    result = []
    for core, count in enumerate(lens):
        end = pos + count * tile
        result.append(S2Shard(core, pos, end))
        pos = end
    if result[0].token_end < resident:
        raise ValueError("cannot preserve Resident prefix on shard 0")
    assert result[0].token_start == 0 and result[-1].token_end == actual_k
    return tuple(result)


def rank_tokens(scores, start, end, k=2048):
    """CPU deterministic rank; NOT necessarily equal to Native tie order."""
    if not (0 <= start <= end <= len(scores)) or k < 1:
        raise ValueError("bad score interval or TopK count")
    return sorted(range(start, end), key=lambda i: (-float(scores[i]), i))[:k]


def merge_shard_topk(scores, shards, k=2048):
    """Mathematically exact TopK union under rank_tokens total ordering."""
    candidates = []
    for shard in shards:
        candidates.extend(rank_tokens(scores, shard.token_start, shard.token_end, k))
    return sorted(candidates, key=lambda i: (-float(scores[i]), i))[:k]


def main():
    for cores in (1, 2, 4):
        shards = plan_static_s2(131072, cores)
        print(
            "AIC_GROUPS=", cores,
            "SHARDS=", [(s.token_start, s.token_end) for s in shards],
            "MAX_TILES=", max(s.token_count for s in shards) // 128,
            "PARTIAL_CANDIDATES<=", cores * 2048,
        )


if __name__ == "__main__":
    main()
