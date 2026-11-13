"""CPU planning model for KVIP v9 LI variable-Q/K core allocation.

This is a work proxy, NOT a latency prediction. Current v9 LI has
isLDOpen=False: SplitCore assigns nonempty (request,Q-group) tasks instead
of S2 fragments. Hypothetical splitting ignores TopK merge and sync.
"""
from dataclasses import dataclass, asdict
from math import ceil

TILE = 128


@dataclass(frozen=True)
class CoreTask:
    core: int
    request: int
    query_begin: int
    query_count: int
    actual_k: int
    s2_tiles: int
    query_tile_units: int


def _inputs(q_lengths, k_lengths, aic_count):
    qs = tuple(int(q) for q in q_lengths)
    ks = tuple(int(k) for k in k_lengths)
    if not qs or len(qs) != len(ks) or len(qs) > aic_count:
        raise ValueError("requires 1<=Batch<=available AIC, equal-length Q/K arrays")
    if any(not 0 <= q <= 4 for q in qs):
        raise ValueError("current test fixture supports 0..4 Q per request")
    if any(k < 0 or k % TILE for k in ks):
        raise ValueError("actual K must be nonnegative and 128-aligned")
    if any(q and not k for q, k in zip(qs, ks)):
        raise ValueError("nonzero Query count requires nonzero actual K")
    return qs, ks


def current_qli_partition(q_lengths, k_lengths, aic_count=32, mode="auto"):
    """Source-derived v9 BSND, isLDOpen=False SplitCore task assignments.

    A score-like query_tile_units proxy counts Q x S2 tiles; per-tile
    execution cost is not guaranteed linear in Q on AIC hardware.
    """
    qs, ks = _inputs(q_lengths, k_lengths, aic_count)
    if mode not in ("auto", "q4"):
        raise ValueError("mode must be auto or q4")
    b = len(qs)
    cores_per_request = 2 if mode == "auto" and 2 * b <= aic_count else 1
    group_width = 4 // cores_per_request
    nominal_cores = b * cores_per_request
    tasks = []
    for row, (qcount, k) in enumerate(zip(qs, ks)):
        for qstart in range(0, qcount, group_width):
            if not k:
                continue
            qsize = min(group_width, qcount - qstart)
            tiles = ceil(k / TILE)
            tasks.append(CoreTask(len(tasks), row, qstart, qsize, k, tiles, qsize * tiles))
    assert len(tasks) <= nominal_cores
    return {
        "batch": b,
        "mode": mode,
        "available_aic": aic_count,
        "nominal_cores": nominal_cores,
        "active_cores": len(tasks),
        "idle_nominal_cores": nominal_cores - len(tasks),
        "queries_per_task_max": group_width,
        "total_query_tile_units": sum(t.query_tile_units for t in tasks),
        "max_query_tile_units_per_core": max((t.query_tile_units for t in tasks), default=0),
        "max_serial_s2_tiles_per_core": max((t.s2_tiles for t in tasks), default=0),
        "tasks": [asdict(t) for t in tasks],
    }


def optimistic_request_s2_split(q_lengths, k_lengths, core_budget=8):
    """UNIMPLEMENTED ideal: split each request's Q*S2 tile work freely.

    No TopK merge, memory, CQ, Event, metadata or synchronization cost is
    included. The result is a lower bound on makespan, NOT a speedup claim.
    """
    qs, ks = _inputs(q_lengths, k_lengths, core_budget)
    weights = [q * (k // TILE) for q, k in zip(qs, ks)]
    active = [i for i, w in enumerate(weights) if w]
    if not active:
        return {"quota": [0] * len(qs), "ideal_max_query_tile_units": 0,
                "global_lower_bound": 0, "work": weights}
    low, high = 1, max(weights)
    while low < high:
        limit = (low + high) // 2
        need = sum(ceil(weights[i] / limit) for i in active)
        if need <= core_budget:
            high = limit
        else:
            low = limit + 1
    quota = [0] * len(qs)
    for i in active:
        quota[i] = ceil(weights[i] / low)
    remaining = core_budget - sum(quota)
    while remaining:
        i = max(active, key=lambda j: (weights[j] / quota[j], -j))
        quota[i] += 1
        remaining -= 1
    return {
        "quota": quota,
        "ideal_max_query_tile_units": max(ceil(weights[i] / quota[i]) for i in active),
        "global_lower_bound": ceil(sum(weights) / core_budget),
        "work": weights,
    }


SCENARIOS = {
    "balanced_128k": ((4, 4, 4, 4), (131072,) * 4),
    "mixed_32_64_96_128k": ((4, 4, 4, 4), (32768, 65536, 98304, 131072)),
    "extreme_8_8_8_128k": ((4, 4, 4, 4), (8192, 8192, 8192, 131072)),
    "mixed_qk": ((4, 2, 1, 0), (131072, 98304, 32768, 0)),
}


def main():
    for name, (qs, ks) in SCENARIOS.items():
        auto = current_qli_partition(qs, ks, aic_count=32, mode="auto")
        q4 = current_qli_partition(qs, ks, aic_count=32, mode="q4")
        ideal = optimistic_request_s2_split(qs, ks, core_budget=8)
        print(f"\n{name}: Q={list(qs)}, K={list(ks)}")
        print(f"  current auto: active={auto['active_cores']}/{auto['nominal_cores']}, "
              f"max_query_tile_units={auto['max_query_tile_units_per_core']}, "
              f"max_serial_s2_tiles={auto['max_serial_s2_tiles_per_core']}")
        print(f"  current q4: active={q4['active_cores']}/{q4['nominal_cores']}, "
              f"max_query_tile_units={q4['max_query_tile_units_per_core']}")
        print(f"  hypothetical S2 at fixed 8 AIC: quota={ideal['quota']}, "
              f"optimistic_makespan={ideal['ideal_max_query_tile_units']}, "
              f"global_lower_bound={ideal['global_lower_bound']}")
    print("\nAll unit values are analytical work proxies, NOT timings or speedups.")


if __name__ == "__main__":
    main()
