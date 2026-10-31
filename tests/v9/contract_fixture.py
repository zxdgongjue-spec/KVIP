"""Seeded equal-length fixtures. Token IDs, slots and both page tables differ."""

from dataclasses import asdict, dataclass, replace

import numpy as np
from contract_reference import bitmap
from kvip.v9.contract import TOKEN_BYTES, TOPK, Geometry, align


@dataclass(frozen=True)
class Case:
    name: str
    hot: str = "mixed"
    selection: str = "overlap"
    profile: int = 0
    pad: bool = False
    invalid_directory: bool = False
    target_miss_fraction: float | None = None
    controlled: bool = False
    prefetch_decoys: int = 0


CASES = (
    Case("mapped_mixed"),
    Case("hot_empty", hot="empty"),
    Case("hot_short", hot="short"),
    Case("zero_ties", selection="zero"),
    Case("mtp_disjoint", selection="disjoint"),
    Case("dummy", profile=2),
    Case("all_pad", pad=True),
    Case("all_hit", target_miss_fraction=0.0),
    Case("one_miss", target_miss_fraction=1 / TOPK),
    Case("resident_full", hot="full"),
    Case("invalid_directory", invalid_directory=True),
)


def cases(suite, rates=None, prefetch_decoys=0):
    if suite == "performance":
        rates = [10.0] if rates is None else rates
        if prefetch_decoys and any(rate == 0 for rate in rates):
            raise ValueError("non-winning threshold decoys require a nonzero miss rate")
        return tuple(
            replace(
                CASES[0],
                name=f"performance_miss_{rate:g}",
                target_miss_fraction=rate / 100,
                controlled=True,
                prefetch_decoys=prefetch_decoys,
            )
            for rate in rates
        )
    base = {"contract": CASES[:3], "lifecycle": (CASES[0],), "full": CASES}.get(suite)
    if base is None:
        raise ValueError(f"unknown suite: {suite}")
    if rates is None:
        return base
    return tuple(
        replace(CASES[0], name=f"mapped_miss_{rate:g}", target_miss_fraction=rate / 100) for rate in rates
    ) + tuple(case for case in base if case.name != "mapped_mixed")


def make_fixture(
    *, batch=1, queries=4, candidate=16384, budget=12288, tail=129, layers=1, layer=0, seed=7, case=CASES[0]
):
    if candidate < 1 or tail < 0:
        raise ValueError("invalid lengths")
    ccap, scap = align(candidate), align(budget)
    # Directory covers all layers and tail payload seeds; candidate visibility
    # remains C. The extra non-batch pool rows catch p == b assumptions.
    max_requests = batch + 2
    qcap = max(16, ccap // 128, (candidate + tail + 127) // 128)
    icols, rcols = ccap // 128 + 1, (budget + tail + 127) // 128 + 1
    g = Geometry(
        batch,
        queries,
        ccap,
        scap,
        max_requests + 1,
        icols,
        rcols,
        batch * icols + 2,
        batch * rcols + 2,
        layer,
        layers,
        max_requests,
        qcap,
        max_requests * layers * qcap,
        (batch * rcols + 2) * 128,
        ccap + 128,
    ).validate()
    rng = np.random.default_rng(seed + 101 * layer)
    pids = rng.permutation(np.arange(1, max_requests))[:batch].astype(np.int32)
    arrays = {}
    for name, (shape, dtype) in g.tensor_spec().items():
        ndtype = np.uint8 if dtype == "float8_e4m3fn" else np.float32 if dtype == "bfloat16" else np.dtype(dtype)
        arrays[name] = np.zeros(shape, ndtype)
    for name in ("forward", "reverse", "topk", "destination"):
        arrays[name].fill(-1)
    arrays["resident_output"].fill(0xA5)
    arrays["pool_rows"][:] = pids
    arrays["cumulative_query"][:] = np.arange(1, batch + 1) * queries
    arrays["candidate_lens"].fill(candidate)
    arrays["logical_lens"].fill(candidate + tail)
    arrays["budgets"].fill(budget)
    arrays["row_modes"].fill(0 if case.pad else 2)
    arrays["profile"][0] = case.profile
    arrays["index_table"][:] = rng.permutation(np.arange(1, batch * icols + 1)).reshape(batch, icols)
    arrays["resident_table"][:] = rng.permutation(np.arange(1, batch * rcols + 1)).reshape(batch, rcols)
    # Synthetic addresses belong exclusively to the CPU/simulator model.
    # RealFetchSession replaces them with KVAF's actual prepared table.
    arrays["address_table"][:] = np.arange(g.table_capacity, dtype=np.int64) * (128 * TOKEN_BYTES) + 0x100000000
    for pool in range(max_requests):
        for layer_index in range(layers):
            begin = (pool * layers + layer_index) * qcap
            arrays["valid_table"][begin : begin + (candidate + 127) // 128] = 1
    arrays["index_dirty"][:, 1] = candidate
    arrays["query_scale"].fill(1)
    arrays["key_scale"].fill(1)
    arrays["weights"][:, 0] = 1
    scores = np.zeros((batch, queries, candidate), np.float32)
    for b, p in enumerate(pids):
        victim = rng.choice(budget, TOPK, replace=False).astype(np.int32)
        arrays["victims"][b] = victim
        arrays["victim_bitmap"][b] = bitmap(victim)
        nonvictim = np.arange(budget) if case.hot == "full" else np.setdiff1d(np.arange(budget), victim)
        count = min(len(nonvictim), candidate)
        if case.hot == "empty":
            count = 0
        elif case.hot == "short":
            count = min(37 + b, count)
        elif case.hot != "full":
            count = max(0, count - 129 * (b + 1))
        tokens = rng.choice(candidate, count, replace=False)
        slots = rng.permutation(nonvictim)[:count]
        arrays["forward"][p, tokens] = slots
        arrays["reverse"][p, slots] = tokens
        # Valid victim owners are part of H0; only their below-threshold slots may be used.
        remaining = np.setdiff1d(np.arange(candidate), tokens)
        old = remaining[: 0 if case.controlled or case.hot in ("empty", "full") else min(len(remaining), 97)]
        arrays["forward"][p, old] = victim[: len(old)]
        arrays["reverse"][p, victim[: len(old)]] = old
        # Stale reverse aliases are not members of H0 and must not erase the
        # canonical forward entry when the alias slot is overwritten.
        holes = np.setdiff1d(nonvictim, slots)
        if len(tokens) and len(holes):
            arrays["reverse"][p, holes[0]] = tokens[0]
        tokens = np.concatenate((tokens, old))
        arrays["cursors"][p] = budget - 17 - b
        order = rng.permutation(candidate)
        key_scales = rng.choice(np.asarray([0.5, 1.0, 2.0], np.float32), candidate)
        if case.controlled:
            key_scales.fill(1)
        for q in range(queries):
            arrays["query"][b * queries + q, 0, q] = 0x38  # exact FP8 1
            visible = max(0, candidate - queries + q + 1)
            values = np.zeros(candidate, np.uint8)
            if case.selection != "zero":
                values[:] = 0x38  # 1
                if case.controlled:
                    values.fill(0)
                    values[tokens] = 0x38  # hot background gives the strict lower bound
                if case.target_miss_fraction is not None:
                    wanted = min(TOPK, visible)
                    misses = round(wanted * case.target_miss_fraction)
                    h = order[(order < visible) & np.isin(order, tokens)]
                    cold = order[(order < visible) & ~np.isin(order, tokens)]
                    if len(h) < wanted - misses or len(cold) < misses:
                        raise ValueError("requested exact miss rate is impossible for this candidate/hot size")
                    winners = np.concatenate((h[: wanted - misses], cold[:misses]))
                elif case.selection == "disjoint":
                    # Ordered disjoint winners survive every query's causal mask.
                    winners = np.arange(q * TOPK, min((q + 1) * TOPK, max(0, candidate - queries + 1)))
                else:
                    winners = order[order < visible][: min(TOPK, visible)]
                if case.controlled and case.prefetch_decoys:
                    decoys = cold[~np.isin(cold, winners)][: case.prefetch_decoys]
                    if len(decoys) != case.prefetch_decoys:
                        raise ValueError("not enough visible cold tokens for requested decoys")
                    values[decoys] = 0x38  # equal to hot bound, strictly below final winners
                values[winners] = 0x58  # exact FP8 16; cutoff separated from background
            query_scale = 0.5 if (b + q) % 2 == 0 else 2.0
            arrays["query_scale"][b * queries + q, 0] = query_scale
            scores[b, q] = np.where(values == 0x58, 16, np.where(values == 0x38, 1, 0)) * key_scales * query_scale
            for page in range((candidate + 127) // 128):
                physical = int(arrays["index_table"][b, page])
                n = min(128, candidate - page * 128)
                arrays["key"][physical, :n, 0, q] = values[page * 128 : page * 128 + n]
                arrays["key_scale"][physical, :n, 0] = key_scales[page * 128 : page * 128 + n]
    if case.invalid_directory:
        p = int(pids[0])
        arrays["valid_table"][(p * layers + layer) * qcap] = 0
    return g, arrays, scores


def to_device(torch, arrays, geometry, device):
    result = {}
    for name, (_, dtype) in geometry.tensor_spec().items():
        tensor = torch.from_numpy(arrays[name])
        if dtype == "float8_e4m3fn":
            tensor = tensor.view(torch.float8_e4m3fn)
        else:
            tensor = tensor.to(getattr(torch, dtype))
        result[name] = tensor.to(device)
    return result


def advance_victims(arrays, geometry, step):
    """Advance all requests together; never use outputs from a failed call."""
    for b in range(geometry.batch):
        budget = int(arrays["budgets"][b])
        slots = (np.arange(TOPK, dtype=np.int32) * 5 + step * 257 + b * 31) % budget
        # Supported budgets are multiples of 128; factor five is coprime to them.
        if len(np.unique(slots)) != TOPK:
            slots = (np.arange(TOPK, dtype=np.int32) + step * 257) % budget
        arrays["victims"][b] = slots
        arrays["victim_bitmap"][b] = bitmap(slots)


def manifest(geometry, case):
    return {
        "geometry": asdict(geometry),
        "case": asdict(case),
        "uniform_lengths": True,
        "payload_bytes": TOKEN_BYTES,
        "topk": TOPK,
        "address_model": "pool-layer-page",
        "layout": "TND/PA_BSND",
        "sparse_mode": 3,
    }


def with_layer(geometry, layer):
    return replace(geometry, layer_id=layer).validate()
