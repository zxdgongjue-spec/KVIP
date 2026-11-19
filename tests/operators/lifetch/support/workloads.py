# SPDX-License-Identifier: Apache-2.0
"""Workload configuration, synthetic inputs and captured replay state."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

DEFAULT_BATCHES = (1, 4, 8, 16)

CHECK_SUITES = ('init', 'basic', 'boundary', 'stateful', 'contract', 'real')

STAGES = ('init', 'li', 'resolve', 'update', 'fetch', 'finalize')


TOPK = 2048


WIDTH = 128

QUERIES = 4

# Scalar targets from the user's real-trace summary, not raw captured operands.
GENERATOR_REFERENCE = dict(query_std=164.76, keys_std=169.75, weights_std=4.40,
                 weights_mean=1.10, query_scale_mean=.01972,
                 query_scale_std=.003658, key_scale_mean=.015479,
                 key_scale_std=.098444, q_cosine=.83724,
                 sample_top_jaccard=.87422, sample_union_ratio=1.14014,
                 score_p50=-2037.68, score_p90=-1570.17, score_p99=-477.53,
                 score_zero_pct=.99182)

# Geometric approximation; intentionally not the previous learned covariance model.
GENERATOR_DEFAULTS = dict(alignment=.2, key_direction=.75, scale=.0055, scale_spread=.15)


TOKEN_BYTES = 656

BLOCK = 128

BASE_ARRAYS = frozenset({
    'pre.row_owners', 'pre.logical_ids', 'pre.hot_valid', 'pre.metadata',
    'pre.query_ends', 'pre.tail_bounds', 'pre.lru_slots', 'pre.address_valid',
    'pre.ddr_page_slots', 'post.logical_ids', 'post.hot_valid',
    'post.logical_output', 'post.slot_output', 'post.topk',
    'post.prefetch_records', 'post.final_records', 'post.read_counts',
})

SCORE_ARRAYS = frozenset({
    'post.score_keys', 'post.threshold_records', 'post.window_posted',
    'post.window_processed_ends', 'post.prefetch_ordinals',
})

REAL_ARRAYS = frozenset({
    'real.keys', 'real.key_scales', 'real.query', 'real.weights', 'real.query_scale',
})


HISTORY, SOURCE, STRIDE, TOPK, QUERIES = 131072, 131075, 131328, 2048, 4


def csv_values(value):
    values = value.split(',')
    if any(not item or item.strip() != item for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('expected distinct comma-separated values')
    return tuple(values)


def miss_values(value):
    try:
        values = tuple(float(item) for item in csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError('miss must be comma-separated percentages') from error
    if any(not math.isfinite(item) or not 0 <= item <= 100 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('miss percentages must be distinct finite values in 0..100')
    return values[0] if len(values) == 1 else values


def batch_values(value):
    try:
        batches = tuple(int(item) for item in csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError('BS must be comma-separated integers') from error
    if any(not 1 <= batch <= 16 for batch in batches):
        raise argparse.ArgumentTypeError('LIFetch supports BS1..16')
    return batches


def round_range(value):
    try:
        start, end = map(int, value.split(':'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('round range must be START:END') from error
    if not 0 <= start <= end:
        raise argparse.ArgumentTypeError('round range must be increasing and nonnegative')
    return start, end


def configuration(argv=None):
    parser = argparse.ArgumentParser(description='Unified LIFetch tests')
    commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('all', 'check', 'bench', 'diagnose'):
        command = commands.add_parser(mode)
        command.add_argument('--transport', choices=('urma', 'ubmem'), default='urma')
        command.add_argument('--device', type=int, default=0)
        command.add_argument('--kvaf-root', type=Path, default=ROOT / 'third_party/KVAF')
        command.add_argument('--kvaf-server')
        command.add_argument('--output', type=Path, default=ROOT / 'tests/results/lifetch')
        command.add_argument('--timeout-seconds', type=int, default=300)
        if mode == 'check':
            command.add_argument('--suite', type=csv_values, default=('all',))
            command.add_argument('--cases', type=csv_values)
            command.add_argument('--snapshot', type=Path)
        if mode in ('bench', 'diagnose'):
            command.add_argument('--input', choices=('synthetic', 'real'), default='synthetic')
            # None distinguishes absent arguments from explicitly supplied defaults on real input.
            command.add_argument('--miss-percent', type=miss_values if mode == 'bench' else float)
            command.add_argument('--queries', type=int)
            command.add_argument('--batches', type=batch_values)
            command.add_argument('--seed', type=int)
            command.add_argument('--history', type=int)
            command.add_argument('--resident', type=int)
            command.add_argument('--index-heads', type=int)
            command.add_argument('--trace-root', type=Path)
            command.add_argument('--group', type=int)
            command.add_argument('--round-range', type=round_range)
            command.add_argument('--prefetch-tokens', type=int, default=512)
            command.add_argument('--prefetch-rank-start', type=int, default=1024,
                                 help='URMA initial admission rank (1..2048); progresses to 2048 by per-Q coverage')
            command.add_argument('--li-score-parts', type=int, default=0,
                                 help='experimental URMA history score parts; 0 keeps the default')
            command.add_argument('--li-score-mode', choices=('full', 'window', 'window_query'), default='full',
                                 help='window uses MR23 ranked Service; window_query separates per-Q TopK from request posting')
            command.add_argument('--intermediate-fetches', type=int, default=8)
            command.add_argument('--warmup', type=int, default=4, help='whole-graph warmup repetitions')
            command.add_argument('--iterations', type=int, default=20, help='independent complete invocations captured in one measured graph')
        if mode == 'diagnose':
            command.add_argument('--level', type=int, choices=(1, 2, 3), default=1)
            command.add_argument('--stages', type=csv_values, default=('all',))
    converter = commands.add_parser('convert-trace', help='CPU-only legacy trace conversion')
    converter.add_argument('source', type=Path)
    converter.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'convert-trace':
        return args
    if args.device < 0 or args.timeout_seconds <= 0:
        parser.error('device must be nonnegative and timeout positive')
    if args.mode == 'check':
        if 'all' in args.suite and len(args.suite) != 1:
            parser.error('suite all cannot be combined with individual suites')
        if set(args.suite) - {*CHECK_SUITES, 'all'}:
            parser.error('unknown check suite')
        if 'real' in args.suite and args.snapshot is None:
            parser.error('suite real requires --snapshot')
        if args.suite == ('all',):
            args.suite = CHECK_SUITES if args.snapshot else CHECK_SUITES[:-1]
    if args.mode in ('bench', 'diagnose'):
        synthetic = dict(miss_percent=10., queries=4, batches=DEFAULT_BATCHES, seed=43,
                         history=131072, resident=16384, index_heads=32)
        real = ('trace_root', 'group', 'round_range')
        if args.input == 'real':
            if any(getattr(args, key) is not None for key in synthetic):
                parser.error('real input geometry comes from trace; synthetic options are not allowed')
            if any(getattr(args, key) is None for key in real):
                parser.error('real input requires --trace-root, --group, --round-range')
            if args.group < 0:
                parser.error('group must be nonnegative')
        else:
            if any(getattr(args, key) is not None for key in real):
                parser.error('trace options require --input real')
            for key, default in synthetic.items():
                if getattr(args, key) is None:
                    setattr(args, key, default)
            misses = args.miss_percent if isinstance(args.miss_percent, tuple) else (args.miss_percent,)
            if any(not math.isfinite(miss) or not 0 <= miss <= 100 for miss in misses) or args.queries != 4:
                parser.error('current full-chain fixture supports Q4 and miss in 0..100')
            if (args.history % 256 or args.resident % 256 or args.resident < 2048
                    or args.history <= args.resident + 4 or args.index_heads not in (32, 64)
                    or args.seed < 0):
                parser.error('invalid synthetic geometry or seed')
        if not 1 <= args.prefetch_rank_start <= 2048:
            parser.error('prefetch-rank-start must be in 1..2048')
        if not 0 <= args.li_score_parts <= 32 or (args.li_score_parts and args.transport != 'urma'):
            parser.error('li-score-parts must be 0..32; the experimental entry requires URMA')
        if args.li_score_parts and args.input == 'synthetic' and args.index_heads != 32:
            parser.error('the experimental T2 score entry requires 32 Index heads')
        if args.li_score_mode in ('window', 'window_query') and (not args.li_score_parts or args.input != 'synthetic'):
            parser.error('window score requires li-score-parts and synthetic Q4 inputs; serving replay is not validated')
        if args.transport != 'urma' and args.prefetch_rank_start != 1024:
            parser.error('prefetch-rank-start requires the URMA ranked prefetch path')
        if (args.warmup < 0 or args.iterations <= 0 or not 1 <= args.intermediate_fetches <= 32
                or not 0 <= args.prefetch_tokens <= 2048
                or args.prefetch_tokens % args.intermediate_fetches):
            parser.error('invalid timing or prefetch configuration')
    if args.mode == 'diagnose':
        if 'all' in args.stages and len(args.stages) != 1:
            parser.error('stages all cannot be combined with individual stages')
        if set(args.stages) - {*STAGES, 'all'}:
            parser.error('unknown diagnostic stage')
        if args.stages == ('all',):
            args.stages = STAGES
    return args


def prepare_fixture(root, history, resident, queries, cold_scale=1.0, cold_active_tiles=None, causal=False):
    if history % 256 or not 2048 <= resident <= history or resident % 256:
        raise ValueError("history/resident must be aligned and resident >= Top2048")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(73121)
    hot = rng.choice(history, resident, replace=False)
    cold = np.setdiff1d(np.arange(history), hot)
    actual = history + (queries - 1 if causal else 0)
    storage = (actual + 127) // 128 * 128
    visible = [history + q if causal else history for q in range(queries)]
    order = np.concatenate((hot, cold, np.arange(history, storage))).astype(np.int32)
    values = rng.integers(0, 4, (storage, 128), dtype=np.uint8)
    if causal:
        values[history:actual] = 3
    key = np.array([0, 0x38, 0x40, 0x44], np.uint8)[values][order]
    query = np.zeros((queries, 64, 128), np.uint8)
    # Exact binary fractions expose stale/repeated query scaling without
    # adding CPU-vs-device accumulation-rounding ambiguity to this fixture.
    head = np.arange(64)[None, :]
    query_index = np.arange(queries)[:, None]
    query_scale = np.array([0.5, 1.0, 2.0], np.float32)[(head + query_index) % 3]
    weights = np.array([1.0, 2.0, 0.5], np.float32)[(head * 2 + query_index) % 3]
    logical_scores = []
    scale = rng.uniform(0.25, 2, storage).astype(np.float32)
    if causal:
        scale[history:actual] = 4  # Visible tails must win; invisible ones must never be fetched.
        scale[actual:] = 0
    if not np.isfinite(cold_scale) or cold_scale < 0:
        raise ValueError("cold_scale must be finite and nonnegative")
    scale[cold] *= np.float32(cold_scale)
    if cold_active_tiles is not None:
        if not 0 <= cold_active_tiles <= (history - resident) // 128:
            raise ValueError("cold_active_tiles outside cold region")
        scale[cold[cold_active_tiles * 128 :]] = 0
    for q in range(queries):
        columns = (np.arange(64) + q * 16) % 128
        query[q, np.arange(64), columns] = 0x38
        score = (values[:, columns].astype(np.float32) * (weights[q] * query_scale[q])).sum(-1) * scale
        # Ascend CAST_ROUND is ties-away, unlike NumPy/PyTorch BF16 defaults.
        logical_scores.append(((score.view(np.uint32) + 0x8000) >> 16).astype(np.uint16))
    metadata = np.zeros(1024, np.uint32)
    metadata[:8] = [1, 0, 0, 0, 0, 0, storage // 128, 0]
    arrays = [
        query,
        key,
        weights,
        query_scale,
        scale[order],
        np.array([queries], np.uint32),
        np.array([actual], np.uint32),
        np.arange(storage // 128, dtype=np.int32),
        metadata,
        np.full((queries, 2048), -1, np.int32),
        np.zeros(128, np.uint8),
        np.zeros(128, np.uint8),
        np.zeros(16 * 1024 * 1024, np.uint8),
        np.zeros(queries * 2048 * 6 + 1024 + queries * history * 2 + 128, np.uint8),
    ]
    address_offset = queries * 2048 * 6 + 1024 + queries * storage * 2
    sq_offset = address_offset + storage * 8
    arrays[13] = np.zeros(sq_offset + queries * 2048 * 64 + queries * 128, np.uint8)
    addresses = np.uint64(0x200000000) + order.astype(np.uint64) * np.uint64(656)
    arrays[13][address_offset:sq_offset] = addresses.view(np.uint8)
    for i, array in enumerate(arrays):
        array.tofile(root / f"arg{i}.bin")
    np.save(root / "logical_scores.npy", np.stack(logical_scores))
    np.save(root / "physical_to_logical.npy", order)
    (root / "fixture.json").write_text(
        json.dumps(
            {
                "history": history,
                "resident": resident,
                "queries": queries,
                "batch": 1,
                "aic": 1,
                "aiv": 2,
                "seed": 73121,
                "dense_heads": True,
                "query_inputs": "distinct_nonunit_exact_binary_scales",
                "cold_scale": cold_scale,
                "cold_active_tiles": cold_active_tiles,
                "storage": storage,
                "visible_lengths": visible,
                "visibility": "causal_history_plus_query" if causal else "same_history_controlled_baseline",
                "network": False,
            }
        )
    )


def prepare_miss_fixture(
    root,
    fraction,
    layout,
    history=8192,
    resident=4096,
    mean_per_aiv=None,
    tile_pattern="uniform",
    independent_queries=False,
):
    if not 0 <= fraction <= 1 or layout not in ("dispersed", "clustered"):
        raise ValueError("invalid fraction/layout")
    prepare_fixture(root, history, resident, 4)
    order = np.load(root / "physical_to_logical.npy")
    scale = np.fromfile(root / "arg4.bin", np.float32)
    weights = np.fromfile(root / "arg2.bin", np.float32).reshape(4, 64)
    query_scale = np.fromfile(root / "arg3.bin", np.float32).reshape(4, 64)
    tile_count = (history - resident) // 128
    if mean_per_aiv is None:
        selected_by_tile = np.full(tile_count, int(np.floor(128 * fraction)), dtype=int)
    else:
        if not 0 <= mean_per_aiv <= 256 or tile_pattern not in ("uniform", "bursty"):
            raise ValueError("invalid mean-per-AIV/tile pattern")
        # Both query masks are controlled together. Fractional means use
        # temporally spaced pairs, never silently claim one hit per tile.
        cumulative = np.floor(np.arange(tile_count + 1) * mean_per_aiv / 2).astype(int)
        selected_by_tile = np.diff(cumulative)
        if tile_pattern == "bursty":
            total = int(selected_by_tile.sum())
            selected_by_tile = np.array([min(128, max(0, total - t * 128)) for t in range(tile_count)])
    for tile, start in enumerate(range(resident, history, 128)):
        selected = selected_by_tile[tile]
        lanes = (
            np.arange(selected)
            if layout == "clustered"
            else np.floor(np.arange(selected) * 128 / max(selected, 1)).astype(int)
        )
        scale[start : start + 128] = 0
        scale[start + lanes] = 8
    scale.tofile(root / "arg4.bin")
    encoded = np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128)
    values = np.zeros_like(encoded, dtype=np.float32)
    for byte, value in ((0x38, 1), (0x40, 2), (0x44, 3)):
        values[encoded == byte] = value
    columns_by_query = [(np.arange(64) + q * 16) % 128 for q in range(4)]
    expected_counts = [selected_by_tile] * 4
    if independent_queries:
        if mean_per_aiv is None:
            raise ValueError("independent queries require mean_per_aiv")
        # Disjoint K feature bands allow one query to win without forcing its
        # partner to win. Keep all 64 heads active; only this fixture's inputs
        # change. Compare stage4/stage5 with the SAME fixture, never paired vs
        # independent inputs as a causal performance comparison.
        columns_by_query = [np.arange(64) % 32 + q * 32 for q in range(4)]
        cumulative = np.floor(np.arange(tile_count + 1) * mean_per_aiv).astype(int)
        total_by_tile = np.diff(cumulative)
        if tile_pattern == "bursty":
            total = int(total_by_tile.sum())
            total_by_tile = np.array([min(256, max(0, total - t * 256)) for t in range(tile_count)])
        first = total_by_tile // 2 + (total_by_tile % 2) * (np.arange(tile_count) % 2 == 0)
        expected_counts = [first, total_by_tile - first, first, total_by_tile - first]
        query = np.zeros((4, 64, 128), np.uint8)
        values[resident:] = 0
        scale[resident:] = 8
        for q, columns in enumerate(columns_by_query):
            query[q, np.arange(64), columns] = 0x38
            for tile, selected in enumerate(expected_counts[q]):
                lanes = (
                    np.arange(selected)
                    if layout == "clustered"
                    else np.floor(np.arange(selected) * 128 / max(selected, 1)).astype(int)
                )
                values[np.ix_(resident + tile * 128 + lanes, np.unique(columns))] = 3
        query.tofile(root / "arg0.bin")
        np.array([0, 0x38, 0x40, 0x44], np.uint8)[values.astype(np.uint8)].tofile(root / "arg1.bin")
        scale.tofile(root / "arg4.bin")
    logical = []
    actual = []
    for q in range(4):
        columns = columns_by_query[q]
        score = (values[:, columns] * (weights[q] * query_scale[q])).sum(-1) * scale
        bits = ((score.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
        row = np.empty_like(bits)
        row[order] = bits
        logical.append(row)
        threshold = np.partition(bits[:resident], -2048)[-2048]
        counts = (bits[resident:] >= threshold).reshape(-1, 128).sum(-1)
        assert np.array_equal(counts, expected_counts[q]), "constructed scores missed requested lane count"
        actual.append(counts)
    np.save(root / "logical_scores.npy", np.stack(logical))
    config = json.loads((root / "fixture.json").read_text())
    config["stage5_gate"] = {
        "requested_fraction": fraction,
        "layout": layout,
        "actual_fraction": float(np.mean(actual) / 128),
        "scores_per_aiv_tile": 256,
        "requested_mean_per_aiv_tile": mean_per_aiv,
        "actual_mean_per_aiv_tile": float(np.mean(actual) * 2),
        "temporal_pattern": tile_pattern,
        "query_mask_correlation": (
            "independent_feature_bands; odd hits rotate query ownership"
            if independent_queries
            else "paired_query_masks; fractional means alternate zero and two hits"
        ),
        "eligible_per_aiv_tile": [(actual[0] + actual[1]).tolist(), (actual[2] + actual[3]).tolist()],
        "eligible_per_query_tile": [counts.tolist() for counts in actual],
        "slot_capacity_per_query": 2048,
        "measurement_must_exclude_capacity_bypass": True,
    }
    (root / "fixture.json").write_text(json.dumps(config, indent=2))
    return config["stage5_gate"]


def prepare_compact_fixture(root, history, resident, miss_percent, index_heads=32, seed=20261104):
    import torch
    if not 0 <= miss_percent <= 100:
        raise ValueError('miss-percent must be in [0,100]')
    root = Path(root)
    operands = generate_operands(history, index_heads, seed)
    query, keys = operands['query'], operands['keys']
    # Chunk CPU scoring to keep full-history temporary storage bounded.
    scores = np.empty((4, history), np.float32)
    for begin in range(0, history, 4096):
        dots = np.einsum('qhd,nd->qhn', query, keys[begin:begin+4096], optimize=True)
        scores[:, begin:begin+4096] = (np.maximum(dots, 0) *
            (operands['weights'] * operands['query_scale'])[:, :, None]).sum(1) * operands['key_scale'][None, begin:begin+4096]
    scores = torch.from_numpy(scores).to(torch.bfloat16).float().numpy()
    selected = np.argpartition(scores, -TOPK, axis=1)[:, -TOPK:]
    union = np.unique(selected)
    rng = np.random.default_rng(seed)
    hot_count = round(len(union) * (1-miss_percent/100))
    if hot_count > resident or len(union)-hot_count > history-resident:
        raise ValueError('Requested miss/resident cannot hold the selected union')
    hot_winners = rng.choice(union, hot_count, replace=False)
    background = np.setdiff1d(np.arange(history), union)
    if resident-hot_count > len(background):
        raise ValueError('Requested miss rate leaves insufficient nonselected hot rows')
    hot = np.concatenate((hot_winners, rng.choice(background, resident-hot_count, replace=False)))
    rng.shuffle(hot)
    cold = np.setdiff1d(np.arange(history), hot)
    rng.shuffle(cold)
    permutation = np.concatenate((hot, cold))
    prepare_fixture(root, history, resident, 4)
    order = np.load(root/'physical_to_logical.npy')
    for name, index in (('query', 0), ('keys', 1), ('weights', 2), ('query_scale', 3), ('key_scale', 4)):
        value = operands[name]
        if name in ('keys', 'key_scale'):
            value = value[permutation]
        if name in ('query', 'keys'):
            value = torch.from_numpy(value).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        value.tofile(root/f'arg{index}.bin')
    physical_scores = scores[:, permutation]
    logical_scores = np.empty_like(scores)
    logical_scores[:, order] = physical_scores
    np.save(root/'logical_scores.npy', logical_scores)
    actual = [int(np.count_nonzero(~np.isin(ids, hot))) for ids in selected]
    info = dict(model='compact_builtin_v1', seed=seed, index_heads=index_heads,
                requested_miss_percent=miss_percent, final_misses_per_query=actual,
                actual_miss_percent_per_query=[n*100/TOPK for n in actual],
                hot_background='compact_builtin_v1', final_boundary='BF16 cutoff ties allowed',
                temporal_trace=False, miss_control='selected-union residency; per-Q rate measured',
                prefetch_decoys=0, expected_final_supplement=None)
    config = json.loads((root/'fixture.json').read_text())
    config['full_chain_fixture'] = info
    (root/'fixture.json').write_text(json.dumps(config, indent=2))
    return info


def fp8(value):
    import torch
    return torch.from_numpy(np.clip(value, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn).float().numpy()


def generate_operands(history=131072, heads=32, seed=20261104, parameters=None):
    import torch
    if history < 128 or heads < 2:
        raise ValueError('history >=128 and heads >=2 required')
    p = GENERATOR_DEFAULTS if parameters is None else parameters
    rng = np.random.default_rng(seed)
    direction = rng.choice([-1., 1.], WIDTH)
    weights = rng.normal(size=heads)
    weights = (weights-weights.mean()) / weights.std() * 4.4 + 1.1
    weights = np.broadcast_to(weights, (QUERIES, heads)).copy().astype(np.float32)
    # Larger responses on negative-weight heads retain the negative score regime.
    base = rng.normal(size=(heads, WIDTH))
    base -= (base @ direction / WIDTH)[:, None] * direction
    base *= np.sqrt(WIDTH) / np.linalg.norm(base, axis=1)[:, None]
    orientation = -np.sign(weights[0]) * p['alignment']
    base = np.sqrt(1-orientation[:, None]**2) * base + orientation[:, None] * direction
    residual = rng.normal(size=(QUERIES, heads, WIDTH))
    query = fp8(166 * (np.sqrt(.84)*base[None, :, :] + np.sqrt(.16)*residual))
    qscale_head = rng.normal(.01972, .003658, heads)
    qscale = np.maximum(np.broadcast_to(qscale_head, (QUERIES, heads)), .001).astype(np.float32).copy()
    keys = rng.normal(size=(history, WIDTH))
    keys = fp8(172 * (np.sqrt(1-p['key_direction']**2)*keys + p['key_direction']*direction))
    key_scale = (p['scale'] * np.exp(rng.normal(-p['scale_spread']**2/2, p['scale_spread'], history))).astype(np.float32)
    weak = rng.choice(history, max(1, round(history*.005)), replace=False)
    key_scale[weak] *= .3
    # Fixed zero-row count avoids random 1% mass crossing the p99 boundary.
    zero_count = round(history * .0099182)
    zero = rng.choice(history, zero_count, replace=False)
    keys[zero] = 0
    key_scale[zero] = 1
    weights = torch.from_numpy(weights).to(torch.bfloat16).float().numpy()
    return dict(query=query, keys=keys, weights=weights, query_scale=qscale, key_scale=key_scale)


def payload(pool_layer, tokens):
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1, 1)
    columns = np.arange(512, dtype=np.int64)[None, :]
    code = np.asarray([0xC4, 0xC0, 0xB8, 0, 0x38, 0x40, 0x44], np.uint8)
    nope = code[((tokens >> ((columns // 8) % 18)) + columns + pool_layer * 3) % 7]
    rope_columns = np.arange(64, dtype=np.int64)[None, :]
    rope = 0x3E80 | (((tokens >> ((rope_columns // 4) % 18)) + rope_columns + pool_layer * 11) & 127)
    rope = rope.astype("<u2").view(np.uint8)
    scales = np.broadcast_to(np.asarray([0.03125, 0.0625, 0.125, 0.25], "<f4"), (len(tokens), 4)).copy()
    return np.concatenate((nope, rope, scales.view(np.uint8)), axis=1).reshape(-1, TOKEN_BYTES)


def unpack(packed):
    packed = np.asarray(packed, np.uint8).reshape(-1, TOKEN_BYTES)
    raw = packed[:, :512]
    sign = np.where(raw & 128, -1.0, 1.0)
    exponent = ((raw >> 3) & 15).astype(np.int32)
    mantissa = raw & 7
    values = sign * np.where(exponent == 0, mantissa / 8.0 * 2.0**-6, (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    scales = packed[:, 640:].copy().view("<f4")
    nope = (values * np.repeat(scales, 128, axis=1)).astype(np.float32)
    rope = (packed[:, 512:640].copy().view("<u2").astype(np.uint32) << 16).view(np.float32)
    return np.concatenate((nope, rope), axis=1), nope


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Trace member missing or outside capture: {relative}')
    return path


def score_values(keys):
    """Decode sortable BF16 keys; a numeric uint16 cast is not a score."""
    keys = np.asarray(keys, dtype=np.uint16)
    bits = np.where(keys & 0x8000, keys ^ 0x8000, ~keys).astype(np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


@dataclass
class ScoreCapture:
    path: Path
    manifest: dict
    arrays: dict

    @property
    def executable(self):
        return self.manifest['real_inputs_saved']

    def logical_keys(self, row):
        """Return LI rows in logical-token order for subsequent layout remapping."""
        if not self.executable:
            raise ValueError('Score-only capture has no executable Q/Index inputs')
        ids = self.arrays['pre.logical_ids'][row]
        keys = np.empty_like(self.arrays['real.keys'][row])
        scales = np.empty_like(self.arrays['real.key_scales'][row])
        keys[ids] = self.arrays['real.keys'][row]
        scales[ids] = self.arrays['real.key_scales'][row]
        return keys, scales


def load_score_capture(manifest_path, *, require_real=True):
    path = Path(manifest_path).resolve()
    if path.is_dir():
        path /= 'manifest.json'
    manifest = json.loads(path.read_text())
    legacy = manifest.get('kind') == 'lifetch_legacy_replay_snapshot'
    if legacy:
        if (manifest.get('schema_version') != 1 or manifest.get('status') != 'complete'
                or manifest.get('provenance', {}).get('identity_scope') != 'independent_snapshot_only'):
            raise ValueError('Invalid legacy executable snapshot')
    elif (manifest.get('schema_version') != 4
            or manifest.get('kind') != 'kvaf_decode_score_trace'
            or manifest.get('content') != 'scores'
            or manifest.get('status') != 'complete'
            or manifest.get('validation', {}).get('passed') is not True
            or manifest['validation'].get('kind') != 'score_trace_integrity'):
        raise ValueError(f'Expected complete schema-4 score capture: {path}')
    real = manifest.get('real_inputs_saved')
    if type(real) is not bool or real != manifest['metadata'].get('capture_real_inputs', False):
        raise ValueError('Real LI capture marker is inconsistent')
    if require_real and not real:
        raise ValueError(f'No real Q/Index saved in {path}; score-only data cannot replay LI')
    description = manifest['archive']
    archive = checked_path(path.parent, description['file'])
    with archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if archive.stat().st_size != description['bytes'] or digest != description['sha256']:
        raise ValueError(f'Trace archive size/checksum differs: {archive}')
    expected = BASE_ARRAYS | SCORE_ARRAYS | (REAL_ARRAYS if real else frozenset())
    if legacy:
        if not real:
            raise ValueError('Legacy replay requires real operands')
        expected = REAL_ARRAYS | {name for name in BASE_ARRAYS if name.startswith('pre.')} | {'post.topk', 'post.score_keys'}
    with np.load(archive, allow_pickle=False) as source:
        if set(source.files) != expected or set(manifest['arrays']) != expected:
            raise ValueError('Missing or unexpected capture arrays')
        arrays = {name: source[name] for name in source.files}
    for name, array in arrays.items():
        spec = manifest['arrays'][name]
        if array.dtype.hasobject or array.dtype.name != spec['dtype'] or list(array.shape) != spec['shape']:
            raise ValueError(f'Trace array differs from manifest: {name}')
    capture = ScoreCapture(path, manifest, arrays)
    validate_inputs(capture)
    return capture


def validate_inputs(capture):
    """Validate executable shapes, logical identities and captured TopK evidence."""
    a, g = capture.arrays, capture.manifest['geometry']
    b, h, r, heads = (g[name] for name in ('batch', 'history', 'resident', 'heads'))
    if (not all(type(n) is int for n in (b, h, r, heads))
            or not 1 <= b <= 16 or not TOPK <= r <= h
            or h % BLOCK or r % BLOCK or heads <= 0):
        raise ValueError('Unsupported capture geometry')
    specs = {
        'pre.logical_ids': ('int32', (b, h)),
        'pre.hot_valid': ('uint8', (b, r)),
        'pre.address_valid': ('uint8', (b, h)),
        'pre.row_owners': ('int32', (b,)),
        'pre.metadata': ('int64', (5, b * QUERIES)),
        'pre.query_ends': ('int32', (b, QUERIES)),
        'pre.lru_slots': ('int32', (b, TOPK)),
        'pre.tail_bounds': ('int32', (b, 2)),
        'pre.ddr_page_slots': ('int32', (b, h // BLOCK)),
        'post.topk': ('int32', (b, QUERIES, TOPK)),
        'post.score_keys': ('uint16', (b, QUERIES, h)),
    }
    if capture.executable:
        specs.update({
            'real.keys': ('uint8', (b, h, 128)),
            'real.key_scales': ('float32', (b, h)),
            'real.query': ('uint8', (b, QUERIES, heads, 128)),
            'real.weights': ('uint16', (b, QUERIES, heads)),
            'real.query_scale': ('float32', (b, QUERIES, heads)),
        })
    for name, (dtype, shape) in specs.items():
        if a[name].dtype.name != dtype or a[name].shape != shape:
            raise ValueError(f'Capture input dtype/shape invalid: {name}')
    if np.any(a['pre.hot_valid'] > 1) or np.any(a['pre.address_valid'] > 1):
        raise ValueError('Validity fields must be binary')
    if not np.array_equal(np.sort(a['pre.metadata'][2]), np.arange(b * QUERIES)):
        raise ValueError('Packed query mapping is not a permutation')
    if not np.array_equal(a['pre.metadata'][4].view(np.int32)[:b], a['pre.row_owners']):
        raise ValueError('Packed owner mapping differs from capture rows')
    for row in range(b):
        if not np.array_equal(np.sort(a['pre.logical_ids'][row]), np.arange(h)):
            raise ValueError('Logical Index IDs are not a permutation')
        lru = a['pre.lru_slots'][row]
        if np.any(lru < 0) or np.any(lru >= r) or len(np.unique(lru)) != TOPK:
            raise ValueError('Invalid LRU candidate slots')
        for q in range(QUERIES):
            end = int(a['pre.query_ends'][row, q])
            selected = a['post.topk'][row, q]
            if not TOPK <= end <= h or np.any(selected < 0) or np.any(selected >= end):
                raise ValueError('TopK exceeds valid query prefix')
            if len(np.unique(selected)) != TOPK:
                raise ValueError('Duplicate positions in captured TopK')
            scores = a['post.score_keys'][row, q, :end]
            if np.any(np.isnan(score_values(scores))):
                raise ValueError('NaN in captured valid score prefix')
            cutoff = np.partition(scores, end - TOPK)[end - TOPK]
            chosen = scores[selected]
            if np.min(chosen) != cutoff or np.sum(chosen > cutoff) != np.sum(scores > cutoff):
                raise ValueError('Captured TopK differs from scores (cutoff ties allowed)')


@dataclass(frozen=True)
class CaptureReference:
    dp_rank: int
    source_round: int
    scheduler_round: int
    group: int
    manifest_path: Path
    request_signature: tuple


def select_series(root, group, round_range):
    """Select capture rounds through series.json, preserving DP/request identity.

    Source round means round_index in the capture, not the scheduler's decode
    counter. Payloads remain lazy so a long series is not all loaded into RAM.
    """
    root = Path(root).resolve()
    first, last = round_range
    if type(group) is not int or group < 0 or not 1 <= first <= last:
        raise ValueError('Invalid group or source round range')
    series_paths = sorted(root.glob('dp[0-9]*/series.json'))
    if not series_paths:
        raise ValueError(f'No DP series manifests under {root}')
    references, seen_ranks, expected_ranks = [], set(), None
    for path in series_paths:
        series = json.loads(path.read_text())
        rank = series.get('dp_rank')
        if (type(rank) is not int or rank < 0 or path.parent.name != f'dp{rank}'
                or rank in seen_ranks or series.get('schema_version') != 1
                or series.get('kind') != 'kvaf_decode_snapshot_series'
                or series.get('status') != 'complete' or series.get('content') != 'scores'):
            raise ValueError(f'Invalid or incomplete score series: {path}')
        seen_ranks.add(rank)
        configured = set(series['dp_ranks'])
        if expected_ranks is None:
            expected_ranks = configured
        if configured != expected_ranks:
            raise ValueError('DP series disagree on configured ranks')
        rounds = series['rounds']
        if series['completed_rounds'] != series['expected_rounds'] or len(rounds) != series['expected_rounds']:
            raise ValueError(f'Incomplete round coverage: {path}')
        if last > len(rounds):
            raise ValueError(f'Requested round {last} absent in {path}')
        previous_scheduler = None
        signature = tuple((item['request_id'], item['owner'])
                          for item in series['request_owner_signature'])
        if len(signature) != series['batch'] or len(set(signature)) != len(signature):
            raise ValueError('Invalid request identity signature')
        for number, entry in enumerate(rounds, 1):
            schedule = entry['scheduler_round']
            if (entry['round_index'] != number or type(schedule) is not int
                    or (previous_scheduler is not None and schedule != previous_scheduler + 1)):
                raise ValueError(f'Nonconsecutive captured rounds: {path}')
            previous_scheduler = schedule
            if not first <= number <= last:
                continue
            matches = [item for item in entry['groups'] if item['group'] == group]
            if len(matches) != 1:
                raise ValueError(f'Round {number} requires exactly one group {group}')
            manifest_path = checked_path(path.parent, matches[0]['manifest'])
            manifest = json.loads(manifest_path.read_text())
            metadata = manifest.get('metadata', {})
            requests = metadata.get('requests', [])
            actual_signature = tuple((item['request_id'], item['owner']) for item in requests)
            if (metadata.get('dp_rank') != rank or metadata.get('group') != group
                    or metadata.get('context', {}).get('round') != schedule
                    or metadata.get('context', {}).get('global_decode_confirmed') is not True
                    or actual_signature != signature
                    or manifest.get('geometry', {}).get('batch') != series['batch']
                    or series['queries'] != QUERIES
                    or any(item['query_stop'] - item['query_begin'] != QUERIES for item in requests)):
                raise ValueError(f'Round identity differs from series: {manifest_path}')
            references.append(CaptureReference(rank, number, schedule, group, manifest_path, signature))
    if seen_ranks != expected_ranks:
        raise ValueError('Configured DP capture is missing')
    return sorted(references, key=lambda ref: (ref.source_round, ref.dp_rank))


@dataclass
class ReplayRound:
    source_round: int
    scheduler_round: int
    identities: tuple
    geometry: dict
    logical_keys: np.ndarray
    logical_scales: np.ndarray
    query: np.ndarray
    weights: np.ndarray
    query_scale: np.ndarray
    query_ends: np.ndarray
    tail_bounds: np.ndarray
    logical_address_valid: np.ndarray
    page_keys: tuple
    captured_layout: np.ndarray
    captured_hot_valid: np.ndarray
    captured_lru: np.ndarray
    source_paths: tuple
    logical_score_keys: np.ndarray
    logical_score_valid: np.ndarray

    def validate_topk(self, selected):
        selected = np.asarray(selected)
        if selected.shape != (self.geometry['batch'], 4, 2048) or selected.dtype.kind not in 'iu':
            raise AssertionError('Replay TopK shape/type differs from Q4 contract')
        for row in range(self.geometry['batch']):
            for query in range(4):
                ids = selected[row, query]
                if np.any(ids < 0) or np.any(ids >= self.geometry['history']):
                    raise AssertionError('Replay TopK logical token outside history')
                valid = self.logical_score_valid[row, query]
                scores = self.logical_score_keys[row, query]
                if len(np.unique(ids)) != 2048 or not np.all(valid[ids]):
                    raise AssertionError('Replay TopK duplicates or exceeds causal prefix')
                population = scores[valid]
                cutoff = np.partition(population, len(population) - 2048)[-2048]
                if np.min(scores[ids]) != cutoff or np.sum(scores[ids] > cutoff) != np.sum(population > cutoff):
                    chosen = np.zeros(len(scores), dtype=bool)
                    chosen[ids] = True
                    missing = np.flatnonzero(valid & ~chosen & (scores > cutoff))
                    below = ids[scores[ids] < cutoff]
                    detail = dict(cutoff=int(cutoff), selected_min=int(np.min(scores[ids])),
                                  missing_above=int(len(missing)), below_cutoff=int(len(below)),
                                  missing=[(int(i), int(scores[i])) for i in missing[:8]],
                                  below=[(int(i), int(scores[i])) for i in below[:8]])
                    raise AssertionError(f'Replay TopK ranking differs: request={row} Q={query + 1}; {detail}')

    def index_for_layout(self, layout):
        """Map real LI input to a path's current layout without resetting residency."""
        layout = np.asarray(layout)
        batch, history = self.logical_scales.shape
        if (layout.shape != (batch, history) or layout.dtype.kind not in 'iu'
                or not np.all(np.sort(layout, axis=1) == np.arange(history))):
            raise ValueError('Replay Index layout must be a per-request token permutation')
        rows = np.arange(batch)[:, None]
        return (self.logical_keys[rows, layout], self.logical_scales[rows, layout],
                self.logical_address_valid[rows, layout])


def assemble_round(references):
    """Merge selected DPs; retain page aliasing within DP, separate DP arenas."""
    if not references:
        raise ValueError('Cannot assemble an empty replay round')
    source_round = references[0].source_round
    scheduler_round = references[0].scheduler_round
    if any(ref.source_round != source_round or ref.scheduler_round != scheduler_round for ref in references):
        raise ValueError('DP capture rounds are not aligned')
    identities, keys, scales, queries, weights, query_scales = [], [], [], [], [], []
    ends, tails, valid, pages, layouts, hot, lru, sources = [], [], [], [], [], [], [], []
    score_keys, score_valid = [], []
    geometry = None
    seen_ranks = set()
    for ref in references:
        if ref.dp_rank in seen_ranks:
            raise ValueError('Duplicate DP in a replay round')
        seen_ranks.add(ref.dp_rank)
        capture = load_score_capture(ref.manifest_path)
        arrays, source_geometry = capture.arrays, capture.manifest['geometry']
        current = {name: source_geometry[name] for name in
                   ('history', 'resident', 'heads', 'token_bytes', 'output_stride')}
        if geometry is None:
            geometry = current
        if current != geometry:
            raise ValueError('Selected DPs have incompatible replay geometry')
        if len(ref.request_signature) != source_geometry['batch']:
            raise ValueError('Request signature size differs from captured batch')
        for row, (request, owner) in enumerate(ref.request_signature):
            if int(arrays['pre.row_owners'][row]) != owner:
                raise ValueError('Captured row owner differs from series request identity')
            identities.append((ref.dp_rank, request, owner))
            logical_keys, logical_scales = capture.logical_keys(row)
            keys.append(logical_keys)
            scales.append(logical_scales)
            logical_valid = np.empty(source_geometry['history'], dtype=np.uint8)
            logical_valid[arrays['pre.logical_ids'][row]] = arrays['pre.address_valid'][row]
            valid.append(logical_valid)
            ordered_scores = np.empty_like(arrays['post.score_keys'][row])
            ordered_valid = np.zeros(ordered_scores.shape, dtype=bool)
            order = arrays['pre.logical_ids'][row]
            for query in range(4):
                ordered_scores[query, order] = arrays['post.score_keys'][row, query]
                ordered_valid[query, order[:arrays['pre.query_ends'][row, query]]] = True
            score_keys.append(ordered_scores)
            score_valid.append(ordered_valid)
            # Capture page slots are logical pages. Do not renumber per request:
            # repeated slots are a shared prefix, but same slot in another DP is not.
            slots = arrays['pre.ddr_page_slots'][row]
            page_valid = logical_valid.reshape(-1, 128)
            if (np.any(page_valid != page_valid[:, :1])
                    or not np.array_equal(slots >= 0, page_valid[:, 0] != 0)
                    or np.any(slots < -1)):
                raise ValueError('Published page slots differ from logical address validity')
            pages.append(tuple(None if slot < 0 else (ref.dp_rank, int(slot)) for slot in slots))
        queries.append(arrays['real.query'])
        weights.append(arrays['real.weights'])
        query_scales.append(arrays['real.query_scale'])
        ends.append(arrays['pre.query_ends'])
        tails.append(arrays['pre.tail_bounds'])
        layouts.append(arrays['pre.logical_ids'])
        hot.append(arrays['pre.hot_valid'])
        lru.append(arrays['pre.lru_slots'])
        sources.append(str(ref.manifest_path))
    if len(identities) > 16 or len(set(identities)) != len(identities):
        raise ValueError('Replay requires distinct requests and batch <= 16')
    return ReplayRound(
        source_round, scheduler_round, tuple(identities), dict(geometry, batch=len(identities)),
        np.stack(keys), np.stack(scales), np.concatenate(queries), np.concatenate(weights),
        np.concatenate(query_scales), np.concatenate(ends), np.concatenate(tails),
        np.stack(valid), tuple(pages), np.concatenate(layouts), np.concatenate(hot),
        np.concatenate(lru), tuple(sources), np.stack(score_keys), np.stack(score_valid),
    )


class ReplaySequence:
    """Lazy complete-sequence loader, repeatable without retaining every Q/Index."""
    def __init__(self, root, group, round_range):
        snapshot = Path(root) / 'manifest.json'
        if snapshot.is_file() and json.loads(snapshot.read_text()).get('kind') == 'lifetch_legacy_replay_snapshot':
            if tuple(round_range) != (1, 1):
                raise ValueError('Legacy snapshots are independent: use --round-range 1:1')
            capture = load_score_capture(snapshot)
            if capture.manifest['metadata']['group'] != group:
                raise ValueError('Requested group differs from legacy snapshot')
            self.rounds = self.from_snapshot(snapshot).rounds
            return
        references = select_series(root, group, round_range)
        self.rounds = tuple(tuple(ref for ref in references if ref.source_round == number)
                            for number in range(round_range[0], round_range[1] + 1))

    @classmethod
    def from_snapshot(cls, path):
        capture = load_score_capture(path)
        metadata = capture.manifest['metadata']
        signature = tuple((item['request_id'], item['owner']) for item in metadata['requests'])
        reference = CaptureReference(metadata['dp_rank'], 1, metadata['context']['round'],
                                     metadata['group'], capture.path, signature)
        result = cls.__new__(cls)
        result.rounds = ((reference,),)
        return result

    def __iter__(self):
        previous = None
        for references in self.rounds:
            current = assemble_round(references)
            if previous is not None:
                if current.identities != previous.identities or current.geometry != previous.geometry:
                    raise ValueError('Request identity/geometry changed within replay sequence')
                if np.any(current.query_ends < previous.query_ends):
                    raise ValueError('Query causal bounds regressed within replay sequence')
                for before, after in zip(previous.page_keys, current.page_keys):
                    if any(old is not None and old != new for old, new in zip(before, after)):
                        raise ValueError('Committed DDR page identity changed within replay sequence')
            yield current
            previous = current


class ReplayPagePlan:
    """Stable synthetic KV identities for real captured page sharing.

    Real captures do not export DDR KV bytes. Use deterministic C8 records and
    state this in results. A page that starts in ABC and is published later
    keeps the same bytes; aliases within a DP share bytes and a Put key.
    """
    def __init__(self, sequence):
        parents = {}
        def root(key):
            parents.setdefault(key, key)
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key
        def join(left, right):
            left, right = root(left), root(right)
            if left != right:
                parents[right] = left
        identities = None
        geometry = None
        for inputs in sequence:
            if identities is None:
                identities, geometry = inputs.identities, inputs.geometry
            if inputs.identities != identities or inputs.geometry != geometry:
                raise ValueError('Replay page plan requires stable request geometry')
            for identity, pages in zip(inputs.identities, inputs.page_keys):
                for logical_page, shared in enumerate(pages):
                    key = ('logical', *identity, logical_page)
                    root(key)
                    if shared is not None:
                        join(key, ('published', *shared))
        if identities is None:
            raise ValueError('Replay page plan requires at least one round')
        self.identities, self.geometry = identities, geometry
        groups = {}
        rows = []
        for identity in identities:
            row = []
            for page in range(geometry['history'] // 128):
                representative = root(('logical', *identity, page))
                if representative not in groups:
                    groups[representative] = len(groups)
                row.append(groups[representative])
            rows.append(row)
        self.page_indices = np.asarray(rows, dtype=np.int32)
        self.page_count = len(groups)

    def page_payload(self, page_indices):
        indices = np.asarray(page_indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.page_count):
            raise ValueError('Synthetic KV page identity outside replay plan')
        tokens = (indices[:, None] * 128 + np.arange(128)).reshape(-1)
        width = self.geometry['token_bytes']
        if width not in (656, 2624):
            raise ValueError('Replay requires supported packed C8 token width')
        return np.concatenate([payload(layer, tokens) for layer in range(width // 656)], axis=1)

    def token_payload(self, row, logical_ids):
        logical_ids = np.asarray(logical_ids, dtype=np.int64)
        if not 0 <= row < len(self.identities) or np.any(logical_ids < 0) or np.any(logical_ids >= self.geometry['history']):
            raise ValueError('Replay payload request outside captured history')
        tokens = self.page_indices[row, logical_ids // 128] * 128 + logical_ids % 128
        width = self.geometry['token_bytes']
        return np.concatenate([payload(layer, tokens) for layer in range(width // 656)], axis=1)


def prepare_full_chain_fixture(root, history, resident, miss_percent, prefetch_decoys=0, index_heads=64, seed=20261104):
    """Reuse controlled cold masks, give each final TopK an unambiguous boundary.

    Hot feature bands independently choose exactly the remaining winners for
    each query. This is an explicitly synthetic controlled-overlap fixture;
    no tie-breaking rule is added to either implementation.
    """

    target = round(2048 * miss_percent / 100)
    if not 0 <= target <= min(2048, history - resident):
        raise ValueError("requested miss count exceeds available cold candidates")
    tiles = (history - resident) // 128
    prepare_miss_fixture(root, 0, "dispersed", history, resident, min(256, 2 * (target + 0.125) / tiles), independent_queries=True)
    root = Path(root)
    encoded = np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128)
    scale = np.fromfile(root / "arg4.bin", np.float32)
    weights = np.fromfile(root / "arg2.bin", np.float32).reshape(4, 64)
    query_scale = np.fromfile(root / "arg3.bin", np.float32).reshape(4, 64)
    if index_heads == 32:
        # The controlled fixture repeats each feature head twice. Merge exact
        # binary weight*scale factors; both native and fused read 32 real heads.
        weights = (weights * query_scale).reshape(4, 2, 32).sum(axis=1)
        query_scale = np.ones((4, 32), np.float32)
        query = np.fromfile(root / "arg0.bin", np.uint8).reshape(4, 64, 128)
        query[:, :32].copy().tofile(root / "arg0.bin")
        weights.tofile(root / "arg2.bin")
        query_scale.tofile(root / "arg3.bin")
    elif index_heads != 64:
        raise ValueError("index_heads must be 32 or 64")
    # Spread background scores before BF16 quantization instead of making all
    # nonwinning hot rows equal to the resident TopK threshold. Binary scales
    # keep the independent score oracle reproducible. Winner value3 stays
    # above every background value (<=1), even at the minimum scale0.5.
    rng = np.random.default_rng(seed)
    for query in range(4):
        background = rng.integers(0x08, 0x39, resident, dtype=np.uint8)
        encoded[:resident, query * 32 : (query + 1) * 32] = background[:, None]
    scale[:resident] = rng.integers(256, 513, resident).astype(np.float32) / 512
    if prefetch_decoys < 0:
        raise ValueError("prefetch_decoys must be nonnegative")
    logical, actual_misses, selected_cold, prefetched = [], [], set(), set()
    order = np.load(root / "physical_to_logical.npy")
    for query in range(4):
        columns = np.arange(index_heads) % 32 + query * 32
        # Per-window paired allocation can give the two queries different
        # totals. Enforce the requested per-query count before choosing hot
        # winners, especially at 100% where an excess creates a cutoff tie.
        cold_rows = np.flatnonzero(encoded[resident:, query * 32]) + resident
        if len(cold_rows) > target:
            encoded[np.ix_(cold_rows[target:], np.unique(columns))] = 0
        elif len(cold_rows) < target:
            empty = np.flatnonzero(encoded[resident:, query * 32] == 0) + resident
            take = np.linspace(0, len(empty) - 1, target - len(cold_rows), dtype=int)
            encoded[np.ix_(empty[take], np.unique(columns))] = 0x44
        cold = np.count_nonzero(encoded[resident:, query * 32])
        candidates = np.flatnonzero(encoded[resident:, query * 32] == 0) + resident
        if prefetch_decoys > len(candidates):
            raise ValueError("prefetch_decoys exceeds available nonwinning cold candidates")
        if prefetch_decoys:
            take = np.floor(np.arange(prefetch_decoys) * len(candidates) / prefetch_decoys).astype(int)
            # FP8 0x20 = 0.125; cold scale8 gives value1, above the
            # background threshold and below every final hot winner (>=1.5).
            encoded[np.ix_(candidates[take], np.unique(columns))] = 0x20
        winners = (np.arange(2048 - cold) + query * 1024) % resident
        encoded[np.ix_(winners, np.unique(columns))] = 0x44
        # Decode positive finite E4M3 values used by the actual input bytes.
        codes = encoded[:, columns]
        exponent = (codes >> 3).astype(np.int32)
        mantissa = (codes & 7).astype(np.float32)
        values = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7)).astype(
            np.float32
        )
        scores = (values * (weights[query] * query_scale[query])).sum(-1) * scale
        bits = ((scores.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
        cutoff = np.partition(bits, -2048)[-2048]
        if int(np.count_nonzero(bits >= cutoff)) != 2048:
            raise ValueError(f"fixture_final_boundary_not_unique:query={query}")
        mapped = np.empty_like(bits)
        mapped[order] = bits
        logical.append(mapped)
        actual_misses.append(int(cold))
        selected_cold.update((np.flatnonzero(bits[resident:] >= cutoff) + resident).tolist())
        resident_cutoff = np.partition(bits[:resident], -2048)[-2048]
        prefetched.update((np.flatnonzero(bits[resident:] > resident_cutoff)[:2048] + resident).tolist())
    encoded.tofile(root / "arg1.bin")
    scale.tofile(root / "arg4.bin")
    np.save(root / "logical_scores.npy", np.stack(logical))
    config = json.loads((root / "fixture.json").read_text())
    config["full_chain_fixture"] = dict(
        model="controlled_query_feature_bands",
        seed=seed,
        index_heads=index_heads,
        requested_miss_percent=miss_percent,
        final_misses_per_query=actual_misses,
        hot_background="seed20261104_positive_fp8_binary_scales_v2",
        final_boundary="exactly_2048_at_or_above_cutoff",
        temporal_trace=False,
        prefetch_decoys=prefetch_decoys,
        expected_final_supplement=len(selected_cold - prefetched),
    )
    (root / "fixture.json").write_text(json.dumps(config, indent=2))
    return config["full_chain_fixture"]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def convert_capture_bundle(source, destination):
    import torch
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    arrays, specs = {}, {}
    if source.suffix == '.pt':
        # Only tensor/primitive archives. No unrestricted pickle fallback.
        obj = torch.load(source, map_location='cpu', weights_only=True)
        meta = obj['metadata']
        if meta.get('schema') != 1:
            raise ValueError('Expected legacy capture schema=1')
        for phase in ('before', 'after'):
            for name, tensor in obj[phase].items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'Not a tensor: {phase}.{name}')
                tensor = tensor.detach().cpu().contiguous()
                dtype = str(tensor.dtype)
                try:
                    array = tensor.numpy()
                    encoding = 'native'
                except (TypeError, RuntimeError):
                    array = tensor.view(torch.uint8).numpy()
                    encoding = 'raw_bytes'  # BF16/FP8 retain bit patterns, not numeric casts.
                key = phase + '.' + name
                arrays[key] = array
                specs[key] = dict(original_dtype=dtype, original_shape=list(tensor.shape), encoding=encoding)
        kind = 'legacy_operand_v1'
        replay = False
    else:
        if source.is_dir():
            source /= 'manifest.json'
        capture = load_score_capture(source, require_real=False)
        arrays = capture.arrays
        meta = capture.manifest
        specs = {key: dict(original_dtype=str(a.dtype), original_shape=list(a.shape), encoding='native')
                 for key, a in arrays.items()}
        kind = 'score_trace_v4'
        replay = capture.executable
    destination.mkdir(parents=True)
    try:
        archive = destination / 'arrays.npz'
        np.savez(archive, **arrays)
        manifest = dict(schema_version=1, kind='lifetch_analysis_bundle', source_kind=kind,
            source=str(source), source_sha256=digest(source), source_metadata=meta,
            executable_via_original_schema4=replay,
            archive=dict(file=archive.name, sha256=digest(archive)),
            arrays={key: dict(**specs[key], storage_dtype=str(a.dtype), storage_shape=list(a.shape))
                    for key, a in arrays.items()})
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # Keep a valid current-loader input alongside the analysis bundle.
        if kind == 'score_trace_v4':
            replay_dir = destination / 'schema4'
            replay_dir.mkdir()
            original = dict(meta)
            original['archive'] = dict(meta['archive'], file='scores.npz')
            shutil.copyfile(source.parent / meta['archive']['file'], replay_dir / 'scores.npz')
            (replay_dir / 'manifest.json').write_text(json.dumps(original, indent=2))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return manifest


def bits(array, spec, dtype, shape):
    import torch
    tensor = torch.from_numpy(np.array(array, copy=True))
    if spec['encoding'] == 'raw_bytes' or tensor.dtype == torch.uint8:
        tensor = tensor.view(dtype)
    elif tensor.dtype != dtype:
        raise ValueError(f'Expected captured {dtype}, got {tensor.dtype}')
    storage = np.uint8 if dtype == torch.float8_e4m3fn else np.uint16
    return tensor.view(torch.uint8).numpy().view(storage).reshape(shape)


def convert_legacy_replay(folder, output):
    import torch
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['source_kind'] != 'legacy_operand_v1':
        raise ValueError('Expected converted legacy operand bundle')
    archive = folder/manifest['archive']['file']
    if digest(archive) != manifest['archive']['sha256']:
        raise ValueError('Input checksum differs')
    with np.load(archive, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    specs = manifest['arrays']
    owners = data['before.row_owners'].reshape(-1)
    active = data['before.query_valid'].reshape(-1, 4).astype(bool)
    if np.any(active.any(1) != active.all(1)):
        raise ValueError('Replay currently requires four active queries per request')
    rows = np.flatnonzero(active.all(1))
    if not 1 <= len(rows) <= 16:
        raise ValueError('Expected 1..16 active requests')
    selected_owners = owners[rows].astype(np.int32)
    layout_pool = data['before.logical_ids']
    h = layout_pool.shape[1]
    r = data['before.hot_valid'].shape[1]
    if np.any(selected_owners < 0) or np.any(selected_owners >= len(layout_pool)):
        raise ValueError('Invalid active owner')
    query_shape = specs['before.query']['original_shape']
    heads = query_shape[-2]
    q = bits(data['before.query'], specs['before.query'], torch.float8_e4m3fn, (len(owners)*4, heads, 128))
    w = bits(data['before.weights'], specs['before.weights'], torch.bfloat16, (len(owners)*4, heads))
    qs = decode_capture_array(data['before.query_scale'], specs['before.query_scale'], 'query_scale').reshape(len(owners)*4, heads)
    unpack = data['before.unpack_indices'].reshape(-1, 4)[rows].astype(int)
    if np.any(unpack < 0) or np.any(unpack >= len(q)) or len(np.unique(unpack)) != len(rows)*4:
        raise ValueError('Invalid active packed Q mapping')
    kpool = bits(data['before.keys'], specs['before.keys'], torch.float8_e4m3fn, (-1, h, 128))
    spool = decode_capture_array(data['before.key_scale'], specs['before.key_scale'], 'key_scale').reshape(-1, h)
    layout = layout_pool[selected_owners].astype(np.int32)
    hot = data['before.hot_valid'][selected_owners]
    if np.any((hot != 0) & (hot != 1)):
        raise ValueError('Nonbinary hot-valid state')
    tails = data['before.tail_bounds'].reshape(-1, 2)[rows].astype(np.int32)
    if np.any(tails[:, 0] < 0) or np.any(tails[:, 1] > h) or np.any(tails[:, 1] < tails[:, 0]):
        raise ValueError('Invalid ABC bounds')
    ends = data['before.query_ends'].reshape(-1, 4)[rows].astype(np.int32)
    b = len(rows)
    meta = np.zeros((5, b*4), np.int64)
    meta[2] = np.arange(b*4)
    meta[4].view(np.int32)[:b] = selected_owners
    arrays = {'pre.row_owners': selected_owners, 'pre.logical_ids': layout,
              'pre.hot_valid': hot.astype(np.uint8), 'pre.metadata': meta,
              'pre.query_ends': ends, 'pre.tail_bounds': tails,
              'pre.lru_slots': data['before.lru_slots'].reshape(-1, 2048)[rows].astype(np.int32),
              'pre.address_valid': np.ones((b, h), np.uint8),
              'pre.ddr_page_slots': np.arange(b*(h//128), dtype=np.int32).reshape(b, -1),
              'real.keys': kpool[selected_owners].copy(), 'real.key_scales': spool[selected_owners].copy(),
              'real.query': q[unpack].copy(), 'real.weights': w[unpack].copy(),
              'real.query_scale': qs[unpack].copy()}
    captured = data['after.logical_topk'].reshape(len(owners), 4, 2048)[rows]
    topk = np.empty((b, 4, 2048), np.int32)
    score_keys = np.empty((b, 4, h), np.uint16)
    for i in range(b):
        if not np.array_equal(np.sort(layout[i]), np.arange(h)):
            raise ValueError('Captured Index layout is not a permutation')
        if np.any(captured[i] < 0) or np.any(captured[i] >= h):
            raise ValueError('Captured logical TopK outside history')
        inverse = np.empty(h, np.int32)
        inverse[layout[i]] = np.arange(h)
        topk[i] = inverse[captured[i]]
        query = torch.from_numpy(arrays['real.query'][i]).view(torch.float8_e4m3fn).float().numpy()
        weights = torch.from_numpy(arrays['real.weights'][i].view(np.uint8)).view(torch.bfloat16).float().numpy()
        for begin in range(0, h, 4096):
            keys = torch.from_numpy(arrays['real.keys'][i, begin:begin+4096]).view(torch.float8_e4m3fn).float().numpy()
            dots = np.einsum('qhd,nd->qhn', query, keys, optimize=True)
            score = (np.maximum(dots, 0)*(weights*arrays['real.query_scale'][i])[:, :, None]).sum(1)
            score *= arrays['real.key_scales'][i, None, begin:begin+4096]
            # CAST_ROUND ties-away, matching the existing fixture oracle.
            bf16 = ((score.view(np.uint32) + np.uint32(0x8000)) >> 16).astype(np.uint16)
            score_keys[i, :, begin:begin+4096] = np.where(bf16 & 0x8000, ~bf16, bf16 ^ 0x8000)
        print(f'RECONSTRUCT {folder.name} request={i+1}/{b}', flush=True)
    arrays['post.topk'], arrays['post.score_keys'] = topk, score_keys
    source_meta = manifest['source_metadata']
    group = int(source_meta.get('group', 0))
    result = dict(schema_version=1, kind='lifetch_legacy_replay_snapshot', status='complete',
                  real_inputs_saved=True,
                  geometry=dict(batch=b, history=h, resident=r, heads=heads, token_bytes=656,
                                output_stride=r+max(384, int(np.max(tails[:, 1]-tails[:, 0])))),
                  metadata=dict(capture_real_inputs=True, dp_rank=0, group=group, context=dict(round=1),
                                requests=[dict(request_id=f'{folder.name}:row{row}', owner=int(owner))
                                          for row, owner in zip(rows, selected_owners)]),
                  provenance=dict(source=manifest.get('source', str(folder)),
                                  source_sha256=manifest.get('source_sha256'), source_metadata=source_meta,
                                  identity_scope='independent_snapshot_only',
                                  scores='CPU reconstructed; captured logical TopK checked',
                                  kv='synthetic C8 identity payload; all pages published independently',
                                  unknown=['original DDR validity', 'DDR page sharing', 'stable request identity', 'KV payload']),
                  arrays={name: dict(dtype=a.dtype.name, shape=list(a.shape)) for name, a in arrays.items()})
    output.mkdir(parents=True, exist_ok=False)
    path = output/'inputs.npz'
    np.savez(path, **arrays)
    result['archive'] = dict(file=path.name, bytes=path.stat().st_size, sha256=digest(path))
    (output/'manifest.json').write_text(json.dumps(result, indent=2))
    try:
        capture = load_score_capture(output)
        sequence = ReplaySequence(output, group, (1, 1))
        initial = next(iter(sequence))
        initial.validate_topk(captured.astype(np.int32))
        pages = ReplayPagePlan(sequence)
        pages.token_payload(0, captured[0, 0])
    except Exception:
        result['status'] = 'invalid'
        (output/'manifest.json').write_text(json.dumps(result, indent=2))
        raise
    return dict(path=str(capture.path), batch=b, group=group, pages=pages.page_count,
                validation='PASS input integrity, captured TopK, layout remap and KV page plan')


def decode_capture_array(array, spec, field):
    """Decode bits before taking numerical statistics, including byte ABI inputs."""
    import torch
    tensor = torch.from_numpy(np.array(array, copy=True))
    original = spec['original_dtype'].removeprefix('torch.')
    if spec['encoding'] == 'raw_bytes':
        dtype = getattr(torch, original, None)
        if dtype is None:
            raise ValueError(f'Unsupported original dtype {original}')
        tensor = tensor.view(dtype)
    elif tensor.dtype == torch.uint8 and field in ('query', 'keys'):
        tensor = tensor.view(torch.float8_e4m3fn)
    elif tensor.dtype == torch.uint8 and field == 'weights':
        tensor = tensor.view(torch.bfloat16)
    elif tensor.dtype == torch.uint8 and field in ('query_scale', 'key_scale'):
        tensor = tensor.view(torch.float32)
    return tensor.float().numpy().reshape(-1)


def convert_replay_inputs(source, output):
    """Convert raw PT or analysis bundles into independently runnable snapshots.

    Existing schema-4 inputs already support continuous replay and need no
    conversion. Legacy captures have no stable request identity across rounds.
    """
    import tempfile

    from reporting import atomic_json

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    if source.is_file() and source.suffix == '.pt':
        inputs = [source]
    elif (source / 'index.json').is_file():
        index = json.loads((source / 'index.json').read_text())
        if index['status'] != 'COMPLETE':
            raise ValueError('Incomplete input conversion')
        inputs = [source / row['bundle'] for row in index['captures']]
    elif (source / 'manifest.json').is_file():
        inputs = [source]
    else:
        inputs = sorted(source.rglob('pid*-sample*.pt'))
    if not inputs:
        raise ValueError('No legacy PT or converted captures found')
    output.mkdir(parents=True)
    status = dict(status='RUNNING', captures=[])
    atomic_json(output / 'conversion.json', status)
    try:
        for i, item in enumerate(inputs):
            name = f'capture-{i:05d}'
            if item.suffix == '.pt':
                with tempfile.TemporaryDirectory(prefix='lifetch-convert-') as temporary:
                    bundle = Path(temporary) / name
                    convert_capture_bundle(item, bundle)
                    result = convert_legacy_replay(bundle, output / name)
            else:
                result = convert_legacy_replay(item, output / name)
            status['captures'].append(result)
            atomic_json(output / 'conversion.json', status)
            print(f"{name}: BS={result['batch']} group={result['group']} {result['validation']}", flush=True)
    except BaseException as error:
        status.update(status='FAILED', error=str(error))
        atomic_json(output / 'conversion.json', status)
        raise
    status['status'] = 'COMPLETE'
    atomic_json(output / 'conversion.json', status)
    print(f'PASS {len(inputs)} independent snapshots; replay each with --round-range 1:1')
    return 0
