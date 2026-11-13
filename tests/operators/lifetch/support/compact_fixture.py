"""Shared compact operands, residency control and tie-aware score validation."""
import json
from pathlib import Path

import numpy as np
import torch

from compact_generator import generate
from fixture import prepare

TOPK = 2048


def validate_topk(ids, scores):
    ids = np.asarray(ids).reshape(-1, TOPK)
    scores = np.asarray(scores).reshape(-1, scores.shape[-1])
    for i, selected in enumerate(ids):
        row = scores[i % len(scores)]
        if selected.min() < 0 or selected.max() >= len(row) or len(np.unique(selected)) != TOPK:
            raise AssertionError('Compact TopK has invalid/duplicate positions')
        cutoff = np.partition(row, -TOPK)[-TOPK]
        # One BF16 spacing bounds CPU/device reduction differences at the cutoff.
        tolerance = max(float(abs(cutoff)) * 2**-7, np.finfo(np.float32).tiny)
        if row[selected].min() < cutoff-tolerance:
            raise AssertionError('Compact TopK selected a score below the cutoff')
        mandatory = np.flatnonzero(row > cutoff+tolerance)
        if not np.isin(mandatory, selected).all():
            raise AssertionError('Compact TopK omitted a score above the cutoff')


def prepare_compact_fixture(root, history, resident, miss_percent, index_heads=32, seed=20261104):
    if not 0 <= miss_percent <= 100:
        raise ValueError('miss-percent must be in [0,100]')
    root = Path(root)
    operands = generate(history, index_heads, seed)
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
    prepare(root, history, resident, 4)
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
