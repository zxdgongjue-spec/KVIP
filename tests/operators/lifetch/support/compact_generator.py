#!/usr/bin/env python3
"""Small built-in operand generator; no trace files or fitted parameter archive."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

WIDTH = 128
QUERIES = 4
# Scalar targets from the user's real-trace summary, not raw captured operands.
REFERENCE = dict(query_std=164.76, keys_std=169.75, weights_std=4.40,
                 weights_mean=1.10, query_scale_mean=.01972,
                 query_scale_std=.003658, key_scale_mean=.015479,
                 key_scale_std=.098444, q_cosine=.83724,
                 sample_top_jaccard=.87422, sample_union_ratio=1.14014,
                 score_p50=-2037.68, score_p90=-1570.17, score_p99=-477.53,
                 score_zero_pct=.99182)
# Geometric approximation; intentionally not the previous learned covariance model.
DEFAULTS = dict(alignment=.2, key_direction=.75, scale=.0055, scale_spread=.15)


def fp8(value):
    return torch.from_numpy(np.clip(value, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn).float().numpy()


def generate(history=131072, heads=32, seed=20261104, parameters=None):
    if history < 128 or heads < 2:
        raise ValueError('history >=128 and heads >=2 required')
    p = DEFAULTS if parameters is None else parameters
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


def evaluate(requests=32, history=8192, seed=20261104, parameters=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'temporary_trace_tools'))
    from fit_generator import metrics
    rows = [metrics(generate(history, seed=seed+i, parameters=parameters), 2048/131072)
            for i in range(requests)]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--sample-keys', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--save-operands', action='store_true', help='Save one synthetic full-history Q/Index NPZ')
    parser.add_argument('--history', type=int, default=131072)
    args = parser.parse_args()
    if args.output.exists() or args.requests < 1:
        parser.error('Use a new output directory and positive requests')
    if args.history < 128 or args.sample_keys < 128:
        parser.error('history and sample-keys must be >=128')
    measured = evaluate(args.requests, args.sample_keys, args.seed)
    args.output.mkdir(parents=True)
    if args.save_operands:
        np.savez(args.output/'operands.npz', **generate(args.history, seed=args.seed))
    lines = ['COMPACT GENERATOR: built-in scalars, no external parameter files',
             f'CPU generated requests={args.requests}; keys/request={args.sample_keys}',
             'Reference: supplied real-trace aggregate log; not paired replay or NPU performance.',
             'Metric | Real reference | Compact generated']
    for key, reference in REFERENCE.items():
        factor = 1000 if key.startswith(('query_scale', 'key_scale')) else 1
        label = key + (' x1000' if factor != 1 else '')
        lines.append(f'{label:28s} | {reference*factor:12.2f} | {measured[key]*factor:12.2f}')
    (args.output/'summary.log').write_text('\n'.join(lines)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(dict(reference=REFERENCE, generated=measured), indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
