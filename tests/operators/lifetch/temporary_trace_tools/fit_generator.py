#!/usr/bin/env python3
"""Fit aggregate Q/Index parameters and compare held-out sampled LI scores on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from convert_trace import digest
from profile_inputs import cosine_matrix, decode

PROBABILITIES = np.linspace(0, 1, 101)
WIDTH = 128
QUERIES = 4


def quantize(values, dtype):
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dtype).float().numpy()


def samples(root, limit, seed):
    rng = np.random.default_rng(seed)
    index = json.loads((root / 'index.json').read_text())
    if index['status'] != 'COMPLETE':
        raise ValueError('Incomplete conversion')
    for entry in index['captures']:
        folder = root / entry['bundle']
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['source_kind'] != 'legacy_operand_v1':
            raise ValueError('Expected legacy operand bundles')
        archive = folder / manifest['archive']['file']
        if digest(archive) != manifest['archive']['sha256']:
            raise ValueError(f'Checksum differs: {archive}')
        with np.load(archive, allow_pickle=False) as data:
            specs = manifest['arrays']
            def decoded(field):
                return decode(data['before.' + field], specs['before.' + field], field)
            owners = data['before.row_owners'].reshape(-1)
            active = data['before.query_valid'].reshape(-1, QUERIES).astype(bool)
            unpack = data['before.unpack_indices'].reshape(-1, QUERIES)
            mapping = data['before.logical_ids']
            lengths = data['before.key_lengths'].reshape(-1)
            queries = decoded('query').reshape(len(owners) * QUERIES, -1, WIDTH)
            weights = decoded('weights').reshape(len(owners) * QUERIES, -1)
            scales = decoded('query_scale').reshape(weights.shape)
            keys = data['before.keys'].reshape(len(mapping), -1, WIDTH)
            key_scales = decoded('key_scale').reshape(keys.shape[:2])
            for request in np.flatnonzero(active.all(axis=1)):
                owner = int(owners[request])
                if not 0 <= owner < len(mapping):
                    raise ValueError('Active owner outside pool')
                valid = np.flatnonzero((mapping[owner] >= 0) & (mapping[owner] < lengths[request]))
                take = rng.choice(valid, min(limit, len(valid)), replace=False)
                if len(take) < 128:
                    raise ValueError('Too few valid Index rows')
                qrows = unpack[request].astype(int)
                if np.any(qrows < 0) or np.any(qrows >= len(queries)):
                    raise ValueError('Invalid Q mapping')
                k = decode(keys[owner, take], specs['before.keys'], 'keys').reshape(-1, WIDTH)
                yield dict(query=queries[qrows], weights=weights[qrows], query_scale=scales[qrows],
                           keys=k, key_scale=key_scales[owner, take], capture=entry['bundle'],
                           selected=np.stack([np.isin(mapping[owner, take], ids) for ids in
                               data['after.logical_topk'].reshape(len(owners), QUERIES, -1)[request]]),
                           eligible=mapping[owner, take][None, :] < data['before.query_ends'].reshape(-1, QUERIES)[request, :, None])
        print(f'INPUT {entry["bundle"]}: sampled valid owner rows', flush=True)


def covariance_model(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(0)
    return dict(mean=values.mean(0).tolist(),
                covariance=(centered.T @ centered / len(values)).tolist())


def draw_model(model, count, rng):
    covariance = np.asarray(model['covariance'])
    eigenvalues, vectors = np.linalg.eigh(covariance)
    factor = vectors * np.sqrt(np.maximum(eigenvalues, 0))
    return (rng.standard_normal((count, len(factor))) @ factor.T + model['mean']).astype(np.float32)


def fit(rows):
    # Preserve coordinate directions and head identity instead of pooling all heads.
    q = np.stack([r['query'] for r in rows]).astype(np.float64)
    mean = q.mean(0)
    residual = (q - mean).transpose(0, 2, 3, 1).reshape(-1, QUERIES)
    keys = np.concatenate([r['keys'] for r in rows])
    scales = np.concatenate([r['key_scale'] for r in rows])
    zero = np.all(keys == 0, axis=1)
    # Scale buckets retain zero rows and the conditional key direction/covariance.
    edges = np.unique(np.quantile(scales[~zero], np.linspace(0, 1, 9))) if (~zero).any() else []
    labels = np.searchsorted(edges[1:-1], scales, side='right')
    groups = []
    for mask in [zero] + [(~zero) & (labels == i) for i in range(max(1, len(edges)-1))]:
        if not mask.any():
            continue
        groups.append(dict(probability=float(mask.mean()), keys=covariance_model(keys[mask]),
                           scales=np.quantile(scales[mask], PROBABILITIES).tolist()))
    # Joint per-head weight / scale / Q-norm variation, without raw input samples.
    norms = np.linalg.norm(q, axis=-1)
    heads = q.shape[2]
    joint = [covariance_model(np.stack([np.concatenate((r['weights'][:, h],
                  r['query_scale'][:, h], norms[i, :, h])) for i, r in enumerate(rows)]))
             for h in range(heads)]
    return dict(schema_version=2, model='head_coordinate_conditional_index_v2',
                heads=heads, probabilities=PROBABILITIES.tolist(), query_mean=mean.tolist(),
                query_residual=covariance_model(residual), head_joint=joint, key_groups=groups)


def generate(parameters, history, seed):
    """Only aggregate JSON is needed; never reads original tensor captures."""
    if parameters.get('schema_version') != 2:
        raise ValueError('Refit version 1 parameters: joint input model requires schema 2')
    rng = np.random.default_rng(seed)
    heads = parameters['heads']
    q = np.asarray(parameters['query_mean']) + draw_model(
        parameters['query_residual'], heads * WIDTH, rng).T.reshape(QUERIES, heads, WIDTH)
    weights, scales = np.empty((QUERIES, heads)), np.empty((QUERIES, heads))
    for h, model in enumerate(parameters['head_joint']):
        joint = draw_model(model, 1, rng)[0]
        weights[:, h], scales[:, h] = joint[:4], np.maximum(joint[4:8], 0)
        q[:, h] *= (np.maximum(joint[8:12], 0) / np.maximum(np.linalg.norm(q[:, h], axis=1), 1e-12))[:, None]
    groups = parameters['key_groups']
    labels = rng.choice(len(groups), history, p=[g['probability'] for g in groups])
    keys, key_scale = np.empty((history, WIDTH), np.float32), np.empty(history, np.float32)
    for i, group in enumerate(groups):
        mask = labels == i
        keys[mask] = draw_model(group['keys'], int(mask.sum()), rng)
        key_scale[mask] = np.interp(rng.random(mask.sum()), parameters['probabilities'], group['scales'])
    return dict(query=quantize(np.clip(q, -448, 448), torch.float8_e4m3fn),
                keys=quantize(np.clip(keys, -448, 448), torch.float8_e4m3fn),
                weights=quantize(weights, torch.bfloat16), query_scale=scales.astype(np.float32),
                key_scale=key_scale)


def score_values(row):
    dots = np.einsum('qhd,nd->qhn', row['query'], row['keys'], optimize=True)
    return (np.maximum(dots, 0) * (row['weights'] * row['query_scale'])[:, :, None]).sum(1) * row['key_scale'][None, :]


def audit_topk(row):
    # Necessary sampled ordering check only, allowing BF16 cutoff ties.
    scores = quantize(score_values(row), torch.bfloat16)
    selected = row.get('selected')
    if selected is None:
        return dict(checked_queries=0, inversions=0)
    checked, inversions = 0, 0
    for q in range(QUERIES):
        valid = row['eligible'][q]
        yes, no = selected[q] & valid, ~selected[q] & valid
        if yes.any() and no.any():
            checked += 1
            inversions += int(scores[q, yes].min() < scores[q, no].max())
    return dict(checked_queries=checked, inversions=inversions)


def metrics(row, top_fraction):
    # CPU mathematical score proxy, not an Ascend instruction/rounding oracle.
    scores = score_values(row)
    count = max(1, round(scores.shape[1] * top_fraction))
    top = [set(np.argpartition(score, -count)[-count:]) for score in scores]
    result = {}
    for name in ('query', 'keys', 'weights', 'query_scale', 'key_scale'):
        result[name + '_std'] = float(row[name].std())
        result[name + '_mean'] = float(row[name].mean())
    result['q_cosine'] = float(cosine_matrix(row['query'].reshape(QUERIES, -1))[np.triu_indices(QUERIES, 1)].mean())
    result['sample_top_jaccard'] = float(np.mean([len(top[a] & top[b]) / len(top[a] | top[b])
                                               for a in range(QUERIES) for b in range(a + 1, QUERIES)]))
    result['sample_union_ratio'] = len(set.union(*top)) / count
    for percentile in (50, 90, 99):
        result[f'score_p{percentile}'] = float(np.percentile(scores, percentile))
    result['score_zero_pct'] = float(np.mean(scores == 0) * 100)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parameters', type=Path, help='Reuse edited parameters instead of refitting')
    parser.add_argument('--sample-keys', type=int, default=8192)
    parser.add_argument('--top-fraction', type=float, default=2048 / 131072)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    if args.output.exists() or args.sample_keys < 128 or not 0 < args.top_fraction <= 1:
        parser.error('Use a new output directory, sample-keys >=128, and 0 < top-fraction <=1')
    rows = list(samples(args.bundle_root, args.sample_keys, args.seed))
    captures = list(dict.fromkeys(row['capture'] for row in rows))
    if len(captures) < 2:
        parser.error('At least two captures required for fit/evaluation split')
    fit_captures = set(captures[::2])
    train = [row for row in rows if row['capture'] in fit_captures]
    held_out = [row for row in rows if row['capture'] not in fit_captures]
    parameters = json.loads(args.parameters.read_text()) if args.parameters else fit(train)
    actual, synthetic, audits = [], [], []
    for i, row in enumerate(held_out):
        audits.append(audit_topk(row))
        actual.append(metrics(row, args.top_fraction))
        generated = generate(parameters, len(row['keys']), args.seed + i)
        synthetic.append(metrics(generated, args.top_fraction))
        print(f'COMPARE {i + 1}/{len(held_out)}', flush=True)
    args.output.mkdir(parents=True)
    (args.output / 'generator.json').write_text(json.dumps(parameters, indent=2, allow_nan=False))
    comparison = {key: dict(real=float(np.mean([r[key] for r in actual])),
                            generated=float(np.mean([r[key] for r in synthetic]))) for key in actual[0]}
    (args.output / 'score_audit.json').write_text(json.dumps(audits, indent=2))
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))
    lines = [f'FIT requests={len(train)} EVAL requests={len(held_out)} keys/request={args.sample_keys}',
             'CPU sampled scores; sampled TopK ratio is not full-history TopK or device correctness.',
             'Capture split only: adjacent captures may share requests.',
             f'SCORE AUDIT sampled queries={sum(a["checked_queries"] for a in audits)} inversions={sum(a["inversions"] for a in audits)} (0 required before interpreting score fit)',
             'Metric | Real | Generated | Delta %']
    if any(a['inversions'] for a in audits) or not sum(a['checked_queries'] for a in audits):
        lines.insert(0, 'SCORE AUDIT NOT VALIDATED: resolve sampled ordering before accepting generator fit')
    for key, values in comparison.items():
        real, generated = values['real'], values['generated']
        delta = f'{100 * (generated - real) / abs(real):+.2f}' if real and not key.endswith('_mean') else 'N/A'
        lines.append(f'{key:24s} | {real:12.2f} | {generated:12.2f} | {delta}')
    lines += ['PARAMETERS: generator.json (aggregate moments/conditional distributions only)',
              'COMPARE_JSON_BEGIN', json.dumps(comparison), 'COMPARE_JSON_END']
    (args.output / 'summary.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
