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
                           keys=k, key_scale=key_scales[owner, take], capture=entry['bundle'])
        print(f'INPUT {entry["bundle"]}: sampled valid owner rows', flush=True)


def fit(rows):
    fields = ('query', 'keys', 'weights', 'query_scale', 'key_scale')
    distributions = {name: np.quantile(np.concatenate([r[name].ravel() for r in rows]),
                                      PROBABILITIES).tolist() for name in fields}
    correlations = {name: np.mean([cosine_matrix(r[name].reshape(QUERIES, -1))
                                   for r in rows], axis=0).tolist()
                    for name in ('query', 'weights', 'query_scale')}
    if not all(np.isfinite(value).all() for value in distributions.values()):
        raise ValueError('Nonfinite fitting inputs')
    return dict(schema_version=1, model='quantile_correlated_q_v1',
                probabilities=PROBABILITIES.tolist(), distributions=distributions,
                correlations=correlations, heads=rows[0]['query'].shape[1])


def generate(parameters, history, seed):
    """Only aggregate JSON is needed; never reads original tensor captures."""
    rng = np.random.default_rng(seed)
    def draw(name, shape):
        return np.interp(rng.random(shape), parameters['probabilities'],
                         parameters['distributions'][name]).astype(np.float32)
    heads = parameters['heads']
    result = {}
    for name, shape in [('query', (QUERIES, heads, WIDTH)),
                        ('weights', (QUERIES, heads)), ('query_scale', (QUERIES, heads))]:
        # Gaussian copula gives correlated ranks; empirical quantiles give marginals.
        correlation = np.asarray(parameters['correlations'][name])
        eigenvalues, vectors = np.linalg.eigh(correlation)
        transform = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-6)))
        z = transform @ rng.standard_normal((QUERIES, int(np.prod(shape[1:]))))
        ranks = np.argsort(np.argsort(z, axis=1), axis=1)
        uniform = (ranks + .5) / z.shape[1]
        result[name] = np.interp(uniform, parameters['probabilities'],
                                 parameters['distributions'][name]).reshape(shape).astype(np.float32)
    result['keys'] = draw('keys', (history, WIDTH))
    result['key_scale'] = draw('key_scale', (history,))
    for name in ('query', 'keys'):
        result[name] = quantize(np.clip(result[name], -448, 448), torch.float8_e4m3fn)
    result['weights'] = quantize(result['weights'], torch.bfloat16)
    return result


def metrics(row, top_fraction):
    # CPU mathematical score proxy, not an Ascend instruction/rounding oracle.
    dots = np.einsum('qhd,nd->qhn', row['query'], row['keys'], optimize=True)
    scores = (np.maximum(dots, 0) * (row['weights'] * row['query_scale'])[:, :, None]).sum(1)
    scores *= row['key_scale'][None, :]
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
    actual, synthetic = [], []
    for i, row in enumerate(held_out):
        actual.append(metrics(row, args.top_fraction))
        generated = generate(parameters, len(row['keys']), args.seed + i)
        synthetic.append(metrics(generated, args.top_fraction))
        print(f'COMPARE {i + 1}/{len(held_out)}', flush=True)
    args.output.mkdir(parents=True)
    (args.output / 'generator.json').write_text(json.dumps(parameters, indent=2, allow_nan=False))
    comparison = {key: dict(real=float(np.mean([r[key] for r in actual])),
                            generated=float(np.mean([r[key] for r in synthetic]))) for key in actual[0]}
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))
    lines = [f'FIT requests={len(train)} EVAL requests={len(held_out)} keys/request={args.sample_keys}',
             'CPU sampled scores; sampled TopK ratio is not full-history TopK or device correctness.',
             'Capture split only: adjacent captures may share requests.',
             'Metric | Real | Generated | Delta %']
    for key, values in comparison.items():
        real, generated = values['real'], values['generated']
        delta = f'{100 * (generated - real) / abs(real):+.2f}' if real else 'N/A'
        lines.append(f'{key:24s} | {real:12.2f} | {generated:12.2f} | {delta}')
    lines += ['PARAMETERS: generator.json (aggregate quantiles/correlations only)',
              'COMPARE_JSON_BEGIN', json.dumps(comparison), 'COMPARE_JSON_END']
    (args.output / 'summary.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
