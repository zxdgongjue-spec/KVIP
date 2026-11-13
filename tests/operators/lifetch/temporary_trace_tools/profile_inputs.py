#!/usr/bin/env python3
"""Round-one CPU input profile. Emits small aggregate parameters, never raw Q/Index."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from convert_trace import digest
from collect_fit_targets import legacy_rows

QUANTILES = [0, .01, .1, .5, .9, .99, 1]


def decode(array, spec, field):
    """Decode bits before taking numerical statistics, including byte ABI inputs."""
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


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(samples=len(values), finite=0)
    return dict(samples=len(values), finite=len(finite), zero_rate=float(np.mean(finite == 0)),
        mean=float(finite.mean()), std=float(finite.std()),
        quantiles=np.quantile(finite, QUANTILES).tolist(),
        abs_quantiles=np.quantile(np.abs(finite), QUANTILES).tolist())


def cosine_matrix(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1)
    denom = norms[:,None] * norms[None,:]
    result = np.full(denom.shape, np.nan)
    np.divide(vectors @ vectors.T, denom, out=result, where=denom != 0)
    return result


def profile(root, *, samples=8192, seed=20261104):
    rng = np.random.default_rng(seed)
    index = json.loads((root/'index.json').read_text())
    if index['status'] != 'COMPLETE':
        raise ValueError('Input conversion incomplete')
    buckets, norm_buckets, layouts, q_pairs, demands = {}, {}, [], [], []
    notes = set()
    for entry in index['captures']:
        folder = root/entry['bundle']
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['source_kind'] != 'legacy_operand_v1':
            raise ValueError('This first-round profiler expects the legacy operand captures')
        archive = folder/manifest['archive']['file']
        if digest(archive) != manifest['archive']['sha256']:
            raise ValueError(f'Checksum differs: {folder}')
        with np.load(archive, allow_pickle=False) as z:
            arrays = {key:z[key] for key in z.files}
        active = arrays['before.query_valid'].reshape(-1,4).astype(bool)
        batch = len(active)
        layout = dict(capture=entry['bundle'], active_requests=int(active.any(axis=1).sum()),
            capacity=batch, shapes={name:spec for name,spec in manifest['arrays'].items()
                                   if name.startswith('before.')}, query_mapping='packed')
        unpack = arrays.get('before.unpack_indices')
        query_rows = np.flatnonzero(active.reshape(-1))
        if unpack is not None:
            unpack = unpack.reshape(-1)
            if len(unpack) != batch*4:
                raise ValueError('Unexpected unpack_indices shape; preserve inputs and inspect manifest')
            query_rows = unpack[query_rows].astype(np.int64)
            if np.any(query_rows < 0) or np.any(query_rows >= batch*4):
                raise ValueError('Active query mapping outside packed queries')
            layout['query_mapping'] = 'unpack_indices'
        else:
            notes.add('No unpack_indices: within-request Q correlation omitted; packed active rows assumed for scalar statistics')
        for field in ('query','keys','weights','query_scale','key_scale'):
            name = 'before.'+field
            if name not in arrays:
                notes.add('Missing '+name)
                continue
            a, spec = arrays[name], manifest['arrays'][name]
            if field == 'keys':
                # Sampling raw complete rows first avoids decoding a multi-GB pool.
                width = 128
                storage = a.reshape(-1, width)
                take = rng.choice(len(storage), min(samples,len(storage)), replace=False)
                decoded = decode(storage[take], spec, field).reshape(-1,128)
                notes.add('Index statistics sample the captured physical pool, not only active request rows; zero-vector fraction reported separately')
                nonzero = np.any(decoded != 0, axis=1)
                layout['index_sample_zero_vector_fraction'] = float(np.mean(~nonzero))
                norm_buckets.setdefault('keys_pool',[]).extend(np.linalg.norm(decoded,axis=1).tolist())
                norm_buckets.setdefault('keys_nonzero',[]).extend(np.linalg.norm(decoded[nonzero],axis=1).tolist())
                populated = decoded[nonzero].reshape(-1)
                if len(populated):
                    pick = rng.choice(len(populated),min(samples,len(populated)),replace=False)
                    buckets.setdefault('keys_nonzero_vectors',[]).extend(populated[pick].tolist())
            elif field == 'key_scale':
                flat = a.reshape(-1)
                # Native FP32 scale; raw-byte storage must first preserve 4-byte units.
                if a.dtype == np.uint8:
                    storage = flat.reshape(-1,4)
                    take = rng.choice(len(storage),min(samples,len(storage)),replace=False)
                    decoded = decode(storage[take],spec,field)
                else:
                    take = rng.choice(len(flat),min(samples,len(flat)),replace=False)
                    decoded = decode(flat[take],spec,field)
            else:
                decoded = decode(a,spec,field).reshape(batch*4,-1)[query_rows]
                if field == 'query':
                    if decoded.shape[1] % 128:
                        raise ValueError('Q width is not a multiple of 128')
                    norm_buckets.setdefault('query_head',[]).extend(np.linalg.norm(decoded.reshape(-1,128),axis=1).tolist())
                    if unpack is not None:
                        packed = decode(a,spec,field).reshape(batch*4,-1)
                        for b in range(batch):
                            if active[b].all():
                                q_pairs.append(cosine_matrix(packed[unpack.reshape(batch,4)[b].astype(int)]).tolist())
            flat = decoded.reshape(-1)
            take = rng.choice(len(flat),min(samples,len(flat)),replace=False)
            buckets.setdefault(field,[]).extend(flat[take].tolist())
        records = legacy_rows(arrays,manifest['source_metadata'])
        for record in records:
            demands.append({key:record[key] for key in ('selected_unique','initial_cold','initial_hot_valid','q_jaccard')})
        layouts.append(layout)
        print(f'PROFILE {entry["bundle"]}: active={layout["active_requests"]}/{batch}', flush=True)
    scalar = {key:summarize(values) for key,values in buckets.items()}
    norms = {key:summarize(values) for key,values in norm_buckets.items()}
    def matrix_mean(values):
        if not values:
            return None
        a=np.asarray(values,dtype=float)
        count=np.isfinite(a).sum(axis=0)
        result=np.full(a.shape[1:],np.nan)
        np.divide(np.nansum(a,axis=0),count,out=result,where=count>0)
        return [[float(v) if np.isfinite(v) else None for v in row] for row in result]
    return dict(schema_version=1,kind='lifetch_input_fit_profile', seed=seed,
        quantile_probabilities=QUANTILES, captures=len(layouts), active_requests=len(demands),
        scalar=scalar, norms=norms, q_cosine_mean=matrix_mean(q_pairs),
        q_topk_jaccard_mean=matrix_mean([r['q_jaccard'] for r in demands]),
        demand={key:summarize([r[key] for r in demands]) for key in
                ('selected_unique','initial_cold','initial_hot_valid')},
        cold_rate=summarize([r['initial_cold']/r['selected_unique'] for r in demands if r['selected_unique']]),
        notes=sorted(notes), layouts=layouts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle_root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-capture',type=int,default=8192)
    p.add_argument('--seed',type=int,default=20261104)
    args=p.parse_args()
    if args.samples_per_capture < 1 or args.output.exists():
        p.error('Use positive samples and a new output directory')
    result=profile(args.bundle_root,samples=args.samples_per_capture,seed=args.seed)
    args.output.mkdir(parents=True)
    layouts=result.pop('layouts')
    (args.output/'layouts.json').write_text(json.dumps(layouts,indent=2))
    (args.output/'profile.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=[f'INPUT PROFILE captures={result["captures"]} active_requests={result["active_requests"]}',
           'field | zero% | mean | std | abs_p50 | abs_p90 | abs_p99']
    for field,value in result['scalar'].items():
        if value.get('finite',0):
            lines.append(f'{field} | {100*value["zero_rate"]:.2f} | {value["mean"]:.2f} | {value["std"]:.2f} | '+
                         ' | '.join(f'{value["abs_quantiles"][i]:.2f}' for i in (3,4,5)))
    for name in ('q_cosine_mean','q_topk_jaccard_mean'):
        lines.append(name)
        for row in result[name] or []:
            lines.append(' '.join('N/A' if v is None else f'{v:.2f}' for v in row))
    if layouts:
        lines.append('INPUT SHAPES (first capture)')
        for name,spec in layouts[0]['shapes'].items():
            lines.append(f'{name}: {spec["original_dtype"]} {spec["original_shape"]}')
    lines += result['notes']
    lines.append('PROFILE_JSON_BEGIN')
    lines.append(json.dumps(result,separators=(',',':'),allow_nan=False))
    lines.append('PROFILE_JSON_END')
    (args.output/'summary.log').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
