#!/usr/bin/env python3
"""Convert legacy operand PT or schema-4 captures into lossless analysis bundles.
No ACL/NPU required. Never invent missing scores, identities or transfer events.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def convert(source, destination, support=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    arrays, specs = {}, {}
    if source.suffix == '.pt':
        import torch
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
        if support is None:
            raise ValueError('--kvip-root required to validate schema-4 input')
        sys.path.insert(0, str(Path(support) / 'tests/operators/lifetch/support'))
        from trace_inputs import load_score_capture
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kvip-root', type=Path)
    args = p.parse_args()
    if args.input.is_file():
        sources = [args.input]
    else:
        sources = sorted(args.input.rglob('pid*-sample*.pt'))
        sources += [path for path in sorted(args.input.rglob('manifest.json'))
                    if json.loads(path.read_text()).get('kind') == 'kvaf_decode_score_trace']
    if not sources:
        raise SystemExit('No legacy PT or schema-4 manifests found')
    if args.output.exists():
        raise SystemExit('Choose a new output directory')
    args.output.mkdir(parents=True)
    index = []
    try:
        for i, source in enumerate(sources):
            name = f'capture-{i:05d}'
            info = convert(source, args.output / name, args.kvip_root)
            index.append(dict(bundle=name, source=str(source.resolve()), kind=info['source_kind']))
            print(f'{name}: {info["source_kind"]}', flush=True)
        (args.output / 'index.json').write_text(json.dumps(dict(status='COMPLETE', captures=index), indent=2))
    except BaseException:
        (args.output / 'index.json').write_text(json.dumps(dict(status='FAILED', captures=index), indent=2))
        raise


if __name__ == '__main__':
    main()
