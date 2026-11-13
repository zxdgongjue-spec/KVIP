#!/usr/bin/env python3
"""Convert lossless legacy bundles into independent executable LI snapshots."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from convert_trace import digest
from profile_inputs import decode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from trace_inputs import load_score_capture  # noqa: E402
from replay_inputs import ReplaySequence, ReplayPagePlan  # noqa: E402


def bits(array, spec, dtype, shape):
    tensor = torch.from_numpy(np.array(array, copy=True))
    if spec['encoding'] == 'raw_bytes' or tensor.dtype == torch.uint8:
        tensor = tensor.view(dtype)
    elif tensor.dtype != dtype:
        raise ValueError(f'Expected captured {dtype}, got {tensor.dtype}')
    storage = np.uint8 if dtype == torch.float8_e4m3fn else np.uint16
    return tensor.view(torch.uint8).numpy().view(storage).reshape(shape)


def convert(folder, output):
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
    qs = decode(data['before.query_scale'], specs['before.query_scale'], 'query_scale').reshape(len(owners)*4, heads)
    unpack = data['before.unpack_indices'].reshape(-1, 4)[rows].astype(int)
    if np.any(unpack < 0) or np.any(unpack >= len(q)) or len(np.unique(unpack)) != len(rows)*4:
        raise ValueError('Invalid active packed Q mapping')
    kpool = bits(data['before.keys'], specs['before.keys'], torch.float8_e4m3fn, (-1, h, 128))
    spool = decode(data['before.key_scale'], specs['before.key_scale'], 'key_scale').reshape(-1, h)
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
                  provenance=dict(source=str(folder), source_metadata=source_meta,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    index = json.loads((args.bundle_root/'index.json').read_text())
    if index['status'] != 'COMPLETE':
        parser.error('Incomplete input conversion')
    args.output.mkdir(parents=True)
    results = []
    for entry in index['captures']:
        result = convert(args.bundle_root/entry['bundle'], args.output/entry['bundle'])
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.output/'summary.json').write_text(json.dumps(results, indent=2))
    print(f'PASS {len(results)} independent executable snapshots; source rounds are not chained')


if __name__ == '__main__':
    main()
