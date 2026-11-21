"""Use the real DP inputs already on this host, including merged single-layer replay snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from workloads import CaptureReference, ScoreCapture, assemble_round, checked_path, validate_inputs


COUNTS = {2: (45138, 955), 7: (57293, 11355), 10: (69708, 2319), 17: (86406, 4982), 4: (48088, 4717)}
RAW = {2: 'real-input-r51-g2g7.dKZUcF', 7: 'real-input-r51-g2g7.dKZUcF',
       10: 'score-trace-141.62.38.24-7lg40xtj', 17: 'real-input-r51-miss5-miss10.hj_6zeym',
       4: 'real-input-r51-miss5-miss10.hj_6zeym'}


def locate(work, group):
    data = work.parent
    step = 51 if group == 10 else 2
    gname = f'group{group:03}'
    def series(root, number):
        return all((root / f'dp{i}/series.json').is_file() and
                   all((root / f'dp{i}/round{number:04}/{gname}' / name).is_file()
                       for name in ('manifest.json', 'scores.npz')) for i in range(8))
    for suite in (work / 'peer-inputs', work / 'lifetch-real-suite',
                  Path(os.environ.get('PEER_SUITE_ROOT', data / 'lifetch-real-suite')),
                  Path('/home/x00965276/KVIP/lifetch-real-suite')):
        for path in sorted(suite.glob(f'case*-r{step}-g{group}')):
            if series(path, 1):
                return 'prepared', path
    if series(data / RAW[group] / 'trace', step):
        return 'raw', data / RAW[group] / 'trace'
    def merged(path):
        manifest = path / 'manifest.json'
        if not manifest.is_file():
            return False
        m = json.loads(manifest.read_text())
        return (m.get('content') == 'real_li_synthetic_kv' and m.get('status') == 'complete'
                and m.get('metadata', {}).get('group') == group and m['geometry']['token_bytes'] == 656
                and 'post.score_keys' in m['tensors'])
    home = work / 'skip-results.po80Nw' / f'g{group}-input'
    if merged(home):
        return 'merged', home
    # Only descend into dp0. Skip builds, run logs and blob stores.
    for directory, dirs, files in os.walk(data):
        path = Path(directory)
        dirs[:] = sorted(d for d in dirs if not d.startswith(('.', 'build', 'cpu-')) and
                         not (d.startswith('dp') and d[2:].isdigit() and d != 'dp0') and
                         d not in ('ddr', 'tensors', 'objects', 'KVAF', 'fused-scan-sources', 'adaptive-sources', 'results'))
        if path.name == f'g{group}-input' and merged(path):
            return 'merged', path
        if path.name == 'dp0' and 'series.json' in files:
            dirs[:] = []
            if series(path.parent, step):
                return 'raw', path.parent
            if path.parent.name.endswith(f'-r{step}-g{group}') and series(path.parent, 1):
                return 'prepared', path.parent
    raise ValueError(f'G{group}: no matching 8-DP capture or merged real LI snapshot in {data}')


def merged_refs(path, m, group):
    sources = m['metadata']['merged_from']
    g = m['geometry']
    if (m['status'] != 'complete' or not m['oracle']['passed'] or m['metadata']['group'] != group
            or (g['batch'], g['history'], g['resident'], g['heads'], g['token_bytes'], g['output_stride'])
            != (16, 131328, 16384, 32, 656, 16896)
            or [s['dp_rank'] for s in sources] != list(range(8))):
        raise ValueError(f'Wrong G{group} single-layer BS16 snapshot: {path}')
    refs = []
    for source in sources:
        dp = source['dp_rank']
        rows = [r for r in m['metadata']['requests'] if r['source_dp_rank'] == dp]
        signature = tuple((r['source_request_id'], r['source_owner']) for r in rows)
        if (len(rows) != 2 or source['group'] != group or not source['context']['global_decode_confirmed']
                or [r['owner'] for r in rows] != [2*dp + 1, 2*dp + 2]
                or set(signature) != {(r['request_id'], r['owner']) for r in source['requests']}):
            raise ValueError(f'Invalid DP{dp} request provenance')
        refs.append(CaptureReference(dp, 1, source['context']['round'], group, path, signature))
    return refs


def merged_round(refs):
    path = refs[0].manifest_path
    m = json.loads(path.read_text())
    def tensor(name, dtype):
        spec = m['tensors'][name]
        p = checked_path(path.parent, spec['file'])
        with p.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError(f'{p}: tensor checksum differs')
        raw = np.load(p, mmap_mode='r', allow_pickle=False)
        if raw.dtype != np.uint8 or raw.nbytes != spec['nbytes'] or spec['dtype'] != dtype:
            raise ValueError(f'{p}: wrong tensor representation')
        bits = {'float8_e4m3fn': 'uint8', 'bfloat16': 'uint16'}.get(dtype, dtype)
        return raw.view(bits).reshape(spec['shape'])

    g = m['geometry']
    r = g['resident']
    meta = tensor('pre.metadata', 'int64')
    owners = meta[4].view(np.int32)[:16]
    if not np.array_equal(owners, np.arange(1, 17)) or not np.array_equal(np.sort(meta[2]), np.arange(64)):
        raise ValueError('Invalid merged owner or unpack order')
    layout = tensor('pre.logical_ids', 'int64')
    hot = tensor('pre.hot_valid', 'int32')
    addr = tensor('pre.addresses', 'int64')
    lru = tensor('pre.lru_slots', 'int32').reshape(16, 2048)
    ends = tensor('pre.query_ends', 'int32').reshape(16, 4)
    tails = tensor('pre.tail_bounds', 'int32').reshape(16, 2)
    keys = tensor('pre.keys', 'uint8')
    scales = tensor('pre.key_scales', 'float32')
    scores = tensor('post.score_keys', 'uint16')
    selected = tensor('post.logical_output', 'int32').reshape(64, 2048)[np.argsort(meta[2])].reshape(16, 4, 2048)
    operands = {target: tensor('pre.' + field, dtype).reshape((16, 4) + shape) for target, field, dtype, shape in (
        ('real.query', 'query', 'float8_e4m3fn', (32, 128)), ('real.query_scale', 'query_scale', 'float32', (32,)),
        ('real.weights', 'weights', 'bfloat16', (32,)))}
    ddr = m['ddr']
    base, stride = ddr['region_base'], ddr['slot_bytes']
    captures = {}
    unique = cold = 0
    arenas = {x['dp_rank']: x for x in m['metadata']['ddr_subarenas']}
    for ref in refs:
        dp = ref.dp_rank
        rows = slice(dp*2, dp*2+2)
        own = owners[rows]
        ids = layout[own].astype(np.int32)
        inverse = np.argsort(ids, axis=1)
        logical = np.take_along_axis(addr[own], inverse, axis=1).reshape(2, -1, 128)
        starts = logical[:, :, 0]
        present = starts != 0
        slots = (starts - base) // stride - arenas[dp]['slot_offset']
        if (np.any((starts[present] - base) % stride) or np.any(slots[present] < 0)
                or np.any(slots[present] >= arenas[dp]['region_bytes'] // stride)
                or np.any(logical[~present]) or np.any(logical[present] != starts[present][:, None] + np.arange(128)*656)):
            raise ValueError(f'DP{dp}: inconsistent physical-to-logical DDR page mapping')
        order = meta[2, dp*8:dp*8+8] - dp*8
        local = meta[:, dp*8:dp*8+8].copy()
        local[0] -= dp*8
        local[2] = order
        local[4] = 0
        old_owners = np.array([o for _, o in ref.request_signature], dtype=np.int32)
        local[4].view(np.int32)[:2] = old_owners
        top = np.take_along_axis(inverse[:, None, :], selected[rows], axis=2).astype(np.int32)
        hot_rows = hot[own].astype(np.uint8)
        arrays = {'pre.logical_ids': ids, 'pre.hot_valid': hot_rows, 'pre.metadata': local, 'pre.row_owners': old_owners,
                  'pre.address_valid': (addr[own] != 0).astype(np.uint8),
                  'pre.ddr_page_slots': np.where(present, slots, -1).astype(np.int32),
                  'pre.lru_slots': lru[rows], 'pre.query_ends': ends[rows], 'pre.tail_bounds': tails[rows],
                  'real.keys': keys[own], 'real.key_scales': scales[own], 'post.topk': top, 'post.score_keys': scores[rows],
                  **{k: v[rows] for k, v in operands.items()}}
        capture = ScoreCapture(path, dict(real_inputs_saved=True, geometry=dict(g, batch=2)), arrays)
        validate_inputs(capture)
        for i in range(2):
            p = np.unique(top[i])
            hit = p < r
            hit[hit] = hot_rows[i, p[hit]] != 0
            lo, hi = tails[2*dp+i]
            unique += p.size
            cold += np.count_nonzero(~(hit | ((lo <= ids[i,p]) & (ids[i,p] < hi))))
        captures[dp] = capture
    if (unique, cold) != COUNTS[refs[0].group]:
        raise ValueError(f'G{refs[0].group}: reconstructed selected/cold differ: {unique}/{cold}')
    return assemble_round(refs, captures=captures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('group', type=int)
    args = parser.parse_args()
    try:
        kind, root = locate(args.work.resolve(), args.group)
        if kind == 'merged':
            manifest = root / 'manifest.json'
            refs = merged_refs(manifest, json.loads(manifest.read_text()), args.group)
            merged_round(refs)
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(f'INPUT_FAILED G{args.group}: {exc}') from exc
    print(kind, root, sep='\n')
