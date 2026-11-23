#!/usr/bin/env python3
"""Sample actual scoring operands of the archived five-case mainline replay on CPU."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import sys
import tempfile

import numpy as np


GROUPS = (10, 4, 17, 7, 2)
MAIN = "609fa26fd715f9409bbe74e1b2d3685c58dfb0cf"
SAMPLES_PER_ROW = 32
WIDTH = 128


def sampled_arrays(inputs, group):
    query = np.asarray(inputs.query)
    weights = np.asarray(inputs.weights)
    layout = np.asarray(inputs.captured_layout)
    b, q, heads, dim = query.shape
    history, resident = inputs.geometry['history'], inputs.geometry['resident']
    if (b, q, heads, dim, history, resident) != (16, 4, 32, 128, 131328, 16384):
        raise ValueError(f'G{group} wrong Q/Key geometry')
    pages, keys, scales, score_words, thresholds = [], [], [], [], []
    rng = np.random.default_rng(54931 + group)
    for row, row_layout in enumerate(layout):
        # Take one page at random from each equal-depth stratum. Every sampled
        # cold page is wholly valid for all four queries in this row.
        first = resident // WIDTH
        past = int(inputs.query_ends[row].min()) // WIDTH
        strata = np.linspace(first, past, SAMPLES_PER_ROW + 1, dtype=np.int32)
        if np.any(np.diff(strata) <= 0):
            raise ValueError('too few complete pages for stratified sampling')
        selected = rng.integers(strata[:-1], strata[1:], dtype=np.int32)
        ids = row_layout[selected[:, None] * WIDTH + np.arange(WIDTH)]
        pages.append(selected)
        keys.append(inputs.logical_keys[row][ids])
        scales.append(inputs.logical_scales[row][ids])
        scores = inputs.logical_score_keys[row]
        score_words.append(np.take(scores, ids, axis=1))
        bounds = inputs.tail_bounds[row]
        recent_begin = max(resident, int(bounds[0]))
        recent_end = min(history, int(bounds[1]), recent_begin + inputs.geometry['output_stride'] - resident)
        row_thresholds = []
        for query_id, end in enumerate(inputs.query_ends[row]):
            seed = np.r_[row_layout[:min(resident, int(end))],
                         row_layout[recent_begin:min(recent_end, int(end))]]
            seed_scores = scores[query_id, seed]
            if len(seed_scores) < 2048:
                raise ValueError('missing seed scores')
            row_thresholds.append(np.partition(seed_scores, len(seed_scores) - 2048)[-2048])
        thresholds.append(row_thresholds)
    return dict(query=query, weights=weights, query_scale=inputs.query_scale,
                pages=np.stack(pages), query_ends=inputs.query_ends,
                tail_bounds=inputs.tail_bounds, threshold=np.asarray(thresholds, dtype=np.uint16),
                keys=np.stack(keys), key_scale=np.stack(scales), score_keys=np.stack(score_words))


def operand_digest(array):
    data = np.ascontiguousarray(array)
    h = hashlib.sha256(f'{data.dtype.str} {data.shape}'.encode())
    h.update(memoryview(data).cast('B'))
    return h.hexdigest()


def file_digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def run(work, repo):
    archive = work / 'peer-baseline/main-five.hF6qQ8'
    if (archive/'main.txt').read_text().strip() != MAIN or (archive/'physical_npu.txt').read_text().strip() != '4':
        raise ValueError(f'unexpected archive {archive}')
    reader = (archive/'reader.txt').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{40}', reader):
        raise ValueError(f'invalid archived reader {reader!r}')
    output = Path(tempfile.mkdtemp(prefix='li-page-sample.', dir=work))
    # Load the same immutable decoder the mainline run used. Only these two
    # modules are needed for archived DP captures and merged captures.
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        support = Path(scratch) / 'repo/tests/operators/lifetch/support'
        support.mkdir(parents=True)
        for module in ('workloads.py', 'query_owner_input.py'):
            (support/module).write_bytes(subprocess.check_output([
                'git', '-C', str(repo), 'show', f'{reader}:tests/operators/lifetch/support/{module}']))
        sys.path.insert(0, str(support))
        from workloads import ReplaySequence
        output_hashes = {}
        for group in GROUPS:
            folder = archive / f'g{group}/mainline/results/mainline/cases/diagnose'
            config = json.loads((folder/'config.json').read_text())
            receipt = json.loads((folder/'replay_input.json').read_text())
            if (config['group'], config['physical_npu'], config['production_revision']) != (group, 4, MAIN):
                raise ValueError(f'G{group} incorrect archived configuration')
            trace_root = Path(config['trace_root'])
            seq = ReplaySequence(trace_root, group, (1, 1))
            actual = [str(Path(ref.manifest_path).resolve()) for round_refs in seq.rounds for ref in round_refs]
            if actual != [str(Path(path).resolve()) for path in receipt['source_paths']]:
                raise ValueError(f'G{group} source manifest differs from replay_input.json')
            inputs = next(iter(seq))
            if (inputs.geometry != receipt['geometry'] or
                    json.loads(json.dumps(inputs.identities)) != receipt['identities']):
                raise ValueError(f'G{group} decoded identities/geometry differ from archived replay')
            checked = json.loads((folder/'fused/mainline_checks.json').read_text())
            names = ('query', 'weights', 'query_scale', 'logical_keys', 'logical_scales',
                     'query_ends', 'captured_layout')
            hashes = {name: operand_digest(getattr(inputs, name)) for name in names}
            if checked['production_revision'] != MAIN or hashes != checked['capture_operand_hashes']:
                raise ValueError(f'G{group} operands differ from the tensors used for the timed run')
            export = Path(scratch)/f'g{group}.npz'
            np.savez_compressed(export, **sampled_arrays(inputs, group))
            output_hashes[export.name] = file_digest(export)
            print(f'G{group} rows=16 pages_per_row={SAMPLES_PER_ROW} compressed_bytes={export.stat().st_size}', flush=True)
            del inputs, seq
        metadata = Path(scratch)/'manifest.json'
        metadata.write_text(json.dumps(
            dict(archive=str(archive), main=MAIN, reader=reader, pages_per_row=SAMPLES_PER_ROW,
                 payload='Lightning Indexer Q, Index keys, weights, scales, score keys and physical page numbers',
                 files=output_hashes), indent=2) + '\n')
        result = output/'li-page-sample.tar'
        staging = Path(scratch)/result.name
        with tarfile.open(staging, 'w') as tar:
            for path in sorted(Path(scratch).glob('*.npz')) + [metadata]:
                tar.add(path, arcname=path.name)
        staging.replace(result)
    print(f'RETURN_FILE={result} bytes={result.stat().st_size} sha256={file_digest(result)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold'))
    p.add_argument('--repo', type=Path, default=Path('/home/c00982601/CacheOS/vLLM-DSA/third_party/KVIP'))
    args = p.parse_args()
    run(args.work, args.repo)
