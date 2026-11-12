"""Exercise offline calibration with the same on-disk schema as service capture."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibration import capture_rows
from test_trace_inputs import capture_file
from trace_inputs import load_score_capture


def scored_capture(directory):
    path = capture_file(directory, real=False)
    capture = load_score_capture(path, require_real=False)
    a = capture.arrays
    a.update({
        'post.window_posted': np.array([[1, 1]], dtype=np.int32),
        'post.window_processed_ends': np.array([[[2048] * 4, [4096] * 4]], dtype=np.int32),
        'post.prefetch_records': np.array([[0, 0, 4095], [0, 3000, 1095]], dtype=np.int64),
        'post.prefetch_ordinals': np.array([1, 0], dtype=np.int32),
        'post.final_records': np.array([[0, 4095, 0, 0]], dtype=np.int64),
        'post.threshold_records': np.zeros((1, 4, 2, 4), dtype=np.int32),
    })
    a['post.threshold_records'][:, :, 0] = [2048, 2048, 0xBF80, 1]
    a['post.threshold_records'][:, :, 1] = [4096, 4096, 0xBF80, 4]
    return capture


def test_prefetch_admission_order_and_ties(tmp_path):
    capture = scored_capture(tmp_path)
    row, selected = capture_rows(capture)[0]
    assert selected == set(range(2048, 4096))
    assert (row['prefetch'], row['unused_prefetch'], row['final'], row['initial_cold']) == (2, 1, 1, 0)
    assert [window['unused'] for window in row['windows']] == [1, 0]
    assert [query['unique_new'] for query in row['queries']] == [2048, 0, 0, 0]
    assert row['q_jaccard'] == [[1.] * 4] * 4
    assert row['queries'][0]['cutoff_tie_population'] == 4096
    assert row['queries'][0]['selected_cutoff_margin']['mean'] == 0
    assert sum(row['queries'][0]['physical_position_hist']) == 2048


@pytest.mark.parametrize('mutation,error', [
    ('ordinal', 'ordinals'), ('identity', 'identity'), ('threshold', 'Threshold'),
    ('kind', 'threshold'), ('processed', 'bounds'), ('dtype', 'shapes'),
])
def test_invalid_evidence_is_rejected(tmp_path, mutation, error):
    capture = scored_capture(tmp_path)
    a = capture.arrays
    if mutation == 'ordinal':
        a['post.prefetch_ordinals'][:] = 0
    elif mutation == 'identity':
        a['post.prefetch_records'][0, 2] = 0
    elif mutation == 'threshold':
        a['post.threshold_records'][0, 0, 0, 2] = 0
    elif mutation == 'kind':
        a['post.threshold_records'][0, 0, 0, 3] = 7
    elif mutation == 'processed':
        a['post.window_processed_ends'][0, 1] = 1024
    elif mutation == 'dtype':
        a['post.threshold_records'] = a['post.threshold_records'].astype(np.float32)
    with pytest.raises(ValueError, match=error):
        capture_rows(capture)


def test_short_threshold_prefix_uses_native_zero_sentinel(tmp_path):
    capture = scored_capture(tmp_path)
    capture.arrays['post.threshold_records'][0, 0, 0] = [1024, 2048, 0, 1]
    assert capture_rows(capture)[0][0]['queries'][0]['thresholds'][0]['key'] == 0


def test_offline_cli_real_schema_and_temporal_statistics(tmp_path):
    dp = tmp_path / 'dp2'
    dp.mkdir()
    rounds = []
    for number in (1, 2):
        directory = dp / str(number)
        directory.mkdir()
        capture = scored_capture(directory)
        if number == 2:
            # All scores tie: replacing a selected token remains legal TopK.
            capture.arrays['post.topk'][:, :, -1] = 2048
        archive = directory / 'scores.npz'
        np.savez_compressed(archive, **capture.arrays)
        manifest = capture.manifest
        manifest['archive'].update(bytes=archive.stat().st_size,
                                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        manifest['arrays'] = {name: {'dtype': value.dtype.name, 'shape': list(value.shape)}
                              for name, value in capture.arrays.items()}
        manifest['metadata'].update(dp_rank=2, group=10,
            context={'round': 100 + number, 'global_decode_confirmed': True},
            requests=[{'request_id': 'private-request', 'owner': 0, 'query_begin': 0, 'query_stop': 4}])
        capture.path.write_text(json.dumps(manifest))
        rounds.append({'round_index': number, 'scheduler_round': 100 + number,
                       'groups': [{'group': 10, 'manifest': f'{number}/manifest.json'}]})
    (dp / 'series.json').write_text(json.dumps(dict(
        dp_rank=2, dp_ranks=[2], schema_version=1, kind='kvaf_decode_snapshot_series',
        status='complete', content='scores', batch=1, queries=4,
        completed_rounds=2, expected_rounds=2,
        request_owner_signature=[{'request_id': 'private-request', 'owner': 0}], rounds=rounds)))
    output = tmp_path / 'analysis'
    command = [sys.executable, str(Path(__file__).with_name('calibration.py')),
               '--trace-root', str(tmp_path), '--group', '10', '--round-range', '1:2', '--output', str(output)]
    run = subprocess.run(command, text=True, capture_output=True, check=True)
    assert 'adjacent_round_pairs=1' in run.stdout
    targets = json.loads((output / 'fit_targets.json').read_text())
    assert json.loads((output / 'analysis_status.json').read_text())['status'] == 'COMPLETE'
    assert targets['fitted_generator'] is False
    assert targets['request_rounds'] == 2
    assert targets['temporal_jaccard']['mean'] == pytest.approx(2047 / 2049)
    assert targets['totals']['initial_cold'] == 1
    rows = json.loads((output / 'request_rounds.json').read_text())
    assert rows[0]['source_round'] == 1 and rows[1]['scheduler_round'] == 102
    assert rows[0]['temporal'] is None
    assert 'private-request' not in (output / 'request_rounds.json').read_text()
    # Artifact corruption must fail before any fitting claims are produced.
    with (dp / '2/scores.npz').open('ab') as file:
        file.write(b'bad')
    failed = subprocess.run(command, text=True, capture_output=True)
    assert failed.returncode != 0 and 'checksum' in failed.stderr
    assert json.loads((output / 'analysis_status.json').read_text())['status'] == 'FAILED'
