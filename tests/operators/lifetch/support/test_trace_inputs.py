import hashlib
import json

import numpy as np
import pytest

from workloads import (
    BASE_ARRAYS, REAL_ARRAYS, SCORE_ARRAYS, checked_path,
    load_score_capture, score_values,
)


def capture_file(tmp_path, *, real=True):
    b, h, r, heads = 1, 4096, 2048, 2
    arrays = {name: np.zeros(0, dtype=np.int32) for name in BASE_ARRAYS | SCORE_ARRAYS}
    metadata = np.zeros((5, 4), dtype=np.int64)
    metadata[2] = np.arange(4)
    arrays.update({
        'pre.logical_ids': np.arange(h, dtype=np.int32)[None, ::-1].copy(),
        'pre.hot_valid': np.ones((b, r), dtype=np.uint8),
        'pre.address_valid': np.ones((b, h), dtype=np.uint8),
        'pre.row_owners': np.zeros(b, dtype=np.int32),
        'pre.metadata': metadata,
        'pre.query_ends': np.full((b, 4), h, dtype=np.int32),
        'pre.lru_slots': np.arange(r, dtype=np.int32)[None],
        'pre.tail_bounds': np.array([[h - 128, h]], dtype=np.int32),
        'pre.ddr_page_slots': np.arange(h // 128, dtype=np.int32)[None],
        'post.topk': np.broadcast_to(np.arange(2048, dtype=np.int32), (b, 4, 2048)).copy(),
        'post.score_keys': np.full((b, 4, h), 0xBF80, dtype=np.uint16),
    })
    if real:
        arrays.update({
            'real.keys': np.broadcast_to((np.arange(h) % 256).astype(np.uint8)[None, :, None], (b, h, 128)).copy(),
            'real.key_scales': np.arange(h, dtype=np.float32)[None],
            'real.query': np.zeros((b, 4, heads, 128), dtype=np.uint8),
            'real.weights': np.zeros((b, 4, heads), dtype=np.uint16),
            'real.query_scale': np.ones((b, 4, heads), dtype=np.float32),
        })
    archive = tmp_path / 'scores.npz'
    np.savez_compressed(archive, **arrays)
    manifest = {
        'schema_version': 4, 'kind': 'kvaf_decode_score_trace', 'content': 'scores',
        'status': 'complete', 'validation': {'passed': True, 'kind': 'score_trace_integrity'},
        'real_inputs_saved': real, 'metadata': {'capture_real_inputs': real},
        'geometry': {'batch': b, 'history': h, 'resident': r, 'heads': heads},
        'archive': {'file': archive.name, 'bytes': archive.stat().st_size,
                    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
        'arrays': {name: {'dtype': value.dtype.name, 'shape': list(value.shape)}
                   for name, value in arrays.items()},
    }
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def test_actual_npz_roundtrip_remaps_physical_index_by_logical_identity(tmp_path):
    capture = load_score_capture(capture_file(tmp_path))
    keys, scales = capture.logical_keys(0)
    assert capture.executable and REAL_ARRAYS <= capture.arrays.keys()
    np.testing.assert_array_equal(scales, np.arange(4096, dtype=np.float32)[::-1])
    assert keys[0, 0] == 255 and keys[-1, 0] == 0


def test_score_only_can_analyze_but_cannot_replay(tmp_path):
    path = capture_file(tmp_path, real=False)
    with pytest.raises(ValueError, match='No real Q/Index'):
        load_score_capture(path)
    capture = load_score_capture(path, require_real=False)
    assert not capture.executable
    with pytest.raises(ValueError, match='no executable'):
        capture.logical_keys(0)


def test_tampered_archive_and_manifest_rejected(tmp_path):
    path = capture_file(tmp_path)
    with (tmp_path / 'scores.npz').open('ab') as output:
        output.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_score_capture(path)
    path = capture_file(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['arrays']['real.query']['shape'][0] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='real.query'):
        load_score_capture(path)


def test_prefix_topk_and_identity_validation(tmp_path):
    capture = load_score_capture(capture_file(tmp_path))
    from workloads import validate_inputs
    capture.arrays['post.topk'][0, 0, 0] = 4096
    with pytest.raises(ValueError, match='prefix'):
        validate_inputs(capture)
    capture.arrays['post.topk'][0, 0, 0] = 0
    capture.arrays['pre.logical_ids'][0, 0] = 0
    with pytest.raises(ValueError, match='permutation'):
        validate_inputs(capture)


def test_path_escape_and_bf16_sortable_decoding(tmp_path):
    outside = tmp_path.parent / 'outside-trace-test'
    outside.write_text('not part of capture')
    try:
        with pytest.raises(ValueError, match='outside capture'):
            checked_path(tmp_path, '../outside-trace-test')
    finally:
        outside.unlink()
    np.testing.assert_array_equal(score_values([0x407F, 0x8000, 0xBF80]), [-1., 0., 1.])


def test_series_uses_source_round_and_preserves_request_identity(tmp_path):
    from workloads import select_series
    dp = tmp_path / 'dp2'
    dp.mkdir()
    rounds = []
    for number in (1, 2):
        directory = dp / f'captured-{number}'
        directory.mkdir()
        path = capture_file(directory)
        manifest = json.loads(path.read_text())
        manifest['metadata'].update({
            'dp_rank': 2, 'group': 10,
            'context': {'round': 100 + number, 'global_decode_confirmed': True},
            'requests': [{'request_id': 'req-a', 'owner': 0, 'query_begin': 0, 'query_stop': 4}],
        })
        path.write_text(json.dumps(manifest))
        rounds.append({'round_index': number, 'scheduler_round': 100 + number,
                       'groups': [{'group': 10, 'manifest': str(path.relative_to(dp))}]})
    series = {
        'dp_rank': 2, 'dp_ranks': [2], 'schema_version': 1,
        'kind': 'kvaf_decode_snapshot_series', 'status': 'complete', 'content': 'scores',
        'batch': 1, 'queries': 4, 'completed_rounds': 2, 'expected_rounds': 2,
        'request_owner_signature': [{'request_id': 'req-a', 'owner': 0}], 'rounds': rounds,
    }
    series_path = dp / 'series.json'
    series_path.write_text(json.dumps(series))
    references = select_series(tmp_path, 10, (2, 2))
    assert len(references) == 1
    assert references[0].source_round == 2 and references[0].scheduler_round == 102
    assert references[0].dp_rank == 2 and references[0].request_signature == (('req-a', 0),)
    with pytest.raises(ValueError, match='absent'):
        select_series(tmp_path, 10, (2, 3))
    series['dp_ranks'] = [0, 2]
    series_path.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='DP capture is missing'):
        select_series(tmp_path, 10, (1, 2))
    series['dp_ranks'] = [2]
    series['rounds'][1]['scheduler_round'] = 104
    series_path.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='Nonconsecutive'):
        select_series(tmp_path, 10, (1, 2))
