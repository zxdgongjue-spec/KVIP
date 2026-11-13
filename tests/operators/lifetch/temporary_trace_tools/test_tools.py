import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from convert_trace import convert
from collect_fit_targets import collect


def test_legacy_conversion_and_metrics(tmp_path):
    source = tmp_path/'pid1-sample0000.pt'
    selected = torch.arange(2048, dtype=torch.int32).repeat(1,4,1)
    torch.save(dict(metadata=dict(schema=1,sample=0), before=dict(
        row_owners=torch.tensor([0]), logical_ids=torch.arange(4096).reshape(1,-1),
        hot_valid=torch.ones(1,2048,dtype=torch.uint8), weights=torch.ones(1,4,2,dtype=torch.bfloat16)),
        after=dict(logical_topk=selected)), source)
    root = tmp_path/'converted'
    m = convert(source, root/'capture-00000')
    assert m['arrays']['before.weights']['encoding'] == 'raw_bytes'
    with np.load(root/'capture-00000/arrays.npz') as z:
        assert z['before.weights'].nbytes == 16
    (root/'index.json').write_text(json.dumps(dict(status='COMPLETE',captures=[dict(bundle='capture-00000')])))
    rows = collect(root)
    assert rows[0]['selected_unique'] == 2048 and rows[0]['initial_cold'] == 0
    assert rows[0]['queries'][1]['cross_q_duplicates'] == 2048
    assert rows[0]['temporal'] is None and rows[0]['prefetch'] is None
    with pytest.raises(ValueError, match='already exists'):
        convert(source, root/'capture-00000')
    with (root/'capture-00000/arrays.npz').open('ab') as f:
        f.write(b'x')
    with pytest.raises(ValueError, match='checksum'):
        collect(root)


def test_current_schema_preserved(tmp_path):
    support = Path(__file__).resolve().parents[1] / 'support'
    sys.path.insert(0, str(support))
    from test_trace_inputs import capture_file
    from trace_inputs import load_score_capture
    source_dir = tmp_path/'source'
    source_dir.mkdir()
    source = capture_file(source_dir)
    convert(source, tmp_path/'output', support.parents[3])
    capture = load_score_capture(tmp_path/'output/schema4/manifest.json')
    assert capture.executable
    assert capture.arrays['real.keys'].shape == (1,4096,128)


def test_padding_owners_and_topk_are_not_requests():
    from collect_fit_targets import legacy_rows
    arrays = {
        'before.row_owners': np.array([1, -1, 0, 999]),
        'before.logical_ids': np.tile(np.arange(4096), (2,1)),
        'before.hot_valid': np.ones((2,2048), dtype=np.uint8),
        'before.query_valid': np.array([[1,1,1,1],[0,0,0,0],[1,0,0,0],[0,0,0,0]]),
        'after.logical_topk': np.full((4,4,2048), -1, dtype=np.int32),
    }
    arrays['after.logical_topk'][0] = np.arange(2048)
    arrays['after.logical_topk'][2,0] = np.arange(2048)
    rows = legacy_rows(arrays, {'sample':0})
    assert [r['row'] for r in rows] == [0,2]
    assert [r['active_queries'] for r in rows] == [4,1]
    assert all(r['selected_unique'] == 2048 for r in rows)


def test_input_profile_decodes_bf16_and_excludes_padding(tmp_path):
    from profile_inputs import profile, decode
    value=np.array([0x80,0x3f],dtype=np.uint8)
    np.testing.assert_array_equal(decode(value,dict(original_dtype='torch.bfloat16',encoding='raw_bytes'),'weights'),[1.])
    source=tmp_path/'pid1-sample0000.pt'
    query=torch.zeros(8,2,128,dtype=torch.uint8)
    query[:4]=0x38  # e4m3fn 1.0; padding zeros must not dilute Q statistics.
    torch.save(dict(metadata=dict(schema=1,sample=0),before=dict(
        row_owners=torch.tensor([0,-1]),logical_ids=torch.arange(4096).reshape(1,-1),
        hot_valid=torch.ones(1,2048,dtype=torch.uint8),query_valid=torch.tensor([1,1,1,1,0,0,0,0]),
        query_ends=torch.full((8,),4096),key_lengths=torch.tensor([4096,0]),unpack_indices=torch.arange(8),query=query,keys=torch.full((4096,128),0x38,dtype=torch.uint8),
        weights=torch.ones(8,2,dtype=torch.bfloat16),query_scale=torch.ones(8,2),key_scale=torch.ones(4096)),
        after=dict(logical_topk=torch.arange(2048,dtype=torch.int32).repeat(2,4,1))),source)
    root=tmp_path/'converted'
    convert(source,root/'capture-00000')
    (root/'index.json').write_text(json.dumps(dict(status='COMPLETE',captures=[dict(bundle='capture-00000')])))
    result=profile(root,samples=16)
    assert result['active_requests']==1
    assert result['scalar']['query']['mean']==1.
    np.testing.assert_allclose(result['q_cosine_mean'],np.ones((4,4)))
    assert result['cold_rate']['mean']==0.
    assert result==profile(root,samples=16)
    from fit_generator import samples
    paired = list(samples(root, 128, 7))
    assert len(paired) == 1
    assert paired[0]['keys'].shape == (128, 128)
    np.testing.assert_array_equal(paired[0]['key_scale'], np.ones(128))


def test_generator_reproducible_and_score_metrics():
    from fit_generator import fit, generate, metrics
    rng = np.random.default_rng(7)
    row = dict(query=rng.normal(size=(4, 2, 128)).astype(np.float32),
               keys=rng.normal(size=(256, 128)).astype(np.float32),
               weights=np.ones((4, 2), np.float32), query_scale=np.ones((4, 2), np.float32),
               key_scale=np.ones(256, np.float32))
    parameters = fit([row])
    first = generate(parameters, 256, 7)
    second = generate(json.loads(json.dumps(parameters)), 256, 7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    measured = metrics(first, .1)
    assert all(np.isfinite(v) for v in measured.values())
    assert 0 <= measured['sample_top_jaccard'] <= 1
    identical = dict(row, query=np.repeat(row['query'][:1], 4, axis=0))
    assert metrics(identical, .1)['sample_top_jaccard'] == 1



def test_joint_generator_preserves_zero_scale_pairing_and_negative_heads():
    from fit_generator import fit, generate, score_values, audit_topk
    row = dict(query=np.zeros((4, 2, 128), np.float32),
               keys=np.zeros((256, 128), np.float32),
               weights=np.tile([-4., 1.], (4, 1)).astype(np.float32),
               query_scale=np.ones((4, 2), np.float32),
               key_scale=np.full(256, .005, np.float32))
    row['query'][:, 0, 0] = 8
    row['query'][:, 1, 1] = 1
    row['keys'][:240, 0] = 8
    row['key_scale'][240:] = 1
    generated = generate(fit([row, row]), 4096, 9)
    high_scale = generated['key_scale'] > .5
    assert high_scale.any()
    assert np.all(generated['keys'][high_scale] == 0)
    assert np.median(score_values(generated)) < 0
    scores = score_values(row)
    row['selected'] = scores == 0
    row['eligible'] = np.ones(scores.shape, bool)
    assert audit_topk(row) == dict(checked_queries=4, inversions=0)
    row['selected'] = ~row['selected']
    assert audit_topk(row)['inversions'] == 4


def test_generator_cli_writes_comparison_and_audit(tmp_path, monkeypatch):
    import fit_generator
    rng = np.random.default_rng(3)
    rows = []
    for capture in ('a', 'b'):
        row = dict(query=rng.normal(size=(4, 2, 128)).astype(np.float32),
                   keys=rng.normal(size=(128, 128)).astype(np.float32),
                   weights=np.ones((4, 2), np.float32), query_scale=np.ones((4, 2), np.float32),
                   key_scale=np.ones(128, np.float32), capture=capture)
        scores = fit_generator.score_values(row)
        row['selected'] = scores >= np.sort(scores, axis=1)[:, -8, None]
        row['eligible'] = np.ones(scores.shape, bool)
        rows.append(row)
    monkeypatch.setattr(fit_generator, 'samples', lambda *args: iter(rows))
    output = tmp_path / 'result'
    monkeypatch.setattr(sys, 'argv', ['fit_generator.py', str(tmp_path), '--output', str(output)])
    fit_generator.main()
    assert json.loads((output/'generator.json').read_text())['schema_version'] == 2
    assert json.loads((output/'score_audit.json').read_text())[0]['inversions'] == 0
    assert 'score_p99' in json.loads((output/'comparison.json').read_text())
