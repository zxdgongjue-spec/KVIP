import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import native_li_reference as ref


def fixture():
    h = 2054
    layout = np.roll(np.arange(h, dtype=np.int32), 1)[None]
    return SimpleNamespace(
        geometry=dict(batch=1, heads=32, history=h), captured_layout=layout, source_paths=('/capture',),
        query_ends=np.array([[h-3, h-2, h-1, h]], dtype=np.int32),
        logical_score_keys=np.full((1, 4, h), 18344, dtype=np.uint16),
        logical_score_valid=np.ones((1, 4, h), dtype=bool))


def test_actual_native_order_and_domains(tmp_path):
    data = fixture()
    for q, end in enumerate(data.query_ends[0]):
        data.logical_score_valid[0, q, data.captured_layout[0, end:]] = False
    serial = np.broadcast_to(np.arange(2048), (1, 4, 2048)).copy().astype(np.int32)
    physical = np.roll(serial, 7, axis=-1)
    physical[..., np.where(physical[0, 0] == 3)[0][0]] = 2048
    refs = dict(strong_serial=serial, strong_serial_repeat=serial[..., ::-1], same_layout_native=physical)
    fused = np.repeat(physical[None, ..., ::-1], 20, axis=0)
    out = ref.write_comparison(tmp_path, 17, data, refs, {'origin':'test'}, fused)
    assert out['comparisons']['strong_serial']['matched_queries'] == 0
    assert out['comparisons']['strong_serial']['cutoff_ties_only_mismatches'] == 80
    assert out['comparisons']['same_layout_native']['matched_queries'] == 80
    assert out['all_same_layout_sets_match'] and not out['all_serial_sets_match']
    assert out['serial_reference_stability']['stable_queries'] == 4
    assert out['domains']['strong_equals_fused_queries'] == 1
    assert out['domains']['same_layout_equals_fused_queries'] == 4
    row = out['samples']['strong_serial'][0]
    assert row['missing']['examples'] == [dict(token=3,captured_score_key=18344)]
    assert row['extra']['examples'] == [dict(token=2048,captured_score_key=18344)]
    with np.load(tmp_path/'native_li_ids.npz') as ids:
        np.testing.assert_array_equal(ids['fused'], fused)
    refs['strong_serial_repeat'] = physical
    serial_output = np.repeat(serial[None], 20, axis=0)
    unstable = ref.write_comparison(tmp_path,17,data,refs,{},serial_output)
    assert unstable['primary_serial_sets_match'] and not unstable['all_serial_sets_match']
    assert unstable['serial_reference_stability']['unstable'][0]['repeat_only'] == [2048]
    with pytest.raises(ValueError, match='20 complete'):
        ref.write_comparison(tmp_path,17,data,refs,{},fused[:1])
    data.logical_score_valid[0, 0, 3] = False
    row = ref.write_comparison(tmp_path,17,data,refs,{},fused)['samples']['strong_serial'][0]
    assert row['missing']['outside_score_domain'] == 1
    assert row['missing']['examples'][0]['captured_score_key'] is None and not row['cutoff_ties_only']
    data.logical_score_keys[0,1,3] += 1
    rows = ref.write_comparison(tmp_path,17,data,refs,{},fused)['samples']['strong_serial']
    row = next(r for r in rows if r['query'] == 2)
    assert row['missing']['above_cutoff'] == 1 and not row['cutoff_ties_only']


def test_native_cache_integrity_and_physical_mapping(tmp_path, monkeypatch):
    calls = []
    class Operands:
        def __init__(self, *args):
            self.fixture = {}
        def prepare(self, inputs, *, physical):
            self.fixture['physical'] = physical
    inputs = fixture()
    positions = np.broadcast_to(np.arange(2048, dtype=np.int32), (1,4,2048)).copy()

    def native_li(_, fixture):
        calls.append(fixture['physical'])
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: positions))

    monkeypatch.setattr(ref, 'NativeReplayLI', Operands)
    monkeypatch.setattr(ref, 'identities', lambda *args: {'layout': ref.digest(inputs.captured_layout)})
    torch = SimpleNamespace(npu=SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setenv('KVIP_Q4_NATIVE_LI_CACHE', str(tmp_path))
    get = lambda: ref.get_reference(inputs, torch, None, None, native_li, tmp_path)
    ids, metadata = get()
    assert calls == [False, False, True] and metadata['origin'] == 'fresh'
    np.testing.assert_array_equal(ids['strong_serial'], positions)
    np.testing.assert_array_equal(ids['same_layout_native'], inputs.captured_layout[0, positions])
    assert get()[1]['origin'] == 'validated-cache' and len(calls) == 3
    path = Path(metadata['cache_path'])
    with np.load(path) as archive:
        bad = dict(archive)
    bad['strong_serial_sha256'] = '0'*64
    np.savez_compressed(path, **bad)
    assert get()[1]['origin'] == 'fresh' and len(calls) == 6
    inputs.captured_layout = np.roll(inputs.captured_layout, 1, axis=1)
    assert get()[1]['cache_path'] != str(path) and len(calls) == 9
    single, receipt = ref.get_reference(inputs, torch, None, None, native_li, tmp_path, strong_only=True)
    assert calls[9:] == [False] and receipt['origin'] == 'fresh' and set(single) == {'strong_serial'}
    np.testing.assert_array_equal(single['strong_serial'], positions)
    cached = ref.get_reference(inputs, torch, None, None, native_li, tmp_path, strong_only=True)[1]
    assert cached['origin'] == 'validated-cache' and len(calls) == 10


def test_sparse_lengths_and_scope():
    inputs = fixture()
    ref.check_positions(np.tile(np.arange(2048, dtype=np.int32), (1,4,1)), ref.native_ends(inputs))
    inputs.query_ends[0,0] -= 1
    with pytest.raises(ValueError, match='consecutive'):
        ref.native_ends(inputs)
    inputs.geometry['sparse_mode'] = 0
    assert np.all(ref.native_ends(inputs) == inputs.query_ends[0,-1])


def test_shared_serial_fixture_preserves_bits_and_physical_layout():
    torch = pytest.importorskip('torch')
    b, h, heads = 2, 2176, 32
    g = dict(batch=b, history=h, heads=heads)
    rng = np.random.default_rng(17)
    keys = rng.integers(0, 256, (b,h,128), dtype=np.uint8)
    scales = rng.random((b,h), dtype=np.float32)
    layout = np.stack([rng.permutation(h), rng.permutation(h)])
    rows = np.arange(b)[:, None]
    data = SimpleNamespace(
        query=rng.integers(0, 256, (b,4,heads,128), dtype=np.uint8),
        weights=rng.integers(0, 65536, (b,4,heads), dtype=np.uint16),
        query_scale=rng.random((b,4,heads), dtype=np.float32), query_ends=np.array(
            [[h-3,h-2,h-1,h],[h-7,h-6,h-5,h-4]], dtype=np.int32),
        logical_keys=keys, logical_scales=scales, captured_layout=layout,
        index_for_layout=lambda order:(keys[rows,order],scales[rows,order],None))
    args = ref.NativeReplayLI(torch, torch.device('cpu'), g)
    for physical in (False, True):
        args.prepare(data, physical=physical)
        f = args.fixture
        np.testing.assert_array_equal(f['actual_key'].numpy(), data.query_ends[:,-1])
        np.testing.assert_array_equal(f['li_query_lengths'].numpy(), [4,4])
        np.testing.assert_array_equal(f['block_table'].numpy(), np.arange(b*h//128).reshape(b,-1))
        np.testing.assert_array_equal(f['query'].view(torch.uint8).numpy(), data.query)
        np.testing.assert_array_equal(f['key'].view(torch.uint8).numpy().reshape(keys.shape),
                                      keys[rows,layout] if physical else keys)
        np.testing.assert_array_equal(f['native_inputs']['key_scale'].numpy().reshape(b,h),
                                      scales[rows,layout] if physical else scales)
        np.testing.assert_array_equal(f['native_inputs']['weights'].view(torch.int16).numpy(),
                                      data.weights.view(np.int16))
        np.testing.assert_array_equal(f['native_inputs']['query_scale'].numpy(), data.query_scale)


def test_raw_fused_topk_uses_preupdate_layout_and_unpack_map():
    def tensor(array):
        return SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:array))
    h = 2176
    layout = np.arange(h)[::-1][None]
    positions = np.stack([np.arange(i,i+2048) for i in range(4)]).reshape(1,4,2048)
    unpack = np.array([3,0,2,1])
    expected = layout[0,positions.reshape(4,2048)[unpack]].reshape(1,4,2048)
    state = SimpleNamespace(batch=1, history=h, input_layout=layout,
                            workspace=SimpleNamespace(topk=tensor(positions),
                                inputs=SimpleNamespace(unpack_indices=tensor(unpack))),
                            update=SimpleNamespace(logical_output=tensor(expected[...,::-1])))
    np.testing.assert_array_equal(ref.fused_li_output(state),expected)
    state.update.logical_output = tensor(np.zeros_like(positions))
    with pytest.raises(ValueError, match='Raw fused LI and the final output'):
        ref.fused_li_output(state)


def test_logical_priority_is_only_a_diagnostic_and_eager_has_no_timings(tmp_path):
    inputs = fixture()
    ids = np.tile(np.arange(2048, dtype=np.int32), (1, 4, 1))
    ids[0, 2, 0] = 2051
    rule = ref.native_priority(inputs, ids)
    assert rule['matched_queries'] == 3 and rule['split_cutoff_queries'] == 4
    assert rule['mismatches'][0]['native_only'] == [2051]
    refs = {mode: ids for mode in ref.MODES}
    fused = ids.copy()
    fused[0, 0, 0] = 2050
    out = ref.write_comparison(tmp_path, 17, inputs, refs, {}, [fused], eager=True)
    assert out['phase'] == 'eager' and out['measured_calls'] == 0
    assert out['comparisons']['strong_serial']['matched_queries'] == 3
