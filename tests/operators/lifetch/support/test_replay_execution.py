import json
from types import SimpleNamespace

import numpy as np
import pytest

import execution
from reporting import timing_summary


def setup_run(monkeypatch, *, mode='bench'):
    initial = SimpleNamespace(geometry={'batch': 1}, identities=((0, 'request', 0),))
    ref = SimpleNamespace(manifest_path='/source/manifest.json')
    class Sequence:
        rounds = ((ref,),)
        def __iter__(self):
            return iter([initial])
    sequence = Sequence()
    monkeypatch.setattr(execution, 'ReplaySequence', lambda *args: sequence)
    monkeypatch.setattr(execution, 'ReplayPagePlan', lambda seq: SimpleNamespace(page_count=32))
    return SimpleNamespace(trace_root='/trace', group=10, round_range=(1, 1), mode=mode)


def test_real_pair_saves_serial_before_fused_failure(tmp_path, monkeypatch):
    config = setup_run(monkeypatch)
    calls = []
    def run(config, directory, sequence, pages, *, fused):
        calls.append(fused)
        if fused:
            persisted = json.loads((tmp_path / 'completed_timings.json').read_text())
            assert persisted['execution'] == 'INCOMPLETE'
            assert persisted['timing']['native_chain']['mean_us'] == 20
            raise RuntimeError('fused failed')
        return {'all_rounds': timing_summary([20, 20]), 'correctness': 'PASS'}
    monkeypatch.setattr(execution, 'run_path', run)
    with pytest.raises(RuntimeError, match='fused failed'):
        execution.run_real(config, tmp_path)
    assert calls == [False, True]
    assert not (tmp_path / 'report.json').exists()


def test_real_numerical_failure_keeps_paired_timing(tmp_path, monkeypatch):
    config = setup_run(monkeypatch)
    def run(config, directory, sequence, pages, *, fused):
        return {'all_rounds': timing_summary([10 if fused else 20]),
                'correctness': 'FAIL' if fused else 'PASS'}
    monkeypatch.setattr(execution, 'run_path', run)
    assert execution.run_real(config, tmp_path) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['correctness'] == 'FAIL' and report['execution'] == 'COMPLETED'
    assert report['timing']['prototype_with_update']['mean_us'] == 10
    assert report['sequence']['serial']['correctness'] == 'PASS'


@pytest.mark.parametrize('width', [656, 2624])
def test_existing_native_mtp_supports_group_payload_width(monkeypatch, width):
    import torch
    from execution import NativeMtp
    calls = []
    runtime = SimpleNamespace(batch_get=lambda *args: calls.append(args))
    device_table = SimpleNamespace(addresses=torch.ones(16, dtype=torch.int64), valid=torch.ones(16, dtype=torch.uint8))
    session = SimpleNamespace(cpu_payload=torch.from_numpy(np.zeros((1, 2048, width), dtype=np.uint8)),
                              runtime=runtime, device_table=device_table, batch_get_attempted=False)
    mapping = torch.full((1, 2048), -1, dtype=torch.int32)
    mapping[0, :128] = torch.arange(128, dtype=torch.int32)
    owner = SimpleNamespace(batch=1, resident_budget=128, cache_map=mapping, session=session, lease=None)
    native = NativeMtp(torch, torch.device('cpu'), owner, {}, source_tokens=2048, padded_tokens=2048,
                       token_bytes=width)
    assert native.output.shape == (128, width)
    native.offsets.fill_(656)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda: None), raising=False)
    native.fetch()
    assert len(calls) == 1 and torch.all(calls[0][3] == width)
    assert calls[0][6].shape == (128, width)


def test_native_li_keeps_synthetic_contract_and_supports_real_causal_q4():
    from execution import native_li
    calls = []
    def invoke(**kwargs):
        calls.append(kwargs)
        return np.zeros((1, 4, 2048), dtype=np.int32)
    fixture = dict(query=None, key=None, native_inputs={'weights': None, 'query_scale': None, 'key_scale': None},
                   li_query_lengths=None, actual_key=None, block_table=None)
    backend = SimpleNamespace(npu_quant_lightning_indexer=invoke)
    native_li(backend, fixture)
    native_li(backend, dict(fixture, sparse_mode=3))
    assert [call['sparse_mode'] for call in calls] == [0, 3]
    assert calls[0]['layout_query'] == calls[1]['layout_query'] == 'BSND'
