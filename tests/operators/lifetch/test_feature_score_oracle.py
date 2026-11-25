# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import feature_score_oracle as module
from cap_pair_report import oracle_for


def test_native_only_oracle_shared_across_candidates(tmp_path, monkeypatch):
    import workloads

    ends = np.tile(np.arange(2048, 2052), (16, 1))
    outputs = np.tile(np.arange(2048, dtype=np.int32), (16, 4, 1))
    inputs = SimpleNamespace(geometry={'batch': 16, 'token_bytes': 656}, identities=[[0, 0]])

    class Sequence:
        rounds = [[SimpleNamespace(manifest_path=tmp_path / 'manifest.json')]]

        def __iter__(self):
            return iter([inputs])

        def cache_ranked_round(self):
            return

    def reference(row, torch, torch_npu, device, native_li, directory, *, strong_only):
        assert strong_only and row is inputs and device == 'npu:0'
        return {'strong_serial': outputs}, dict(origin='validated-cache',
                    identity=dict(visible='4', operator='torch_npu.npu_quant_lightning_indexer'),
                    sparse_count=2048, effective_native_ends=ends.tolist())

    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '4')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(device=str, npu=SimpleNamespace(set_device=lambda d: None)))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(workloads, 'ReplaySequence', lambda *args: Sequence())
    monkeypatch.setattr(workloads, 'ReplayPagePlan', lambda seq: SimpleNamespace(page_count=42))
    monkeypatch.setattr(module, 'get_reference', reference)
    directory = tmp_path / 'oracle'
    module.prepare(tmp_path, 7, directory)
    actual, receipt, capture = oracle_for(directory, 7)
    np.testing.assert_array_equal(actual, outputs)
    assert receipt['reference_kind'] == 'native-only'
    assert capture['page_count'] == 42 and capture['source_paths'] == [str(tmp_path/'manifest.json')]
    with pytest.raises(ValueError, match='unexpected standalone'):
        oracle_for(directory, 2)
    outputs[0, 1, 0] = 2048
    np.save(directory/'strong_serial.npy', outputs)
    with pytest.raises(ValueError, match='checksum mismatch'):
        oracle_for(directory, 7)
