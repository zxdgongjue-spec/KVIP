"""Exercise shared DDR ownership with real CPU tensors and a fake transport."""
import sys
from types import SimpleNamespace

import pytest
import torch

from bench_pool import BenchPool


@pytest.mark.parametrize('transport,workers', [('urma', 40), ('ubmem', 2)])
def test_pool_puts_once_and_retains_tables_until_all_cases_complete(tmp_path, monkeypatch, transport, workers):
    events = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            assert kwargs['npu_core_count'] == workers
            events.append('create')

        def link_to(self, server):
            events.append('link')

        def initialize(self, size, slot):
            events.append('initialize')

        def batch_put(self, keys, source, offsets, lengths):
            events.append(('put', keys))
            assert len(keys) <= source.shape[0]
            assert offsets == [i * 128 * 656 for i in keys]

        def prepare_access_table(self, keys):
            events.append(('table', len(keys)))
            return keys

        def access_table_view(self, table, **kwargs):
            return SimpleNamespace(addresses=torch.arange(len(table)), valid=torch.ones(len(table)))

        def release_access_table(self, table):
            events.append(('release_table', len(table)))

        def shutdown(self):
            events.append('shutdown')

    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(NpuRuntime=Runtime))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None), raising=False)
    pool = BenchPool(SimpleNamespace(batches=(1, 2), history=256, transport=transport,
                                    kvaf_root=tmp_path, kvaf_server='test:1'))
    first = pool.borrow(torch, torch.device('cpu'), 1, 256, 2)
    original = first.cpu_payload.clone()
    with pytest.raises(ValueError):
        pool.borrow(torch, torch.device('cpu'), 2, 256, 2)
    with pytest.raises(RuntimeError):
        pool.close()
    pool.release()
    for batch in (2, 1, 2):
        item = pool.borrow(torch, torch.device('cpu'), batch, 256, 2)
        assert item.runtime is first.runtime
        assert len(item.device_table.addresses) == batch * 2
        assert torch.equal(item.cpu_payload[:1], original)
        pool.release()
    assert len([event for event in events if isinstance(event, tuple) and event[0] == 'put']) == 1
    assert 'shutdown' not in events
    pool.close()
    assert events.count('create') == events.count('shutdown') == 1
    assert sorted(event[1] for event in events if isinstance(event, tuple) and event[0] == 'release_table') == [2, 4]


def test_native_payload_check_uses_device_reference_and_detects_corruption():
    from native_mtp import NativeMtp
    native = NativeMtp.__new__(NativeMtp)
    native.torch = torch
    native.batch, native.hot, native.token_bytes, native.source_tokens = 1, 2048, 656, 2048
    selected = torch.arange(2048).repeat(1, 4, 1)
    native.destinations = selected.clone()
    native.forward = torch.arange(2048).reshape(1, -1)
    native.counts = torch.tensor([0, 0])
    reference = torch.ones((1, 2048, 656), dtype=torch.uint8)
    native.output = reference.clone().reshape(-1, 656)
    # Deliberately different CPU data proves validation uses the retained reference.
    native.owner = SimpleNamespace(session=SimpleNamespace(reference=reference, cpu_payload=torch.zeros_like(reference)))
    assert native.verify(selected) == [0, 0]
    native.output[100, 655] = 0
    with pytest.raises(ValueError, match='payload_mismatch'):
        native.verify(selected)
    assert reference[0, 100, 655] == 1


def test_put_chunks_keep_one_registered_source_and_global_offsets(tmp_path, monkeypatch):
    calls = []
    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def link_to(self, *args):
            pass
        def initialize(self, *args):
            pass
        def batch_put(self, keys, source, offsets, lengths):
            calls.append((source.data_ptr(), keys, offsets, lengths))
        def prepare_access_table(self, keys):
            return keys
        def access_table_view(self, table, **kwargs):
            return SimpleNamespace(addresses=torch.arange(len(table)), valid=torch.ones(len(table)))
        def release_access_table(self, table):
            pass
        def shutdown(self):
            pass
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(NpuRuntime=Runtime))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None), raising=False)
    pool = BenchPool(SimpleNamespace(batches=(2,), history=16384, transport='urma',
                                    kvaf_root=tmp_path, kvaf_server='test:1'))
    pool.borrow(torch, torch.device('cpu'), 2, 16384, 2)
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert [key for _, keys, _, _ in calls for key in keys] == list(range(256))
    for _, keys, offsets, lengths in calls:
        assert offsets == [key * 128 * 656 for key in keys]
        assert lengths == [128 * 656] * len(keys)
    from contract_payload import payload
    import numpy as np
    for row in range(2):
        assert np.array_equal(pool.cpu_payload[row].numpy(), payload(row, np.arange(16384)))
    pool.release()
    pool.close()
