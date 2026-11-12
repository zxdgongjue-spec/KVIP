"""Numerical failures keep subsequent UBmem cases scheduled and return failure."""
import sys
from types import SimpleNamespace

import ubmem_smoke


def test_main_keeps_cases_after_numerical_failure_and_returns_nonzero(tmp_path, monkeypatch):
    import torch
    build = tmp_path / 'build-npu'
    library = tmp_path / 'kernel.so'
    library.write_bytes(b'test artifact')
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(__file__=str(build / 'python/kvaf/__init__.py')))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True, set_device=lambda device: None),
                        raising=False)
    monkeypatch.setattr(ubmem_smoke, 'kvaf_cann', lambda build: tmp_path)
    monkeypatch.setattr(ubmem_smoke, 'query_device_resources', lambda *args: {'effective_aic': 32})
    monkeypatch.setattr(ubmem_smoke, 'build_ubmem_library', lambda **kwargs: library)
    calls = []
    def check(*args, **kwargs):
        assert args[4].intermediate_fetches == 4
        batch = args[4].batch
        calls.append(batch)
        return batch != 1
    monkeypatch.setattr(ubmem_smoke, 'check_shape', check)
    status = ubmem_smoke.main(['--build-dir', str(build), '--soc', 'Ascend950PR_9599',
                              '--kvaf-server', '127.0.0.1:1234', '--batches', '1', '2',
                              '--token-bytes', '656', '--history', '131072', '--resident', '16384',
                              '--intermediate-fetches', '4'])
    assert calls == [1, 2] and status == 1
