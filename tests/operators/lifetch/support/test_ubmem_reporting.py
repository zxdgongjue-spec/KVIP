"""Numerical failures keep subsequent UBmem cases scheduled and return failure."""
import sys
from types import SimpleNamespace

import execution
from workloads import configuration


def test_main_keeps_cases_after_numerical_failure_and_returns_nonzero(tmp_path, monkeypatch):
    import torch
    build = tmp_path / 'build-npu'
    library = tmp_path / 'kernel.so'
    library.write_bytes(b'test artifact')
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(__file__=str(build / 'python/kvaf/__init__.py')))
    monkeypatch.setitem(sys.modules, 'acl', SimpleNamespace(get_soc_name=lambda: 'Ascend950PR_9599'))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True, set_device=lambda device: None),
                        raising=False)
    monkeypatch.setattr(execution, 'kvaf_cann', lambda build: tmp_path)
    monkeypatch.setattr(execution, 'query_device_resources', lambda *args: {'effective_aic': 32})
    monkeypatch.setattr(execution, 'build_ubmem_library', lambda **kwargs: library)
    calls = []
    def check(*args, **kwargs):
        assert args[4].intermediate_fetches == 4
        batch = args[4].batch
        calls.append(batch)
        return batch != 1
    monkeypatch.setattr(execution, 'run_ubmem_case', check)
    config = configuration(['bench', '--transport', 'ubmem', '--kvaf-root', str(tmp_path),
                            '--kvaf-server', '127.0.0.1:1234', '--batches', '1,2',
                            '--history', '131072', '--resident', '16384', '--intermediate-fetches', '4'])
    status = execution.run_ubmem(config, tmp_path)
    assert calls == [1, 2] and status == 1
