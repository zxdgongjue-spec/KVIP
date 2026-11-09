# SPDX-License-Identifier: Apache-2.0
"""Host checks for real initialization entry orchestration and safe teardown."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import init_transport
import kvip.adapters.kvaf
import kvip.runtime.build


@pytest.mark.parametrize('transport', ['urma', 'ubmem'])
def test_initialization_acquires_and_releases_selected_lease(monkeypatch, tmp_path, transport):
    runtime = SimpleNamespace(link_to=Mock(), initialize=Mock(), shutdown=Mock(), initialized=True)
    lease = SimpleNamespace(check_stream=Mock(), close=Mock(), region=Mock())
    constructor = Mock(return_value=runtime)
    kvaf = SimpleNamespace(__file__=str(tmp_path / 'build-npu/python/kvaf/__init__.py'), NpuRuntime=constructor)
    monkeypatch.setitem(sys.modules, 'kvaf', kvaf)
    monkeypatch.setitem(sys.modules, 'acl', SimpleNamespace(get_soc_name=lambda: 'Ascend950PR_9599'))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    torch = SimpleNamespace(npu=SimpleNamespace(set_device=Mock(), current_stream=lambda device: 'stream'),
                            device=lambda value: value, uint8='uint8', zeros=Mock(return_value='output'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(kvip.adapters.kvaf, 'KvafGetLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.adapters.kvaf, 'UbmemLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.runtime.build, 'kvaf_build_paths', lambda path: ('cann', 'headers'))
    monkeypatch.setattr(kvip.runtime.build, 'build_libraries', lambda **kwargs: ('library', 'update'))
    kernels = tmp_path / 'build-npu/bin'
    kernels.mkdir(parents=True)
    for name in ('kvaf_execute_batch_get_kernels.o', 'kvaf_execute_batch_put_kernels.o'):
        (kernels / name).touch()
    assert init_transport.main(transport, ['--device', '5', '--kvaf-server', 'host:1',
                                           '--kvaf-root', str(tmp_path)]) == 0
    runtime.link_to.assert_called_once_with('host:1')
    runtime.initialize.assert_called_once()
    lease.check_stream.assert_called_once()
    lease.close.assert_called_once()
    runtime.shutdown.assert_called_once()
    assert constructor.call_args.kwargs.get('transport', 'urma') == transport
    if transport == 'ubmem':
        lease.region.assert_called_once_with('output')
    else:
        lease.region.assert_not_called()
