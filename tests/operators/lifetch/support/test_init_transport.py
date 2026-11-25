# SPDX-License-Identifier: Apache-2.0
"""Host checks for real initialization entry orchestration and safe teardown."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np

import pytest
from execution import initialize_transport
import kvip.adapters.kvaf
import kvip.runtime.build


@pytest.mark.parametrize('transport', ['urma', 'ubmem'])
def test_initialization_acquires_and_releases_selected_lease(monkeypatch, tmp_path, transport):
    class Tensor(np.ndarray):
        def cpu(self):
            return self
        def to(self, device):
            return self
        def contiguous(self):
            return self
    def tensor(value, **kwargs):
        return np.asarray(value, dtype=kwargs.get('dtype')).view(Tensor)
    stored = {}
    def put(keys, source, offsets, sizes):
        assert keys == [0] and offsets == [0] and sizes == [128 * 656]
        stored['bytes'] = source.copy().reshape(-1)
    def get(addresses, valid, indices, offsets, count, destinations, output, stream, flag):
        assert count.tolist() == [128]
        assert np.array_equal(offsets, np.arange(128) * 656)
        for offset, destination in zip(offsets, destinations):
            output[destination] = stored['bytes'][offset:offset + 656]
    runtime = SimpleNamespace(link_to=Mock(), initialize=Mock(), shutdown=Mock(), initialized=True,
                              batch_put=Mock(side_effect=put), batch_get=Mock(side_effect=get),
                              prepare_access_table=Mock(return_value='table'), release_access_table=Mock(),
                              access_table_view=Mock(return_value=SimpleNamespace(addresses='addresses', valid='valid')))
    lease = SimpleNamespace(check_stream=Mock(), close=Mock(), region=Mock())
    constructor = Mock(return_value=runtime)
    kvaf = SimpleNamespace(
        __file__=str(tmp_path / 'build-npu/python/kvaf/__init__.py'), NpuRuntime=constructor,
        PutExecutionConfig=Mock(return_value='put'), GetExecutionConfig=Mock(return_value='get'),
        ExecutionResources=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, 'kvaf', kvaf)
    monkeypatch.setitem(sys.modules, 'acl', SimpleNamespace(get_soc_name=lambda: 'Ascend950PR_9599'))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    stream = SimpleNamespace(synchronize=Mock())
    torch = SimpleNamespace(npu=SimpleNamespace(set_device=Mock(), current_stream=lambda device: stream),
                            device=lambda value: value, uint8=np.uint8, int32=np.int32,
                            zeros=lambda shape, **kwargs: tensor(np.zeros(shape, dtype=kwargs['dtype'])),
                            from_numpy=tensor, tensor=tensor, equal=np.array_equal,
                            arange=lambda n, **kwargs: tensor(np.arange(n), dtype=kwargs.get('dtype')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(kvip.adapters.kvaf, 'KvafGetLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.adapters.kvaf, 'UbmemLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.runtime.build, 'kvaf_build_paths', lambda path: ('cann', 'headers'))
    monkeypatch.setattr(kvip.runtime.build, 'build_libraries', lambda **kwargs: ('library',))
    kernels = tmp_path / 'build-npu/bin'
    kernels.mkdir(parents=True)
    for name in ('kvaf_execute_batch_get_kernels.o', 'kvaf_execute_batch_put_kernels.o'):
        (kernels / name).touch()
    assert initialize_transport(transport, ['--device', '5', '--kvaf-server', 'host:1',
                                           '--kvaf-root', str(tmp_path)]) == 0
    runtime.link_to.assert_called_once_with('host:1')
    runtime.initialize.assert_called_once()
    lease.check_stream.assert_called_once()
    lease.close.assert_called_once()
    runtime.shutdown.assert_called_once()
    runtime.release_access_table.assert_called_once_with('table')
    runtime.batch_get.assert_called_once()
    kvaf.PutExecutionConfig.assert_called_once_with(transport, 'aiv')
    kvaf.GetExecutionConfig.assert_called_once_with(transport, 'aiv', 'direct')
    assert constructor.call_args.kwargs['transport_mode'] == transport
    if transport == 'ubmem':
        lease.region.assert_called_once()
        assert lease.region.call_args.args[0].shape == (128, 656)
    else:
        lease.region.assert_not_called()
