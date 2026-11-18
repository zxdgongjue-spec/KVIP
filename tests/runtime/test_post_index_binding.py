# SPDX-License-Identifier: Apache-2.0
"""Post/Index/CQ needs exclusive queues and a retained, graph-stable ABI."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kvip.operators.lifetch.streaming import StreamingLIFetchForward
from test_threshold_binding import Device, Tensor, fixture


@pytest.mark.parametrize("feature", (None, 0, 1))
@pytest.mark.parametrize("batch", (1, 2, 16))
def test_post_index_workspace_keeps_default_abi_and_graph_pointers(monkeypatch, feature, batch):
    f = fixture(monkeypatch, batch)
    default = f.construct()
    assert default.post_index is False  # No new export in old native libraries.
    if feature is not None:
        f.library.kvip_low_miss_post_index = Mock(return_value=feature)
    if feature == 1:
        owner = default.owner
        owner.build_device_args.return_value = Tensor(np.zeros((80, 128), np.uint8))
        owner.core_status = Tensor(np.zeros((80, 1216), np.uint8))
    rotating = f.construct()
    assert rotating.post_index == (feature == 1)
    f.library.kvip_streaming_resolver_cores = Mock(return_value=3)
    f.library.kvip_streaming_flag_count = Mock(return_value=16)
    f.library.kvip_lifetch_streaming_forward = Mock(return_value=0)
    f.torch.empty = Mock(side_effect=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype), device))
    f.torch.npu = SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=987))
    update = SimpleNamespace(device=Device(), bindings=Tensor(np.zeros(28, np.uint64)),
                             storage=[Tensor(np.zeros(16, np.uint64)) for _ in range(4)])
    forward = StreamingLIFetchForward(rotating, update, f.torch, 512, fused_final=True, q_workers=1)
    blob = forward.workspace.values
    assert len(blob) == 256
    expected_cores = 2 * rotating.runtime_shape.aicCount if feature == 1 else 3 * batch + 4
    assert forward.times.numel() == batch * 256 + expected_cores * 32
    pointer = int.from_bytes(blob[144:152], "little")
    if feature == 1:
        assert forward.post_index.shape == (expected_cores, 16)
        assert pointer == forward.post_index.data_ptr()
        assert not forward.post_index.values.any()
    else:
        assert forward.post_index is None and pointer == 0
    assert not blob[152:].any()
    # Graph replay calls retain every tensor and only invoke the native entry.
    allocations = f.torch.empty.call_count, f.torch.zeros.call_count
    workspace_pointer = forward.workspace.data_ptr()
    for _ in range(2):
        forward()
    assert (f.torch.empty.call_count, f.torch.zeros.call_count) == allocations
    assert forward.workspace.data_ptr() == workspace_pointer
    assert int.from_bytes(forward.workspace.values[144:152], "little") == pointer
    assert f.library.kvip_lifetch_streaming_forward.call_count == 2
    # Offline order control survives diagnostic toggles without changing graph
    # addresses. Ordinary callers retain the original overlap order by default.
    for wait_first in (False, True, False):
        forward.post_index_wait_first = wait_first
        for diagnostics in (True, False):
            forward.set_options(diagnostics=diagnostics)
            flags = int.from_bytes(forward.workspace.values[96:100], "little")
            assert flags == (512 | (1 << 16) | (0x02000000 if wait_first else 0)
                             | (0x80000000 if diagnostics else 0))
            assert forward.workspace.data_ptr() == workspace_pointer
            assert int.from_bytes(forward.workspace.values[144:152], "little") == pointer

    # The optional HIGH workspace is introduced only when explicitly requested.
    forward.high_post_index = True
    with pytest.raises(RuntimeError, match="eight-window Index"):
        forward.set_options()
    if feature == 1:
        f.library.kvip_high_post_index = Mock(return_value=1)
        forward.set_options()
        high_pointer = forward._high_index_storage.data_ptr()
        expected_bytes = batch * 20480 * 160 + expected_cores * rotating.owner.abi_sizes["status"]
        assert forward._high_index_storage.shape == (expected_bytes,)
        assert int.from_bytes(blob[152:160], "little") == high_pointer
        allocation_count = f.torch.empty.call_count
        for enabled in (True, False, True):
            forward.high_post_index = enabled
            forward.set_options(diagnostics=True)
            flags = int.from_bytes(blob[96:100], "little")
            assert bool(flags & 0x01000000) == enabled
            assert int.from_bytes(blob[152:160], "little") == high_pointer
            assert forward.workspace.data_ptr() == workspace_pointer
        assert f.torch.empty.call_count == allocation_count


def test_post_index_requires_every_physical_aiv_to_have_an_exclusive_queue(monkeypatch):
    f = fixture(monkeypatch, batch=2)
    baseline = f.construct()
    f.library.kvip_low_miss_post_index = Mock(return_value=1)
    with pytest.raises(ValueError, match="exclusive prefetch and per-final-AIV queues"):
        f.construct()
    baseline.owner.build_device_args.return_value = Tensor(np.zeros((64, 128), np.uint8))
    baseline.owner.core_status = Tensor(np.zeros((64, 1216), np.uint8))
    assert f.construct().post_index is True
