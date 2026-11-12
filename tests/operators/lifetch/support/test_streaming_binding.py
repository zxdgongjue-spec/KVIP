"""Public graph-safe launch contract and pre-allocation capacity failures."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kvip.operators.lifetch.forward import LIFetchForward
from kvip.operators.lifetch.tiling import RuntimeShape


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def data_ptr(self):
        return self.values.ctypes.data

    def numel(self):
        return self.values.size

    def to(self, device):
        return self

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def copy_(self, other):
        self.values[...] = other.values


def binding_fixture(transport="urma", batch=1, history=131072, resident=16384, heads=32):
    stream = SimpleNamespace(npu_stream=303)
    torch = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        uint64=np.uint64,
        zeros=Mock(side_effect=lambda shape, dtype, device: Tensor(np.zeros(shape, dtype=dtype))),
        empty=Mock(side_effect=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype))),
        empty_like=lambda value: Tensor(np.empty_like(value.values)),
        tensor=lambda data, dtype, device: Tensor(np.asarray(data, dtype=dtype)),
        from_numpy=Tensor,
        npu=SimpleNamespace(current_stream=Mock(return_value=stream)),
    )
    rotating = SimpleNamespace(
        torch=torch,
        device=5,
        prefetch_limit=512,
        runtime_shape=RuntimeShape(
            batch=batch, history=history, resident=resident, indexHeads=heads, aicCount=32, finalWorkers=4
        ),
        context=Tensor(np.zeros(320, dtype=np.uint8)),
        hot_valid=Tensor(np.zeros(32, dtype=np.int32)),
        arguments=[Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(15)],
        library=SimpleNamespace(
            kvip_lifetch_streaming_forward=Mock(return_value=0),
            kvip_streaming_resolver_cores=Mock(return_value=3),
            kvip_streaming_flag_count=Mock(return_value=16),
        ),
    )
    if transport == "ubmem":
        rotating.lease = SimpleNamespace(check_stream=Mock(return_value=stream))
    update = SimpleNamespace(
        device=5,
        bindings=Tensor(np.zeros(28, dtype=np.uint64)),
        storage=[Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(4)],
    )
    return rotating, update


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
def test_public_forward_only_launches_retained_workspace(transport):
    rotating, update = binding_fixture(transport)
    forward = LIFetchForward(rotating, update)
    allocations = (rotating.torch.zeros.call_count, rotating.torch.empty.call_count)
    pointer = forward.workspace.data_ptr()
    for _ in range(2):
        forward()
        args = rotating.library.kvip_lifetch_streaming_forward.call_args.args
        assert args[:4] == (rotating.bindings.data_ptr(), update.bindings.data_ptr(), pointer, 303)
        assert args[5] == 2
        assert ctypes.cast(args[4], ctypes.POINTER(RuntimeShape)).contents.batch == 1
    assert allocations == (rotating.torch.zeros.call_count, rotating.torch.empty.call_count)
    if transport == "ubmem":
        rotating.lease.check_stream.side_effect = RuntimeError("released lease")
        with pytest.raises(RuntimeError, match="released lease"):
            forward()
        assert rotating.library.kvip_lifetch_streaming_forward.call_count == 2
    rotating.library.kvip_lifetch_streaming_forward.return_value = 42
    if transport == "ubmem":
        rotating.lease.check_stream.side_effect = None
    with pytest.raises(RuntimeError, match="42"):
        forward()


@pytest.mark.parametrize("batch", (0, 17, 32))
def test_batch_limit_fails_before_device_allocation(batch):
    rotating, update = binding_fixture(batch=batch)
    with pytest.raises(ValueError, match="BS1..16"):
        LIFetchForward(rotating, update)
    rotating.torch.zeros.assert_not_called()
    rotating.torch.empty.assert_not_called()


@pytest.mark.parametrize("history,resident,heads", ((33024, 8192, 64), (210944, 131072, 32), (65536, 2048, 64)))
def test_runtime_workspace_scales_without_recompilation(history, resident, heads):
    rotating, update = binding_fixture(history=history, resident=resident, heads=heads)
    rotating.prefetch_limit = 2048
    forward = LIFetchForward(rotating, update)
    assert forward.query_bitmaps.values.shape == (1, 4, resident // 32)
    assert forward.events_src.values.shape == (1, 10240)
    assert forward.dense_shadow.values.shape == (1, 3 * history)
    assert np.frombuffer(forward.workspace.values, dtype="<u4", count=1, offset=96)[0] & 0xFFFF == 2048


def test_device_and_aiv_budget_are_checked():
    rotating, update = binding_fixture(batch=16)
    update.device = 6
    with pytest.raises(ValueError, match="same device"):
        LIFetchForward(rotating, update)
    update.device = 5
    rotating.runtime_shape.aicCount = 24
    with pytest.raises(ValueError, match="AIV budget"):
        LIFetchForward(rotating, update)
    rotating.torch.empty.assert_not_called()


def test_layer_groups_share_only_ephemeral_storage():
    first, update1 = binding_fixture()
    second, update2 = binding_fixture()
    pool = {}
    first.streaming_scratch = second.streaming_scratch = pool
    a, b = LIFetchForward(first, update1), LIFetchForward(second, update2)
    assert a.dense_shadow is b.dense_shadow
    assert a.packed_fetch is b.packed_fetch
    assert a.flags is b.flags
    assert a.workspace.data_ptr() != b.workspace.data_ptr()
    assert a.rotating.context.data_ptr() != b.rotating.context.data_ptr()
    second.torch.empty.assert_not_called()
