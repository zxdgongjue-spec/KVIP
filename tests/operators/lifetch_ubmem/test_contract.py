# SPDX-License-Identifier: Apache-2.0
"""Host checks for the fusion lease and static-shape contract; no NPU required."""

from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kvip.adapters.ubmem import UbmemLease
from kvip.operators.lifetch_ubmem.forward import UbmemForward
from kvip.operators.lifetch_ubmem.specialization import Specialization
from kvip.operators.lifetch_ubmem.workspace import UbmemQueryInputs, UbmemWorkspace, device_words, rotation_spec


def lease_fixture(token_bytes=656):
    device = SimpleNamespace(type="npu", index=1)
    output = SimpleNamespace(
        device=device, data_ptr=lambda: 0x800000, numel=lambda: 128 * token_bytes, element_size=lambda: 1
    )
    stream = SimpleNamespace(npu_stream=17)
    slot_bytes = (128 * token_bytes + 4095) // 4096 * 4096
    context = dict(
        abi_version=1,
        lease_id=9,
        device_id=1,
        region_base=0x100000,
        region_bytes=slot_bytes * 4,
        slot_bytes=slot_bytes,
        output_address=output.data_ptr(),
        output_bytes=output.numel(),
        token_bytes=token_bytes,
        tokens_per_block=128,
    )
    runtime = SimpleNamespace(
        transport="ubmem",
        acquire_ubmem_fused_context=Mock(return_value=context),
        release_ubmem_fused_context=Mock(),
        retain_ubmem_fused_graph=Mock(),
    )
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=Mock(return_value=stream), synchronize=Mock()))
    return torch, runtime, output, stream


@pytest.mark.parametrize("token_bytes", (656, 2624))
def test_padded_kvaf_slots_and_failed_release_retain_owners(token_bytes):
    torch, runtime, output, stream = lease_fixture(token_bytes)
    lease = UbmemLease(torch, runtime, output, stream)
    binding = object()
    lease.bindings.append(binding)
    graph = object()
    lease.retain_graph(graph)
    runtime.release_ubmem_fused_context.side_effect = RuntimeError("device fence failed")
    with pytest.raises(RuntimeError, match="fence"):
        lease.close()
    assert not lease.closed and lease.bindings == [binding] and lease.graphs == [graph]
    assert lease.outputs
    runtime.release_ubmem_fused_context.side_effect = None
    lease.close()
    assert lease.closed and not lease.outputs and not lease.bindings and not lease.graphs


def test_capture_stream_only_allowed_inside_fenced_scope():
    torch, runtime, output, stream = lease_fixture()
    lease = UbmemLease(torch, runtime, output, stream)
    capture = SimpleNamespace(npu_stream=23)
    with lease.capture_scope():
        torch.npu.current_stream.return_value = capture
        assert lease.check_stream() is capture
    with pytest.raises(RuntimeError, match="execution stream"):
        lease.check_stream()
    torch.npu.current_stream.return_value = stream
    torch.npu.synchronize.side_effect = [None, RuntimeError("failed fence")]
    with pytest.raises(RuntimeError, match="failed fence"):
        with lease.capture_scope():
            pass
    with pytest.raises(RuntimeError, match="unknown"):
        lease.close()
    runtime.release_ubmem_fused_context.assert_not_called()


def test_static_shape_and_ubmem_scratch_contract():
    for batch in (1, 2):
        for width in (656, 2624):
            shape = Specialization(batch, 33024, 8192, 64, width, 32, 8, "Ascend950DT_9582")
            assert shape.options("/cann")["SIM_HISTORY"] == 33024
            scratch = rotation_spec(batch, shape.history, shape.resident, shape.aic_count, shape.intermediate_fetches)
            assert scratch["ubmem_status"] == ((batch, 16), "int64")
            assert "final_status" not in scratch and "prefetch_status" not in scratch
    with pytest.raises(ValueError, match="cold tiles"):
        Specialization(2, 8448, 8192, 64, 656, 32, 8, "Ascend950DT_9582")


def test_optimization_switches_are_explicit_content_addressed_build_options():
    baseline = Specialization(16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582")
    optimized = Specialization(
        16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582", parallel_init=True, balanced_index=True
    )
    assert baseline.options("/cann")["UBMEM_PARALLEL_INIT"] == 0
    assert baseline.options("/cann")["UBMEM_BALANCED_INDEX"] == 0
    assert optimized.options("/cann")["UBMEM_PARALLEL_INIT"] == 1
    assert optimized.options("/cann")["UBMEM_BALANCED_INDEX"] == 1
    assert baseline.options("/cann")["SKIP_INACTIVE"] == 1
    assert baseline.options("/cann")["COMPACT_RESERVED"] == 0
    compact = Specialization(16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582", compact_reserved=True)
    assert compact.options("/cann")["COMPACT_RESERVED"] == 1
    paired = Specialization(16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582", grouped_copy=True)
    assert paired.options("/cann")["UBMEM_GROUPED_COPY"] == 1
    assert baseline.options("/cann")["UBMEM_GROUPED_COPY"] == 0
    tagged = replace(paired, tagged_map=True, final_copy_stages=8)
    assert tagged.options("/cann")["UBMEM_TAGGED_MAP"] == 1
    assert tagged.options("/cann")["UBMEM_FINAL_COPY_STAGES"] == 8
    for overrides in (
        dict(tagged_map=1),
        dict(tagged_map=True, parallel_init=True),
        dict(final_copy_stages=5),
        dict(final_copy_stages=True),
        dict(final_copy_stages=8, grouped_copy=False),
    ):
        with pytest.raises(ValueError):
            replace(paired, **overrides)
    with pytest.raises(ValueError, match="booleans"):
        Specialization(16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582", parallel_init=1)
    with pytest.raises(ValueError, match="booleans"):
        Specialization(16, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582", grouped_copy=1)
    with pytest.raises(ValueError, match="four-layer"):
        Specialization(16, 131328, 16384, 32, 656, 32, 8, "Ascend950DT_9582", grouped_copy=True)


def test_workspace_initializes_with_npu_supported_zero_dtypes():
    def zeros(size, *, dtype, device):
        if dtype == np.uint64:
            raise RuntimeError("aclnnInplaceZero does not support DT_UINT64")
        return np.zeros(size, dtype=dtype)

    torch = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        uint8=np.uint8,
        uint64=np.uint64,
        zeros=zeros,
        full=lambda size, value, *, dtype, device: np.full(size, value, dtype=dtype),
        empty=lambda size, *, dtype, device: np.empty(size, dtype=dtype),
    )
    with patch("kvip.operators.lifetch_ubmem.workspace.UbmemQueryInputs"):
        workspace = UbmemWorkspace(torch, batch=2, history=33024, resident=8192, aics=32, windows=8, device="npu:0")
    status = workspace.scratch["ubmem_status"]
    assert status.shape == (2, 16) and status.itemsize == 8
    assert status.tobytes() == bytes(2 * 128)
    workspace.inputs.block_table = SimpleNamespace(shape=(2, 33024 // 128))
    storage = SimpleNamespace(history=33024, keys=object(), scales=object())
    assert len(workspace.arguments(storage)) == 14
    assert workspace.index_plan.shape == (2, 40968)
    assert workspace.index_work.shape == (2, 40960)


def test_device_abi_words_preserve_all_pointer_bits_in_signed_storage():
    words = [0, 17, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]

    def from_numpy(values):
        assert values.dtype == np.int64
        return SimpleNamespace(to=lambda device: (device, values.copy()))

    device, values = device_words(SimpleNamespace(from_numpy=from_numpy), words, "npu:1")
    assert device == "npu:1" and values.shape == (len(words),)
    assert values.tobytes() == np.asarray(words, dtype="<u8").tobytes()
    assert values.view(np.uint64).tolist() == words


def test_unique_active_owners_accepts_repeated_padding_without_device_read():
    inputs = object.__new__(UbmemQueryInputs)
    with patch("kvip.adapters.vllm.inputs.QueryInputs.prepare", return_value=None) as original:
        inputs.prepare([0, 0], query_owners=[0, 0])
        inputs.prepare([4, 0], query_owners=[1, 1])
        assert original.call_count == 2
        with pytest.raises(ValueError, match="distinct"):
            inputs.prepare([4, 1], query_owners=[1, 1])
        assert original.call_count == 2


def test_forward_only_submits_bound_pointers_on_the_current_lease_stream():
    launch = Mock(return_value=0)
    device = object()
    rotating = SimpleNamespace(
        device=device,
        library=SimpleNamespace(kvip_ubmem_forward=launch),
        bindings=SimpleNamespace(data_ptr=lambda: 101),
        lease=SimpleNamespace(check_stream=Mock(return_value=SimpleNamespace(npu_stream=303))),
    )
    update = SimpleNamespace(device=device, bindings=SimpleNamespace(numel=lambda: 28, data_ptr=lambda: 202))
    forward = UbmemForward(rotating, update)
    forward()
    launch.assert_called_once_with(101, 202, 303)
    launch.return_value = 4
    with pytest.raises(RuntimeError, match="launch failed: 4"):
        forward()
