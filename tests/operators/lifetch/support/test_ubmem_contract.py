# SPDX-License-Identifier: Apache-2.0
"""Host checks for the fusion lease and static-shape contract; no NPU required."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kvip.adapters.kvaf import UbmemLease
from kvip.operators.lifetch_ubmem.build import UbmemGeometry
from kvip.operators.lifetch.workspace import UbmemQueryInputs, UbmemWorkspace, device_words, ubmem_rotation_spec as rotation_spec

def test_host_copy_and_launch_models():
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler unavailable")
    script = Path(__file__).parent / "host_ubmem/run_host_models.py"
    subprocess.run([sys.executable, str(script)], check=True, timeout=180)

def lease_fixture(monkeypatch, token_bytes=656):
    device = SimpleNamespace(type="npu", index=1)
    output = SimpleNamespace(
        device=device, data_ptr=lambda: 0x800000, numel=lambda: 128 * token_bytes, element_size=lambda: 1
    )
    stream = SimpleNamespace(npu_stream=17)
    slot_bytes = (128 * token_bytes + 4095) // 4096 * 4096
    runtime = SimpleNamespace(
        device_id=1, batch_get_token_bytes=token_bytes, batch_get_tokens_per_block=128,
        remote_region=dict(base=0x100000, length=slot_bytes * 4, slot_size=slot_bytes),
        begin_get_graph=Mock(return_value=9),
        get_graph_mapped_base=Mock(return_value=0x400000),
        end_get_graph=Mock(),
    )
    monkeypatch.setitem(sys.modules, "kvaf", SimpleNamespace(GetExecutionConfig=Mock(return_value="ubmem-get")))
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=Mock(return_value=stream), synchronize=Mock()))
    return torch, runtime, output, stream

@pytest.mark.parametrize("token_bytes", (656, 2624))
def test_padded_kvaf_slots_and_failed_release_retain_owners(monkeypatch, token_bytes):
    torch, runtime, output, stream = lease_fixture(monkeypatch, token_bytes)
    lease = UbmemLease(torch, runtime, output, stream)
    runtime.begin_get_graph.assert_called_once_with("ubmem-get", output, 1, stream)
    binding = object()
    lease.bindings.append(binding)
    graph = object()
    lease.retain_graph(graph)
    runtime.end_get_graph.side_effect = RuntimeError("device fence failed")
    with pytest.raises(RuntimeError, match="fence"):
        lease.close()
    assert not lease.closed and lease.bindings == [binding] and lease.graphs == [graph]
    assert lease.outputs
    runtime.end_get_graph.side_effect = None
    lease.close()
    assert lease.closed and not lease.outputs and not lease.bindings and not lease.graphs

def test_invalid_native_lease_descriptor_releases_acquired_lease(monkeypatch):
    torch, runtime, output, stream = lease_fixture(monkeypatch)
    runtime.device_id = 7
    with pytest.raises(ValueError, match="ABI/device"):
        UbmemLease(torch, runtime, output, stream)
    runtime.end_get_graph.assert_called_once_with(9, stream, discard_result=True)

def test_peer_and_device_mapping_bases_are_kept_distinct(monkeypatch):
    torch, runtime, output, stream = lease_fixture(monkeypatch)

    class Wrapped:
        def __init__(self, array):
            self.array = array

        def to(self, device):
            return self.array

    torch.from_numpy = Wrapped
    lease = UbmemLease(torch, runtime, output, stream)
    words = lease.region(output).view("<u8")
    assert words[2] == 0x100000  # addresses supplied by CPU service
    assert words[3] == 0x400000  # actual imported NPU mapping
    assert words[4] == runtime.remote_region["length"]

def test_capture_stream_only_allowed_inside_fenced_scope(monkeypatch):
    torch, runtime, output, stream = lease_fixture(monkeypatch)
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
    runtime.end_get_graph.assert_not_called()

def test_static_shape_and_ubmem_scratch_contract():
    for batch in (1, 2):
        for width in (656, 2624):
            shape = UbmemGeometry(batch, 33024, 8192, 64, width, 32, 8, "Ascend950DT_9582")
            assert set(shape.options("/cann")) == {"RUN_MODE", "ASCEND_CANN_PACKAGE_PATH", "SOC_VERSION"}
            scratch = rotation_spec(batch, shape.history, shape.resident, shape.aic_count, shape.intermediate_fetches)
            assert scratch["ubmem_status"] == ((batch, 16), "int64")
            assert "final_status" not in scratch and "prefetch_status" not in scratch
    with pytest.raises(ValueError, match="cold tiles"):
        UbmemGeometry(2, 8448, 8192, 64, 656, 32, 8, "Ascend950DT_9582")

@pytest.mark.parametrize("batch,resident", [(1, 2048), (16, 2048), (24, 16384), (32, 2048)])
def test_extended_ubmem_geometry(batch, resident):
    if batch > 16:
        with pytest.raises(ValueError, match="BS1..16"):
            UbmemGeometry(batch, 131072, resident, 32, 656, 32, 8, "Ascend950PR_9599")
        return
    shape = UbmemGeometry(batch, 131072, resident, 32, 656, 32, 8, "Ascend950PR_9599")
    spec = rotation_spec(batch, shape.history, resident, shape.aic_count, shape.intermediate_fetches)
    assert spec["mapping"][0] == (batch, shape.history)
    assert spec["reserved_hot"][0] == (batch, resident)

def test_shared_workspace_keeps_ubmem_transport_specific_status():
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry
    from kvip.operators.lifetch.workspace import rotation_spec as urma_spec

    for batch in (1, 8, 16, 32):
        geometry = li_geometry(batch, "auto", 8, 32)
        original = urma_spec(batch, 131072, 16384, final_aivs(batch, "auto", 32),
                             geometry["intermediate_fetches"], geometry["service_aivs_per_window"], 1216)
        ubmem = rotation_spec(batch, 131072, 16384, 32, 8)
        urma_only = {"final_status", "mapping_journal", "window_topk_storage"}
        assert set(original) - urma_only == set(ubmem) - {"ubmem_status"}
        assert all(ubmem[name] == spec for name, spec in original.items() if name not in urma_only)
        assert ubmem["ubmem_status"] == ((batch, 16), "int64")

def test_ubmem_supports_sixteen_requests_and_many_windows():
    shape = UbmemGeometry(16, 131072, 16384, 32, 656, 32, 32, "Ascend950PR_9599")
    assert shape.options("/cann")["SOC_VERSION"] == "Ascend950PR_9599"

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
    with patch("kvip.operators.lifetch.workspace.UbmemQueryInputs"):
        workspace = UbmemWorkspace(torch, batch=2, history=33024, resident=8192, aics=32, windows=8, device="npu:0")
    status = workspace.scratch["ubmem_status"]
    assert status.shape == (2, 16) and status.itemsize == 8
    assert status.tobytes() == bytes(2 * 128)
    workspace.inputs.block_table = SimpleNamespace(shape=(2, 33024 // 128))
    storage = SimpleNamespace(history=33024, keys=object(), scales=object())
    assert len(workspace.arguments(storage)) == 14
    assert not hasattr(workspace, "index_plan")
    assert not hasattr(workspace, "index_work")

def test_device_abi_words_preserve_all_pointer_bits_in_signed_storage():
    words = [0, 17, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]

    def from_numpy(values):
        assert values.dtype == np.int64
        return SimpleNamespace(to=lambda device: (device, values.copy()))

    device, values = device_words(SimpleNamespace(from_numpy=from_numpy), words, "npu:1")
    assert device == "npu:1" and values.shape == (len(words),)
    assert values.tobytes() == np.asarray(words, dtype="<u8").tobytes()
    assert values.view(np.uint64).tolist() == words

def test_shared_index_update_abi_preserves_transport_descriptors():
    from kvip.operators.lifetch.index_update import _update_pointers

    def tensor(address):
        return SimpleNamespace(data_ptr=lambda: address)

    rotating = SimpleNamespace(
        change_slots=tensor(5), changes=tensor(6), rows=tensor(9), row_status=tensor(11),
        final_posts=tensor(12), final_workers=15, hot_valid=tensor(17),
        arguments=[None] * 9 + [tensor(19)], locations=tensor(20), change_count=tensor(27),
    )
    addresses = tensor(3)
    addresses.stride = lambda dim: 8
    query = SimpleNamespace(unpack_indices=tensor(21), output_valid=tensor(22))
    prefix = (tensor(1), tensor(2), addresses, tensor(4), tensor(10), rotating, tensor(7), tensor(18))
    for statuses, expected in (((13, 14, 16), (13, 14, 16)), ((113, 114, 128), (113, 114, 128))):
        values = _update_pointers(*prefix, tensor(statuses[0]), tensor(statuses[1]), statuses[2],
                                  tensor(28), query, tensor(23), tensor(24), 25, 26)
        assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                          expected[0], expected[1], 15, expected[2], 17, 18,
                          19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    values = _update_pointers(*prefix, tensor(13), tensor(14), 16, tensor(28), None, None, None, 25, 26)
    assert values[18:24] == [0] * 6

def test_service_update_arguments_use_matching_layer_group_buffers():
    from kvip.adapters.vllm.selector import index_update_arguments

    storage = SimpleNamespace(keys=object(), scales=object(), addresses=object(), logical_ids=object())
    workspace = SimpleNamespace(inputs=SimpleNamespace(batch=2, row_owners=object()))
    group_outputs = np.arange(24).reshape(2, 12)
    output_buffer = Mock()
    output_buffer.unbind.return_value = tuple(group_outputs)
    policy = SimpleNamespace(fused_errors=np.arange(6).reshape(2, 3), fused_outputs=[object(), output_buffer])
    rotating = object()
    values = index_update_arguments(policy, workspace, storage, rotating, 1)
    np.testing.assert_array_equal(values.pop("status"), policy.fused_errors[1, :2])
    np.testing.assert_array_equal(values.pop("outputs"), group_outputs[:, :8])
    output_buffer.unbind.assert_called_once_with(0)
    assert values == dict(keys=storage.keys, scales=storage.scales, addresses=storage.addresses,
                          logical_ids=storage.logical_ids, row_owners=workspace.inputs.row_owners,
                          rotating=rotating, query_inputs=workspace.inputs)

def test_unique_active_owners_accepts_repeated_padding_without_device_read():
    inputs = object.__new__(UbmemQueryInputs)
    with patch("kvip.adapters.vllm.inputs.QueryInputs.prepare", return_value=None) as original:
        inputs.prepare([0, 0], query_owners=[0, 0])
        inputs.prepare([4, 0], query_owners=[1, 1])
        assert original.call_count == 2
        with pytest.raises(ValueError, match="distinct"):
            inputs.prepare([4, 1], query_owners=[1, 1])
        assert original.call_count == 2

@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_public_selector_dispatches_transport(transport):
    import kvip

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch = SimpleNamespace(inference_mode=InferenceMode)
    setup, workspace = object(), object()
    policy = SimpleNamespace(fused_setup=setup, fused_workspace=workspace, writer=object())
    config = SimpleNamespace(transport=transport)
    with patch("kvip.adapters.vllm.binding._bind_selectors") as urma, \
         patch("kvip.adapters.vllm.binding._bind_ubmem_selectors") as ubmem:
        selected, other = (ubmem, urma) if transport == "ubmem" else (urma, ubmem)
        assert kvip.bind_selectors(torch, policy, config) is selected.return_value
        assert policy.indexer_selectors is selected.return_value
        selected.assert_called_once_with(torch, policy, config, setup, workspace)
        other.assert_not_called()

@pytest.mark.parametrize("percent", [10, 30, 50])
def test_controlled_fixture_matches_urma(percent):
    from execution import controlled_fixture

    fixture = controlled_fixture(8192, 4096, 32, percent)
    order = fixture["order"]
    assert fixture["expected"].shape == (4, 2048)
    assert fixture["native_miss_union"] > 0
    assert len(set(order)) == len(order) == 8192
    assert len(fixture["info"]["final_misses_per_query"]) == 4

def test_shared_q4_fixture_topk_and_lru_match_urma():
    from execution import controlled_fixture, layer_payload as payload
    from kvip.cache.lru import SlotLRU

    fixture = controlled_fixture(8192, 4096, 32, 10)
    scores, order = fixture["scores"], fixture["order"]
    expected = fixture["expected"]
    physical_scores = scores[:, order]
    physical = np.stack([np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in physical_scores])
    assert np.array_equal(np.sort(physical, axis=-1), expected)
    union = np.unique(physical[physical >= 4096])
    assert len(union) == fixture["native_miss_union"]
    policy = SlotLRU()
    policy.resize(4096, 4096)
    values = physical.reshape(-1)
    policy.commit(values[values < 4096])
    selected = policy.slots[0, :2048]
    assert len(set(selected)) == 2048 and np.all((selected >= 0) & (selected < 4096))
    assert payload(1, np.arange(3), 656).shape == (3, 656)
    assert payload(1, np.arange(3), 2624).shape == (3, 2624)
    np.testing.assert_array_equal(payload(1, np.arange(3), 2624)[:, :656], payload(1, np.arange(3), 656))

def test_mixed_query_counts_cover_padding_and_query_lengths():
    from execution import mixed_query_counts

    assert mixed_query_counts(2) == [1, 2]
    counts = mixed_query_counts(16)
    assert counts[:4] == [1, 2, 3, 4]
    assert counts[-1] == 0
    assert set(counts) == {0, 1, 2, 3, 4}
