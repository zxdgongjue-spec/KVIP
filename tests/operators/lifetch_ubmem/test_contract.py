# SPDX-License-Identifier: Apache-2.0
"""Host checks for the fusion lease and static-shape contract; no NPU required."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kvip.adapters.ubmem import UbmemLease
from kvip.operators.lifetch_ubmem.build import UbmemGeometry
from kvip.operators.lifetch.workspace import UbmemQueryInputs, UbmemWorkspace, device_words, ubmem_rotation_spec as rotation_spec


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
        mapped_base=0x400000,
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



def test_invalid_native_lease_descriptor_releases_acquired_lease():
    torch, runtime, output, stream = lease_fixture()
    context = runtime.acquire_ubmem_fused_context.return_value
    context["device_id"] = 7
    with pytest.raises(ValueError, match="ABI/device"):
        UbmemLease(torch, runtime, output, stream)
    runtime.release_ubmem_fused_context.assert_called_once_with(context, stream)

def test_peer_and_device_mapping_bases_are_kept_distinct():
    torch, runtime, output, stream = lease_fixture()

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
    assert words[4] == runtime.acquire_ubmem_fused_context.return_value["region_bytes"]


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
            shape = UbmemGeometry(batch, 33024, 8192, 64, width, 32, 8, "Ascend950DT_9582")
            assert set(shape.options("/cann")) == {"RUN_MODE", "ASCEND_CANN_PACKAGE_PATH", "SOC_VERSION"}
            scratch = rotation_spec(batch, shape.history, shape.resident, shape.aic_count, shape.intermediate_fetches)
            assert scratch["ubmem_status"] == ((batch, 16), "int64")
            assert "final_status" not in scratch and "prefetch_status" not in scratch
    with pytest.raises(ValueError, match="cold tiles"):
        UbmemGeometry(2, 8448, 8192, 64, 656, 32, 8, "Ascend950DT_9582")


@pytest.mark.parametrize("batch,resident", [(1, 2048), (16, 2048), (24, 16384), (32, 2048)])
def test_extended_ubmem_geometry(batch, resident):
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
        assert set(original) - {"final_status"} == set(ubmem) - {"ubmem_status"}
        assert all(ubmem[name] == spec for name, spec in original.items() if name != "final_status")
        assert ubmem["ubmem_status"] == ((batch, 16), "int64")


def test_ubmem_supports_wide_batch_and_many_windows():
    shape = UbmemGeometry(32, 131072, 16384, 32, 656, 32, 32, "Ascend950PR_9599")
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
    workspace = SimpleNamespace(inputs=SimpleNamespace(row_owners=object()))
    policy = SimpleNamespace(fused_errors=[object(), object()], fused_outputs=[object(), object()])
    rotating = object()
    values = index_update_arguments(policy, workspace, storage, rotating, 1)
    assert values == dict(keys=storage.keys, scales=storage.scales, addresses=storage.addresses,
                          logical_ids=storage.logical_ids, row_owners=workspace.inputs.row_owners,
                          rotating=rotating, status=policy.fused_errors[1], query_inputs=workspace.inputs,
                          outputs=policy.fused_outputs[1])


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
    import kvip

    launch = Mock(return_value=0)
    device = object()
    rotating = SimpleNamespace(
        device=device,
        runtime_shape=ctypes.c_uint32(7),
        library=SimpleNamespace(kvip_ubmem_forward=launch),
        bindings=SimpleNamespace(data_ptr=lambda: 101),
        lease=SimpleNamespace(check_stream=Mock(return_value=SimpleNamespace(npu_stream=303))),
    )
    update = SimpleNamespace(device=device, bindings=SimpleNamespace(numel=lambda: 28, data_ptr=lambda: 202))
    forward = kvip.LIFetch(rotating, update)
    forward()
    args = launch.call_args.args
    assert args[:3] == (101, 202, 303)
    assert ctypes.cast(args[3], ctypes.POINTER(ctypes.c_uint32)).contents.value == 7
    launch.return_value = 4
    with pytest.raises(RuntimeError, match="launch failed: 4"):
        forward()


def test_public_forward_preserves_urma_launch_and_stream():
    import kvip

    launch = Mock(return_value=0)
    device = object()
    stream = SimpleNamespace(npu_stream=303)
    rotating = SimpleNamespace(
        device=device,
        shape_pointer=404,
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=Mock(return_value=stream))),
        library=SimpleNamespace(kvip_lifetch_forward=launch),
        bindings=SimpleNamespace(data_ptr=lambda: 101),
    )
    update = SimpleNamespace(device=device, bindings=SimpleNamespace(numel=lambda: 28, data_ptr=lambda: 202))
    kvip.LIFetch(rotating, update)()
    launch.assert_called_once_with(101, 202, 303, 404)
    rotating.torch.npu.current_stream.assert_called_once_with(device)


@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_public_selector_dispatches_transport(transport):
    import kvip

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch = SimpleNamespace(inference_mode=InferenceMode)
    policy = object()
    config = SimpleNamespace(transport=transport)
    with patch("kvip.adapters.vllm.binding._bind_selectors") as urma, \
         patch("kvip.adapters.vllm.ubmem_binding._bind_selectors") as ubmem:
        kvip.bind_selectors(torch, policy, config)
        (ubmem if transport == "ubmem" else urma).assert_called_once_with(torch, policy, config)
        (urma if transport == "ubmem" else ubmem).assert_not_called()


@pytest.mark.parametrize("percent", [10, 30, 50])
def test_controlled_fixture_matches_urma(percent):
    from smoke import controlled_fixture

    fixture = controlled_fixture(8192, 4096, 32, percent)
    order = fixture["order"]
    assert fixture["expected"].shape == (4, 2048)
    assert fixture["native_miss_union"] > 0
    assert len(set(order)) == len(order) == 8192
    assert len(fixture["info"]["final_misses_per_query"]) == 4


def test_shared_q4_fixture_topk_and_lru_match_urma():
    from smoke import controlled_fixture, payload
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
    from smoke import mixed_query_counts

    assert mixed_query_counts(2) == [1, 2]
    counts = mixed_query_counts(16)
    assert counts[:4] == [1, 2, 3, 4]
    assert counts[-1] == 0
    assert set(counts) == {0, 1, 2, 3, 4}


def test_runtime_batch_bounds_compact_outputs(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3]
    text = (root / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/index_cache_update.cpp").read_text()
    begin = text.index("__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs")
    end = text.index("template <bool Owned>", begin)
    source = r'''
#include <cstdint>
#include <vector>
#include <cassert>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
FUNCTION
int main() {
 std::vector<int32_t> topk(8192,-1), slots(8192,-1), out(8192,99), dest(8192,99);
 std::vector<int64_t> unpack={0,3,4,0}, valid={1,1,1,0}, ids(64);
 uint32_t owners[]={0},status[]={0}; uint64_t a[26]={};a[24]=64;
 for(unsigned i=0;i<64;++i)ids[i]=100+i;
 topk[0]=7;slots[0]=11;topk[3*2048]=8;slots[3*2048]=12;
 a[3]=(uint64_t)ids.data();a[6]=(uint64_t)status;a[9]=(uint64_t)owners;
 a[18]=(uint64_t)topk.data();a[19]=(uint64_t)slots.data();a[20]=(uint64_t)unpack.data();
 a[21]=(uint64_t)valid.data();a[22]=(uint64_t)out.data();a[23]=(uint64_t)dest.data();
 for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishOutputs(a,0,1);
 assert(out[0]==107 && dest[0]==11);assert(out[2048]==108 && dest[2048]==12);
 assert(out[4096]==-1 && dest[4096]==-1);assert(out[6144]==-1 && dest[6144]==-1);
}
'''.replace("FUNCTION", text[begin:end])
    cpp, exe = tmp_path / "outputs.cpp", tmp_path / "outputs"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(exe)], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True)
