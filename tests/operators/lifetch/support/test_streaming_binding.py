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

    def element_size(self):
        return self.values.itemsize

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def clone(self):
        return Tensor(self.values.copy())

    def view(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            return Tensor(self.values.view(shape[0]))
        return Tensor(self.values.reshape(*shape))

    def fill_(self, value):
        self.values.fill(value)
        return self

    def zero_(self):
        return self.fill_(0)

    def to(self, device):
        return self

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def __setitem__(self, key, value):
        self.values[key] = value

    def copy_(self, other):
        self.values[...] = other.values


def binding_fixture(transport="urma", batch=1, history=131072, resident=16384, heads=32):
    stream = SimpleNamespace(npu_stream=303)
    torch = SimpleNamespace(
        uint8=np.uint8,
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
        rows=Tensor(np.zeros((batch, 128), dtype=np.uint64)),
        events=Tensor(np.zeros((32, 16), dtype=np.uint64)),
        mapping=Tensor(np.full((batch, history), -1, dtype=np.int32)),
        mapping_journal=Tensor(np.zeros((batch, 8192 + 32), dtype=np.uint32)),
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
        assert args[:4] == (forward.bindings.data_ptr(), update.bindings.data_ptr(), pointer, 303)
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


def test_single_jetty_workspace_survives_diagnostic_updates():
    rotating, update = binding_fixture()
    forward = LIFetchForward(rotating, update, fetch_jettys_per_aiv=1)
    for diagnostics in (False, True, False):
        forward.set_options(diagnostics=diagnostics)
        assert np.frombuffer(forward.workspace.values, dtype="<u4", count=1, offset=192)[0] == 1
        snapshots = np.frombuffer(forward.workspace.values, dtype="<u8", count=6, offset=144)
        assert snapshots.tolist() == [forward.flags.data_ptr(), forward._live_count.data_ptr(),
                                     rotating.rows.data_ptr(), rotating.events.data_ptr(),
                                     rotating.mapping.data_ptr(), rotating.mapping_journal.data_ptr()]
        forward()
        assert rotating.library.kvip_lifetch_streaming_forward.call_args.args[5] == 2


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


def test_device_failure_invalidates_only_failed_owner(tmp_path):
    """Execute the retained production error handler after removing old update."""
    import subprocess
    from pathlib import Path

    header = Path(__file__).resolve().parents[4] / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/index_cache_update.h'
    source = tmp_path / 'failure.cpp'
    source.write_text(r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct { unsigned x; } threadIdx;
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T value){*p=value;}
void asc_threadfence(){}
''' + f'#include "{header}"\n' + r'''
int main(){
 uint64_t args[28]={};uint32_t status[2]={0,0},owners[2]={2,0};
 std::vector<uint32_t> valid(3*1024,1);
 args[6]=(uintptr_t)status;args[9]=(uintptr_t)owners;
 args[16]=(uintptr_t)valid.data();args[25]=1024;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,0,0);
 for(auto v:valid)assert(v==1);
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,0,9);
 assert(status[0]==4 && status[1]==0);
 for(unsigned i=0;i<valid.size();++i)assert(valid[i]==(i<2048?1U:0U));
 status[1]=2;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,1,0);
 assert(status[1]==2);
 for(unsigned i=0;i<1024;++i)assert(valid[i]==0 && valid[1024+i]==1);
}
''')
    binary = tmp_path / 'failure'
    subprocess.run(['c++', '-std=c++17', '-fsanitize=address,undefined', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
