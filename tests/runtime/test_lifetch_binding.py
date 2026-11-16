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


@pytest.fixture(autouse=True)
def physical_device(monkeypatch):
    monkeypatch.setattr("kvip.operators.lifetch.streaming.query_device_resources", lambda *a: dict(aic=32, aiv=64))


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
        npu=SimpleNamespace(current_stream=Mock(return_value=stream), current_device=lambda: 5),
    )
    rotating = SimpleNamespace(
        torch=torch,
        device=5,
        prefetch_limit=512,
        queue_count=128,
        runtime_shape=RuntimeShape(
            batch=batch,
            history=history,
            resident=resident,
            indexHeads=heads,
            aicCount=32,
            finalWorkers=4,
            liCores=batch * (2 if batch <= 16 else 1),
        ),
        context=Tensor(np.zeros(320, dtype=np.uint8)),
        hot_valid=Tensor(np.zeros(32, dtype=np.int32)),
        arguments=[Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(15)],
        library=SimpleNamespace(
            kvip_lifetch_streaming_forward_v2=Mock(return_value=0),
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
        args = rotating.library.kvip_lifetch_streaming_forward_v2.call_args.args
        assert args[:4] == (rotating.bindings.data_ptr(), update.bindings.data_ptr(), pointer, 303)
        assert args[5:] == (2, 64)
        assert ctypes.cast(args[4], ctypes.POINTER(RuntimeShape)).contents.batch == 1
    assert allocations == (rotating.torch.zeros.call_count, rotating.torch.empty.call_count)
    if transport == "ubmem":
        rotating.lease.check_stream.side_effect = RuntimeError("released lease")
        with pytest.raises(RuntimeError, match="released lease"):
            forward()
        assert rotating.library.kvip_lifetch_streaming_forward_v2.call_count == 2
    rotating.library.kvip_lifetch_streaming_forward_v2.return_value = 42
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
    rotating.runtime_shape.liCores = 16
    rotating.aiv_budget = 48
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

    header = (
        Path(__file__).resolve().parents[2]
        / "csrc/operators/lightning_indexer_with_fetch/op_kernel/index_cache_update.h"
    )
    source = tmp_path / "failure.cpp"
    source.write_text(
        r"""
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
"""
        + f'#include "{header}"\n'
        + r"""
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
"""
    )
    binary = tmp_path / "failure"
    subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_c28_b16_uses_four_records_with_independent_v56(monkeypatch):
    monkeypatch.setattr("kvip.operators.lifetch.streaming.query_device_resources", lambda *a: dict(aic=28, aiv=56))
    rotating, update = binding_fixture(batch=16)
    rotating.runtime_shape.aicCount = 28
    rotating.runtime_shape.liCores = 16
    rotating.queue_count = 48
    forward = LIFetchForward(rotating, update, aiv_budget=56)
    assert forward.schedule.tail_aiv == 56
    forward()
    assert rotating.library.kvip_lifetch_streaming_forward_v2.call_args.args[5:] == (2, 56)
    with pytest.raises(ValueError, match="AIV budget"):
        LIFetchForward(rotating, update, q_workers=4, aiv_budget=56)
    with pytest.raises(ValueError, match="actual device"):
        LIFetchForward(rotating, update, aiv_budget=64)


def test_queue_requirement_tracks_fetch_workers():
    rotating, update = binding_fixture(batch=1)
    rotating.queue_count = 18
    with pytest.raises(ValueError, match="queues"):
        LIFetchForward(rotating, update, q_workers=4)
    rotating.queue_count = 34
    assert LIFetchForward(rotating, update, q_workers=4).schedule.urma_queues == 34


def test_unified_score_configuration_and_worker_options():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/operators/lifetch/support"))
    from execution import runner_options, uses_compact_fixture
    from workloads import configuration

    default = configuration(["bench"])
    assert default.score_route == "legacy" and default.q_workers == 2
    assert default.aic_budget == 0 and default.aiv_budget is None
    assert uses_compact_fixture(default)
    explicit = configuration(
        ["bench", "--batches", "16", "--score-route", "auto", "--aic-budget", "28", "--aiv-budget", "56"]
    )
    worker = runner_options(explicit)
    assert worker.batches == "16" and worker.score_route == "auto"
    assert (worker.aic_budget, worker.aiv_budget, worker.q_workers) == (28, 56, 2)
    for argv in (["bench", "--exact-miss"], ["bench", "--history", "39424", "--valid-history", "39322"]):
        assert not uses_compact_fixture(configuration(argv))
    assert worker.trace_rounds == 0
    from execution import require_paired_batch

    require_paired_batch(1)
    require_paired_batch(16)
    for batch in (0, 17, 24, 32):
        with pytest.raises(ValueError, match="paired benchmarks support BS1..16"):
            require_paired_batch(batch)
    for argv in (
        ["bench", "--transport", "ubmem", "--score-route", "auto"],
        ["bench", "--batches", "32"],
        ["bench", "--batches", "32", "--score-route", "auto"],
        ["diagnose", "--score-route", "bsq", "--level", "1"],
        ["all", "--score-route", "auto"],
        ["diagnose", "--score-route", "bsq", "--level", "2"],
        ["bench", "--input", "real", "--trace-root", "/tmp", "--group", "0", "--round-range", "0:1", "--exact-miss"],
    ):
        try:
            configuration(argv)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"unsupported configuration accepted: {argv}")


def test_controlled_partial_fixture_exact_overlap_and_seed(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/operators/lifetch/support"))
    from workloads import prepare_full_chain_fixture

    history, valid, resident, target = 39424, 39322, 16384, 205
    values = []
    for seed in (43, 47):
        root = tmp_path / f"seed-{seed}"
        info = prepare_full_chain_fixture(
            root, history, resident, 10, index_heads=32, seed=seed, exact_miss=True, valid_history=valid
        )
        order = np.load(root / "physical_to_logical.npy")
        scores = np.load(root / "logical_scores.npy")[:, order]
        selected_cold = set()
        for row in scores:
            cutoff = np.partition(row[:valid], -2048)[-2048]
            selected = np.flatnonzero(row >= cutoff)
            assert len(selected) == 2048 and np.all(selected < valid)
            cold = selected[selected >= resident]
            assert len(cold) == target
            selected_cold.update(cold.tolist())
        assert len(selected_cold) == info["unique_misses_per_request"] == 2 * target
        assert info["seed"] == seed and info["final_misses_per_query"] == [target] * 4
        assert np.fromfile(root / "arg6.bin", np.uint32).tolist() == [valid]
        assert np.all(scores[:, valid:] == 0)
        assert np.all(np.fromfile(root / "arg4.bin", np.float32)[valid:] == 0)
        assert np.all(order[:valid] < valid) and np.array_equal(order[valid:], np.arange(valid, history))
        values.append(scores[:, :resident])
    assert not np.array_equal(*values)  # Preserve the main fixture's public seed contract.


def test_unified_ubmem_constructor_calls_preserve_public_signature():
    import ast
    import inspect
    from pathlib import Path

    from kvip.operators.lifetch_ubmem.binding import UbmemBinding

    source = Path(__file__).resolve().parents[2] / "tests/operators/lifetch/support/execution.py"
    calls = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "UbmemBinding"
    ]
    assert len(calls) == 2  # Synthetic and captured replay both use the actual constructor.
    for call in calls:
        keywords = {item.arg: None for item in call.keywords if item.arg is not None}
        if any(item.arg is None for item in call.keywords):
            keywords.update(
                query_ends=None,
                hot_valid=None,
                row_owners=None,
                token_addresses=None,
                tail_bounds=None,
                output_stride=None,
                scratch=None,
            )
        inspect.signature(UbmemBinding).bind(*[None for _ in call.args], **keywords)
