"""Check the graph's real launch policy, device tile partition and capture-owned ready flags."""
import ctypes
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.operators.lifetch.terminal import TerminalLeaseBinding
from kvip.operators.lifetch.tiling import RuntimeShape


def test_fixed_score_geometry_and_binding(tmp_path):
    operator = Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch"
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for fixed Q4 scheduling")
    source = tmp_path / "policy.cpp"
    source.write_text(
        """
#include "fixed_score_groups.h"
#include <cassert>
#include <initializer_list>
extern "C" uint32_t kvip_fixed_q4_score_groups(const rotating::RuntimeShape* s) {
    return fixed_score::LaunchGroups(*s);
}
extern "C" void prove_tiles() {
    for (uint32_t batch : {1u, 2u, 4u, 8u}) {
        const uint32_t parts = fixed_score::GroupsPerRow(batch) / 2;
        for (uint32_t n = 0; n <= 1025; ++n) {
            uint32_t next = 128;
            for (uint32_t p = 0; p < parts; ++p) {
                auto range = fixed_score::Split(128, 128+n, p, parts);
                assert(range.begin == next && range.end >= range.begin);
                assert(range.end - range.begin >= n / parts);
                assert(range.end - range.begin <= (n + parts-1) / parts);
                next = range.end;
            }
            assert(next == 128+n);
        }
    }
}
"""
    )
    lib = tmp_path / "policy.so"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(operator / "include"),
                    str(source), "-o", str(lib)], check=True)
    library = ctypes.CDLL(str(lib))
    library.prove_tiles()

    class Tensor:
        def __init__(self, dims=None, data=None):
            self.dims, self.data = dims, data

        def data_ptr(self):
            return id(self)

    torch = SimpleNamespace(
        zeros=lambda dims, **kw: Tensor(dims),
        empty=lambda dims, **kw: Tensor(dims),
        tensor=lambda data, **kw: Tensor(data=data),
        int32=None, int64=None, uint8=None,
    )
    shapes = [
        (1, 131328, 8), (2, 131072, 16), (4, 131328, 32), (8, 131328, 32),
        (16, 131328, 0), (20, 131328, 0), (24, 131328, 0), (28, 131328, 0),
        (32, 131328, 0), (4, 524544, 0),
    ]
    for batch, history, expected in shapes:
        shape = RuntimeShape()
        for key, value in dict(batch=batch, history=history, resident=16384, aicCount=32,
                               indexHeads=32, splitQ=1, rounds=min(4, batch*2),
                               liCores=batch*2, finalWorkers=4, tokenBytes=656).items():
            setattr(shape, key, value)
        lease = object.__new__(TerminalLeaseBinding)
        lease.rotating = SimpleNamespace(runtime_shape=shape, torch=torch, device=None, library=library)
        lease.args, lease.templates = Tensor(), Tensor()
        lease.attach_workspace(SimpleNamespace(rotating=lease.rotating, workspace=Tensor()))
        fields = struct.unpack("<6Q2I9Q", bytes(lease.workspace.data))
        if expected:
            assert lease.fixed_score_ready.dims == (2 * expected, 32)
            assert fields[11] == lease.fixed_score_ready.data_ptr()
        else:
            assert lease.fixed_score_ready is None
            assert fields[11] == 0


def test_public_forward_uses_extended_entry_only_with_ready_state(monkeypatch):
    from kvip.operators.lifetch import forward

    class Lease:
        def __init__(self, rotating):
            self.fixed_score_ready = rotating.ready

        def attach_workspace(self, owner):
            pass

    def legacy(*args):
        pass

    def sharded(*args):
        pass

    monkeypatch.setattr(forward, "TerminalLeaseBinding", Lease)
    monkeypatch.setattr(forward.StreamingLIFetchForward, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(forward.LIFetchForward, "set_options", lambda self: None)
    for ready in (None, object()):
        library = SimpleNamespace(kvip_parallel_tail_supported=lambda shape: 1, kvip_parallel_tail_forward=legacy)
        if ready is not None:
            library.kvip_fixed_q4_small_forward = sharded
        rotating = SimpleNamespace(library=library, shape_pointer=None, torch=None, prefetch_limit=256, ready=ready)
        op = forward.LIFetchForward(rotating, None)
        assert op.terminal_function is (sharded if ready is not None else legacy)


def test_disjoint_windows_return_key_bank_and_keep_query(tmp_path):
    """Run the production range loop against a model of the cube's two-tile K loads."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for native LI scheduling")
    operators = Path(__file__).resolve().parents[3] / "csrc/operators"
    indexer = operators / "lightning_indexer_with_fetch/op_kernel/indexer/indexer.h"
    native = indexer.read_text()
    start = native.index("template <typename QLIT>\n__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ScanSequenceHalf(")
    end = native.index("\n#if !V9_UBMEM\n}\n\n// Score the recent range", start)
    source = tmp_path / "bank.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include "fixed_score_groups.h"
namespace rotating {
#include "sequence_schedule.h"
}
#define __aicore__
#define V9_UBMEM 0
#define KVIP_FIXED_SCORE_SHARDS 1
#define ASCEND_IS_AIC (true)
namespace QLICommon {
constexpr unsigned KScaleTiles=4;
struct RunInfo {
    unsigned tile;
    bool isFirstS2InnerLoop, isLastS2InnerLoop, reuseQuery=false, retainQuery=false;
};
}
namespace fixed_window_q {
bool DeferInWindow(bool enabled, unsigned owner, unsigned epoch, unsigned tile) {
    return enabled && (epoch == 2 || epoch == 3) && owner + 1 == epoch && tile < 3;
}
}
struct Cube {
    unsigned history, begin=0, end=0, loads=0, releases=0, deferred=0;
    bool occupied=false;
    template<bool Defer = false> void ComputeMm1(const QLICommon::RunInfo& i) {
        if constexpr (Defer) ++deferred;
        if (!occupied || i.tile < begin || i.tile >= end) {
            // A cache miss needs the current KEY_MTE1_MTE2 credit.
            assert(!occupied);
            begin=i.tile;
            end=std::min(history, begin+2);
            occupied=true;
        }
        if (i.tile+1 == end || i.isLastS2InnerLoop) occupied=false;
        if (i.isFirstS2InnerLoop && !i.reuseQuery) ++loads;
        if (i.isLastS2InnerLoop && !i.retainQuery) ++releases;
    }
};
template<class T> struct QuantLightningIndexerKernel {
    struct { unsigned s2Start; } splitCoreInfo;
    struct { unsigned s2LoopEnd, validS2Len; } tempLoopInfo;
    Cube matmulService{};
    struct { void ProcessVec1(QLICommon::RunInfo, unsigned, bool) {} } vectorService;
    static constexpr unsigned S2_BASE_SIZE=128;
    void CalcRunInfo(unsigned, unsigned tile, QLICommon::RunInfo& out, unsigned, unsigned) { out.tile=tile; }
    bool fixedDeferCredit_ = false;
    void ScanSequenceHalf(unsigned,unsigned,unsigned,unsigned&,unsigned&,bool,bool=false,unsigned=0,unsigned=0);
};
"""
        + native[start:end] + "\n}\n"
        + r"""
int main() {
    // A disjoint recent range can reuse the same Q after the hot range.
    QuantLightningIndexerKernel<int> recent;
    recent.matmulService.history = 32;
    unsigned loop = 0, scale = 0;
    recent.ScanSequenceHalf(0, 3, 0, loop, scale, false);
    assert(!recent.matmulService.occupied && !recent.matmulService.releases);
    recent.ScanSequenceHalf(20, 21, 0, loop, scale, true);
    assert(!recent.matmulService.occupied && recent.matmulService.loads == 1 && recent.matmulService.releases == 1);
    assert(recent.matmulService.deferred == 0);
    QuantLightningIndexerKernel<int> overlap;
    overlap.fixedDeferCredit_ = true;
    overlap.matmulService.history = 32;
    loop = scale = 0;
    overlap.ScanSequenceHalf(0, 4, 0, loop, scale, false, false, 2, 1);
    assert(overlap.matmulService.deferred == 3 && !overlap.matmulService.occupied);
    overlap.ScanSequenceHalf(12, 17, 0, loop, scale, true, false, 3, 1);
    assert(overlap.matmulService.deferred == 3 && !overlap.matmulService.occupied);
    for (unsigned history : {131072u,131328u}) for (unsigned batch : {1u,2u,4u,8u}) {
        rotating::RuntimeShape s{};
        s.batch=batch; s.history=history; s.resident=16384; s.indexHeads=32; s.aicCount=32;
        s.rounds=batch==1 ? 2 : 4; s.splitQ=1; s.tokenBytes=656;
        assert(!rotating::PrepareShape(&s));
        auto half=rotating::SequenceHalfWindow(s)/128, parts=fixed_score::LaunchGroups(s)/(2*batch);
        for (unsigned side=0; side<2; ++side) for (unsigned p=0; p<parts; ++p) {
            QuantLightningIndexerKernel<int> op;
            op.matmulService.history=history/128;
            unsigned loop=0, scale=0;
            for (unsigned epoch=1; epoch<s.rounds; ++epoch) {
                auto begin=s.resident/128+(2*(epoch-1)+side)*half;
                auto slice=fixed_score::Split(begin,std::min(begin+half,history/128),p,parts);
                bool last=epoch+1==s.rounds;
                op.ScanSequenceHalf(slice.begin,slice.end,side,loop,scale,last);
                assert(!op.matmulService.occupied);
                assert(op.matmulService.loads==1 && op.matmulService.releases==unsigned(last));
            }
        }
    }
}
"""
    )
    exe = tmp_path / "bank"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(operators / "lightning_indexer_with_fetch/include"),
                    "-I", str(operators / "common/lifetch"), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=5)


def test_fixed_last_window_uses_only_logical_score_lanes(tmp_path):
    operator = Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch"
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for fixed Q4 window mapping")
    source = tmp_path / "combo.cpp"
    source.write_text(
        """
#include "fixed_score_groups.h"
#include <array>
#include <cassert>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace pipeline_ab { struct Args { uint64_t words[32]; }; }
namespace kvaf_device { void CleanCacheLines(GM_ADDR, uint32_t) {} }
namespace rotating {
uint32_t ServiceRank(uint32_t batch, uint32_t split, uint32_t g) {
    return split ? (g % 2) * batch + g / 2 : g;
}
uint32_t ServiceLane(uint32_t b, uint32_t split, uint32_t g, uint32_t core, uint32_t sub) {
    return ServiceRank(b, split, core) % g * 2 + sub;
}
}
#include "fixed_window_q.h"

int main() {
    alignas(128) std::array<uint8_t, 8 * 5 * 128> mailbox{};
    alignas(128) std::array<uint64_t, 32> workspace{};
    auto args = reinterpret_cast<GM_ADDR>(workspace.data());
    for (uint32_t batch : {1U, 2U, 4U, 8U}) {
        rotating::RuntimeShape s{};
        s.batch = batch;
        s.history = 131328;
        s.resident = 16384;
        s.aicCount = s.indexHeads = 32;
        s.finalWorkers = 4;
        s.splitQ = 1;
        s.liCores = 2 * batch;
        s.rounds = batch == 1 ? 2 : 4;
        s.serviceAics = (batch + 1) / 2;
        s.serviceLanes = 2 * s.serviceAics;
        s.tokenBytes = 656;
        auto &credit = workspace[fixed_window_q::kCreditOptionOffset / 8];
        assert(!fixed_window_q::DeferCredits(args, s));
        credit = 1;
        assert(fixed_window_q::DeferCredits(args, s) == (batch == 4));
        for (uint32_t owner = 0; owner < 4; ++owner)
            for (uint32_t epoch = 0; epoch < 4; ++epoch)
                for (uint32_t tile = 0; tile < 8; ++tile) {
                    auto eligible = (owner == 1 || owner == 2) && epoch == owner + 1 && tile < 3;
                    assert(fixed_window_q::DeferInWindow(true, owner, epoch, tile) == eligible);
                    assert(!fixed_window_q::DeferInWindow(false, owner, epoch, tile));
                }
        credit = 0;
        auto &option = workspace[fixed_window_q::kScratchOffset / 8];
        assert(!fixed_window_q::Scratch(args, s));
        option = reinterpret_cast<uint64_t>(mailbox.data());
        auto scratch = fixed_window_q::Scratch(args, s);
        assert(scratch == mailbox.data());
        std::array<std::array<unsigned, 4>, 8> assigned{};
        for (uint32_t aiv = 0; aiv < 2 * fixed_score::LaunchGroups(s); ++aiv) {
            auto [row, q] = fixed_window_q::Helper(s, aiv);
            if (aiv >= s.liCores * 2) { assert(q == 0); continue; }
            assert(row < batch && q < 4);
            if (q) ++assigned[row][q];
            fixed_window_q::Publish(scratch, aiv / 4, aiv % 4, 1);
        }
        for (uint32_t aiv = 0; aiv < s.liCores * 2; ++aiv) {
            fixed_window_q::Reset(args, s, aiv);
            assert(fixed_window_q::Signal(scratch, aiv / 4, aiv % 4)->ready == 0);
        }
        for (uint32_t row = 0; row < batch; ++row)
            assert((assigned[row] == std::array<unsigned, 4>{0, 1, 1, 1}));
        ++s.rounds;
        assert(!fixed_window_q::Scratch(args, s));
        option = 0;
    }
}
"""
    )
    binary = tmp_path / "combo"
    subprocess.run([compiler, "-std=c++17", "-I", str(operator / "include"),
                    "-I", str(operator / "op_kernel/indexer"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
