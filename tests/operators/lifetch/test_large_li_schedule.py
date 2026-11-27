"""Compile the same schedule functions as the native LI entry points."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_large_li_coverage(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for native LI scheduling")
    include = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = tmp_path / "coverage.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include "lightning_indexer_with_fetch/include/runtime_shape.h"
#include "lightning_indexer_with_fetch/include/large_batch_li.h"
namespace rotating {
#include "common/lifetch/sequence_schedule.h"
}
int main() {
    for(uint32_t b : {20,24,28,32}) {
        uint32_t costs[32]{}, order[32];
        for(uint32_t r=0;r<b;++r) costs[r]=128;
        large_li::RankRows(costs,b,order);
        for(uint32_t r=0;r<b;++r) assert(order[r]==r);
        if (b==32) continue;
        uint32_t h=32-b, longs=0;
        for(uint32_t r=0;r<b;r+=h) {costs[r]=1024; ++longs;}
        large_li::RankRows(costs,b,order);
        uint32_t onHelper[12]{};
        for(uint32_t i=0;i<longs;++i) {assert(costs[order[i]]==1024); ++onHelper[i%h];}
        assert(*std::max_element(onHelper,onHelper+h)-*std::min_element(onHelper,onHelper+h)<=1);
    }
    // Uniform lengths and deliberately skewed groups. Verify setting the
    // ceiling one tile lower would exceed the helper's capacity.
    for (uint32_t rows : {1,2,3,7}) for(uint32_t mixed : {0,1}) {
        uint32_t total[7]{},fixed[7]{};
        for(uint32_t r=0;r<rows;++r) {total[r]=mixed?(r+1)*128:128; fixed[r]=r+1;}
        auto budget=large_li::OwnerBudget(total,fixed,rows);
        uint32_t helper=0,lowerHelper=0;
        for(uint32_t r=0;r<rows;++r) {
            auto take=large_li::SuffixQuota(total[r]-fixed[r],fixed[r],budget);
            helper+=take;
            if(total[r]>=budget) lowerHelper+=total[r]-(budget-1);
            assert(take<=total[r]-fixed[r] && total[r]-take<=budget);
        }
        assert(helper<=budget && lowerHelper>=budget);
    }
    assert(!large_li::HasStagedMisses(0) && !large_li::HasStagedMisses(1));
    for (unsigned n=0;n<=2048;++n) assert(large_li::HasStagedMisses(large_li::kStagedBase+n));
    assert(!large_li::HasStagedMisses(2051));
    assert(large_li::CanStagePublishedMisses(131071,131072,1,131072,1));
    assert(!large_li::CanStagePublishedMisses(131071,131072,1,131071,1)); // remaining suffix
    assert(!large_li::CanStagePublishedMisses(131071,131072,1,131072,0)); // boundary ties
    assert(!large_li::CanStagePublishedMisses(131073,131200,1,131200,7)); // packed bitmap limit
    assert(!large_li::CanStagePublishedMisses(0,131072,1,131072,3));
    assert(!large_li::CanStagePublishedMisses(3,131072,0,131072,3));
    for (uint32_t n=0;n<256;++n) for (uint32_t quota=0;quota<=n;++quota)
        for (uint32_t rounds=2;rounds<=8;++rounds) if (large_li::CanOverlapBanks(n,quota,rounds))
            for (uint32_t e=1;e<rounds;++e) {
                auto span=large_li::Window(128,128+n,rounds,e);
                for (uint32_t rows=1;rows<=7;++rows)
                    assert(large_li::LaterSuffix(span,128,n,quota,rows,e)<span.end-span.begin);
            }
    // 890 cold pages on a pair of owners plus their helper.
    const uint32_t expected[] = {98,117,117};
    for(uint32_t e=1;e<=3;++e) {
        auto span=large_li::Window(128,1018,4,e);
        assert(large_li::LaterSuffix(span,128,890,296,2,e)==expected[e-1]);
    }
    for (uint32_t b : {20,24,28,32}) for (uint32_t history : {131328,524544}) {
        rotating::RuntimeShape s{};
        s.batch=b; s.aicCount=32; s.splitQ=1; s.rounds=4; s.history=history;
        s.resident=16384; s.indexHeads=32; s.tokenBytes=656;
        assert(rotating::PrepareShape(&s)==0);
        assert(s.liCores==32 && !s.splitQ && s.finalWorkers==4 && s.serviceLanes==b);
        assert(s.finalCores==4*b && s.postGroups==1 && s.batchedRows);
        uint32_t ends[128]{};
        for (uint32_t r=0;r<b;++r) ends[r*4+3]=r%2?history:(r*4127+1)%history;
        for (uint32_t r=0;r<b;++r) {
            uint32_t valid=ends[4*r+3];
            const auto tiles=large_li::CeilDiv(std::max(valid,s.resident),128);
            assert(large_li::ScanHistory(s,ends,r)==tiles*128);
            const auto helpers=32-b;
            const auto shares=helpers?(1+(b-1-r%helpers)/helpers):0;
            if (helpers) {
                uint32_t served=0;
                for(uint32_t j=r%helpers;j<b;j+=helpers) ++served;
                assert(shares==served);
            }
            uint32_t fixed[7]{},cost[7]{};
            for(uint32_t i=0;i<(shares?shares:1);++i) {
                const auto peer=helpers?r%helpers+i*helpers:r;
                const auto end=large_li::CeilDiv(std::max(ends[4*peer+3],s.resident),128);
                cost[i]=end-s.resident/128;
            }
            const auto budget=large_li::OwnerBudget(cost,fixed,shares?shares:1);
            uint32_t quota=helpers?large_li::SuffixQuota(tiles-s.resident/128,0,budget):0;
            for(uint32_t rounds: {2,4,8,32}) {
                uint32_t cursor=s.resident/128, baseSum=0, assigned=0;
                for(uint32_t e=1;e<rounds;++e) {
                    const auto span=large_li::Window(s.resident/128,tiles,rounds,e);
                    const uint32_t base=large_li::SuffixInWindow(span,s.resident/128,tiles-s.resident/128,quota);
                    const uint32_t suffix=large_li::LaterSuffix(span,s.resident/128,tiles-s.resident/128,quota,shares,e);
                    assert(suffix>=base);
                    if (e==1 || !shares || shares>3 || !base) assert(suffix==base);
                    assert(suffix-base<=(span.end-span.begin)/16 && suffix-base<=base/(shares+1));
                    assert(cursor==span.begin && suffix<=span.end-span.begin);
                    if(!helpers) assert(!suffix);
                    cursor=span.end;
                    baseSum+=base;
                    assigned+=suffix+(span.end-suffix-span.begin);
                }
                assert(cursor==tiles && baseSum==quota && assigned==tiles-s.resident/128);
            }
        }
    }
}
""")
    exe = tmp_path / "coverage"
    subprocess.run([compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_large_window_policy_covers_queries_under_lane_budget(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for native LI scheduling")
    include = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = tmp_path / "window_policy.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <random>
#include "lightning_indexer_with_fetch/op_kernel/indexer/large_window_policy.h"

using namespace large_window_policy;

void verify(const Cost* costs, uint32_t rows) {
    uint32_t assignments[64]{};
    Plan(costs, rows, assignments);
    uint32_t widths[32]{}, coverage[32]{}, peak = 0, pairPeak = 0, used = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        pairPeak = std::max(pairPeak, costs[row].q2);
    }
    for (uint32_t lane : assignments) {
        if (!lane) continue;
        const uint32_t row = (lane >> 3) / 4, first = (lane >> 3) % 4, width = lane & 7;
        assert(row < rows && (width == 1 || width == 2 || width == 4));
        assert(first % width == 0 && first + width <= 4);
        const uint32_t mask = ((1U << width) - 1U) << first;
        assert(!(coverage[row] & mask));
        assert(!widths[row] || widths[row] == width);
        widths[row] = width;
        coverage[row] |= mask;
        peak = std::max(peak, width == 1 ? costs[row].q1 : width == 2 ? costs[row].q2 : costs[row].q4);
        ++used;
    }
    assert(used <= 64);
    for (uint32_t row = 0; row < rows; ++row) assert(coverage[row] == 15);

    uint32_t optimum = 0;
    for (; optimum <= pairPeak; ++optimum) {
        uint32_t needs = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            const auto c = costs[row];
            needs += c.q4 <= optimum ? 1 : c.q2 <= optimum ? 2 : c.q1 <= optimum ? 4 : 65;
        }
        if (needs <= 64) break;
    }
    assert(peak == optimum);
    if (optimum == pairPeak) {
        for (uint32_t row = 0; row < rows; ++row) assert(widths[row] == 2);
    }
}

int main() {
    Cost costs[32]{};
    std::minstd_rand rng(43);
    for (uint32_t rows : {20, 24, 28, 32}) {
        std::fill_n(costs, rows, Estimate(24, 24, 24, 24));
        verify(costs, rows);
        for (uint32_t i = 0; i < 100; ++i) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t q0 = 1 + rng() % 64, q1 = 1 + rng() % 64;
                const uint32_t q2 = 1 + rng() % 64, q3 = 1 + rng() % 64;
                costs[row] = Estimate(q0, q1, q2, q3);
            }
            verify(costs, rows);
        }
    }
    std::fill_n(costs, 8, Estimate(50, 50, 50, 50));
    std::fill_n(costs + 8, 24, Estimate(12, 12, 12, 12));
    verify(costs, 32);
    uint32_t lanes[64]{};
    Plan(costs, 32, lanes);
    assert(std::count_if(lanes, lanes + 64, [](auto lane) { return (lane & 7) == 1; }) == 32);
    assert(std::count_if(lanes, lanes + 64, [](auto lane) { return (lane & 7) == 4; }) == 24);
}
""")
    binary = tmp_path / "window_policy"
    subprocess.run([compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_cube_vector_flags_drain_between_rows(tmp_path):
    """Exercise native range start/end around a serializable two-bank MM tile."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for mixed-core protocol")
    native = (Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch"
              / "op_kernel/indexer_large.cpp").read_text()
    finish = native.index("} // namespace QLIKernel")
    begin = native.index("__aicore__ inline void PrimeLargeBanks(")
    range_impl = native[begin:finish]
    source = tmp_path / "mixed_flags.cpp"
    source.write_text(
        r"""
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>
#define __aicore__
#define ASCEND_IS_AIV (lane>=0)
thread_local int lane;
std::atomic<int> flags[32]{};
constexpr int PIPE_V=1, PIPE_FIX=2, PIPE_ALL=3;
template<int> void PipeBarrier() {}
template<int,int> void CrossCoreSetFlag(int id) {
    int bit=id + (lane==1?16:0);
    assert(flags[bit].exchange(1)==0);
}
template<int,int> void CrossCoreWaitFlag(int id) {
    int bit=id + (lane==1?16:0);
    while (!flags[bit].exchange(0)) std::this_thread::yield();
}
namespace QLICommon {
struct ConstInfo {
    static constexpr int QLI_SYNC_MODE4=4, CROSS_VC_EVENT=0, CROSS_CV_EVENT=2, AIV0_AIV1_OFFSET=16;
};
}
struct Service {
    void AllocEventID() {}
    void FreeEventID() {}
    unsigned thresholdsLoaded=0;
    void LoadSequenceSplitThresholds(uint32_t) { ++thresholdsLoaded; }
};
template<class T> struct QuantLightningIndexerKernel {
    void ProcessLargeScore(uint32_t,uint32_t,bool,bool,bool);
    Service vectorService, matmulService;
    uint32_t assignedRow_=0;
    void CalcGS1LoopParams(uint32_t) {}
    void CalcS2LoopParams(uint32_t,uint32_t) {}
    void ScanSequenceHalf(uint32_t begin,uint32_t end,uint32_t,uint32_t&,uint32_t&,bool) {
        for (uint32_t i=begin;i<end;++i) {
            auto bank=(i-begin)%2;
            if (lane<0) {
                for (uint32_t v : {0,16}) CrossCoreWaitFlag<4,PIPE_FIX>(v+bank);
                for (uint32_t v : {0,16}) CrossCoreSetFlag<4,PIPE_FIX>(v+bank+2);
            } else {
                CrossCoreWaitFlag<4,PIPE_V>(bank+2);
                CrossCoreSetFlag<4,PIPE_V>(bank);
            }
        }
    }
};
"""
        + range_impl
        + r"""
int main() {
    std::vector<std::thread> workers;
    for(int worker=-1;worker<2;++worker) workers.emplace_back([=]{
        lane=worker;
        QuantLightningIndexerKernel<int> op;
        for (int r=0;r<100;++r) {
            bool loaded = false;
            for (uint32_t n:{0,1,2,3,7,16}) {
                const bool early = (r%2 == 0) && n != 0;
                if (lane>=0 && early) { PrimeLargeBanks(); std::this_thread::yield(); }
                op.ProcessLargeScore(32,32+n,true,!loaded,early);
                loaded |= n != 0;
            }
        }
        assert(op.vectorService.thresholdsLoaded == (worker < 0 ? 0U : 100U));
    });
    for(auto& worker:workers) worker.join();
    for(auto& flag:flags) assert(flag.load()==0);
}
"""
    )
    exe = tmp_path / "mixed_flags"
    subprocess.run([compiler, "-pthread", "-O2", "-std=c++17", str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], timeout=15, check=True)
