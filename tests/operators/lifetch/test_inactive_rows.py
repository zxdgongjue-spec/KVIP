# SPDX-License-Identifier: Apache-2.0
"""Execute the production LI control loops with host compute/event probes."""

import shutil
import subprocess
from pathlib import Path

import pytest


def method(source, name):
    start = source.index("template <typename QLIT>\n__aicore__ inline void QuantLightningIndexerKernel<QLIT>::" + name)
    end = source.find("\ntemplate <typename QLIT>", start + 1)
    return source[start : end if end >= 0 else source.index("}  // namespace QLIKernel", start)]


@pytest.mark.parametrize("backend", ["lightning_indexer_with_fetch", "lightning_indexer_with_fetch_ubmem"])
@pytest.mark.parametrize("skip", [0, 1])
def test_actual_control_loop_keeps_all_service_lanes_without_idle_compute(tmp_path, backend, skip):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    operator = root / backend
    source = (operator / "op_kernel/indexer/indexer.h").read_text()
    methods = method(source, "ProcessMain()") + method(source, "ProcessInactiveGroup(")
    # The runtime header's otherwise-unused native ABI include is not needed
    # by this host model. All service geometry comes from production context.h.
    (tmp_path / "batch_get_abi.h").write_text("#pragma once\n")
    is_ubmem = backend.endswith("_ubmem")
    shape = "" if is_ubmem else "rotating::RuntimeShape runtimeShape_{};"
    configure = (
        ""
        if is_ubmem
        else """
        op.runtimeShape_.batch=B;op.runtimeShape_.history=H;op.runtimeShape_.resident=R;
        op.runtimeShape_.splitQ=SPLIT;op.runtimeShape_.rounds=ROUNDS;
        op.runtimeShape_.serviceAics=SERVICE;"""
    )
    epoch = "rotating::EpochAt(tile,core)" if is_ubmem else "rotating::EpochAt(B,H,R,SPLIT,ROUNDS,SERVICE,tile,core)"
    harness = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include "context.h"
#include "inactive_row.h"
#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (side<2)
constexpr unsigned B=SIM_BATCH,H=SIM_HISTORY,R=SIM_RESIDENT,SPLIT=V9_SPLIT_Q;
constexpr unsigned CORES=B*(1+SPLIT),ROUNDS=V9_INTERMEDIATE_FETCHES,SERVICE=CORES/ROUNDS;
constexpr unsigned PIPE_V=0,PIPE_FIX=1;
unsigned side=0,core=0,allocated=0,freed=0,computes=0,topks=0;
bool checkIdle=false;
std::array<unsigned,10> credits{};
std::array<unsigned,B*4> ends{},progress{},outputs{},bitmap{};
std::array<unsigned,ROUNDS*SERVICE*2> services{};
namespace QLICommon {
struct ConstInfo {static constexpr unsigned QLI_SYNC_MODE4=4,CROSS_VC_EVENT=2;};
struct RunInfo {unsigned bIdx=0,gS1Idx=0;};
}
template<unsigned,unsigned> void CrossCoreSetFlag(unsigned flag) {++credits[flag];}
template<unsigned,unsigned> void CrossCoreWaitFlag(unsigned flag) {
    assert(side==2 && credits[flag]==2);credits[flag]-=2;
}
int Epoch(unsigned tile) {return EPOCH;}
struct Vector {
    int gated=-1;
    void AllocEventID(){++allocated;}void FreeEventID(){++freed;}
    void WaitInactiveServiceWindow(unsigned epoch){gated=epoch;}
    void ProcessTopK(const QLICommon::RunInfo& info) {
        ++topks;
        unsigned count=SPLIT?1:2,first=info.bIdx*4+info.gS1Idx*(SPLIT?2:4)+side*count;
        for(unsigned q=first;q<first+count;++q){assert(ends[q]==0);outputs[q]=UINT32_MAX;}
    }
    void RotateFetch(unsigned epoch) {
        unsigned own=core/(1+SPLIT);
        if(checkIdle && !(ends[own*4]|ends[own*4+1]|ends[own*4+2]|ends[own*4+3]))
            assert(gated==int(epoch));
        unsigned rank=SPLIT?(core%2)*B+core/2:core,lane=(rank%SERVICE)*2+side;
        ++services[epoch*SERVICE*2+lane];
        for(unsigned b=lane;b<B;b+=SERVICE*2)for(unsigned q=b*4;q<b*4+4;++q) {
            // Service can visit another producer's row before it runs. Its
            // initialized zero readiness must exclude every stale bitmap bit.
            if(!progress[q])assert(std::min(progress[q],std::max(ends[q],R))==0);
            if(checkIdle && !(ends[b*4]|ends[b*4+1]|ends[b*4+2]|ends[b*4+3]))
                assert(progress[q]==0 && bitmap[q]==0xdeadbeef);
        }
    }
};
struct Cube {void AllocEventID(){++allocated;}void FreeEventID(){++freed;}};
template<class QLIT> struct QuantLightningIndexerKernel {
    Vector vectorService;Cube matmulService;SHAPE
    uint64_t queryEnds_=0;unsigned aiCoreIdx=0;bool rotationServed_=false;
    struct {bool isCoreEnable=true;unsigned bN2Start=0,bN2End=0,gS1Start=0,gS1End=0,s2Start=0;}splitCoreInfo;
    struct {bool curActSeqLenIsZero=false,needDealActS1LessThanS1=false;
        unsigned bIdx=0,n2Idx=0,gS1LoopEnd=0,gS1Idx=0,s2LoopEnd=H/128-1,actS1Size=4;}tempLoopInfo;
    void CalcGS1LoopParams(unsigned b){tempLoopInfo.bIdx=b;tempLoopInfo.gS1LoopEnd=splitCoreInfo.gS1End;}
    void CalcS2LoopParams(unsigned,unsigned group){tempLoopInfo.gS1Idx=group;}
    void CalcRunInfo(unsigned,unsigned,QLICommon::RunInfo& info,unsigned,unsigned){
        info.bIdx=tempLoopInfo.bIdx;info.gS1Idx=tempLoopInfo.gS1Idx;}
    void DealActSeqLenIsZero(unsigned,unsigned,unsigned){assert(false);}
    void ProcessBaseBlock(unsigned,unsigned tile,QLICommon::RunInfo,unsigned,unsigned){
        ++computes;
        if(side<2){unsigned count=SPLIT?1:2,first=tempLoopInfo.bIdx*4+tempLoopInfo.gS1Idx*(SPLIT?2:4)+side*count;
            for(unsigned q=first;q<first+count;++q)progress[q]=(tile+1)*128;}
        int e=Epoch(tile+1);
        if(!rotationServed_ && e>=0){
            if(side<2){vectorService.RotateFetch(e);CrossCoreSetFlag<4,PIPE_V>(8);}
            else CrossCoreWaitFlag<4,PIPE_FIX>(8);
            rotationServed_=true;
        }
    }
    void ProcessMain();void ProcessInactiveGroup(const QLICommon::RunInfo&);
};
METHODS
int main(){
    for(unsigned mask=0;mask<16;++mask)
        assert(QLICommon::InactiveQueryRow(mask&1,mask&2,mask&4,mask&8)==bool(V9_SKIP_INACTIVE && !mask));
    for(unsigned active:{0U,1U,B/2,B})for(bool missingPointer:{false,true}){
        checkIdle=V9_SKIP_INACTIVE && !missingPointer;
        ends.fill(0);progress.fill(0);outputs.fill(17);bitmap.fill(0xdeadbeef);services.fill(0);
        for(unsigned b=0;b<active;++b)for(unsigned q=0;q<4;++q)ends[b*4+q]=H-4+q;
        // Partially padded Q4 must stay on the original path.
        if(active && active<B){ends[(active-1)*4]=0;}
        computes=topks=allocated=freed=0;
        for(core=0;core<CORES;++core){
            credits.fill(0);
            for(side=0;side<3;++side){
                QuantLightningIndexerKernel<int> op;op.aiCoreIdx=core;
                op.queryEnds_=missingPointer?0:reinterpret_cast<uintptr_t>(ends.data());
                op.splitCoreInfo.bN2Start=op.splitCoreInfo.bN2End=core/(1+SPLIT);
                op.splitCoreInfo.gS1Start=op.splitCoreInfo.gS1End=SPLIT?core%2:0;
                CONFIGURE
                op.ProcessMain();
            }
            assert(std::all_of(credits.begin(),credits.end(),[](unsigned n){return !n;}));
        }
        const bool skipping=V9_SKIP_INACTIVE && !missingPointer;
        unsigned activeCores=(skipping?active:B)*(1+SPLIT);
        assert(computes==activeCores*3*(H/128));
        assert(topks==(skipping?B-active:0)*(1+SPLIT)*2);
        assert(allocated==3*CORES && freed==allocated);
        assert(std::all_of(services.begin(),services.end(),[](unsigned n){return n==1;}));
        if(skipping)for(unsigned q=active*4;q<B*4;++q)
            assert(outputs[q]==UINT32_MAX && progress[q]==0 && bitmap[q]==0xdeadbeef);
    }
}
"""
    harness = (
        harness.replace("SHAPE", shape)
        .replace("CONFIGURE", configure)
        .replace("EPOCH", epoch)
        .replace("METHODS", methods)
    )
    path = tmp_path / "inactive.cpp"
    path.write_text(harness)
    common = dict(
        SIM_HISTORY=131328,
        SIM_RESIDENT=16384,
        INDEX_HEADS=32,
        SIM_Q=4,
        TOKEN_BYTES=2624,
        V9_AIC_COUNT=32,
        V9_FINAL_AIVS=0,
        V9_SERVICE_AICS=0,
        V9_FINAL_REPLICATED=1,
        V9_DEDICATED_SERVICE=0,
        V9_UBPIPE_DEPTH=8,
        V9_UBPIPE_ROWS=4,
        V9_SKIP_INACTIVE=skip,
    )
    for batch, split, rounds in ((1, 0, 1), (1, 1, 2), (3, 1, 3), (16, 0, 8), (16, 1, 8)):
        definitions = common | dict(SIM_BATCH=batch, V9_SPLIT_Q=split, V9_INTERMEDIATE_FETCHES=rounds)
        binary = tmp_path / f"inactive-{batch}-{split}"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-fsanitize=address,undefined",
                *[f"-D{k}={v}" for k, v in definitions.items()],
                "-I" + str(tmp_path),
                "-I" + str(operator / "include"),
                "-I" + str(root / "common/lightning_indexer"),
                str(path),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run([str(binary)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("backend", ["lightning_indexer_with_fetch", "lightning_indexer_with_fetch_ubmem"])
def test_actual_idle_window_gate_waits_for_active_rows_only(tmp_path, backend):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    operator = root / backend
    source = (operator / "op_kernel/indexer/indexer_vector.h").read_text()
    start = source.index("    __aicore__ inline void WaitInactiveServiceWindow(")
    gate = source[start : source.index("    __aicore__ inline void RotateFetch(", start)]
    static = backend.endswith("_ubmem")
    shape = "" if static else "rotating::RuntimeShape runtimeShape_{};"
    configure = (
        ""
        if static
        else """
        op.runtimeShape_.batch=B;op.runtimeShape_.history=H;op.runtimeShape_.resident=R;
        op.runtimeShape_.splitQ=1;op.runtimeShape_.rounds=8;
        op.runtimeShape_.serviceAics=4;op.runtimeShape_.serviceLanes=8;"""
    )
    trigger = "rotating::Trigger(epoch)*128" if static else "rotating::Trigger(H,R,8,epoch)*128"
    (tmp_path / "batch_get_abi.h").write_text("#pragma once\n")
    harness = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include "context.h"
#include "inactive_row.h"
#define __aicore__
#define __gm__
constexpr unsigned B=16,H=131328,R=16384,EVENT_ID0=0,PIPE_ALL=0;
std::array<unsigned,B*4> ends{},targets{},reads{};
unsigned stage=0,localValue=0;
enum class HardEvent {S_MTE2,MTE2_S};
template<unsigned> void PipeBarrier(){}
template<HardEvent E> void SetFlag(unsigned){if constexpr(E==HardEvent::S_MTE2)stage=1;else assert(stage==2);}
template<HardEvent E> void WaitFlag(unsigned){if constexpr(E==HardEvent::MTE2_S)stage=3;}
struct Tensor {
    template<class T>Tensor ReinterpretCast(){return {};}
    unsigned GetValue(unsigned){assert(stage==3);return localValue;}
    unsigned operator[](unsigned word){assert(word%32==0);return word/32;}
};
void DataCopy(Tensor,unsigned query,unsigned words){
    assert(stage==1 && words==32 && targets[query]>0 && ends[query]>0);
    // Two incomplete DMA snapshots precede maturity. GetValue must follow
    // MTE2_S, and the loop must poll again instead of admitting this window.
    localValue=++reads[query]<3?targets[query]-1:targets[query];stage=2;
}
struct Vector {
    SHAPE
    unsigned blockId_=0;uint8_t* rotationContext_=nullptr;
    Tensor scoreOutLocal_,rotationProgressGm_;
    GATE
};
int main(){
    for(bool lru:{false,true})for(unsigned epoch=0;epoch<8;++epoch)for(unsigned block=0;block<64;++block){
        rotating::Context context{};context.queryEnds=reinterpret_cast<uintptr_t>(ends.data());context.lruSlots=lru;
        Vector op;op.rotationContext_=reinterpret_cast<uint8_t*>(&context);op.blockId_=block;CONFIGURE
        for(unsigned q=0;q<B*4;++q)ends[q]=(q%3==0?0:(q%3==1?R/2:H-3));
        targets.fill(0);reads.fill(0);
        unsigned core=block/2,rank=(core%2)*B+core/2,lane=(rank%4)*2+block%2;
        const unsigned trigger=TRIGGER;
        for(unsigned b=lane;b<B;b+=8)for(unsigned q=b*4;q<b*4+4;++q)
            if(ends[q])targets[q]=std::min(trigger,lru?std::max(ends[q],R):ends[q]);
        op.WaitInactiveServiceWindow(epoch);
        for(unsigned q=0;q<B*4;++q)assert(reads[q]==(targets[q]?3U:0U));
    }
}
"""
    harness = (
        harness.replace("SHAPE", shape)
        .replace("CONFIGURE", configure)
        .replace("TRIGGER", trigger)
        .replace("GATE", gate)
    )
    path, binary = tmp_path / "gate.cpp", tmp_path / "gate"
    path.write_text(harness)
    definitions = dict(
        SIM_BATCH=16,
        SIM_HISTORY=131328,
        SIM_RESIDENT=16384,
        TOKEN_BYTES=2624,
        V9_AIC_COUNT=32,
        V9_SPLIT_Q=1,
        V9_INTERMEDIATE_FETCHES=8,
        V9_FINAL_AIVS=0,
        V9_SERVICE_AICS=0,
        V9_FINAL_REPLICATED=1,
        V9_DEDICATED_SERVICE=0,
        V9_UBPIPE_DEPTH=8,
        V9_UBPIPE_ROWS=4,
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=address,undefined",
            *[f"-D{k}={v}" for k, v in definitions.items()],
            "-I" + str(tmp_path),
            "-I" + str(operator / "include"),
            "-I" + str(root / "common/lightning_indexer"),
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_idle_maturity_dependencies_are_acyclic_with_every_service_owner_retained():
    # Worst case: every nonzero query waits for the full trigger. Short causal
    # bounds remove dependencies, so cannot create a new cycle. A producer
    # publishes progress before entering its own service window; only later
    # progress depends on that window completing.
    for batch in (1, 2, 3, 4, 8, 16):
        for split in (0, 1):
            cores = batch * (1 + split)
            for rounds in sorted({1, min(3, cores), min(8, cores), cores}):
                service = cores // rounds
                lanes = service * 2

                def rank(core):
                    return (core % 2) * batch + core // 2 if split else core

                def owner(value):
                    return (value % batch) * 2 + value // batch if split else value

                for active in (set(), {0}, {batch - 1}, set(range(0, batch, 2)), set(range(batch))):
                    graph = {}
                    for core in range(cores):
                        if core // (1 + split) not in active:
                            continue
                        own_epoch = rank(core) // service
                        for sub in range(2):
                            own_lane = rank(core) % service * 2 + sub
                            for epoch in range(rounds):
                                deps = set()
                                if epoch:
                                    deps.add(("compute", core * 2 + sub, epoch - 1))
                                if own_epoch < epoch:
                                    deps.add(("event", own_lane, own_epoch))
                                graph["compute", core * 2 + sub, epoch] = deps
                    for epoch in range(rounds):
                        for lane in range(lanes):
                            core = owner(epoch * service + lane // 2)
                            deps = {("event", lane, epoch - 1)} if epoch else set()
                            if core // (1 + split) in active:
                                deps.add(("compute", core * 2 + lane % 2, epoch))
                            else:
                                for row in range(lane, batch, lanes):
                                    if row in active:
                                        for producer in range(row * (1 + split), (row + 1) * (1 + split)):
                                            deps.update(("compute", producer * 2 + sub, epoch) for sub in range(2))
                            graph["event", lane, epoch] = deps
                    completed = set()
                    while len(completed) < len(graph):
                        ready = {node for node, deps in graph.items() if node not in completed and deps <= completed}
                        assert ready, (batch, split, rounds, active)
                        completed.update(ready)
