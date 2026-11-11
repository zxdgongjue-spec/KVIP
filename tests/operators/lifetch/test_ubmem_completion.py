# SPDX-License-Identifier: Apache-2.0
"""Execute completion and the original Index planner across the merged boundary."""

import ast
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("bitmap_index", (0, 1))
def test_complete_plan_preserves_checks_publication_and_plan(tmp_path, bitmap_index):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for completion model")
    root = Path(__file__).resolve().parents[3]
    operator = root / "csrc/operators/lightning_indexer_with_fetch_ubmem"
    kernel = operator / "op_kernel"

    def function(path, name):
        source = path.read_text()
        begin = source.rfind("\n", 0, source.index(name + "(")) + 1
        brace = source.index("{", begin)
        end, depth = brace + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[begin:end]

    completion = kernel / "completion.h"
    update = kernel / "index_cache_update.cpp"
    functions = (
        "namespace rotating {\n"
        + "\n".join(function(completion, name) for name in ("PublishCompletedResidency", "CompleteBatch"))
        + "\n}\n"
    )
    functions += "\n".join(
        "template<uint32_t Threads>\n" + function(update, name) for name in ("PublishOutputRow", "PublishOwnedOutputs")
    )
    functions += "\ntemplate<bool Extended,bool WithOutputs=false>\n" + function(update, "Update")
    functions += "\n" + function(update, "PublishOutputs")
    functions += "\n" + function(update, "PlanOwned")
    functions += "\n" + function(update, "CompleteAndPlan")
    functions += "\n" + function(update, "kvip_ubmem_complete_plan_kernel")
    functions += "\n" + function(update, "kvip_ubmem_complete_plan_outputs_kernel")
    source = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __simt_callee__
#define __global__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
#define KERNEL_TASK_TYPE_DEFAULT(x)
#define ASCEND_IS_AIV (true)
using GM_ADDR=uint8_t*;
struct {uint32_t x;} threadIdx;
uint32_t block=0;
uint32_t GetBlockIdx(){return block;}
bool publicationReady=false;
bool allPlansComplete=false;
uintptr_t errorsBegin=0,errorsEnd=0;
uintptr_t unpublishedBegin=0,unpublishedEnd=0;
std::vector<std::pair<uintptr_t,uintptr_t>> watched;
template<class T>T asc_ldcg(T* p) {
    assert(uintptr_t(p)<unpublishedBegin || uintptr_t(p)>=unpublishedEnd);
    for(auto range:watched)if(uintptr_t(p)>=range.first && uintptr_t(p)<range.second)assert(publicationReady);
    if(uintptr_t(p)>=errorsBegin && uintptr_t(p)<errorsEnd)
        assert(allPlansComplete || (uintptr_t(p)-errorsBegin)/sizeof(uint32_t)==block);
    return *p;
}
template<class T>void asc_stcg(T* p,T value){*p=value;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t value){auto v=*p;if(v==old)*p=value;return v;}
uint32_t asc_atomic_or(uint32_t* p,uint32_t value){auto old=*p;*p|=value;return old;}
uint32_t asc_shfl_up(uint32_t value,uint32_t){return value;}
void asc_syncthreads(){}
void asc_threadfence(){}
namespace AscendC {
constexpr uint32_t EVENT_ID0=0;
enum class HardEvent{V_S,S_V};
std::vector<std::function<void()>> pending;
bool events[2]{};
void CompleteV(){for(auto& f:pending)f();pending.clear();}
template<HardEvent E>void SetFlag(uint32_t){assert(!events[unsigned(E)]);events[unsigned(E)]=true;}
template<HardEvent E>void WaitFlag(uint32_t){
    assert(events[unsigned(E)]);events[unsigned(E)]=false;
    if constexpr(E==HardEvent::V_S)CompleteV();
    else publicationReady=true;
}
template<class T>struct LocalTensor{T* data;void* GetPhyAddr(){return data;}};
enum class TPosition{VECCALC};
// Native TPipe retains its UB through pending VF work. The host model joins
// at destruction before the matching buffer goes out of scope.
struct TPipe{template<class T>void InitBuffer(T&,uint32_t){} ~TPipe(){CompleteV();}};
template<TPosition>struct TBuf{
    std::vector<uint32_t> data=std::vector<uint32_t>(40000);
    template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}
    ~TBuf(){CompleteV();}
};
namespace Simt {
struct Dim3{uint32_t x,y,z;};
template<auto F,class... A>void VF_CALL(Dim3 grid,A... args){
    pending.push_back([=]{for(threadIdx.x=0;threadIdx.x<grid.x;++threadIdx.x)F(args...);});
}
}
}
namespace ubmem_device {void CleanCacheLines(uint8_t*,uint64_t){publicationReady=false;}}
using namespace AscendC;
#include "device_context.h"
#include "index_cache_update.h"
// Planner's existing one-thread host mode runs its real path/cycle algorithm.
constexpr uint32_t kCandidates=kExtendedCandidates,kThreads=1;
FUNCTIONS
using namespace rotating;
struct Result {
    std::vector<uint8_t> rows;
    std::vector<uint32_t> info,hot,errors,plan;
    std::vector<int32_t> output;
    std::vector<uint64_t> status;
    bool operator==(const Result& o)const{return rows==o.rows && info==o.info && hot==o.hot && errors==o.errors && plan==o.plan && status==o.status && output==o.output;}
};
Result Run(uint32_t mode,uint32_t fault) {
    std::vector<Row> rows(SIM_BATCH);
    std::memset(rows.data(),0,rows.size()*sizeof(Row));
    std::vector<FinalPost> ownerStorage((kFinalOwnerBytes+sizeof(FinalPost)-1)/sizeof(FinalPost));
    std::memset(ownerStorage.data(),0,ownerStorage.size()*sizeof(FinalPost));
    std::vector<uint32_t> info(SIM_BATCH*kRowStatusWords),hot((SIM_BATCH+1)*SIM_RESIDENT),owners(SIM_BATCH);
    std::vector<uint32_t> lru(SIM_BATCH*kLruSlots),changes(SIM_BATCH*kChangeCapacity,UINT32_MAX);
    std::vector<uint32_t> changeSlots(SIM_BATCH*kChangeCapacity),counts(SIM_BATCH,kLruSlots),errors(SIM_BATCH,999);
    std::vector<uint32_t> plan(SIM_BATCH*kExtendedTransferWords,777),work(SIM_BATCH*kExtendedWorkWords);
    std::vector<int32_t> destinations(SIM_BATCH*kSlots,-1);
    std::vector<uint64_t> status(SIM_BATCH*kUbmemStatusWords);
    std::vector<int64_t> logical((SIM_BATCH+1)*SIM_HISTORY),unpack(SIM_BATCH*4),valid(SIM_BATCH*4,1);
    std::vector<int32_t> topk(SIM_BATCH*kSlots),locations(SIM_BATCH*kSlots),output(SIM_BATCH*kSlots*2,9191);
    std::iota(logical.begin(),logical.end(),100000);
    for(uint32_t q=0;q<SIM_BATCH*4;++q) {
        unpack[q]=(q*3+5)%(SIM_BATCH*4);
        for(uint32_t k=0;k<2048;++k){topk[q*2048+k]=k%SIM_HISTORY;locations[q*2048+k]=(q+k)%SIM_RESIDENT;}
    }
    if(fault%3==0)valid.back()=0;
    if(fault%3==1)unpack[0]=-1;
    if(fault%3==2)unpack.back()=SIM_BATCH*4;
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x100000,0x200000,0x100000,7,0x800000,
        uint64_t(SIM_BATCH+1)*(SIM_RESIDENT+384)*kTokenBytes,0,kUbmemHostLocal};
    Context ctx{};
    ctx.rows=uint64_t(rows.data());ctx.rowStatus=uint64_t(info.data());ctx.owners=uint64_t(ownerStorage.data());
    ctx.ubmemStatus=uint64_t(status.data());ctx.ubmemRegion=uint64_t(&region);
    ctx.rowOwners=uint64_t(owners.data());ctx.hotValid=uint64_t(hot.data());ctx.lruSlots=uint64_t(lru.data());
    ctx.changes=uint64_t(changes.data());ctx.destinations=uint64_t(destinations.data());ctx.outputStride=SIM_RESIDENT+384;
    for(uint32_t b=0;b<SIM_BATCH;++b) {
        owners[b]=SIM_BATCH-b;rows[b].requests=1;rows[b].finalPending=1;
        status[b*kUbmemStatusWords+2]=1;info[b*kRowStatusWords+1]=2;
        for(uint32_t i=0;i<kLruSlots;++i)lru[b*kLruSlots+i]=changeSlots[b*kChangeCapacity+i]=i;
        changes[b*kChangeCapacity]=SIM_RESIDENT+7;
        std::fill(changes.begin()+b*kChangeCapacity+kLruSlots,changes.begin()+(b+1)*kChangeCapacity,37U);
        for(uint32_t w=0;w<kFinalWorkers;++w) {
            auto post=FinalPostAt(&ctx,b*kFinalWorkers+w);post->count=post->head=w<2?1:0;
        }
        destinations[b*kSlots]=owners[b]*ctx.outputStride+4000;
        destinations[b*kSlots+1]=owners[b]*ctx.outputStride+4001;
    }
    auto post=FinalPostAt(&ctx,0);
    switch(fault) {
        case 1:rows[0].error=32;break;
        case 2:info[0]=8;break;
        case 3:post->pending=1;break;
        case 4:post->deferred=1;break;
        case 5:post->error=16;break;
        case 6:++post->count;break;
        case 7:++info[1];break;
        case 8:rows[0].pending=1;break;
        case 9:++status[2];break;
        case 10:region.generation=0;break;
        case 11:region.magic=0;break;
        case 12: // Planner failure after successful completion remains visible.
            changes[1]=changes[0];break;
        case 13: // Exercise the extended planner with a noncandidate victim.
            counts[0]=kLruSlots+1;changes[kLruSlots]=SIM_RESIDENT+8;changeSlots[kLruSlots]=3000;break;
        case 14: // No final reads, but one valid prefetch completion.
            for(uint32_t w=0;w<kFinalWorkers;++w)FinalPostAt(&ctx,w)->count=FinalPostAt(&ctx,w)->head=0;
            info[1]=0;rows[0].finalPending=0;break;
    }
    // Failed partial final planning may expose a stale high-water mark. The
    // completion error must gate extended planning before any suffix read.
    if(fault>=1 && fault<=11) {
        counts[0]=kChangeCapacity;
        unpublishedBegin=uint64_t(changes.data()+kLruSlots);
        unpublishedEnd=uint64_t(changes.data()+kChangeCapacity);
    }
    uint64_t bindings[15]{};bindings[14]=uint64_t(&ctx);
    uint64_t a[28]{};a[4]=uint64_t(changeSlots.data());a[5]=uint64_t(changes.data());a[6]=uint64_t(errors.data());
    a[8]=ctx.rows;a[9]=ctx.rowOwners;a[10]=ctx.rowStatus;a[11]=uint64_t(FinalPostAt(&ctx,0));
    a[12]=ctx.ubmemStatus;a[13]=ctx.ubmemRegion;a[14]=kFinalWorkers;a[17]=uint64_t(plan.data());
    a[24]=SIM_HISTORY;a[25]=SIM_RESIDENT;a[26]=uint64_t(counts.data());a[27]=uint64_t(work.data());
    a[3]=uint64_t(logical.data());a[18]=uint64_t(topk.data());a[19]=uint64_t(locations.data());
    a[20]=uint64_t(unpack.data());a[21]=uint64_t(valid.data());
    a[22]=uint64_t(output.data());a[23]=uint64_t(output.data()+SIM_BATCH*kSlots);
    errorsBegin=uint64_t(errors.data());errorsEnd=errorsBegin+errors.size()*4;allPlansComplete=false;
    watched={{ctx.rows,ctx.rows+rows.size()*sizeof(Row)},{ctx.rowStatus,ctx.rowStatus+info.size()*4},
        {ctx.ubmemStatus,ctx.ubmemStatus+status.size()*8},{a[11],a[11]+SIM_BATCH*kFinalWorkers*sizeof(FinalPost)}};
    publicationReady=false;
    for(uint32_t i=0;i<SIM_BATCH;++i) {
        block=SIM_BATCH-1-i; // Cross-batch TND rows cannot rely on another block's status.
        if(mode==2)kvip_ubmem_complete_plan_outputs_kernel(reinterpret_cast<GM_ADDR>(bindings),reinterpret_cast<GM_ADDR>(a));
        else if(mode==1)kvip_ubmem_complete_plan_kernel(reinterpret_cast<GM_ADDR>(bindings),reinterpret_cast<GM_ADDR>(a));
        else {CompleteBatch(&ctx,block);publicationReady=true;PlanOwned(reinterpret_cast<GM_ADDR>(a));}
        assert(pending.empty() && !events[0] && !events[1]);
    }
    allPlansComplete=true;
    if(mode!=2)for(uint32_t core=0;core<kOutputCores;++core){threadIdx.x=0;PublishOutputs(a,core);}
    for(uint32_t q=0;q<SIM_BATCH*4;++q)for(uint32_t k=0;k<2048;++k) {
        bool active=valid[q] && unpack[q]>=0 && unpack[q]<SIM_BATCH*4;
        if(active)active=!errors[unpack[q]/4];
        int32_t token=active?topk[unpack[q]*2048+k]:-1;
        int32_t expected=active?int32_t(logical[owners[unpack[q]/4]*SIM_HISTORY+token]):-1;
        assert(output[q*2048+k]==expected);
        assert(output[SIM_BATCH*kSlots+q*2048+k]==(active?locations[unpack[q]*2048+k]:-1));
    }
    assert(errors[0]==0 || plan[0]==0);
    const bool completedSuccessfully=fault==0 || fault>=12;
    assert((status[0]==0)==completedSuccessfully);
    assert(hot[owners[0]*SIM_RESIDENT]==uint32_t(completedSuccessfully));
    assert(hot[owners[0]*SIM_RESIDENT+4000]==uint32_t(completedSuccessfully && fault!=14));
    assert((errors[0]==0)==(completedSuccessfully && fault!=12));
    if(!errors[0])assert(plan[0]>0);
    if(fault!=10 && fault!=11)assert(errors[1]==0); // Failed row never poisons another owner.
    watched.clear();unpublishedBegin=unpublishedEnd=0;
    Result result;result.rows.resize(rows.size()*sizeof(Row));
    std::memcpy(result.rows.data(),rows.data(),result.rows.size());
    result.info=info;result.hot=hot;result.errors=errors;result.plan=plan;result.status=status;result.output=output;return result;
}
int main(){for(uint32_t fault=0;fault<15;++fault){auto baseline=Run(0,fault);assert(baseline==Run(1,fault));assert(baseline==Run(2,fault));}}
""".replace("FUNCTIONS", functions)
    cpp, binary = tmp_path / "complete.cpp", tmp_path / "complete"
    cpp.write_text(source)
    runner = ast.parse((operator / "tests/run_host_models.py").read_text())
    common = next(
        ast.literal_eval(node.value)
        for node in runner.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "common" for t in node.targets)
    )
    definitions = common | {"SIM_BATCH": 2, "TOKEN_BYTES": 2624, "V9_BITMAP_INDEX": bitmap_index}
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            *[f"-D{k}={v}" for k, v in definitions.items()],
            f"-I{operator / 'include'}",
            f"-I{kernel}",
            f"-I{root / 'csrc/operators/common/lifetch'}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
