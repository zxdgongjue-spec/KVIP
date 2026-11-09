#pragma once
#include "prefetch.h"
#include "final_scratch.h"
#include "final_worker_submit.h"
#include "../../../common/lifetch/resolver_candidates.h"
#include "../../../common/lifetch/resident_selection.h"
namespace rotating {
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ReplicatedResolve(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,__gm__ uint32_t* mapping,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch,__gm__ Context* ctx,
    uint32_t history,uint32_t resident,uint32_t finalWorkers,uint32_t tokenBytes) {
#define KV_RESOLVE_HISTORY history
#define KV_RESOLVE_RESIDENT resident
#define KV_RESOLVE_WORKERS finalWorkers
#include "../../../common/lifetch/resolver_plan.h"
#undef KV_RESOLVE_HISTORY
#undef KV_RESOLVE_RESIDENT
#undef KV_RESOLVE_WORKERS

    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    // All final AIVs retain an identical immutable copy plan. Payload writes
    // remain partitioned by destination sector, including scattered hot victims.
    for(uint32_t rank=tid;rank<count;rank+=Threads) {
        uint32_t token=selected[ids[rank]],physical=slots[rank],dst=batch*kSlots+rank;
        uint64_t remote=asc_ldcg(addresses+token);
        keys[rank]=uint32_t(remote);ranks[rank]=uint32_t(remote>>32);
        if(rank>=ownBegin && rank<ownEnd) {
            int32_t ordinal=asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*resident+physical);
            uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
            asc_stcg(changeSlots+record,physical);asc_stcg(changes+record,token);
            StoreFallbackSource(history,tokenBytes,ctx,batch,dst,token,remote,blocks,offsets);
            asc_stcg(destinations+dst,int32_t(ownerBase+physical));
        }
    }
    if(!tid && ownEnd>candidateCount)asc_atomic_max(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+ownEnd);
    // Mapping remains the immutable prefetch snapshot. Final locations and
    // descriptors are authoritative; mutating mapping here races other readers.
    asc_threadfence();
}

__aicore__ inline void ReplicatedFinal(__gm__ Context* ctx) {
    const auto shape=LoadRuntimeShape(ctx);
    const uint32_t rt_history=shape.history,rt_resident=shape.resident,rt_finalWorkers=shape.finalWorkers;
#include "../../../common/lifetch/resolver_front.h"
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*shape.history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx,shape.history,shape.resident,shape.finalWorkers,shape.tokenBytes);
#include "../../../common/lifetch/resolver_output.h"
    PublishFinalWorker(ctx,row,b,w,own,batchCount,batchBase+prefix,error,local,keys,ranks,slots,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalCompleted=FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
