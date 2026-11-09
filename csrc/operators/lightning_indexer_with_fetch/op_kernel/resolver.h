#pragma once
#include "prefetch.h"
#include "../../lidu/op_kernel/hot_slot_selection.h"

// Imported from final_resolve.h

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

#include "../../common/lifetch/resolver_candidates.h"

#include "../../common/lifetch/resident_selection.h"

// Imported from final_worker_submit.h

namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+rt_finalPostOffset)+core;
}
// Resolver publishes compact ranges only. KVAF owns all final transport work.
__aicore__ inline void PrepareFinalFetch(uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,
    __gm__ Context* ctx,uint32_t b,uint32_t w,uint32_t count,uint32_t descriptor,uint32_t error,uint32_t nonconflicting) {
    auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
    post->count=count;post->pending=0;post->error=error;
    post->deferred=count && !error;post->descriptor=descriptor;post->head=0;
    post->nonconflicting=nonconflicting;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}

// Imported from final_scratch.h

#include "../../common/lifetch/final_scratch.h"

namespace rotating {
// Every worker independently builds the same rank/slot plan from immutable
// prefetch mapping. Rank is the first TopK occurrence, NOT atomic arrival order.
// Only disjoint location/descriptor/WQE outputs are partitioned across workers.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ReplicatedResolve(uint32_t rt_history,uint32_t rt_resident,uint32_t rt_finalWorkers,uint32_t rt_tokenBytes,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,__gm__ uint32_t* mapping,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch,__gm__ Context* ctx) {
#define KV_RESOLVE_HISTORY rt_history
#define KV_RESOLVE_RESIDENT rt_resident
#define KV_RESOLVE_WORKERS rt_finalWorkers
#include "../../common/lifetch/resolver_plan.h"
#undef KV_RESOLVE_HISTORY
#undef KV_RESOLVE_RESIDENT
#undef KV_RESOLVE_WORKERS

    // Source addresses are snapshotted before Index update. Prefetch records
    // overwritten by final repair are replaced, not duplicated in the plan.
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    // Stable partition within each worker's existing compact range. Reuse the
    // dead hash scratch and the same warp-prefix scan used above; no new GM pass.
    uint32_t ownCount=ownEnd-ownBegin,perOwn=(ownCount+Threads-1)/Threads;
    uint32_t localSafe=0;
    for(uint32_t i=tid*perOwn;i<(tid+1)*perOwn && i<ownCount;++i) {
        uint32_t slot=slots[ownBegin+i];
        int32_t ordinal=asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*rt_resident+slot);
        keys[i]=uint32_t(ordinal);
        localSafe+=!(ordinal>=0 && uint32_t(ordinal)<used);
    }
    uint32_t safeScan=localSafe;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(safeScan,d);if(lane>=d)safeScan+=prior;}
    if(lane==warpSize-1)cfg[544+warp]=safeScan;
    asc_syncthreads();
    if(!warp){uint32_t v=lane<Threads/warpSize?cfg[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}cfg[544+lane]=v;}
    asc_syncthreads();
    uint32_t safeCount=cfg[544+Threads/warpSize-1];
    uint32_t safeBefore=safeScan-localSafe+(warp?cfg[544+warp-1]:0);
    uint32_t conflictBefore=tid*perOwn-safeBefore;
    if(!tid)cfg[5]=safeCount;
    for(uint32_t i=tid*perOwn;i<(tid+1)*perOwn && i<ownCount;++i) {
        uint32_t rank=ownBegin+i,token=selected[ids[rank]],slot=slots[rank];
        int32_t ordinal=int32_t(keys[i]);
        bool conflict=ordinal>=0 && uint32_t(ordinal)<used;
        uint32_t index=conflict?safeCount+conflictBefore++:safeBefore++;
        uint32_t dst=batch*kSlots+ownBegin+index;
        uint64_t remote=asc_ldcg(addresses+token);
        uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
        asc_stcg(changeSlots+record,slot);asc_stcg(changes+record,token);
        StoreFallbackSource(rt_history,rt_tokenBytes,ctx,batch,dst,token,remote,blocks,offsets);
        asc_stcg(destinations+dst,int32_t(ownerBase+slot));
    }
    // Fallback ranks follow the candidate prefix. Publish one high-water mark
    // per worker, not an atomic on the same GM word for every repaired token.
    if(!tid && ownEnd>candidateCount)
        asc_atomic_max(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+ownEnd);
    // Mapping remains the immutable prefetch snapshot. Final locations and
    // descriptors are authoritative; mutating mapping here races other readers.
    asc_threadfence();
}
__aicore__ inline void ReplicatedFinal(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,uint32_t rt_tokenBytes,__gm__ Context* ctx) {
#include "../../common/lifetch/resolver_front.h"
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},rt_history,rt_resident,rt_finalWorkers,rt_tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
#include "../../common/lifetch/resolver_output.h"
    PrepareFinalFetch(rt_finalWorkers,rt_finalPostOffset,ctx,b,w,own,batchBase+prefix,error,error?0:ctl.GetValue(5));
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalHead=FinalPostAt(rt_finalPostOffset,ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
