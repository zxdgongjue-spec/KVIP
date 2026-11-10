// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_context.h"
#include "copy/device_utils.h"

// Call only after the resolve kernel has completed every final worker.
// The merged Index kernel retains these checks and joins residency publication
// before its same-row plan reads the resulting status and hot validity.
namespace rotating {
using namespace AscendC;
// Publish residency only after all UBmem copy pipelines report completion. Resolver
// workers must all observe the same pre-round validity, not one another's posts.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishCompletedResidency(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t fetched) {
    if(!ctx->hotValid)return;
    uint32_t owner=ctx->rowOwners ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)+b) : b;
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*rt_resident;
    // The original resolver already emits exactly the final Fetch destinations.
    // Hits need no publication; do not rescan all four TopK rows.
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+b*kSlots;
    uint32_t ownerBase=owner*(ctx->outputStride ? uint32_t(ctx->outputStride) : rt_resident);
    for(uint32_t i=threadIdx.x;i<fetched;i+=256) {
        uint32_t slot=uint32_t(asc_ldcg(destinations+i))-ownerBase;
        if(slot<rt_resident)asc_stcg(valid+slot,uint32_t(1));
    }
    if(ctx->changes && ctx->lruSlots) {
        auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
        auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
        for(uint32_t i=threadIdx.x;i<kLruSlots;i+=256)
            if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),uint32_t(1));
    }
    asc_threadfence();
}
__aicore__ inline void CompleteBatch(__gm__ Context* ctx,uint32_t b) {
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*4);
    auto region=UbmemRegionAt(ctx);
    uint64_t error=info[0]|row->error;
    if(!ctx->ubmemStatus || !UbmemRegionValid(region))error|=kUbmemInvalidRegion;
    uint64_t scheduled=0,completed=0;
    for(uint32_t w=0;w<kFinalWorkers;++w) {
        auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(FinalPost));
        scheduled+=post->count;completed+=post->head;error|=post->error;
        if(post->pending || post->deferred || post->count!=post->head)error|=kUbmemIncomplete;
    }
    if(row->pending || scheduled!=info[1] || scheduled!=completed)error|=kUbmemIncomplete;
    if(ctx->ubmemStatus) {
        auto status=UbmemStatusAt(ctx,b);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
        if(status[2]!=row->requests)error|=kUbmemIncomplete;
        status[0]=error;status[1]=row->requests;status[3]=scheduled;status[4]=completed;
        status[5]=(status[2]+completed)*kTokenBytes;
        status[6]=UbmemRegionValid(region)?region->generation:0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
    }
    row->error=static_cast<uint32_t>(error);
    info[0]|=static_cast<uint32_t>(error);info[3]=0;info[4]=static_cast<uint32_t>(completed);
    if(!error) {
        row->finalPending=0;
        Simt::VF_CALL<PublishCompletedResidency>(Simt::Dim3{256,1,1},SIM_RESIDENT,ctx,b,static_cast<uint32_t>(completed));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),32);
}
}
