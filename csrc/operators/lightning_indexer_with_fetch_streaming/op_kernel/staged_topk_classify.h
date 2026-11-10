// SPDX-License-Identifier: Apache-2.0
// Final TopK UB-side classification. One independent bitmap per query;
// the staged resolver merges all four before selecting any victim.
#pragma once
namespace rotating {
constexpr uint32_t kClassifyThreads=512;
__simt_vf__ __aicore__ LAUNCH_BOUND(kClassifyThreads)
inline void ClassifyTopkHalf(uint32_t history,uint32_t resident,uint32_t batch,
    uint32_t used,uint32_t halfBase,uint32_t halfSize,uint32_t tailBegin,uint32_t tailEnd,
    bool hasValid,bool hasReserved,bool hasTail,__gm__ Context* ctx,
    __ubuf__ uint32_t* indices,__ubuf__ uint32_t* locations,
    __ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,
    __ubuf__ uint32_t* bitmap,__ubuf__ uint32_t* error) {
    auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(batch)*history;
    auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots;
    for(uint32_t i=threadIdx.x;i<kTop;i+=kClassifyThreads){
        uint32_t token=indices[i];
        if(token==UINT32_MAX){if(!halfBase)locations[i]=UINT32_MAX;continue;}
        if(token>=history){if(!halfBase){locations[i]=UINT32_MAX;asc_atomic_or(error,1U);}continue;}
        bool inHalf=token>=halfBase && token-halfBase<halfSize;
        bool cold=!halfBase && token>=resident;
        if(!inHalf && !cold)continue;
        bool inTail=hasTail && token>=tailBegin && token<tailEnd;
        uint32_t slot=UINT32_MAX;
        if(inHalf && !inTail && (!hasValid || valid[token-halfBase]) &&
           (!hasReserved || reserved[token-halfBase]<0))slot=token;
        else if(used || hasTail)slot=ResolvePhysicalSlot(resident,lru,asc_ldcg(mapping+token));
        locations[i]=slot;
        if(slot<resident)asc_atomic_or(bitmap+slot/32,1U<<(slot%32));
    }
    asc_syncthreads();
}
}
