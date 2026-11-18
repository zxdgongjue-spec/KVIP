// SPDX-License-Identifier: Apache-2.0
// Final TopK UB-side classification. One independent bitmap per query;
// the staged resolver merges all four before selecting any victim.
#pragma once
namespace rotating {
constexpr uint32_t kClassifyThreads=512;
constexpr uint32_t kBitmapWords=512,kBitmapShards=8;
__simt_vf__ __aicore__ LAUNCH_BOUND(kClassifyThreads)
inline void ClassifyTopkHalf(uint32_t history,uint32_t resident,uint32_t batch,
    uint32_t used,uint32_t halfBase,uint32_t halfSize,uint32_t tailBegin,uint32_t tailEnd,
    bool hasValid,bool hasReserved,bool hasTail,__gm__ Context* ctx,
    __ubuf__ uint32_t* indices,__ubuf__ uint32_t* locations,
    __ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,
    __ubuf__ uint32_t* bitmap,__ubuf__ uint32_t* error) {
    const uint32_t words=(resident+31)/32,shards=4096/words<8?4096/words:8;
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
#if !V9_UBMEM
        else if(inTail)slot=DirectTailSlot(token,resident,tailBegin,tailEnd,OutputStride(resident,ctx)-resident);
#endif
        else if(used || hasTail)slot=ResolvePhysicalSlot(resident,lru,asc_ldcg(mapping+token));
        locations[i]=slot;
        if(slot<resident)asc_atomic_or(bitmap+(threadIdx.x%shards)*words+slot/32,1U<<(slot%32));
    }
    asc_syncthreads();
}
// The first shard becomes the exact per-query protection bitmap. Sharding
// limits contention between lanes without changing which slots are protected.
__simt_vf__ __aicore__ LAUNCH_BOUND(kClassifyThreads)
inline void MergeClassifyBitmap(__ubuf__ uint32_t* bitmap,uint32_t resident) {
    const uint32_t words=(resident+31)/32,shards=4096/words<8?4096/words:8;
    for(uint32_t word=threadIdx.x;word<words;word+=kClassifyThreads){
        uint32_t value=bitmap[word];
        for(uint32_t shard=1;shard<shards;++shard)value|=bitmap[shard*words+word];
        bitmap[word]=value;
    }
    asc_syncthreads();
}
}
#if !V9_UBMEM

namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ClassifyPreparedLocations(uint32_t history,uint32_t resident,uint32_t batch,__gm__ Context* ctx,
    uint32_t tailBegin,uint32_t tailSize,uint32_t ringStart,uint32_t capacity,
    __ubuf__ uint32_t* indices,__ubuf__ uint32_t* locations,__ubuf__ uint32_t* bitmap,__ubuf__ uint32_t* error) {
    const uint32_t words=(resident+31)/32,shards=4096/words<8?4096/words:8;
    auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(batch)*history;
    auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots;
    for(uint32_t i=threadIdx.x;i<kTop;i+=512){
        uint32_t token=indices[i],slot=locations[i];
        if(token==UINT32_MAX){locations[i]=UINT32_MAX;continue;}
        if(token>=history){locations[i]=UINT32_MAX;asc_atomic_or(error,1U);continue;}
        const uint32_t offset=token-tailBegin;
        if(slot==UINT32_MAX && offset<tailSize){
            uint64_t index=uint64_t(ringStart)+offset;
            slot=resident+uint32_t(index>=capacity?index-capacity:index);
        }
        if(slot==UINT32_MAX)slot=ResolvePhysicalSlot(resident,lru,asc_ldcg(mapping+token));
        locations[i]=slot;
        if(slot<resident)asc_atomic_or(bitmap+(threadIdx.x%shards)*words+slot/32,1U<<(slot%32));
    }
    asc_syncthreads();
}
}
#endif
