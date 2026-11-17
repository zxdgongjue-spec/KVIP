// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace rotating {
constexpr uint32_t kCompactThreads=512;
static_assert(kCompactThreads%(4*warpSize)==0,"Q4 prefix boundaries must align to whole warps");

// A no-prefetch round has no speculative writes to reconcile. Resolve the
// complete Q4 selection once, preserving first-occurrence order and the same
// protected LRU/fallback slot order as four StagedQuery calls. The existing
// event ranges and per-Q publications remain the transport/Index contract.
template<bool PostIndex=false>
__simt_vf__ __aicore__ LAUNCH_BOUND(kCompactThreads)
inline void ResolveCompactQ4(
    StagedContextPointer ctx,uint32_t history,uint32_t resident,uint32_t batch,
    __ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* scratch,__ubuf__ uint32_t* ctl,
    __gm__ uint32_t* eventSource,__gm__ uint32_t* eventDest) {
    static_assert(!PostIndex || kSlots<=8192,"post-Index ranks use 13 bits");
    const uint32_t tid=threadIdx.x,buckets=ctl[7],per=kSlots/kCompactThreads;
    auto bitmap=ctl+1024,lru=scratch+4096;
    for(uint32_t i=tid;i<kSlots;i+=kCompactThreads){
        uint32_t token=selected[i];
        if(token==UINT32_MAX || loc[i]!=UINT32_MAX)continue;
        if(token>=history){asc_atomic_or(ctl+1,1U);continue;}
        uint32_t h=SparseHash(token)&(buckets-1),probe=0;
        for(;probe<buckets;++probe){
            uint32_t previous=asc_atomic_cas(keys+h,UINT32_MAX,token);
            if(previous==UINT32_MAX || previous==token){asc_atomic_min(ranks+h,i);break;}
            h=(h+1)&(buckets-1);
        }
        if(probe==buckets)asc_atomic_or(ctl+1,8U);
    }
    asc_syncthreads();
    if(ctl[1])return;
    uint32_t local=0,coldLocal=0;
    for(uint32_t i=tid*per;i<(tid+1)*per;++i){
        if(selected[i]==UINT32_MAX || loc[i]!=UINT32_MAX)continue;
        uint32_t h=SparseHash(selected[i])&(buckets-1),probe=0;
        while(keys[h]!=selected[i] && probe<buckets){h=(h+1)&(buckets-1);++probe;}
        if(probe==buckets){asc_atomic_or(ctl+1,8U);continue;}
        if(ranks[h]==i){loc[i]=UINT32_MAX-1;++local;coldLocal+=selected[i]>=resident;}
    }
    // A missing resident row refills its own KV slot. Its Index already names
    // the right token, so only cold sources consume a new hot destination.
    uint32_t offset=StagedPrefix(PostIndex?local|(coldLocal<<16):local,ctl);
    uint32_t total=ctl[544+kCompactThreads/warpSize-1];
    uint32_t needed=PostIndex?total&0xffffU:total;
    uint32_t neededSlots=PostIndex?total>>16:needed;
    uint32_t coldOffset=offset>>16;
    if constexpr(PostIndex)offset&=0xffffU;
    if(ctl[1])return;
    if(tid<4){
        uint32_t end=ctl[544+(tid+1)*(kCompactThreads/4/warpSize)-1];
        uint32_t begin=tid?ctl[544+tid*(kCompactThreads/4/warpSize)-1]:0;
        ctl[2+tid]=PostIndex?(end-begin)&0xffffU:end-begin;
    }
    if(!tid)ctl[0]=needed;
    for(uint32_t i=tid*per;i<(tid+1)*per;++i){
        if(loc[i]!=UINT32_MAX-1)continue;
        uint32_t h=SparseHash(selected[i])&(buckets-1);
        while(keys[h]!=selected[i])h=(h+1)&(buckets-1);
        uint32_t rank=offset++;
        if constexpr(PostIndex){
            // Two 13-bit ranks fit alongside the existing populated marker.
            ranks[h]=kStagedRankTag|(rank<<13)|coldOffset;
            coldOffset+=selected[i]>=resident;
        }else ranks[h]=kStagedRankTag|rank;
        asc_stcg(eventSource+2048+rank,selected[i]);
    }
    asc_syncthreads();
    if(neededSlots){
        uint32_t available=0;
        constexpr uint32_t candidatesPerThread=kLruSlots/kCompactThreads;
        for(uint32_t i=tid*candidatesPerThread;i<(tid+1)*candidatesPerThread;++i){
            uint32_t slot=lru[i];
            if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
            available+=!(bitmap[slot/32]&(1U<<(slot%32)));
        }
        uint32_t rank=StagedPrefix(available,ctl),allocated=ctl[544+kCompactThreads/warpSize-1];
        if(allocated>neededSlots)allocated=neededSlots;
        if(ctl[1])return;
        for(uint32_t i=tid*candidatesPerThread;i<(tid+1)*candidatesPerThread;++i){
            uint32_t slot=lru[i];
            if(!(bitmap[slot/32]&(1U<<(slot%32)))){
                if(rank<allocated)slots[rank]=slot;
                ++rank;
            }
        }
        asc_syncthreads();
        for(uint32_t i=tid;i<allocated;i+=kCompactThreads)
            asc_atomic_or(bitmap+slots[i]/32,1U<<(slots[i]%32));
        asc_syncthreads();
        if(allocated<neededSlots){
            // Protect every known Q before fallback, including late-query hot
            // hits. Dead scratch may overwrite LRU: it is never scanned again.
            for(uint32_t i=tid;i<kSlots;i+=kCompactThreads)
                if(loc[i]<resident)asc_atomic_or(bitmap+loc[i]/32,1U<<(loc[i]%32));
            asc_syncthreads();
            uint32_t words=(resident+31)/32;
            auto counts=reinterpret_cast<__ubuf__ int32_t*>(scratch),offsets=counts+words;
            kvip_hot_slots::CountAvailableBitmapWordsBody(bitmap,counts,resident,0,words,0);
            asc_syncthreads();
            uint32_t wordsPerThread=(words+kCompactThreads-1)/kCompactThreads,localWords=0;
            for(uint32_t i=tid*wordsPerThread;i<(tid+1)*wordsPerThread && i<words;++i)localWords+=counts[i];
            uint32_t prefix=StagedPrefix(localWords,ctl);
            if(!tid && ctl[544+kCompactThreads/warpSize-1]<neededSlots-allocated)ctl[1]=4;
            for(uint32_t i=tid*wordsPerThread;i<(tid+1)*wordsPerThread && i<words;++i){offsets[i]=prefix;prefix+=counts[i];}
            asc_syncthreads();
            if(ctl[1])return;
            kvip_hot_slots::WriteAvailableBitmapSlotsBody(bitmap,counts,offsets,
                reinterpret_cast<__ubuf__ int32_t*>(slots),resident,0,words,0,neededSlots-allocated,allocated);
            asc_syncthreads();
        }
    }
    for(uint32_t i=tid;i<kSlots;i+=kCompactThreads){
        if(selected[i]==UINT32_MAX || loc[i]<UINT32_MAX-1)continue;
        uint32_t h=SparseHash(selected[i])&(buckets-1);
        while(keys[h]!=selected[i])h=(h+1)&(buckets-1);
        if constexpr(PostIndex)
            loc[i]=selected[i]<resident?selected[i]:slots[ranks[h]&8191U];
        else loc[i]=slots[ranks[h]&~kStagedRankTag];
    }
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*resident;
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
    for(uint32_t h=tid;h<buckets;h+=kCompactThreads){
        if(keys[h]==UINT32_MAX)continue;
        uint32_t rank=ranks[h]&~kStagedRankTag,slot;
        if constexpr(PostIndex){
            slot=keys[h]<resident?keys[h]:slots[rank&8191U];
            rank>>=13;
        }else slot=slots[rank];
        int32_t ordinal=asc_ldcg(reserved+slot);
        uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
        asc_stcg(eventDest+2048+rank,slot);
        asc_stcg(changeSlots+record,slot);asc_stcg(changes+record,keys[h]);
    }
    if(!tid){
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+needed);
        // All selected resident rows are protected before allocation. Thus
        // each non-self event joins a unique cold source to a unique hot slot.
        if constexpr(PostIndex)ctl[9]=0;
    }
    asc_threadfence();asc_syncthreads();
}
}  // namespace rotating
