#pragma once
// SPDX-License-Identifier: Apache-2.0
// Transport-independent final TopK protection and persistent hot-slot selection.
// Both backends provide the same virtual-slot translation from prefetch/ABC.
#include "../../lidu/op_kernel/hot_slot_selection.h"
#include "mapping.h"
namespace rotating {
template<uint32_t Threads,bool CacheBuckets=false>
__simt_callee__ inline bool ResolveResidentPlan(
    uint32_t rt_history, uint32_t rt_resident, uint32_t outputStride,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions, uint32_t coldCount, __ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks, __ubuf__ uint16_t* ids, __ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg, __gm__ uint32_t* mapping, __gm__ uint32_t* lru,
    uint32_t used, bool hasTail, uint32_t& count, uint32_t& candidateCount,
    uint32_t mappingTag=0,uint32_t mappingMask=UINT32_MAX) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    // slots has kSlots entries and is otherwise dead until hot-slot selection.
    // Cache one bucket per cold-list position (not per token/rank): each writer
    // owns a distinct i, even when different queries select the same token.
    for(uint32_t i=tid;i<buckets;i+=Threads){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    if(tid<8)cfg[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<coldCount;i+=Threads){
        uint32_t pos=positions[i],token=selected[pos];
        if(token>=rt_history){asc_atomic_or(cfg+1,1U);continue;}
        if(pos>=2*kTop && selected[pos-2*kTop]==token){loc[pos]=UINT32_MAX-2;continue;}
        uint32_t slot=(used || hasTail)?ReadMappingSlot(mapping+token,mappingTag,mappingMask):UINT32_MAX;loc[pos]=slot;
        if(slot==UINT32_MAX){
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t n=0;n<buckets;++n){uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(old==UINT32_MAX || old==token){asc_atomic_min(ranks+h,pos);
                    if constexpr(CacheBuckets)slots[i]=h;
                    break;}h=(h+1)&(buckets-1);}
        } else if(slot<rt_resident || slot>=kSlots+outputStride || (slot>=rt_resident+kLruSlots && slot<rt_resident+kSlots))asc_atomic_or(cfg+1,2U);
    }
    asc_syncthreads();
    // Mark the first occurrence before replacing hash-table positions by ranks.
    uint32_t per=(coldCount+Threads-1)/Threads,localCount=0;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX)continue;
        uint32_t h;
        if constexpr(CacheBuckets)h=slots[i];
        else {uint32_t token=selected[pos];h=SparseHash(token)&(buckets-1);
            while(keys[h]!=token)h=(h+1)&(buckets-1);}
        if(ranks[h]==pos){loc[pos]=UINT32_MAX-1;++localCount;}}
    uint32_t scan=localCount;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
    if(lane==warpSize-1)cfg[544+warp]=scan;
    asc_syncthreads();
    if(!warp){uint32_t value=lane<Threads/warpSize?cfg[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}cfg[544+lane]=value;}
    asc_syncthreads();
    uint32_t begin=scan-localCount+(warp?cfg[544+warp-1]:0);
    count=cfg[544+Threads/warpSize-1];
    if(!tid)cfg[0]=count;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX-1)continue;
        uint32_t h;
        if constexpr(CacheBuckets)h=slots[i];
        else {uint32_t token=selected[pos];h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);}
        ids[begin]=uint16_t(pos);ranks[h]=begin++;}
    asc_syncthreads();
    // Resolve rank markers before releasing the hash tables. All workers see
    // the same final Q4 set, so each independently chooses identical hot slots.
    constexpr uint32_t missingTag=0x80000000U;
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],slot=loc[pos],token=selected[pos];
        // Earlier query locations may already be physical (including ABC).
        // Read the immutable mapping instead of translating them a second time.
        const bool skipped=slot==UINT32_MAX-2;
        if(skipped)slot=(used || hasTail)?ReadMappingSlot(mapping+token,mappingTag,mappingMask):UINT32_MAX;
        if(slot>=UINT32_MAX-2) {
            uint32_t h;
            if constexpr(CacheBuckets) {
                // The Q2-repeat fast path skipped insertion, so it alone
                // still needs a probe. Every inserted occurrence owns slots[i].
                if(!skipped)h=slots[i];
                else {h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);}
            } else {h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);}
            slot=missingTag|ranks[h];
        } else slot=ResolvePhysicalSlot(rt_resident,lru,slot);
        loc[pos]=slot;
    }
    // All cached bucket reads must finish before hot-slot selection overwrites
    // slots. This existing barrier is also the handoff for the uncached path.
    asc_syncthreads();
    if(!count)return false;
    // Hash scratch is now dead. Reuse it as LIDU's hot protection bitmap and
    // count/prefix scratch; LI tiling and the compacted TopK worklist stay intact.
    uint32_t words=(rt_resident+31)/32;
    if(words>kSlots/2) {if(!tid)cfg[1]=4;return false;}
    for(uint32_t i=tid;i<words;i+=Threads)keys[i]=0;
    asc_syncthreads();
    // Every LRU slot was classified as cold by reservedHot. Only resolved
    // cold entries can protect a candidate; untouched hot hits are outside LRU.
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t slot=loc[positions[i]];
        if(slot<rt_resident)asc_atomic_or(keys+slot/32,1U<<(slot%32));
    }
    asc_syncthreads();
    // First keep candidate order, then use LIDU's bitmap fallback from hot 0.
    uint32_t perCandidate=(kLruSlots+Threads-1)/Threads,available=0;
    for(uint32_t i=tid*perCandidate;i<(tid+1)*perCandidate && i<kLruSlots;++i) {
        uint32_t slot=asc_ldcg(lru+i);
        if(slot>=rt_resident){asc_atomic_or(cfg+1,4U);continue;}
        available+=!(keys[slot/32]&(1U<<(slot%32)));
    }
    uint32_t prefix=available;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(prefix,d);if(lane>=d)prefix+=prior;}
    if(lane==warpSize-1)cfg[544+warp]=prefix;
    asc_syncthreads();
    if(!warp){uint32_t v=lane<Threads/warpSize?cfg[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}cfg[544+lane]=v;}
    asc_syncthreads();
    candidateCount=cfg[544+Threads/warpSize-1];
    if(candidateCount>count)candidateCount=count;
    uint32_t rank=prefix-available+(warp?cfg[544+warp-1]:0);
    for(uint32_t i=tid*perCandidate;i<(tid+1)*perCandidate && i<kLruSlots;++i) {
        uint32_t slot=asc_ldcg(lru+i);
        if(slot<rt_resident && !(keys[slot/32]&(1U<<(slot%32)))) {
            if(rank<candidateCount)slots[rank]=slot;
            ++rank;
        }
    }
    asc_syncthreads();
    for(uint32_t i=tid;i<candidateCount;i+=Threads)asc_atomic_or(keys+slots[i]/32,1U<<(slots[i]%32));
    asc_syncthreads();
    if(candidateCount<count && !cfg[1]) {
        // The fallback scans all hot slots, so also protect untouched hot hits.
        for(uint32_t i=tid;i<kSlots;i+=Threads) {
            uint32_t slot=loc[i];
            if(slot<rt_resident)asc_atomic_or(keys+slot/32,1U<<(slot%32));
        }
        asc_syncthreads();
        auto counts=reinterpret_cast<__ubuf__ int32_t*>(ranks);
        auto offsets=counts+words;
        kvip_hot_slots::CountAvailableBitmapWordsBody(keys,counts,rt_resident,0,words,0);
        asc_syncthreads();
        uint32_t perWord=(words+Threads-1)/Threads,local=0;
        for(uint32_t i=tid*perWord;i<(tid+1)*perWord && i<words;++i)local+=counts[i];
        uint32_t scan=local;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
        if(lane==warpSize-1)cfg[544+warp]=scan;
        asc_syncthreads();
        if(!warp){uint32_t v=lane<Threads/warpSize?cfg[544+lane]:0;
            for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}cfg[544+lane]=v;}
        asc_syncthreads();
        uint32_t offset=scan-local+(warp?cfg[544+warp-1]:0);
        if(!tid && cfg[544+Threads/warpSize-1]<count-candidateCount)cfg[1]=4;
        for(uint32_t i=tid*perWord;i<(tid+1)*perWord && i<words;++i){offsets[i]=offset;offset+=counts[i];}
        asc_syncthreads();
        if(!cfg[1])kvip_hot_slots::WriteAvailableBitmapSlotsBody(keys,counts,offsets,
            reinterpret_cast<__ubuf__ int32_t*>(slots),rt_resident,0,words,0,count-candidateCount,candidateCount);
    }
    asc_syncthreads();
    if(cfg[1])return false;
    return true;
}
}
