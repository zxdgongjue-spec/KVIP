// SPDX-License-Identifier: Apache-2.0
// Experimental single-worker Q4 resolver: each query's result is published
// independently. This file is not used by the production forward path.
#pragma once
namespace rotating {
constexpr uint32_t kStagedRankTag = 0x80000000U;

// Resolve the original prefetch mapping without modifying it.
__simt_callee__ inline uint32_t StagedKnownSlot(uint32_t token, uint32_t resident,
    uint32_t history, uint32_t used, uint32_t batch, __gm__ Context* ctx,
    __ubuf__ uint32_t* valid, __ubuf__ int32_t* reservedLocal) {
    if (token == UINT32_MAX) return UINT32_MAX;
    if (token >= history) return UINT32_MAX;
    bool isHot = token < resident && !IsTailToken(ctx, batch, token) &&
        (!ctx->hotValid || valid[token]) && (!ctx->reservedHot || reservedLocal[token] < 0);
    if (isHot) return token;
    auto mapping = reinterpret_cast<__gm__ uint32_t*>(ctx->mapping) + batch * history;
    uint32_t slot = (used || ctx->tailBounds) ? asc_ldcg(mapping + token) : UINT32_MAX;
    if (slot == UINT32_MAX) return slot;
    auto lru = reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots) + batch * kLruSlots;
    return ResolvePhysicalSlot(resident, lru, slot);
}

// Deterministic prefix of per-thread counts (512 SIMT threads).
__simt_callee__ inline uint32_t StagedPrefix(uint32_t local, __ubuf__ uint32_t* ctl) {
    uint32_t lane=threadIdx.x%warpSize, warp=threadIdx.x/warpSize;
    uint32_t scan=local;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t v=asc_shfl_up(scan,d);if(lane>=d)scan+=v;}
    if(lane==warpSize-1)ctl[544+warp]=scan;
    asc_syncthreads();
    if(!warp){uint32_t v=lane<512/warpSize?ctl[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}ctl[544+lane]=v;}
    asc_syncthreads();
    return scan-local+(warp?ctl[544+warp-1]:0);
}

// Standalone BS16, one worker/request. Global protection runs once;
// the caller invokes StagedQuery and bulk-writes one query at a time.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void StagedInitialize(uint32_t history,uint32_t resident,uint32_t batch,uint32_t used,
    __gm__ Context* ctx,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* ctl) {
    uint32_t tid=threadIdx.x;auto bitmap=ctl+1024;
    if(tid<8)ctl[tid]=0;
    for(uint32_t i=tid;i<(resident+31)/32;i+=512)bitmap[i]=0;
    asc_syncthreads();
    uint32_t localMissing=0;
    // Q1/Q2 are classified first, so Q3/Q4 can reuse an identical position
    // without repeating its mapping load or bitmap update. This is the same
    // same-position shortcut as the baseline; different positions still go
    // through normal classification and cross-Q hash deduplication.
    for(uint32_t half=0;half<2;++half){
        uint32_t begin=half*2*kTop,end=begin+2*kTop;
        for(uint32_t i=begin+tid;i<end;i+=512){
            uint32_t token=selected[i];
            if(half && token==selected[i-2*kTop]){loc[i]=loc[i-2*kTop];continue;}
            if(token!=UINT32_MAX && token>=history)asc_atomic_or(ctl+1,1U);
            uint32_t slot=StagedKnownSlot(token,resident,history,used,batch,ctx,keys,
                reinterpret_cast<__ubuf__ int32_t*>(ids));
            loc[i]=slot;
            localMissing+=(token!=UINT32_MAX && token<history && slot==UINT32_MAX);
            // Ordinary hot hits are outside the candidate list. The fallback
            // marks all known hot slots before scanning the full hot bitmap.
            auto reservedLocal=reinterpret_cast<__ubuf__ int32_t*>(ids);
            if(slot<resident && (!ctx->reservedHot || reservedLocal[slot]>=0))
                asc_atomic_or(bitmap+slot/32,1U<<(slot%32));
        }
        asc_syncthreads();
    }
    // Use the original adaptive hash sizing, based on an upper bound on
    // distinct misses. No two same-position Q3/Q4 entries need a new key.
    StagedPrefix(localMissing,ctl);
    uint32_t missingUpper=ctl[544+512/warpSize-1];
    uint32_t buckets=missingUpper<=1024?2048U:(missingUpper<=2048?4096U:8192U);
    if(!tid)ctl[7]=buckets;
    // Metadata staging is dead; reuse its UB banks for the cross-Q hash.
    for(uint32_t i=tid;i<buckets;i+=512){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void StagedQuery(uint32_t query,uint32_t history,uint32_t resident,uint32_t bytes,
    uint32_t batch,uint32_t used,__gm__ Context* ctx,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* scratch,__ubuf__ uint32_t* ctl) {
    const uint32_t tid=threadIdx.x,owner=OwnerRow(ctx,batch),stride=OutputStride(resident,ctx);
    auto bitmap=ctl+1024;
    uint32_t buckets=ctl[7];
    auto lru=scratch+4096;
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*resident;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+batch;
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
        const uint32_t begin=query*kTop,end=begin+kTop,oldCount=ctl[0];
        for(uint32_t i=begin+tid;i<end;i+=512){
            uint32_t token=selected[i];
            if(token==UINT32_MAX){loc[i]=UINT32_MAX;continue;}
            // Reuse the original resolver's Q3/Q1, Q4/Q2 same-position
            // shortcut: the earlier query has already published its slot.
            if(query>=2 && selected[i-2*kTop]==token){loc[i]=loc[i-2*kTop];continue;}
            if(loc[i]!=UINT32_MAX)continue;
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t probe=0;probe<buckets;++probe){
                uint32_t prev=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(prev==UINT32_MAX || prev==token){
                    uint32_t rank=ranks[h];
                    if(rank==UINT32_MAX || !(rank&kStagedRankTag))asc_atomic_min(ranks+h,i);
                    break;
                }
                h=(h+1)&(buckets-1);
            }
        }
        asc_syncthreads();
        uint32_t local=0,pending=0;
        for(uint32_t i=begin+tid*4;i<begin+(tid+1)*4;++i){
            if(loc[i]!=UINT32_MAX || selected[i]==UINT32_MAX)continue;
            ++pending;
            uint32_t h=SparseHash(selected[i])&(buckets-1);
            uint32_t probes=0;
            while(keys[h]!=selected[i] && probes<buckets){h=(h+1)&(buckets-1);++probes;}
            if(probes==buckets){asc_atomic_or(ctl+1,8U);continue;}
            if(ranks[h]==i){loc[i]=UINT32_MAX-1;++local;}
        }
        // Both counts fit in 16 bits: each query contains 2048 positions.
        // Carrying them in one scan avoids another barrier just to know if
        // there are any unresolved duplicate positions to translate.
        uint32_t offset=StagedPrefix(local|(pending<<16),ctl)&0xffffU;
        uint32_t totals=ctl[544+512/warpSize-1];
        uint32_t added=totals&0xffffU,unresolved=totals>>16;
        for(uint32_t i=begin+tid*4;i<begin+(tid+1)*4;++i){
            if(loc[i]!=UINT32_MAX-1)continue;
            uint32_t h=SparseHash(selected[i])&(buckets-1);
            uint32_t probes=0;
            while(keys[h]!=selected[i] && probes<buckets){h=(h+1)&(buckets-1);++probes;}
            if(probes==buckets){asc_atomic_or(ctl+1,8U);continue;}
            uint32_t rank=oldCount+offset++;ids[rank]=uint16_t(i);ranks[h]=kStagedRankTag|rank;
        }
        asc_syncthreads();
        uint32_t allocated=0;
        if(added) for(uint32_t pass=0;pass<2;++pass){
            uint32_t available=0;
            for(uint32_t i=tid*4;i<(tid+1)*4;++i){
                if((i<used)!=(pass==1))continue;
                uint32_t slot=lru[i];
                if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
                available+=!(bitmap[slot/32]&(1U<<(slot%32)));
            }
            uint32_t rank=StagedPrefix(available,ctl),n=ctl[544+512/warpSize-1];
            if(n>added-allocated)n=added-allocated;
            for(uint32_t i=tid*4;i<(tid+1)*4;++i){
                if((i<used)!=(pass==1))continue;
                uint32_t slot=lru[i];
                if(slot<resident && !(bitmap[slot/32]&(1U<<(slot%32)))){
                    if(rank<n)slots[oldCount+allocated+rank]=slot;
                    ++rank;
                }
            }
            asc_syncthreads();allocated+=n;
            if(allocated==added)break;
        }
        for(uint32_t i=tid;i<allocated;i+=512){uint32_t slot=slots[oldCount+i];asc_atomic_or(bitmap+slot/32,1U<<(slot%32));}
        asc_syncthreads();
        if(allocated<added){
            // Fallback scans beyond LRU candidates, including ordinary hits.
            // loc for future queries was populated by the protection prepass.
            for(uint32_t i=tid;i<kSlots;i+=512){
                uint32_t slot=loc[i];
                if(slot<resident)asc_atomic_or(bitmap+slot/32,1U<<(slot%32));
            }
            asc_syncthreads();
            uint32_t words=(resident+31)/32;
            auto counts=reinterpret_cast<__ubuf__ int32_t*>(scratch);
            auto offsets=counts+words;
            kvip_hot_slots::CountAvailableBitmapWordsBody(bitmap,counts,resident,0,words,0);
            asc_syncthreads();
            uint32_t per=(words+511)/512,localWords=0;
            for(uint32_t i=tid*per;i<(tid+1)*per && i<words;++i)localWords+=counts[i];
            uint32_t off=StagedPrefix(localWords,ctl);
            if(!tid && ctl[544+512/warpSize-1]<added-allocated)ctl[1]=4;
            for(uint32_t i=tid*per;i<(tid+1)*per && i<words;++i){offsets[i]=off;off+=counts[i];}
            asc_syncthreads();
            if(!ctl[1])kvip_hot_slots::WriteAvailableBitmapSlotsBody(bitmap,counts,offsets,
                reinterpret_cast<__ubuf__ int32_t*>(slots),resident,0,words,0,added-allocated,oldCount+allocated);
            asc_syncthreads();
            for(uint32_t i=tid+allocated;i<added;i+=512){uint32_t slot=slots[oldCount+i];asc_atomic_or(bitmap+slot/32,1U<<(slot%32));}
        }
        asc_syncthreads();
        if(ctl[1])return;
        if(unresolved)for(uint32_t i=begin+tid;i<end;i+=512){
            if(loc[i]>=UINT32_MAX-1 && selected[i]!=UINT32_MAX){
                uint32_t h=SparseHash(selected[i])&(buckets-1);
                uint32_t probes=0;
                while(keys[h]!=selected[i] && probes<buckets){h=(h+1)&(buckets-1);++probes;}
                if(probes==buckets){asc_atomic_or(ctl+1,8U);continue;}
                loc[i]=slots[ranks[h]&~kStagedRankTag];
            }
        }
        for(uint32_t i=tid;i<added;i+=512){
            uint32_t rank=oldCount+i, token=selected[ids[rank]], slot=slots[rank];
            int32_t ordinal=asc_ldcg(reserved+slot);
            uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
            uint32_t dst=batch*kSlots+rank;
            asc_stcg(changeSlots+record,slot);asc_stcg(changes+record,token);
            StoreFallbackSource(history,bytes,ctx,batch,dst,token,asc_ldcg(addresses+token),blocks,offsets);
            asc_stcg(destinations+dst,int32_t(owner*stride+slot));
        }
        asc_threadfence();asc_syncthreads();
        if(!tid){ctl[0]=oldCount+added;ctl[2+query]=added;
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+oldCount+added);}
        asc_syncthreads();
}
}
