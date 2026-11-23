// SPDX-License-Identifier: Apache-2.0
// Single-worker Q4 resolver: publish each query independently.
#pragma once
namespace rotating {
using StagedContextPointer=__ubuf__ Context*;
// Control words 0..7 hold counts, errors and the hash size.
constexpr uint32_t kProtectionState = 8;
constexpr uint32_t kProtectionActive = 1;
constexpr uint32_t kReservationMismatch = 2;
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
#if !V9_UBMEM
    uint32_t tailSlot=DirectTailSlot(ctx,batch,resident,token);
    if(tailSlot!=UINT32_MAX)return tailSlot;
#endif
    auto mapping = reinterpret_cast<__gm__ uint32_t*>(ctx->mapping) + batch * history;
    uint32_t slot = (used || ctx->tailBounds) ? asc_ldcg(mapping + token) : UINT32_MAX;
    if (slot == UINT32_MAX) return slot;
    auto lru = reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots) + batch * kLruSlots;
    return ResolvePhysicalSlot(resident, lru, slot);
}

// Deterministic prefix; resolver uses 512 threads, small publication batches use 64.
template<uint32_t Threads=512>
__simt_callee__ inline uint32_t StagedPrefix(uint32_t local, __ubuf__ uint32_t* ctl) {
    uint32_t lane=threadIdx.x%warpSize, warp=threadIdx.x/warpSize;
    uint32_t scan=local;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t v=asc_shfl_up(scan,d);if(lane>=d)scan+=v;}
    if(lane==warpSize-1)ctl[544+warp]=scan;
    asc_syncthreads();
    if(!warp){uint32_t v=lane<Threads/warpSize?ctl[544+lane]:0;
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

// Prepare only the hash needed by the resolver. LI produced loc and bitmap.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void StagedInitializeReady(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint32_t* ctl) {
    uint32_t tid=threadIdx.x,missing=0;
    if(tid<8)ctl[tid]=0;
    for(uint32_t i=tid;i<kSlots;i+=512)
        missing += selected[i]!=UINT32_MAX && loc[i]==UINT32_MAX;
    asc_syncthreads();StagedPrefix(missing,ctl);
    uint32_t n=ctl[544+512/warpSize-1];
    uint32_t buckets=n<=1024?2048U:(n<=2048?4096U:8192U);
    if(!tid)ctl[7]=buckets;
    for(uint32_t i=tid;i<buckets;i+=512){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    asc_syncthreads();
}

// A joined owner can reach the protected allocator before its peers finish
// their TopK. Suspend in UB at that boundary; the scalar core waits for them.
enum class StagedPhase { Complete, UntilPeerBitmap, ResumeWithPeerBitmap };
constexpr uint32_t kPendingAdded = 24, kPendingUnresolved = 25, kPendingAllocated = 26;

template<StagedPhase Phase = StagedPhase::Complete>
__simt_callee__
inline void StagedQuery(uint32_t query,uint32_t history,uint32_t resident,uint32_t bytes,
    uint32_t batch,uint32_t used,StagedContextPointer ctx,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* scratch,__ubuf__ uint32_t* ctl,__gm__ uint32_t* deferredProtection=nullptr, bool peerMapsReady=true) {
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
        // Large hot bitmaps reuse the whole positions bank for fallback
        // prefix counts. Restore LRU candidates before the next Q consumes it.
        if(resident>65536){
            auto lruGm=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots;
            for(uint32_t i=tid;i<kLruSlots;i+=512)lru[i]=asc_ldcg(lruGm+i);
            asc_syncthreads();
        }
        const uint32_t begin=query*kTop,end=begin+kTop,oldCount=ctl[0];
        uint32_t added, unresolved;
        if constexpr(Phase == StagedPhase::ResumeWithPeerBitmap) {
            added=ctl[kPendingAdded];
            unresolved=ctl[kPendingUnresolved];
        } else {
            if constexpr(Phase == StagedPhase::UntilPeerBitmap) {
                if(!tid)ctl[kPendingAdded]=0;
            }
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
            added=totals&0xffffU; unresolved=totals>>16;
            for(uint32_t i=begin+tid*4;i<begin+(tid+1)*4;++i){
                if(loc[i]!=UINT32_MAX-1)continue;
                uint32_t h=SparseHash(selected[i])&(buckets-1);
                uint32_t probes=0;
                while(keys[h]!=selected[i] && probes<buckets){h=(h+1)&(buckets-1);++probes;}
                if(probes==buckets){asc_atomic_or(ctl+1,8U);continue;}
                uint32_t rank=oldCount+offset++;ids[rank%kTop]=uint16_t(i);ranks[h]=kStagedRankTag|rank;
            }
            asc_syncthreads();
        }
        uint32_t allocated=0;
        // A lane may mark a reservation mismatch before another warp reaches
        // this branch. Both the initial and mismatch states enter preflight.
        if(deferredProtection && added && ctl[kProtectionState]!=kProtectionActive){
            if constexpr(Phase == StagedPhase::ResumeWithPeerBitmap) {
                allocated=ctl[kPendingAllocated];
            } else {
                const uint32_t cursor=used+oldCount;
                const uint32_t left=cursor<kLruSlots?kLruSlots-cursor:0;
                allocated=added<left?added:left;
                for(uint32_t i=tid;i<allocated;i+=512){
                    const uint32_t slot=lru[cursor+i];
                    if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
                    // Bindings without a complete inverse LRU reservation can
                    // retain an ordinary hit in this slot. Use protected allocation.
                    if(asc_ldcg(reserved+slot)!=int32_t(cursor+i))
                        asc_atomic_or(ctl+kProtectionState,kReservationMismatch);
                    slots[oldCount+i]=slot;
                }
                asc_syncthreads();
                if(ctl[1])return;
                if(ctl[kProtectionState]==kReservationMismatch)allocated=0;
            }
            if(allocated<added){
                if constexpr(Phase == StagedPhase::UntilPeerBitmap) {
                    if(!peerMapsReady){
                        if(!tid){
                            ctl[kPendingAdded]=added;
                            ctl[kPendingUnresolved]=unresolved;
                            ctl[kPendingAllocated]=allocated;
                        }
                        asc_syncthreads();
                        return;
                    }
                }
                const uint32_t words=(resident+31)/32;
                for(uint32_t i=tid;i<words;i+=512){
                    bitmap[i]=asc_ldcg(deferredProtection+i)|asc_ldcg(deferredProtection+words+i)|
                              asc_ldcg(deferredProtection+2*words+i)|asc_ldcg(deferredProtection+3*words+i);
                }
                asc_syncthreads();
                for(uint32_t i=tid;i<oldCount+allocated;i+=512){
                    uint32_t slot=slots[i];asc_atomic_or(bitmap+slot/32,1U<<(slot%32));
                }
                if(!tid)ctl[kProtectionState]=kProtectionActive;
                asc_syncthreads();
            }
        }
        if(allocated<added) for(uint32_t pass=0;pass<2;++pass){
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
        if(!deferredProtection || ctl[kProtectionState])for(uint32_t i=tid;i<allocated;i+=512){uint32_t slot=slots[oldCount+i];asc_atomic_or(bitmap+slot/32,1U<<(slot%32));}
        asc_syncthreads();
        if(allocated<added){
            // Fallback scans beyond LRU candidates, including ordinary hits.
            // loc for future queries was populated by the protection prepass.
            if(!deferredProtection)for(uint32_t i=tid;i<kSlots;i+=512){
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
            // On exhaustion there are no slots for the suffix. Every lane
            // must exit before reading or marking those unassigned entries.
            if(ctl[1])return;
            kvip_hot_slots::WriteAvailableBitmapSlotsBody(bitmap,counts,offsets,
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
            uint32_t rank=oldCount+i, token=selected[ids[rank%kTop]], slot=slots[rank];
            int32_t ordinal=asc_ldcg(reserved+slot);
            uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
            // PackStagedUrmaQuery publishes the per-Q transport descriptors.
            // Do not write the shared compact descriptor arrays here.
            // Reserved destinations use a prefix record. Invalidate the unused
            // suffix ordinal before publishing a count that includes it.
            if(ordinal>=0)asc_stcg(changes+kLruSlots+rank,UINT32_MAX);
            asc_stcg(changeSlots+record,slot);asc_stcg(changes+record,token);
        }
        asc_threadfence();asc_syncthreads();
        if(!tid){ctl[0]=oldCount+added;ctl[2+query]=added;
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+oldCount+added);}
        asc_syncthreads();
}
}
