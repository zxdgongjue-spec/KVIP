#pragma once
#include "prefetch.h"
#include "../../lidu/op_kernel/hot_slot_selection.h"

// Imported from final_resolve.h

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

namespace rotating {
// The demo's SIMD compaction keeps hot TopK entries out of the SIMT worklist.
// Runtime work scales with selected cold entries, not all four 2048-entry rows.
__simd_vf__ inline void ColdPositions(uint32_t rt_history,uint32_t rt_resident,__ubuf__ uint32_t* selected,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* positions,__ubuf__ uint32_t* count) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> ids,flags,total,packed,errors;
    RegTensor<int32_t> signedIndex;
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(signedIndex);
    Arange(signedIndex,int32_t(0));Duplicate(total,uint32_t(0));Duplicate(errors,uint32_t(0));
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<kSlots;i+=64) {
        LoadAlign(ids,selected+i);StoreAlign(locations+i,ids,all);
        MaskReg invalid,notPadding,valid;
        CompareScalar<uint32_t,CMPMODE::NE>(notPadding,ids,UINT32_MAX,all);
        CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(rt_history),all);
        MaskAnd(invalid,invalid,notPadding,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(rt_resident),all);
        CompareScalar<uint32_t,CMPMODE::LT>(valid,ids,uint32_t(rt_history),all);
        MaskAnd(cold,cold,valid,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),cold);
        Add(total,total,flags,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,cold);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(positions,store);ReduceSum(total,total,all);StoreAlign(count,total,lane);
    ReduceSum(errors,errors,all);StoreAlign(count+8,errors,lane);
}
__simt_callee__ inline uint32_t SparseHash(uint32_t token) {
    return (token*0x9e3779b9U)>>19;
}
}

// Imported from final_worker_submit.h

namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+rt_finalPostOffset)+core;
}
// Resolver publishes compact ranges only. KVAF owns all final transport work.
__aicore__ inline void PrepareFinalFetch(uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,
    __gm__ Context* ctx,uint32_t b,uint32_t w,uint32_t count,uint32_t descriptor,uint32_t error) {
    auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
    post->count=count;post->pending=0;post->error=error;
    post->deferred=count && !error;post->descriptor=descriptor;post->head=0;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}

// Imported from final_scratch.h

#include <cstdint>
#ifndef V9_USER_UB_BYTES
#error "V9_USER_UB_BYTES must match the compiler user-reserved-ub-size option"
#endif
namespace rotating {
// Six uint32 banks, one uint16 first-occurrence-position bank, and control.
// Storing a TopK position (0..8191) instead of its token saves 16 KiB without
// narrowing token IDs. Recover tokens from selected[position] at address gather.
constexpr uint32_t kFinalSlotsOffset = 4 * 8192 + 4096;
constexpr uint32_t kFinalPositionsOffset = kFinalSlotsOffset + 8192;
constexpr uint32_t kFinalControlOffset = kFinalPositionsOffset + 8192;
constexpr uint32_t kFinalScratchWords = kFinalControlOffset + 2560;
constexpr uint32_t kFinalScratchBytes = kFinalScratchWords * sizeof(uint32_t);
static_assert(kFinalScratchBytes <= V9_USER_UB_BYTES,
              "final scratch exceeds the compiler user UB reservation");
static_assert(V9_USER_UB_BYTES + 6144 <= 229376,
              "user plus CANN 9.1 compiler UB exceeds the SIMT limit");
}

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
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    auto lru=ctx->lruSlots ? reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots : nullptr;
    uint32_t outputStride=OutputStride(rt_resident,ctx),ownerBase=OwnerRow(ctx,batch)*outputStride;
    uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    if(tid<8)cfg[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<coldCount;i+=Threads){
        uint32_t pos=positions[i],token=selected[pos];
        if(token>=rt_history){asc_atomic_or(cfg+1,1U);continue;}
        if(pos>=2*kTop && selected[pos-2*kTop]==token){loc[pos]=UINT32_MAX-2;continue;}
        uint32_t slot=(used || ctx->tailBounds)?asc_ldcg(mapping+token):UINT32_MAX;loc[pos]=slot;
        if(slot==UINT32_MAX){
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t n=0;n<buckets;++n){uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(old==UINT32_MAX || old==token){asc_atomic_min(ranks+h,pos);break;}h=(h+1)&(buckets-1);}
        } else if(slot<rt_resident || slot>=kSlots+outputStride || (slot>=rt_resident+kLruSlots && slot<rt_resident+kSlots))asc_atomic_or(cfg+1,2U);
    }
    asc_syncthreads();
    // Mark the first occurrence before replacing hash-table positions by ranks.
    uint32_t per=(coldCount+Threads-1)/Threads,localCount=0;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX)continue;
        uint32_t token=selected[pos],h=SparseHash(token)&(buckets-1);
        while(keys[h]!=token)h=(h+1)&(buckets-1);
        if(ranks[h]==pos){loc[pos]=UINT32_MAX-1;++localCount;}}
    uint32_t scan=localCount;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
    if(lane==warpSize-1)cfg[544+warp]=scan;
    asc_syncthreads();
    if(!warp){uint32_t value=lane<Threads/warpSize?cfg[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}cfg[544+lane]=value;}
    asc_syncthreads();
    uint32_t begin=scan-localCount+(warp?cfg[544+warp-1]:0),count=cfg[544+Threads/warpSize-1];
    if(!tid)cfg[0]=count;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX-1)continue;
        uint32_t token=selected[pos],h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);
        ids[begin]=uint16_t(pos);ranks[h]=begin++;}
    asc_syncthreads();
    // Resolve rank markers before releasing the hash tables. All workers see
    // the same final Q4 set, so each independently chooses identical hot slots.
    constexpr uint32_t missingTag=0x80000000U;
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],slot=loc[pos],token=selected[pos];
        // Earlier query locations may already be physical (including ABC).
        // Read the immutable mapping instead of translating them a second time.
        if(slot==UINT32_MAX-2)slot=(used || ctx->tailBounds)?asc_ldcg(mapping+token):UINT32_MAX;
        if(slot>=UINT32_MAX-2) {
            uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);
            slot=missingTag|ranks[h];
        } else slot=ResolvePhysicalSlot(rt_resident,lru,slot);
        loc[pos]=slot;
    }
    asc_syncthreads();
    if(!count)return;
    // Hash scratch is now dead. Reuse it as LIDU's hot protection bitmap and
    // count/prefix scratch; LI tiling and the compacted TopK worklist stay intact.
    uint32_t words=(rt_resident+31)/32;
    if(words>kSlots/2) {if(!tid)cfg[1]=4;return;}
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
    uint32_t candidateCount=cfg[544+Threads/warpSize-1];
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
    if(cfg[1])return;
    uint32_t ownBegin=count*worker/rt_finalWorkers,ownEnd=count*(worker+1)/rt_finalWorkers;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],slot=loc[pos];
        if(pos/(kSlots/rt_finalWorkers)==worker && (slot&missingTag))loc[pos]=slots[slot&~missingTag];
    }
    asc_syncthreads();
    // Source addresses are snapshotted before Index update. Prefetch records
    // overwritten by final repair are replaced, not duplicated in the plan.
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    for(uint32_t i=tid;i<ownEnd-ownBegin;i+=Threads) {
        uint32_t rank=ownBegin+i,token=selected[ids[rank]],slot=slots[rank],dst=batch*kSlots+rank;
        uint64_t remote=asc_ldcg(addresses+token);keys[i]=uint32_t(remote);ranks[i]=uint32_t(remote>>32);positions[i]=slot;
        int32_t ordinal=asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*rt_resident+slot);
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
    const uint32_t core=GetBlockIdx(),b=core/rt_finalWorkers,w=core%rt_finalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
    uint64_t started=GetSystemCycle();
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    DataCopy(selected,input,kSlots);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(rt_resident<=2*kSlots) {
        static_assert(6*kSlots<=kFinalControlOffset,"metadata staging overlaps resolver control");
        // Hash/ID/slot scratch is not live yet. Reuse it for two dense metadata
        // rows; release it before ColdPositions and the original resolver.
        GlobalTensor<uint32_t> metadata;
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if(ctx->hotValid) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*rt_resident);
            DataCopy(keys,metadata,rt_resident);
        }
        if(ctx->reservedHot) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*rt_resident);
            DataCopy(local[4*kSlots],metadata,rt_resident);
        }
        uint32_t begin=0,end=0;
        if(ctx->tailBounds) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
            begin=bounds[0];end=bounds[1];
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{512,1,1},rt_resident,FinalPtr(selected),FinalPtr(loc),
            FinalPtr(keys),reinterpret_cast<__ubuf__ int32_t*>(local[4*kSlots].GetPhyAddr()),
            ctx->hotValid!=0,ctx->reservedHot!=0,begin,end);
        PipeBarrier<PIPE_V>();
        ColdPositions(rt_history,rt_resident,FinalPtr(loc),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    } else {
        // Larger supported resident layouts cannot hold both rows in this UB.
        Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},rt_resident,ctx,b,FinalPtr(selected),FinalPtr(keys));
        PipeBarrier<PIPE_V>();
        ColdPositions(rt_history,rt_resident,FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),error=ctl.GetValue(2056)|row->error|(row->slots>kSlots?32U:0U);

    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},rt_history,rt_resident,rt_finalWorkers,rt_tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t batchCount=error?0:ctl.GetValue(0),prefix=error?0:ctl.GetValue(2),own=error?0:ctl.GetValue(3),batchBase=b*kSlots;
    if(!error)error=ctl.GetValue(1);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/rt_finalWorkers)],loc[w*(kSlots/rt_finalWorkers)],kSlots/rt_finalWorkers);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if(!w)trace[2]=GetSystemCycle();
    PrepareFinalFetch(rt_finalWorkers,rt_finalPostOffset,ctx,b,w,own,batchBase+prefix,error);
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
