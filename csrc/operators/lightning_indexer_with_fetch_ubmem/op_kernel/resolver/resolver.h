#pragma once
#include "prefetch.h"
#include "final_scratch.h"
#include "final_worker_submit.h"
namespace rotating {
__simd_vf__ inline void ColdPositions(__ubuf__ uint32_t* selected,
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
        CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(SIM_HISTORY),all);
        MaskAnd(invalid,invalid,notPadding,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(SIM_RESIDENT),all);
        CompareScalar<uint32_t,CMPMODE::LT>(valid,ids,uint32_t(SIM_HISTORY),all);
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
#include "../../../common/lifetch/resident_selection.h"
namespace rotating {
template<uint32_t Threads,bool PlanOnly=false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ReplicatedResolve(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,__gm__ uint32_t* mapping,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch,__gm__ Context* ctx) {
    constexpr uint32_t planWorkers=PlanOnly?1:kFinalWorkers;
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    auto lru=ctx->lruSlots ? reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots : nullptr;
    uint32_t outputStride=OutputStride(SIM_RESIDENT,ctx),ownerBase=OwnerRow(ctx,batch)*outputStride;
    uint32_t count=0,candidateCount=0;
    if(!ResolveResidentPlan<Threads,V9_CACHED_BUCKETS>(SIM_HISTORY,SIM_RESIDENT,outputStride,selected,loc,
        positions,coldCount,keys,ranks,ids,slots,cfg,mapping,lru,used,ctx->tailBounds!=0,count,candidateCount,
        CurrentMappingTag(ctx,batch),V9_UBMEM_TAGGED_MAP?kMappingSlotMask:UINT32_MAX))return;
    constexpr uint32_t missingTag=0x80000000U;
    uint32_t ownBegin=count*worker/planWorkers,ownEnd=count*(worker+1)/planWorkers;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],slot=loc[pos];
        if(pos/(kSlots/planWorkers)==worker && (slot&missingTag))loc[pos]=slots[slot&~missingTag];
    }
    asc_syncthreads();
    auto changeSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity;
    auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+batch*kChangeCapacity;
    // All final AIVs retain an identical immutable copy plan. Payload writes
    // remain partitioned by destination sector, including scattered hot victims.
    for(uint32_t rank=tid;rank<count;rank+=Threads) {
        uint32_t token=selected[ids[rank]],physical=slots[rank],dst=batch*kSlots+rank;
        uint64_t remote=asc_ldcg(addresses+token);
        keys[rank]=uint32_t(remote);ranks[rank]=uint32_t(remote>>32);
        if(rank>=ownBegin && rank<ownEnd) {
            int32_t ordinal=asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*SIM_RESIDENT+physical);
            uint32_t record=ordinal>=0?uint32_t(ordinal):kLruSlots+rank;
            // Candidate records live in the prefix, leaving holes in the
            // extended rank range. Its sole rank owner publishes those holes
            // before changeCount exposes the range to the Index planner.
            if(count>candidateCount && ordinal>=0)asc_stcg(changes+kLruSlots+rank,UINT32_MAX);
            asc_stcg(changeSlots+record,physical);asc_stcg(changes+record,token);
            StoreFallbackSource(SIM_HISTORY,kTokenBytes,ctx,batch,dst,token,remote,blocks,offsets);
            asc_stcg(destinations+dst,int32_t(ownerBase+physical));
        }
    }
    if(!tid && ownEnd>candidateCount)asc_atomic_max(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+batch,kLruSlots+ownEnd);
    // Mapping remains the immutable prefetch snapshot. Final locations and
    // descriptors are authoritative; mutating mapping here races other readers.
    asc_threadfence();
}

template<bool PlanOnly=false>
__aicore__ inline void ReplicatedFinal(__gm__ Context* ctx) {
    constexpr uint32_t planWorkers=PlanOnly?1:kFinalWorkers;
    const uint32_t core=GetBlockIdx(),b=core/planWorkers,w=core%planWorkers;
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
    if(SIM_RESIDENT<=2*kSlots) {
        static_assert(6*kSlots<=kFinalControlOffset,"metadata staging overlaps resolver control");
        // Hash/ID/slot scratch is not live yet. Reuse it for two dense metadata
        // rows; release it before ColdPositions and the original resolver.
        GlobalTensor<uint32_t> metadata;
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if(ctx->hotValid) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*SIM_RESIDENT);
            DataCopy(keys,metadata,SIM_RESIDENT);
        }
        if(ctx->reservedHot) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*SIM_RESIDENT);
            DataCopy(local[4*kSlots],metadata,SIM_RESIDENT);
        }
        uint32_t begin=0,end=0;
        if(ctx->tailBounds) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
            begin=bounds[0];end=bounds[1];
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{512,1,1},SIM_RESIDENT,FinalPtr(selected),FinalPtr(loc),
            FinalPtr(keys),reinterpret_cast<__ubuf__ int32_t*>(local[4*kSlots].GetPhyAddr()),
            ctx->hotValid!=0,ctx->reservedHot!=0,begin,end);
        PipeBarrier<PIPE_V>();
        ColdPositions(FinalPtr(loc),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    } else {
        // Larger supported resident layouts cannot hold both rows in this UB.
        Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},SIM_RESIDENT,ctx,b,FinalPtr(selected),FinalPtr(keys));
        PipeBarrier<PIPE_V>();
        ColdPositions(FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),error=ctl.GetValue(2056)|row->error|(row->slots>kSlots?32U:0U);

    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    if(!error)Simt::VF_CALL<ReplicatedResolve<512,PlanOnly>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t batchCount=error?0:ctl.GetValue(0),prefix=error?0:ctl.GetValue(2),own=error?0:ctl.GetValue(3),batchBase=b*kSlots;
    if(!error)error=ctl.GetValue(1);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/planWorkers)],loc[w*(kSlots/planWorkers)],kSlots/planWorkers);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if(!w)trace[2]=GetSystemCycle();
    if constexpr(!PlanOnly)
        PublishFinalWorker(ctx,row,b,w,own,batchCount,batchBase+prefix,error,local,keys,ranks,slots,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;
        row->finalCompleted=PlanOnly?0:FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}

// Read the existing compact descriptors after the plan kernel has finished.
// No planner scratch or pointer survives that kernel; the same stream provides
// the publication boundary. Preserve the optional block/offset source form.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void LoadFinalCopyPlan(__gm__ Context* ctx,__gm__ Row* row,uint32_t b,uint32_t count,
    __ubuf__ uint32_t* remoteLow,__ubuf__ uint32_t* remoteHigh,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* error) {
    const uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(SIM_RESIDENT,ctx);
    for(uint32_t rank=threadIdx.x;rank<count;rank+=256) {
        const uint32_t descriptor=b*kSlots+rank;
        uint64_t peer=0;
        if(ctx->fallbackAddresses)peer=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses)+descriptor);
        else {
            const uint32_t block=uint32_t(asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks)+descriptor));
            const uint32_t offset=uint32_t(asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets)+descriptor));
            const uint32_t localBlock=block-b*(SIM_HISTORY/128);
            if(localBlock>=SIM_HISTORY/128 || offset%kTokenBytes || offset/kTokenBytes>=128)
                asc_atomic_or(error,kUbmemInvalidAddress);
            else peer=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(row->addresses)+localBlock*128+offset/kTokenBytes);
        }
        remoteLow[rank]=uint32_t(peer);remoteHigh[rank]=uint32_t(peer>>32);
        slots[rank]=uint32_t(asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+descriptor))-ownerBase;
    }
}

__aicore__ inline void CopyFinalPlan(__gm__ Context* ctx) {
    const uint32_t core=GetBlockIdx(),b=core/kFinalWorkers,w=core%kFinalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),kRowStatusWords*4);
    uint32_t count=status[1],error=status[0];
    if(count>kSlots){count=0;error|=kUbmemInvalidAddress;}
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);
    auto local=scratch.Get<uint32_t>(),keys=local[2*kSlots],ranks=local[3*kSlots];
    auto slots=local[kFinalSlotsOffset],ctl=local[kFinalControlOffset];
    ctl.SetValue(0,0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if(!error && count)
        Simt::VF_CALL<LoadFinalCopyPlan>(Simt::Dim3{256,1,1},ctx,row,b,count,
            FinalPtr(keys),FinalPtr(ranks),FinalPtr(slots),FinalPtr(ctl));
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    error|=ctl.GetValue(0);
    const uint32_t begin=count*w/kFinalWorkers,end=count*(w+1)/kFinalWorkers;
    PublishFinalWorker(ctx,row,b,w,end-begin,count,b*kSlots+begin,error,local,keys,ranks,slots,ctl);
    if(!w) {
        auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
        row->finalCompleted=FinalPostAt(ctx,core)->head;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
