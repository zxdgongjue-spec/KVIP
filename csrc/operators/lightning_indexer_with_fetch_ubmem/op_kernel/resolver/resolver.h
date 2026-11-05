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

template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ReplicatedResolve(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,__gm__ uint32_t* mapping,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch,__gm__ Context* ctx) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    auto lru=ctx->lruSlots ? reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots : nullptr;
    uint32_t outputStride=OutputStride(SIM_RESIDENT,ctx),ownerBase=OwnerRow(ctx,batch)*outputStride;
    uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    if(tid<256)cfg[32+tid]=0;
    if(tid<8)cfg[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<coldCount;i+=Threads){
        uint32_t pos=positions[i],token=selected[pos];
        if(token>=SIM_HISTORY){asc_atomic_or(cfg+1,1U);continue;}
        if(pos>=2*kTop && selected[pos-2*kTop]==token){loc[pos]=UINT32_MAX-2;continue;}
        uint32_t slot=(used || ctx->tailBounds)?asc_ldcg(mapping+token):UINT32_MAX;loc[pos]=slot;
        if(slot==UINT32_MAX){
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t n=0;n<buckets;++n){uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(old==UINT32_MAX || old==token){asc_atomic_min(ranks+h,pos);break;}h=(h+1)&(buckets-1);}
        } else if(slot<SIM_RESIDENT || slot>=outputStride)asc_atomic_or(cfg+1,2U);
        else if(slot<SIM_RESIDENT+kSlots && used+coldCount>kSlots)asc_atomic_or(cfg+32+(slot-SIM_RESIDENT)/32,1U<<((slot-SIM_RESIDENT)%32));
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
    if(used+count<=kSlots){for(uint32_t i=tid;i<count;i+=Threads)slots[i]=SIM_RESIDENT+used+i;}
    else {
        uint32_t n=tid<256?uint32_t(__popc(~cfg[32+tid])):0,value=n;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}
        if(lane==warpSize-1)cfg[544+warp]=value;
        asc_syncthreads();
        if(!warp){uint32_t v=lane<Threads/warpSize?cfg[544+lane]:0;
            for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}cfg[544+lane]=v;}
        asc_syncthreads();
        if(tid<256)cfg[288+tid]=value-n+(warp?cfg[544+warp-1]:0);
        if(!tid && cfg[544+Threads/warpSize-1]<count)cfg[1]=4;
        asc_syncthreads();
        for(uint32_t slot=tid;slot<kSlots;slot+=Threads){uint32_t bits=cfg[32+slot/32],bit=slot%32;
            uint32_t rank=cfg[288+slot/32]+uint32_t(__popc((~bits)&((1U<<bit)-1U)));
            if(!(bits&(1U<<bit)) && rank<count)slots[rank]=SIM_RESIDENT+slot;}
    }
    asc_syncthreads();
    // Only an empty hot row outside the LRU pool retains its own destination.
    // Reserved hot rows are ordinary misses and keep the allocator's slots.
    for(uint32_t i=tid;i<count;i+=Threads) {
        uint32_t token=selected[ids[i]];
        if(token<SIM_RESIDENT && !IsTailToken(ctx,batch,token)) {
            int32_t ordinal=ctx->reservedHot ? asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*SIM_RESIDENT+token) : -1;
            if(ordinal<0)slots[i]=token;
        }
    }
    asc_syncthreads();
    if(cfg[1])return;
    uint32_t ownBegin=count*worker/kFinalWorkers,ownEnd=count*(worker+1)/kFinalWorkers;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads){uint32_t pos=positions[i];
        if(pos/(kSlots/kFinalWorkers)!=worker)continue;
        uint32_t slot=loc[pos],token=selected[pos];
        if(slot==UINT32_MAX-2)slot=loc[pos-2*kTop];
        if(slot>=UINT32_MAX-1){uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);slot=slots[ranks[h]];}
        loc[pos]=slot;
    }
    asc_syncthreads(); // hash and position arrays may now be reused
    // All duplicate-query reads of logical slots joined above. The output
    // adapter shares the original resolver invocation; no extra VF launch.
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i];
        if(pos/(kSlots/kFinalWorkers)==worker)
            loc[pos]=ResolvePhysicalSlot(SIM_RESIDENT,lru,loc[pos]);
    }
    asc_syncthreads(); // Join position readers before descriptor scratch reuse.
    // The UBmem final workers retain the same complete immutable plan.
    // Ownership is selected from each actual destination sector, so scattered
    // LRU victims cannot make two AIVs overwrite halves of one 32-byte sector.
    for(uint32_t rank=tid;rank<count;rank+=Threads) {
        uint32_t token=selected[ids[rank]],logicalSlot=slots[rank],dst=batch*kSlots+rank;
        uint32_t physical=ResolvePhysicalSlot(SIM_RESIDENT,lru,logicalSlot);
        uint64_t remote=asc_ldcg(addresses+token);
        keys[rank]=uint32_t(remote);ranks[rank]=uint32_t(remote>>32);slots[rank]=physical;
        if(rank>=ownBegin && rank<ownEnd) {
            RecordChange(SIM_RESIDENT,ctx,batch,logicalSlot,token);
            StoreFallbackSource(SIM_HISTORY,kTokenBytes,ctx,batch,dst,token,remote,blocks,offsets);
            asc_stcg(destinations+dst,int32_t(ownerBase+physical));
        }
    }
    // Mapping remains the immutable prefetch snapshot. Final locations and
    // descriptors are authoritative; mutating mapping here races other readers.
    asc_threadfence();
}

__aicore__ inline void ReplicatedFinal(__gm__ Context* ctx) {
    const uint32_t core=GetBlockIdx(),b=core/kFinalWorkers,w=core%kFinalWorkers;
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
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t batchCount=error?0:ctl.GetValue(0),prefix=error?0:ctl.GetValue(2),own=error?0:ctl.GetValue(3),batchBase=b*kSlots;
    if(!error)error=ctl.GetValue(1);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/kFinalWorkers)],loc[w*(kSlots/kFinalWorkers)],kSlots/kFinalWorkers);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if(!w)trace[2]=GetSystemCycle();
    PublishFinalWorker(ctx,row,b,w,own,batchCount,batchBase+prefix,error,local,keys,ranks,slots,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalCompleted=FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
