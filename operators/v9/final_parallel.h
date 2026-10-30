#pragma once
#include "final_scratch.h"
namespace rotating {
__simd_vf__ inline void OwnedColdPositions(__ubuf__ uint32_t* selected,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* positions,__ubuf__ uint32_t* count,uint32_t worker) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> ids,flags,total,packed,errors,ownedTotal,hash,ownerMask;
    RegTensor<int32_t> signedIndex;
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(signedIndex);
    Arange(signedIndex,int32_t(0));Duplicate(total,uint32_t(0));Duplicate(errors,uint32_t(0));
    Duplicate(ownedTotal,uint32_t(0));Duplicate(ownerMask,uint32_t(kFinalWorkers-1));
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<kSlots;i+=64) {
        LoadAlign(ids,selected+i);StoreAlign(locations+i,ids,all);
        MaskReg invalid;CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(SIM_HISTORY),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(SIM_RESIDENT),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),cold);
        Add(total,total,flags,all);
        Muls(hash,ids,uint32_t(0x9e3779b9U),all);ShiftRights(hash,hash,int16_t(16),all);And(hash,hash,ownerMask,all);
        MaskReg owned;CompareScalar<uint32_t,CMPMODE::EQ>(owned,hash,worker,all);MaskAnd(owned,owned,cold,all);
        if(i>=2*kTop){RegTensor<uint32_t> first;LoadAlign(first,selected+i-2*kTop);
            MaskReg distinct;Compare<uint32_t,CMPMODE::NE>(distinct,ids,first,all);MaskAnd(owned,owned,distinct,all);}
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),owned);Add(ownedTotal,ownedTotal,flags,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,owned);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(positions,store);ReduceSum(total,total,all);StoreAlign(count+16,total,lane);
    ReduceSum(ownedTotal,ownedTotal,all);StoreAlign(count,ownedTotal,lane);
    ReduceSum(errors,errors,all);StoreAlign(count+8,errors,lane);
}

// Token ownership keeps all query duplicates on one worker. Only counts and
// protected-slot bitsets cross workers; the hot path has no global hash atomics.
__simt_callee__ inline uint32_t FinalOwner(uint32_t token) {
    return ((token*0x9e3779b9U)>>16)&(kFinalWorkers-1);
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ParallelCollect(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,
    __ubuf__ uint32_t* control,__gm__ uint32_t* map,uint32_t worker,uint32_t used,bool protect) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize;
    const uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads)keys[i]=UINT32_MAX;
    if(tid<256)control[32+tid]=0;
    if(tid<8)control[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<((coldCount+Threads-1)/Threads)*Threads;i+=Threads) {
        bool fresh=false;uint32_t token=0,h=0;
        if(i<coldCount) {
            uint32_t pos=positions[i];token=selected[pos];
            if(token>=SIM_HISTORY)asc_atomic_or(control+1,1U);
            else if(FinalOwner(token)==worker) {
                bool alias=pos>=2*kTop && selected[pos-2*kTop]==token;
                uint32_t slot=alias?UINT32_MAX-1:(used?asc_ldcg(map+token):UINT32_MAX);
                loc[pos]=slot;
                if(slot==UINT32_MAX) {
                    h=SparseHash(token)&(buckets-1);
                    for(uint32_t probe=0;probe<buckets;++probe) {
                        uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                        if(old==UINT32_MAX){fresh=true;break;}
                        if(old==token)break;
                        h=(h+1)&(buckets-1);
                    }
                } else if(alias){}
                else if(slot<SIM_RESIDENT || slot>=SIM_RESIDENT+kSlots)asc_atomic_or(control+1,2U);
                else if(protect)asc_atomic_or(control+32+(slot-SIM_RESIDENT)/32,1U<<((slot-SIM_RESIDENT)%32));
            }
        }
        uint32_t mask=asc_ballot(fresh),base=0;
        if(!lane && mask)base=asc_atomic_add(control,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        if(fresh){uint32_t rank=base+uint32_t(__popc(mask&((1U<<lane)-1U)));ranks[h]=rank;ids[rank]=uint16_t(positions[i]);}
    }
    asc_syncthreads();
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ParallelEmit(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* control,__ubuf__ uint32_t* protections,
    __gm__ uint32_t* map,__gm__ uint64_t* addresses,__gm__ uint32_t* locations,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    uint32_t worker,uint32_t used,uint32_t ownCount,uint32_t batchCount,
    uint32_t prefix,uint32_t globalBase,uint32_t batch) {
    uint32_t tid=threadIdx.x;
    const uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    if(used+batchCount<=kSlots) {
        for(uint32_t i=tid;i<ownCount;i+=Threads)slots[i]=SIM_RESIDENT+used+prefix+i;
    } else {
        if(tid<256) {
            uint32_t bits=0;
            for(uint32_t w=0;w<kFinalWorkers;++w)bits|=protections[w*256+tid];
            control[32+tid]=bits;control[288+tid]=uint32_t(__popc(~bits));
        }
        asc_syncthreads();
        const uint32_t lane=tid%warpSize,warp=tid/warpSize;
        uint32_t n=tid<256?control[288+tid]:0,scan=n;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
        if(lane==warpSize-1)control[544+warp]=scan;
        asc_syncthreads();
        if(!warp){uint32_t value=lane<Threads/warpSize?control[544+lane]:0;
            for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}
            control[544+lane]=value;}
        asc_syncthreads();
        if(tid<256)control[288+tid]=scan-n+(warp?control[544+warp-1]:0);
        if(!tid && control[544+Threads/warpSize-1]<batchCount)control[1]=4;
        asc_syncthreads();
        for(uint32_t s=tid;s<kSlots;s+=Threads) {
            uint32_t bits=control[32+s/32],bit=s%32;
            uint32_t rank=control[288+s/32]+uint32_t(__popc((~bits)&((1U<<bit)-1U)));
            if(!(bits&(1U<<bit)) && rank>=prefix && rank<prefix+ownCount)slots[rank-prefix]=SIM_RESIDENT+s;
        }
    }
    asc_syncthreads();
    if(control[1])return;
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],token=selected[pos];
        if(FinalOwner(token)!=worker)continue;
        uint32_t slot=loc[pos];
        if(slot==UINT32_MAX-1)slot=loc[pos-2*kTop];
        if(slot==UINT32_MAX){uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);slot=slots[ranks[h]];}
        asc_stcg(locations+pos,slot);
        if(pos<2*kTop && selected[pos+2*kTop]==token)asc_stcg(locations+pos+2*kTop,slot);
    }
    asc_syncthreads(); // hash arrays become remote-address vectors only after all lookups
    for(uint32_t i=tid;i<ownCount;i+=Threads) {
        uint32_t token=selected[ids[i]],slot=slots[i],dst=globalBase+i;
        uint64_t remote=asc_ldcg(addresses+token);
        keys[i]=uint32_t(remote);ranks[i]=uint32_t(remote>>32);
        asc_stcg(map+token,slot);
        asc_stcg(blocks+dst,int32_t(batch*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(batch*(SIM_RESIDENT+kSlots)+slot));
    }
    asc_threadfence();
}
__aicore__ inline void ParallelFinal(__gm__ Context* ctx) {
    const uint32_t core=GetBlockIdx(),b=core/kFinalWorkers,w=core%kFinalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
    const uint64_t started=GetSystemCycle();
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;
    static_assert(kSlots==8192,"final scratch bank geometry changed");
    pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output,shared;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    shared.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->owners)+SIM_BATCH*32);
    DataCopy(selected,input,kSlots);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    OwnedColdPositions(FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]),w);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),invalid=ctl.GetValue(2056);
    Simt::VF_CALL<ParallelCollect<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(ctl),reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,w,row->slots,row->slots+ctl.GetValue(2064)>kSlots);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t error=ctl.GetValue(1)|invalid|row->error|(row->slots>kSlots?32U:0U);

    ctl.SetValue(1,error);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(shared[core*kFinalStateWords],ctl,kFinalStateWords);
    // Initialize hot locations exactly once; cold overwrites follow the barrier.
    DataCopy(output[w*(kSlots/kFinalWorkers)],selected[w*(kSlots/kFinalWorkers)],kSlots/kFinalWorkers);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SyncAll<true>();
    auto counts=ctl[1024];
    DataCopyExtParams cp{uint16_t(kFinalCores),32,uint32_t(kFinalStateWords*4-32),0,0};
    DataCopyPad(counts,shared,cp,DataCopyPadExtParams<uint32_t>{false,0,0,0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t globalBase=0,batchBase=0,batchCount=0,prefix=0;
    for(uint32_t c=0;c<kFinalCores;++c){uint32_t n=counts.GetValue(c*8);if(c<core)globalBase+=n;if(c<b*kFinalWorkers)batchBase+=n;if(c/kFinalWorkers==b){batchCount+=n;error|=counts.GetValue(c*8+1);if(c<core)prefix+=n;}}
    uint32_t own=counts.GetValue(core*8);
    if(!core){uint32_t total=0;for(uint32_t c=0;c<kFinalCores;++c)total+=counts.GetValue(c*8);*reinterpret_cast<__gm__ uint32_t*>(ctx->count)=total;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),32);}
    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    // Counts occupy [1024,1536); protection bits occupy [1536,2560).
    auto protect=ctl[1536];
    if(row->slots+batchCount>kSlots){
        DataCopyExtParams bp{uint16_t(kFinalWorkers),1024,128,0,0};
        DataCopyPad(protect,shared[b*kFinalWorkers*kFinalStateWords+32],bp,DataCopyPadExtParams<uint32_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    ctl.SetValue(1,error);SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if(!error)Simt::VF_CALL<ParallelEmit<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),FinalPtr(protect),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots,reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
        w,row->slots,own,batchCount,prefix,globalBase,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
    if(!w)trace[2]=GetSystemCycle();
    PublishFinalWorker(ctx,row,b,w,own,batchCount,globalBase,error,local,keys,ranks,slots,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalHead=FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
