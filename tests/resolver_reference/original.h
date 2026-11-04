#pragma once
// Scratch declarations provided by resolve.h
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
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    if(tid<256)cfg[32+tid]=0;
    if(tid<8)cfg[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<coldCount;i+=Threads){
        uint32_t pos=positions[i],token=selected[pos];
        if(token>=rt_history){asc_atomic_or(cfg+1,1U);continue;}
        if(pos>=2*kTop && selected[pos-2*kTop]==token){loc[pos]=UINT32_MAX-2;continue;}
        uint32_t slot=used?asc_ldcg(mapping+token):UINT32_MAX;loc[pos]=slot;
        if(slot==UINT32_MAX){
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t n=0;n<buckets;++n){uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(old==UINT32_MAX || old==token){asc_atomic_min(ranks+h,pos);break;}h=(h+1)&(buckets-1);}
        } else if(slot<rt_resident || slot>=rt_resident+kSlots)asc_atomic_or(cfg+1,2U);
        else if(used+coldCount>kSlots)asc_atomic_or(cfg+32+(slot-rt_resident)/32,1U<<((slot-rt_resident)%32));
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
    if(used+count<=kSlots){for(uint32_t i=tid;i<count;i+=Threads)slots[i]=rt_resident+used+i;}
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
            if(!(bits&(1U<<bit)) && rank<count)slots[rank]=rt_resident+slot;}
    }
    asc_syncthreads();
    if(cfg[1])return;
    uint32_t ownBegin=count*worker/rt_finalWorkers,ownEnd=count*(worker+1)/rt_finalWorkers;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads){uint32_t pos=positions[i];
        if(pos/(kSlots/rt_finalWorkers)!=worker)continue;
        uint32_t slot=loc[pos],token=selected[pos];
        if(slot==UINT32_MAX-2)slot=loc[pos-2*kTop];
        if(slot>=UINT32_MAX-1){uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);slot=slots[ranks[h]];}
        loc[pos]=slot;
    }
    asc_syncthreads(); // hash and position arrays may now be reused
    for(uint32_t i=tid;i<ownEnd-ownBegin;i+=Threads){uint32_t rank=ownBegin+i,token=selected[ids[rank]],slot=slots[rank],dst=batch*kSlots+rank;
        uint64_t remote=asc_ldcg(addresses+token);keys[i]=uint32_t(remote);ranks[i]=uint32_t(remote>>32);positions[i]=slot;
        asc_stcg(blocks+dst,int32_t(batch*(rt_history/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*rt_tokenBytes));asc_stcg(destinations+dst,int32_t(batch*(rt_resident+kSlots)+slot));
    }
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
    ColdPositions(rt_history,rt_resident,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),error=ctl.GetValue(2056)|row->error|(row->slots>kSlots?32U:0U);

    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},rt_history,rt_resident,rt_finalWorkers,rt_tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t batchCount=error?0:ctl.GetValue(0),prefix=error?0:ctl.GetValue(2),own=error?0:ctl.GetValue(3),batchBase=b*kSlots;
    if(!error)error=ctl.GetValue(1);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/rt_finalWorkers)],loc[w*(kSlots/rt_finalWorkers)],kSlots/rt_finalWorkers);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if(!w)trace[2]=GetSystemCycle();
    PublishFinalWorker(rt_batch,rt_finalWorkers,rt_finalPostOffset,rt_tokenBytes,ctx,row,b,w,own,batchCount,batchBase+prefix,error,local,keys,ranks,positions,ctl);
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
