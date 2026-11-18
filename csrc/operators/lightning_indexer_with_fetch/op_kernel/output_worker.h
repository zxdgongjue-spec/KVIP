// SPDX-License-Identifier: Apache-2.0
// Optional publication AIV. Raw change events remain owned by Resolver; this
// worker owns service outputs, FinalPost and existing KVAF descriptor publication.
#pragma once
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PreparePublicationOutputs(
 __gm__ int64_t* logical,__ubuf__ uint32_t* masks,uint32_t batchCount,uint32_t b,uint32_t q,
 __ubuf__ uint32_t* selected,__ubuf__ uint32_t* work){
 if(threadIdx.x<8)work[threadIdx.x]=0;
 asc_syncthreads();
 if(logical)PrepareStagedQueryOutputs(logical,masks,batchCount,b,q,selected,work,work+64);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PackPublicationEvents(
 StagedContextPointer ctx,uint32_t history,uint32_t resident,uint32_t bytes,uint32_t b,
 uint32_t q,uint32_t used,uint32_t begin,uint32_t count,
 __gm__ uint32_t* source,__gm__ uint32_t* dest,__ubuf__ uint32_t* tokens,
 __ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* ctl,
 __gm__ int32_t* direct,uint32_t stride){
 for(uint32_t i=threadIdx.x;i<count;i+=512){tokens[i]=asc_ldcg(source+begin+i);slots[i]=asc_ldcg(dest+begin+i);ids[i]=uint16_t(i);}
 asc_syncthreads();
 PackStagedUrmaQuery(history,resident,bytes,b,q,used,0,count,ctx,tokens,ids,slots,ctl,direct,stride);
}
#if KVIP_LOW_MISS_POST_INDEX && KVIP_READY_BATCH_FETCH && !V9_UBMEM
// One exclusive publisher owns these four ranges. Keep the parallel consumer
// checks, but read each raw token, slot and hazard ordinal only once when packing.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PackRowEvents(StagedContextPointer ctx,uint32_t history,uint32_t resident,
        uint32_t bytes,uint32_t b,uint32_t q,uint32_t used,uint32_t begin,uint32_t count,
        __gm__ uint32_t* src,__gm__ uint32_t* dst,__ubuf__ uint32_t* tokens,
        __ubuf__ uint32_t* slots,__ubuf__ uint32_t* ctl) {
    const uint32_t tid=threadIdx.x, per=(count+511)/512, first=tid*per;
    const uint32_t end=first+per<count ? first+per : count;
    static_assert(kTop<=512*32,"one conflict bit per lane item");
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(row->addresses);
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident;
    auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto offsets=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto outSlots=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    const uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(resident,ctx);
    uint32_t conflicts=0,localSafe=0;
    for(uint32_t i=first;i<end;++i) {
        tokens[i]=asc_ldcg(src+begin+i);
        uint32_t slot=asc_ldcg(dst+begin+i);
        slots[i]=slot;
        if(slot>=resident){asc_atomic_or(ctl+1,4U);continue;}
        int32_t ordinal=asc_ldcg(reserved+slot);
        bool conflict=ordinal>=0 && uint32_t(ordinal)<used;
        conflicts|=uint32_t(conflict)<<(i-first);
        localSafe+=!conflict;
    }
    uint32_t nextSafe=StagedPrefix(localSafe,ctl);
    const uint32_t safe=ctl[544+512/warpSize-1];
    uint32_t nextConflict=safe+first-nextSafe;
    for(uint32_t i=first;i<end;++i) {
        uint32_t slot=slots[i],token=tokens[i];
        if(slot>=resident) continue;
        if(token>=history){asc_atomic_or(ctl+1,1U);continue;}
        const uint32_t pos=b*kSlots+q*kTop+
            ((conflicts>>(i-first))&1U ? nextConflict++ : nextSafe++);
        StoreFallbackSource(history,bytes,ctx,b,pos,token,asc_ldcg(addresses+token),blocks,offsets);
        asc_stcg(outSlots+pos,int32_t(ownerBase+slot));
    }
    asc_syncthreads();
    if(!tid) {
        ctl[16]=count;ctl[17]=safe;
        if(!ctl[1])asc_atomic_add(reinterpret_cast<__gm__ uint32_t*>(ctx->count),count);
    }
    asc_threadfence();asc_syncthreads();
}
#endif
__aicore__ inline uint32_t AwaitPublication(__gm__ BatchFlag* flag){
 uint64_t start=GetSystemCycle();
 while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),sizeof(BatchFlag));
  if(flag->ready)return flag->error;
  if(GetSystemCycle()-start>5000000000ULL)return 8;
 }
}
// A dedicated HIGH measurement owns its per-core trace lines after the role
// join; suppress this role's legacy per-batch writes to those same lines.
// The full-AIV HIGH tail has a spare lane for TND output. Both specializations
// use the same descriptor and completion publications as the original worker.
enum class OutputRole { Combined, Descriptors, Tnd };
template<bool TraceRole = true, OutputRole Role = OutputRole::Combined>
__aicore__ inline uint32_t OutputWorker(__gm__ Args* a,uint32_t b){
 constexpr bool pack=Role!=OutputRole::Tnd, tnd=Role!=OutputRole::Descriptors;
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 bool profile=TraceRole && (a->prefetchCount&0x80000000U)!=0;
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+uint64_t(b*kResolveCores+2)*32;
 uint32_t error=AwaitPublication(flags);
 if(profile)ticks[2]=GetSystemCycle();
 if(error){if constexpr(pack)for(uint32_t q=0;q<4;++q){auto f=flags+q+1;f->count=0;f->error=error;f->ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(BatchFlag));}
  if(kEarlyFetchReady && Role==OutputRole::Combined){auto done=flags+kOutputsDoneFlag;done->count=0;done->error=error;done->ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(BatchFlag));}
  return error;}
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);
 auto local=scratch.Get<uint32_t>();auto selected=local,loc=local[kSlots],tokens=local[2*kSlots];
 auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
 auto slots=local[kFinalSlotsOffset],work=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
 // Only counters and output masks need initialization. Context occupies
 // ctl[128:128+ceil(sizeof(Context)/32)*8]; clearing all 2560 words races its MTE2 snapshot when auto
 // synchronization is disabled. Keep the regions disjoint and explicitly
 // order Vector initialization before MTE2/SIMT consumers.
 constexpr uint32_t controlInitWords=64,contextOffset=128;
 static_assert(controlInitWords<=contextOffset,"Control clear overlaps Context");
 if constexpr(pack){
  Duplicate(ctl,uint32_t(0),controlInitWords);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
 }
 auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 bool enabled=tnd && u[20]&&u[21]&&u[22]&&u[23];
 auto logical=enabled?reinterpret_cast<__gm__ int64_t*>(u[3])+uint64_t(reinterpret_cast<__gm__ uint32_t*>(u[9])[b])*uint32_t(u[24]):nullptr;
 GlobalTensor<uint32_t> input;
 if constexpr(pack)CopyContextSnapshot(ctl[contextOffset],ctx);
 auto staged=reinterpret_cast<__ubuf__ Context*>(ctl[contextOffset].GetPhyAddr());
 if constexpr(tnd){
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);DataCopy(selected,input,kSlots);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<MergeQueryBitmaps>(Simt::Dim3{512,1,1},FinalPtr(ctl[1024]),FinalPtr(tokens),0,
   reinterpret_cast<__gm__ int64_t*>(u[20]),reinterpret_cast<__gm__ int64_t*>(u[21]),shape.batch,b,FinalPtr(ctl[kOutputMasksOffset]),enabled);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }else{
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 }
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 auto source=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
 auto dest=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
 uint32_t total=0,workers=(a->prefetchCount>>16)&0xffU,stride=shape.batch*kTop,meta=(3*shape.batch+31)&~31U;
 if(profile)ticks[3]=GetSystemCycle();
 for(uint32_t q=0;q<4;++q){
  if(profile)ticks[4+q*6]=GetSystemCycle();
  if constexpr(tnd){
   if(!error && (pack || enabled)){
    Simt::VF_CALL<PreparePublicationOutputs>(Simt::Dim3{512,1,1},logical,FinalPtr(ctl[kOutputMasksOffset+q*kOutputMasksPerQuery]),shape.batch,b,q,FinalPtr(selected),FinalPtr(work));
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   }
  }
  if(profile)ticks[5+q*6]=GetSystemCycle();
  auto ready=flags+(pack?kRawQueryFlag+q:q+1);
  if(!error)error=AwaitPublication(ready);
  if(profile)ticks[6+q*6]=GetSystemCycle();
  uint32_t count=error?0:ready->count;
  if(count>kTop)error=4;
  if(tnd && !error && (pack || enabled)){
   input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots+q*kTop);
   DataCopy(loc[q*kTop],input,kTop);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   if constexpr(!pack){
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
   }
  }
  if constexpr(pack){
   if(!error){
    auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*workers*(3*stride+meta)+(q*workers+b%workers)*(6*stride+64);
#if KVIP_READY_BATCH_FETCH && !V9_UBMEM
    // The ready worker repacks only committed canonical ranges. An atomic
    // reservation in this whole-shard layout is not a publication boundary.
    direct=nullptr;
#endif
#if KVIP_LOW_MISS_POST_INDEX && KVIP_READY_BATCH_FETCH && !V9_UBMEM
    if constexpr(Role==OutputRole::Descriptors) {
     if(!count){ctl.SetValue(16,0);ctl.SetValue(17,0);}
     else {
      Simt::VF_CALL<PackRowEvents>(Simt::Dim3{512,1,1},staged,shape.history,shape.resident,shape.tokenBytes,
          b,q,row->slots,kPrefetch+total,count,source,dest,FinalPtr(tokens),FinalPtr(slots),FinalPtr(ctl));
      SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
     }
    } else
#endif
    {
    Simt::VF_CALL<PackPublicationEvents>(Simt::Dim3{512,1,1},staged,shape.history,shape.resident,shape.tokenBytes,b,q,row->slots,kPrefetch+total,count,source,dest,FinalPtr(tokens),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),(a->prefetchCount&0x20000000U)?nullptr:direct,stride);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
    }
    if(profile)ticks[7+q*6]=GetSystemCycle();
    if(!error){
     if(!kEarlyFetchReady)PublishQueryOutputs(u,shape.batch,b,q,selected,loc,work);
     SyncFetchDescriptors(ctx,b*kSlots+q*kTop,count);
    }
    if(profile)ticks[8+q*6]=GetSystemCycle();
   }
   PrepareFinalFetch(shape.finalWorkers,shape.finalPostOffset,ctx,b,q,error?0:ctl.GetValue(16),b*kSlots+q*kTop,error,error?0:ctl.GetValue(17));
   auto f=flags+q+1;f->count=count;f->error=error;f->ready=1;
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(BatchFlag));
  }
  total+=count;
  if(profile)ticks[9+q*6]=GetSystemCycle();
  // Get consumes immutable descriptors/FinalPost, not these TND output rows.
  // Keep output DMA and its completion fence after fetch-ready, and retain a
  // separate all-output join before either reader may commit logical metadata.
  if(kEarlyFetchReady && tnd && !error && enabled)PublishQueryOutputs(u,shape.batch,b,q,selected,loc,work);
 }
 if(kEarlyFetchReady && Role==OutputRole::Combined){auto done=flags+kOutputsDoneFlag;done->count=total;done->error=error;done->ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(BatchFlag));}
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),32*8);
 pipe.Destroy();
 return error;
}
