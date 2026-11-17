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
// Publish all metadata before the group-ready counter; no per-request polling
// is needed by the direct URMA consumer after it acquires the completed group.
template<class ContextPointer>
__simt_callee__ inline void PublishPreparedQueryBody(
 ContextPointer ctx,__gm__ BatchFlag* flags,uint32_t b,uint32_t q,uint32_t postOffset,
 uint32_t count,uint32_t safe,uint32_t error,__gm__ uint32_t* counters){
 uint32_t tid=threadIdx.x;
 if(count>kTop || safe>count)error|=16U;
 auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+b*4+q;
 if(tid<8){
  uint32_t value=tid==0?(error?0U:count):tid==2?error:tid==3?uint32_t(count && !error):
      tid==4?b*kSlots+q*kTop:tid==5?(error?0U:safe):0U;
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(post)+tid,value);
 }
 if(flags && tid<3){auto f=flags+b*kFlagCount+q+1;
  if(tid==1)asc_stcg(&f->count,count);if(tid==2)asc_stcg(&f->error,error);}
 if(counters && !tid && error)asc_atomic_or(counters+kGroupErrorWord,error);
 asc_threadfence();asc_syncthreads();
 if(!tid){
  if(flags)asc_stcg(&flags[b*kFlagCount+q+1].ready,1U);
  if(counters)asc_atomic_add(counters+kGroupReadyWord,1U);
 }
 asc_threadfence();asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void PublishPreparedQuery(
 __gm__ Context* ctx,__gm__ BatchFlag* flags,uint32_t b,uint32_t q,uint32_t postOffset,
 uint32_t count,uint32_t safe,uint32_t error,__gm__ uint32_t* counters){
 PublishPreparedQueryBody(ctx,flags,b,q,postOffset,count,safe,error,counters);
}
__aicore__ inline __gm__ uint32_t* QueryGroupCounters(__gm__ Args* a,uint32_t batch,uint32_t b,uint32_t q){
#if !V9_UBMEM
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
 if(!(a->prefetchCount&0x20000000U) && ctx->fallbackAddresses && a->packedFetch){
  uint32_t workers=(a->prefetchCount>>16)&0xffU,stride=batch*kTop,meta=(3*batch+31)&~31U;
  auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*workers*(3*stride+meta)+(q*workers+b%workers)*(6*stride+64);
  return reinterpret_cast<__gm__ uint32_t*>(direct+6*stride);
 }
#endif
 return nullptr;
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void PackPublicationEvents(
 StagedContextPointer ctx,uint32_t history,uint32_t resident,uint32_t bytes,uint32_t b,
 uint32_t q,uint32_t used,uint32_t begin,uint32_t count,
 __gm__ uint32_t* source,__gm__ uint32_t* dest,__ubuf__ uint32_t* tokens,
 __ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* ctl,
 __gm__ int32_t* direct,uint32_t stride,__gm__ uint64_t* compactAddresses){
 for(uint32_t i=threadIdx.x;i<count;i+=Threads){tokens[i]=asc_ldcg(source+begin+i);slots[i]=asc_ldcg(dest+begin+i);ids[i]=uint16_t(i);}
 asc_syncthreads();
 PackStagedUrmaQuery<Threads>(history,resident,bytes,b,q,used,0,count,ctx,tokens,ids,slots,ctl,direct,stride,compactAddresses);
 if(compactAddresses)PublishPreparedQueryBody(ctx,static_cast<__gm__ BatchFlag*>(nullptr),b,q,
  ctx->shape.finalPostOffset,ctl[16],ctl[17],ctl[1],reinterpret_cast<__gm__ uint32_t*>(direct+6*stride));
}
__aicore__ inline uint32_t AwaitPublication(__gm__ BatchFlag* flag){
 uint64_t start=GetSystemCycle();
 while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),kBatchFlagPayloadBytes);
  if(flag->ready)return flag->error;
  if(GetSystemCycle()-start>5000000000ULL)return 8;
 }
}
__aicore__ inline void OutputWorker(__gm__ Args* a,uint32_t b){
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 bool profile=(a->prefetchCount&0x80000000U)!=0;
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+uint64_t(b*kResolveCores+2)*32;
 uint32_t error=AwaitPublication(flags);
 if(profile)ticks[2]=GetSystemCycle();
 if(error){for(uint32_t q=0;q<4;++q){
  Simt::VF_CALL<PublishPreparedQuery>(Simt::Dim3{32,1,1},ctx,reinterpret_cast<__gm__ BatchFlag*>(a->flags),b,q,shape.finalPostOffset,0,0,error,QueryGroupCounters(a,shape.batch,b,q));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }return;}
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);
 auto local=scratch.Get<uint32_t>();auto selected=local,loc=local[kSlots],tokens=local[2*kSlots];
 auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
 auto slots=local[kFinalSlotsOffset],work=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
 // Only counters and output masks need initialization. Context occupies
 // ctl[128:208]; clearing all 2560 words races its MTE2 snapshot when auto
 // synchronization is disabled. Keep the regions disjoint and explicitly
 // order Vector initialization before MTE2/SIMT consumers.
 constexpr uint32_t controlInitWords=64,contextOffset=128;
 static_assert(controlInitWords<=contextOffset,"Control clear overlaps Context");
 Duplicate(ctl,uint32_t(0),controlInitWords);
 SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
 auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 bool enabled=u[20]&&u[21]&&u[22]&&u[23];
 auto logical=enabled?reinterpret_cast<__gm__ int64_t*>(u[3])+uint64_t(reinterpret_cast<__gm__ uint32_t*>(u[9])[b])*uint32_t(u[24]):nullptr;
 GlobalTensor<uint32_t> input;
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx));DataCopy(ctl[contextOffset],input,sizeof(Context)/4);
 auto staged=reinterpret_cast<__ubuf__ Context*>(ctl[contextOffset].GetPhyAddr());
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);DataCopy(selected,input,kSlots);
 SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 Simt::VF_CALL<MergeQueryBitmaps>(Simt::Dim3{512,1,1},FinalPtr(ctl[1024]),FinalPtr(tokens),0,
  reinterpret_cast<__gm__ int64_t*>(u[20]),reinterpret_cast<__gm__ int64_t*>(u[21]),shape.batch,b,FinalPtr(ctl[kOutputMasksOffset]),enabled);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 auto source=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
 auto dest=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
 uint32_t total=0,workers=(a->prefetchCount>>16)&0xffU,stride=shape.batch*kTop,meta=(3*shape.batch+31)&~31U;
 if(profile)ticks[3]=GetSystemCycle();
 for(uint32_t q=0;q<4;++q){
  if(profile)ticks[4+q*6]=GetSystemCycle();
  if(!error){
   // Logical lookup can proceed before this Q has finished choosing slots.
   Simt::VF_CALL<PreparePublicationOutputs>(Simt::Dim3{512,1,1},logical,FinalPtr(ctl[kOutputMasksOffset+q*kOutputMasksPerQuery]),shape.batch,b,q,FinalPtr(selected),FinalPtr(work));
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   if(profile)ticks[5+q*6]=GetSystemCycle();
   error=AwaitPublication(flags+kRawQueryFlag+q);
   if(profile)ticks[6+q*6]=GetSystemCycle();
  }
  bool groupPublished=false;
  uint32_t count=error?0:flags[kRawQueryFlag+q].count;
  if(count>kTop)error=4;
  if(!error){
   input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots+q*kTop);
   // loc is disjoint from the descriptor packing buffers; load it while
   // packing publishes the Fetch group, then join before service output mapping.
   DataCopy(loc[q*kTop],input,kTop);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
   auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*workers*(3*stride+meta)+(q*workers+b%workers)*(6*stride+64);
   auto compactAddresses=QueryGroupCounters(a,shape.batch,b,q)?reinterpret_cast<__gm__ uint64_t*>(reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+(q*workers+b%workers)*(3*stride+meta)):nullptr;
   if(count<=64){
   Simt::VF_CALL<PackPublicationEvents<64>>(Simt::Dim3{64,1,1},staged,shape.history,shape.resident,shape.tokenBytes,b,q,row->slots,kPrefetch+total,count,source,dest,FinalPtr(tokens),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),(a->prefetchCount&0x20000000U)?nullptr:direct,stride,compactAddresses);
   }else{
   Simt::VF_CALL<PackPublicationEvents<512>>(Simt::Dim3{512,1,1},staged,shape.history,shape.resident,shape.tokenBytes,b,q,row->slots,kPrefetch+total,count,source,dest,FinalPtr(tokens),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),(a->prefetchCount&0x20000000U)?nullptr:direct,stride,compactAddresses);
   }
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);groupPublished=compactAddresses!=nullptr;
   if(profile)ticks[7+q*6]=GetSystemCycle();
   WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   if(!error){PublishQueryOutputs(u,shape.batch,b,q,selected,loc,work);
    if(!compactAddresses)SyncFetchDescriptors(ctx,b*kSlots+q*kTop,count);}
   if(profile)ticks[8+q*6]=GetSystemCycle();
  }
  auto groupCounters=QueryGroupCounters(a,shape.batch,b,q);
  if(groupCounters && !groupPublished){
   Simt::VF_CALL<PublishPreparedQuery>(Simt::Dim3{32,1,1},ctx,reinterpret_cast<__gm__ BatchFlag*>(a->flags),b,q,shape.finalPostOffset,0,0,error?error:16,groupCounters);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }else{
   if(!groupCounters)PrepareFinalFetch(shape.finalWorkers,shape.finalPostOffset,ctx,b,q,error?0:ctl.GetValue(16),b*kSlots+q*kTop,error,error?0:ctl.GetValue(17));
   auto f=flags+q+1;f->count=count;f->error=error;f->ready=1;
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),kBatchFlagPayloadBytes);
  }
  total+=count;
  if(profile)ticks[9+q*6]=GetSystemCycle();
 }
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),32*8);
 pipe.Destroy();
}
