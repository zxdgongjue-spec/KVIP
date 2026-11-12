// SPDX-License-Identifier: Apache-2.0
// Optional publication AIV. Raw change events remain owned by Resolver; this
// worker owns service outputs, FinalPost and existing KVAF descriptor publication.
#pragma once
#if KVIP_OUTPUT_WORKER
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
__aicore__ inline uint32_t AwaitPublication(__gm__ BatchFlag* flag){
 uint64_t start=GetSystemCycle();
 while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),sizeof(BatchFlag));
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
 if(error){for(uint32_t q=0;q<4;++q){auto f=flags+q+1;f->count=0;f->error=error;f->ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(BatchFlag));}return;}
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
  uint32_t count=error?0:flags[kRawQueryFlag+q].count;
  if(count>kTop)error=4;
  if(!error){
   input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots+q*kTop);
   DataCopy(loc[q*kTop],input,kTop);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*workers*(3*stride+meta)+(q*workers+b%workers)*(6*stride+64);
   Simt::VF_CALL<PackPublicationEvents>(Simt::Dim3{512,1,1},staged,shape.history,shape.resident,shape.tokenBytes,b,q,row->slots,kPrefetch+total,count,source,dest,FinalPtr(tokens),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),(a->prefetchCount&0x20000000U)?nullptr:direct,stride);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
   if(profile)ticks[7+q*6]=GetSystemCycle();
   if(!error){PublishQueryOutputs(u,shape.batch,b,q,selected,loc,work);SyncFetchDescriptors(ctx,b*kSlots+q*kTop,count);}
   if(profile)ticks[8+q*6]=GetSystemCycle();
  }
  PrepareFinalFetch(shape.finalWorkers,shape.finalPostOffset,ctx,b,q,error?0:ctl.GetValue(16),b*kSlots+q*kTop,error,error?0:ctl.GetValue(17));
  total+=count;
  auto f=flags+q+1;f->count=count;f->error=error;f->ready=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(BatchFlag));
  if(profile)ticks[9+q*6]=GetSystemCycle();
 }
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),32*8);
 pipe.Destroy();
}
#endif
