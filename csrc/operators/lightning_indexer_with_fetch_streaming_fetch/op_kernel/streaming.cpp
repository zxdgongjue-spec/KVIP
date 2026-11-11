// SPDX-License-Identifier: Apache-2.0
// Experimental two-AIV in-kernel producer/consumer: no KV transport.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "resolver.h"
#include "resolver_staged_dma2.h"
#include "staged_urma_pack.h"
#include "staged_outputs.h"
#include "pipeline_types.h"
#include "shadow_exchange_parallel.h"
using namespace AscendC;
using namespace rotating;
using namespace pipeline_ab;
#include "staged_fetch_worker.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishQ(
 __ubuf__ uint32_t* selected,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
 __gm__ uint32_t* source,__gm__ uint32_t* dest,uint32_t begin,uint32_t end) {
 for(uint32_t i=begin+threadIdx.x;i<end;i+=512){asc_stcg(source+kPrefetch+i,selected[ids[i]]);asc_stcg(dest+kPrefetch+i,slots[i]);}
 asc_threadfence();asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishPrefetch(
 __gm__ uint32_t* source,__gm__ uint32_t* dest,__gm__ uint32_t* outSource,__gm__ uint32_t* outDest,uint32_t count) {
 for(uint32_t i=threadIdx.x;i<count;i+=512){asc_stcg(outSource+i,asc_ldcg(source+i));asc_stcg(outDest+i,asc_ldcg(dest+i));}
 asc_threadfence();asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void MergeQueryBitmaps(__ubuf__ uint32_t* output,__ubuf__ uint32_t* other,uint32_t words) {
 for(uint32_t i=threadIdx.x;i<words;i+=512)output[i]|=other[i]|other[words+i]|other[2*words+i];
 asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void PublishStreamingUpdateStatus(__gm__ uint32_t* status,uint32_t b,uint32_t error) {
 if(!threadIdx.x)asc_stcg(status+b,error);
 asc_threadfence();
}
__aicore__ inline void PublishQueryOutputs(__gm__ uint64_t* update,uint32_t batchCount,uint32_t batch,uint32_t query,
 LocalTensor<uint32_t> selected,LocalTensor<uint32_t> loc,LocalTensor<uint32_t> work) {
 if(!update[20]||!update[21]||!update[22]||!update[23])return;
 auto values=work[64];
 Duplicate(work,uint32_t(0),8);PipeBarrier<PIPE_V>();
 Simt::VF_CALL<PrepareStagedQueryOutputs>(Simt::Dim3{512,1,1},update,batchCount,batch,query,
  FinalPtr(selected),FinalPtr(work),FinalPtr(values));
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 GlobalTensor<uint32_t> logicalOut,slotOut;
 logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[22]));
 slotOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[23]));
 const uint32_t words=(batchCount*4+31)/32;
 for(uint32_t w=0;w<words;++w){uint32_t mask=work.GetValue(w);while(mask){
  uint32_t row=w*32+__builtin_ctz(mask);
  DataCopy(logicalOut[row*kTop],values,kTop);DataCopy(slotOut[row*kTop],loc[query*kTop],kTop);
  mask&=mask-1;
 }}
 uint32_t pad0=work.GetValue(2),pad1=words>1?work.GetValue(3):0;
 if(pad0||pad1){
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  Duplicate(values,UINT32_MAX,kTop);SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  for(uint32_t w=0;w<words;++w){uint32_t mask=w?pad1:pad0;while(mask){
   uint32_t row=w*32+__builtin_ctz(mask);
   DataCopy(logicalOut[row*kTop],values,kTop);DataCopy(slotOut[row*kTop],values,kTop);
   mask&=mask-1;
  }}
 }
 SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
__aicore__ inline void Producer(__gm__ Args* a, uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+b*64;
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 if(profile)ticks[1]=GetSystemCycle();
 uint32_t inputError=0;
 if(!a->queryBitmaps || !a->queryErrors || !a->updateBindings || !ctx->hotValid || !ctx->changeCount || row->slots>kPrefetch ||
    shape.history!=kHistory || shape.resident!=kResident || shape.finalWorkers!=4 || a->maxEvents<kPrefetch+kSlots)inputError=1;
 if(!inputError){auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+(b*4)*32;
  for(uint32_t q=0;q<4;++q){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+q*32),128);inputError|=errors[q*32];}}
 if(inputError){auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;status[0]=inputError;status[1]=0;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  for(uint32_t q=0;q<kFlagCount;++q){flags[q].count=0;flags[q].error=inputError;flags[q].ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q),sizeof(BatchFlag));}
  return;}
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
 Simt::VF_CALL<PublishPrefetch>(Simt::Dim3{512,1,1},reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*10240,
  reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*10240,reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents,
  reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents,row->slots);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 flags[0].count=row->slots;flags[0].ready=1;if(profile)ticks[0]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[0]),sizeof(BatchFlag));
 auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
 auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
 auto slots=local[kFinalSlotsOffset],work=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
 GlobalTensor<uint32_t> input;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);DataCopy(selected,input,kSlots);
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);DataCopy(loc,input,kSlots);
 uint32_t words=(shape.resident+31)/32;
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(b)*4*words);
 DataCopy(ctl[1024],input,words);
 for(uint32_t q=1;q<4;++q)DataCopy(keys[(q-1)*words],input[q*words],words);
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);DataCopy(work[4096],input,kLruSlots);
 SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 Simt::VF_CALL<MergeQueryBitmaps>(Simt::Dim3{512,1,1},FinalPtr(ctl[1024]),FinalPtr(keys),words);
 PipeBarrier<PIPE_V>();
 if(profile)ticks[2]=GetSystemCycle();
 Simt::VF_CALL<StagedInitializeReady>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),FinalPtr(ctl));
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(profile)ticks[3]=GetSystemCycle();
 GlobalTensor<uint32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
 auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
 for(uint32_t q=0;q<4 && !ctl.GetValue(1);++q){
  if(profile)ticks[10+q*5]=GetSystemCycle();
  Simt::VF_CALL<StagedQuery>(Simt::Dim3{512,1,1},q,shape.history,shape.resident,shape.tokenBytes,b,row->slots,ctx,
   FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(work),FinalPtr(ctl));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  if(profile)ticks[11+q*5]=GetSystemCycle();
  uint32_t total=ctl.GetValue(0),added=ctl.GetValue(2+q),error=ctl.GetValue(1);
  if(!error && a->updateBindings){
   PublishQueryOutputs(reinterpret_cast<__gm__ uint64_t*>(a->updateBindings),shape.batch,b,q,selected,loc,work);
  }
  if(profile)ticks[12+q*5]=GetSystemCycle();
  if(!error){
   Simt::VF_CALL<PackStagedUrmaQuery>(Simt::Dim3{512,1,1},shape.history,shape.resident,shape.tokenBytes,b,row->slots,total-added,added,ctx,
    FinalPtr(selected),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl));
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
   PrepareFinalFetch(shape.finalWorkers,shape.finalPostOffset,ctx,b,q,error?0:ctl.GetValue(16),
      b*kSlots+total-added,error,error?0:ctl.GetValue(17));
  }
  if(profile)ticks[13+q*5]=GetSystemCycle();
  if(!error) Simt::VF_CALL<PublishQ>(Simt::Dim3{512,1,1},FinalPtr(selected),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),
    reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents,reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents,total-added,total);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(output[q*kTop],loc[q*kTop],kTop);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  if(profile)ticks[14+q*5]=GetSystemCycle();
  flags[q+1].count=added;flags[q+1].error=error;flags[q+1].ready=1;
  if(profile)ticks[4+q]=GetSystemCycle();status[4+q]=added;status[8+q]=total;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[q+1]),sizeof(BatchFlag));
 }
 status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);status[2]=b*kSlots;status[3]=0;status[4]=0;
 row->finalBegin=0;row->finalRequests=status[1];row->finalHead=0;row->finalPending=status[1]?1:0;row->finalEnd=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
 // Publish failure to every remaining reader, never leave the consumer waiting.
 for(uint32_t q=0;q<4;++q)if(!flags[q+1].ready){flags[q+1].count=0;flags[q+1].error=status[0]?status[0]:1;flags[q+1].ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[q+1]),sizeof(BatchFlag));}
 if(profile)ticks[8]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);
 pipe.Destroy();
}
__aicore__ inline void Consumer(__gm__ Args* a, uint32_t b) {
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;auto status=reinterpret_cast<__gm__ uint32_t*>(a->workerStatus)+b*32;
 // Producer and consumer must not write the same cache lines.
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+b*64+32;auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,SCRATCH_WORDS*4);auto local=scratch.Get<uint32_t>();
 auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
 uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
 if(profile)ticks[10]=GetSystemCycle();Simt::VF_CALL<InitShadow>(Simt::Dim3{THREADS,1,1},work);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 uint32_t total=0,error=0;
 for(uint32_t phase=0;phase<5 && !error;++phase){
  {uint64_t wait=GetSystemCycle();while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[phase]),sizeof(BatchFlag));if(flags[phase].ready)break;
   if(GetSystemCycle()-wait>5000000000ULL){error=8;break;}}}
  if(error)break;
  uint32_t count=flags[phase].count;
  if(flags[phase].error){error=flags[phase].error;break;}
  uint32_t begin=phase?kPrefetch+total:0;
  auto source=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
  auto dest=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
  if(profile)ticks[12+2*phase]=GetSystemCycle();
  Simt::VF_CALL<ApplyBatch>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,work);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  error=local.GetValue(CONTROL);if(profile)ticks[13+2*phase]=GetSystemCycle();if(phase)total+=count;
 }
 if(!error){auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);uint64_t addressStride=update[7];if(profile)ticks[22]=GetSystemCycle();Simt::VF_CALL<CommitMetadata>(Simt::Dim3{THREADS,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,work);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);if(profile)ticks[23]=GetSystemCycle();}
 status[0]=error;status[1]=total;status[2]=local.GetValue(CONTROL+4);if(profile)ticks[24]=GetSystemCycle();
 if(a->updateBindings){auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(u[6]),b,error);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);pipe.Destroy();
}
// One AIV per request consumes Q-ready descriptors. The prefetch queue is
// drained by this AIV only; each Q's final queue has a single owner.
__aicore__ inline void FetchQueries(__gm__ Args* a,uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 uint64_t error=0;bool joined=false;
 // Diagnostic-only isolation: keep the per-Q packing but start submission
 // after Q4 publication. This uses the existing mailbox, not a host sync.
 if(a->prefetchCount & 0x40000000U){
  uint64_t start=GetSystemCycle();
  while(true){
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+4),sizeof(BatchFlag));
   if(flags[4].ready)break;
   if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}
  }
 }
 for(uint32_t q=0;q<4 && !error;++q) {
  uint64_t start=GetSystemCycle();
  while(true) {
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q+1),sizeof(BatchFlag));
   if(flags[q+1].ready)break;
   if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}
  }
  if(!flags[q+1].ready)break;
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
  error|=flags[q+1].error|post->error;
  if(post->nonconflicting>post->count)error|=kvaf_device::kInvalidArgs;
  auto args=GetArgs(ctx,FinalQueue(shape.batch,shape.finalWorkers,b,q));
  args.indices=ctx->sourceBlocks;args.offsets=ctx->sourceOffsets;
  args.output_indices=ctx->destinations;args.miss_count=ctx->count;
  args.request_capacity=shape.batch*kSlots;
  if(ctx->fallbackAddresses){args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;args.table_capacity=shape.batch*kSlots;}
  uint32_t completed=0;
  for(uint32_t phase=0;phase<2;++phase){
   uint32_t count=phase?post->count-post->nonconflicting:post->nonconflicting;
   uint32_t begin=post->descriptor+(phase?post->nonconflicting:0);
   if(phase && count && !joined){CompletePrefetch(ctx,b);joined=true;error|=row->error;}
#if !V9_SIMULATION
   if(count && !error){kvaf_device::KvafBatchGetDevice(args,0,begin,count);
    auto done=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
    error|=done->result;completed=done->completed_batch_count;
    if(done->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || done->processed_request_count!=count)error|=kvaf_device::kInvalidArgs;
   }
#else
   if(count)error|=kvaf_device::kInvalidArgs;
#endif
  }
  post->pending=completed;post->error=uint32_t(error);post->deferred=0;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
 }
 if(!joined){CompletePrefetch(ctx,b);error|=row->error;}
 if(error){row->error|=uint32_t(error);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);}
}

#endif
extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {auto a=reinterpret_cast<__gm__ Args*>(arg);uint32_t block=GetBlockIdx(),b=block/3;if(b<LoadRuntimeShape(reinterpret_cast<__gm__ Context*>(a->context)).batch){if(block%3==0)Producer(a,b);else if(block%3==1)Consumer(a,b);else FetchQueries(a,b);}}
#endif
}
