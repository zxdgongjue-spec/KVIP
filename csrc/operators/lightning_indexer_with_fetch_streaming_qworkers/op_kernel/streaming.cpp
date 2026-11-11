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
#include "index_cache_update.h"
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
// The producer and Fetch AIV execute in the same kernel. A kernel boundary no
// longer provides descriptor visibility: release and invalidate the exact
// compact range before publishing/consuming its Q-ready flag.
__aicore__ inline void SyncFetchDescriptors(__gm__ Context* ctx, uint32_t begin, uint32_t count) {
 if(!count)return;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->sourceBlocks)+uint64_t(begin)*4,uint64_t(count)*4);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->sourceOffsets)+uint64_t(begin)*4,uint64_t(count)*4);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->destinations)+uint64_t(begin)*4,uint64_t(count)*4);
 if(ctx->fallbackAddresses)
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->fallbackAddresses)+uint64_t(begin)*8,uint64_t(count)*8);
}
__aicore__ inline void Producer(__gm__ Args* a, uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+b*64;
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 if(profile)ticks[1]=GetSystemCycle();
 uint32_t inputError=0;
 if(!a->queryBitmaps || !a->queryErrors || !a->updateBindings || !a->packedFetch || !ctx->hotValid || !ctx->changeCount || row->slots>kPrefetch ||
    shape.history!=kHistory || shape.resident!=kResident || shape.finalWorkers!=4 || a->maxEvents<kPrefetch+kSlots)inputError=1;
 if(!inputError){auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+(b*4)*32;
  for(uint32_t q=0;q<4;++q){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+q*32),128);inputError|=errors[q*32];}}
 if(inputError){auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;status[0]=inputError;status[1]=0;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  for(uint32_t q=0;q<=5;++q){flags[q].count=0;flags[q].error=inputError;flags[q].ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q),sizeof(BatchFlag));}
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
   Simt::VF_CALL<PackStagedUrmaQuery>(Simt::Dim3{512,1,1},shape.history,shape.resident,shape.tokenBytes,b,q,row->slots,total-added,added,ctx,
    FinalPtr(selected),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl));
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
   if(!error)SyncFetchDescriptors(ctx,b*kSlots+q*kTop,added);
   PrepareFinalFetch(shape.finalWorkers,shape.finalPostOffset,ctx,b,q,error?0:ctl.GetValue(16),
      b*kSlots+q*kTop,error,error?0:ctl.GetValue(17));
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
 flags[5].error=status[0];flags[5].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[5]),sizeof(BatchFlag));
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
 uint32_t total=0,error=0,seen=0;bool denseMode=false;
 auto dense=reinterpret_cast<__gm__ uint32_t*>(a->denseShadow)+uint64_t(b)*shape.history*6;
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
  // Each exchange introduces at most two new positions into either map.
  if(!denseMode && uint64_t(seen+count)*2>MAP){
   if(!a->denseShadow){error=1;break;}
   Simt::VF_CALL<InitDenseShadow>(Simt::Dim3{THREADS,1,1},dense,shape.history,work);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);denseMode=true;
  }
  if(denseMode)Simt::VF_CALL<ApplyDenseBatch>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,dense,shape.history,work);
  else Simt::VF_CALL<ApplyBatch>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,work);
  seen+=count;
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  error=local.GetValue(CONTROL);if(profile)ticks[13+2*phase]=GetSystemCycle();if(phase)total+=count;
 }
 if(!error){auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);uint64_t addressStride=update[7];if(profile)ticks[22]=GetSystemCycle();if(denseMode)Simt::VF_CALL<CommitDenseMetadata>(Simt::Dim3{THREADS,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,dense,reinterpret_cast<__gm__ uint64_t*>(dense+2*shape.history),shape.history);
  else Simt::VF_CALL<CommitMetadata>(Simt::Dim3{THREADS,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,work);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);if(profile)ticks[23]=GetSystemCycle();}
 status[0]=error;status[1]=total;status[2]=local.GetValue(CONTROL+4);if(profile)ticks[24]=GetSystemCycle();
 if(a->updateBindings){auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(u[6]),b,error);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);pipe.Destroy();
 flags[6].error=error;flags[6].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[6]),sizeof(BatchFlag));
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishStreamingResidency(uint32_t resident,uint32_t postOffset,__gm__ Context* ctx,uint32_t b) {
    if(!ctx->hotValid)return;
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*resident;
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(resident,ctx);
    for(uint32_t q=0;q<4;++q){
        auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+b*4+q;
        uint32_t count=asc_ldcg(&post->count),begin=asc_ldcg(&post->descriptor);
        for(uint32_t i=threadIdx.x;i<count;i+=256){
            uint32_t slot=uint32_t(asc_ldcg(destinations+begin+i))-ownerBase;
            if(slot<resident)asc_stcg(valid+slot,uint32_t(1));
        }
    }
    if(ctx->changes && ctx->lruSlots){
        auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
        auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
        for(uint32_t i=threadIdx.x;i<kLruSlots;i+=256)
            if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),uint32_t(1));
    }
    asc_threadfence();
}
// Compact each ready request's descriptors into the worker's private slice.
// The source block indices still refer to the original immutable address table.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PackFetchRanges(
 __gm__ Context* ctx,__gm__ int32_t* packed,uint32_t stride,
 __gm__ uint32_t* ranges,uint32_t rows) {
 auto blocks=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
 auto offsets=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
 auto dest=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
 for(uint32_t row=0;row<rows;++row){
  uint32_t begin=asc_ldcg(ranges+3*row),count=asc_ldcg(ranges+3*row+1),out=asc_ldcg(ranges+3*row+2);
  for(uint32_t i=threadIdx.x;i<count;i+=256){
   asc_stcg(packed+out+i,asc_ldcg(blocks+begin+i));
   asc_stcg(packed+stride+out+i,asc_ldcg(offsets+begin+i));
   asc_stcg(packed+2*stride+out+i,asc_ldcg(dest+begin+i));
  }
 }
 asc_threadfence();asc_syncthreads();
}
constexpr uint32_t kBatchGetInflight = 2048;
// Each worker owns one KVAF queue. It gathers its shard of requests for one Q,
// calls the native BatchGet once per safe/conflicting phase, then acknowledges
// all rows covered by that call. The native scheduler owns WQE and CQ handling.
__aicore__ inline void FetchQueryGroup(__gm__ Args* a,uint32_t q,uint32_t lane,uint32_t workers){
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 if(lane>=shape.batch)return;
 auto allFlags=reinterpret_cast<__gm__ BatchFlag*>(a->flags);
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 uint32_t stride=shape.batch*kTop;
 uint32_t metadataWords=(3*shape.batch+31)&~31U;
 auto packed=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+uint64_t(q*workers+lane)*(3*stride+metadataWords);
 auto ranges=reinterpret_cast<__gm__ uint32_t*>(packed+3*stride);
 // The first 2*batch views include the independent prefetch queues. The
 // following views are paired, exclusively owned by this Q worker.
 uint32_t pair=2*shape.batch+2*(q*workers+lane);
 auto args=GetArgs(ctx,pair);args.jettys_per_core=2;
 args.indices=uint64_t(packed);args.offsets=uint64_t(packed+stride);args.output_indices=uint64_t(packed+2*stride);
 args.miss_count=ctx->count;args.request_capacity=stride;
 if(ctx->fallbackAddresses){args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;args.table_capacity=shape.batch*kSlots;}
 if(!args.batch_get_max_inflight_total)args.batch_get_max_inflight_total=kBatchGetInflight;
 uint64_t error=0;uint32_t completed=0;
 for(uint32_t phase=0;phase<2 && !error;++phase){
  uint32_t size=0,rangeCount=0;
  for(uint32_t b=lane;b<shape.batch;b+=workers){
   auto flags=allFlags+b*kFlagCount;auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   if(profile && !phase)ft[0]=GetSystemCycle();
   uint64_t start=GetSystemCycle();
   while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q+1),sizeof(BatchFlag));
    if(flags[q+1].ready)break;if(GetSystemCycle()-start>5000000000ULL){error=kvaf_device::kCqTimeout;break;}}
   if(profile && !phase)ft[1]=GetSystemCycle();
   auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
   error|=flags[q+1].error|post->error;
   if(post->count>kTop || post->descriptor!=b*kSlots+q*kTop || post->nonconflicting>post->count ||
      post->descriptor>shape.batch*kSlots || post->count>shape.batch*kSlots-post->descriptor)error|=kvaf_device::kInvalidArgs;
   uint32_t count=phase?post->count-post->nonconflicting:post->nonconflicting;
   uint32_t begin=post->descriptor+(phase?post->nonconflicting:0);
   if(phase && count && !error){
    if(profile)ft[5]=GetSystemCycle();start=GetSystemCycle();
    while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),sizeof(BatchFlag));
     if(flags[7].ready){error|=flags[7].error;break;}if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}}
    if(profile)ft[6]=GetSystemCycle();
   }
   if(size>stride || count>stride-size)error|=kvaf_device::kInvalidArgs;
   if(error)break;
   if(count){SyncFetchDescriptors(ctx,begin,count);
    ranges[3*rangeCount]=begin;ranges[3*rangeCount+1]=count;ranges[3*rangeCount+2]=size;
    ++rangeCount;size+=count;}
   if(profile && !phase)ft[2]=GetSystemCycle();
  }
  if(error)break;
  if(rangeCount){
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ranges),uint64_t(rangeCount)*12);
   Simt::VF_CALL<PackFetchRanges>(Simt::Dim3{256,1,1},ctx,packed,stride,ranges,rangeCount);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  uint64_t start=GetSystemCycle(),end=start,postCycles=0,waitCycles=0,groups=0;
#if !V9_SIMULATION
  if(size){
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed),uint64_t(size)*4);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed+stride),uint64_t(size)*4);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(packed+2*stride),uint64_t(size)*4);
   auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
   auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),2*sizeof(*queue));
   uint64_t oldProfile=0;
   if(profile){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));oldProfile=status->profile.magic;
    status->profile.magic=KVAF_AIV_BATCH_GET_PROFILE_MAGIC;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&status->profile),sizeof(status->profile));}
   start=GetSystemCycle();kvaf_device::KvafBatchGetDevice(args,0,0,size);end=GetSystemCycle();
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),2*sizeof(*queue));
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
   if(profile){postCycles=status->profile.post_active_cycles;waitCycles=status->profile.drain_wait_cycles;groups=status->profile.submitted_group_count;
    status->profile.magic=oldProfile;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&status->profile),sizeof(status->profile));}
   error|=status->result;completed+=status->completed_batch_count;
   if(status->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || status->processed_request_count!=size)error|=kvaf_device::kInvalidArgs;
  }
#else
  if(size)error|=kvaf_device::kInvalidArgs;
#endif
  if(profile)for(uint32_t b=lane;b<shape.batch;b+=workers){
   auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   ft[phase?7:3]=start;ft[phase?8:4]=end;ft[12+2*phase]=postCycles;ft[13+2*phase]=waitCycles;
   ft[16+phase]=groups;ft[20]=3; // shared Q-worker group, not per-request time
  }
 }
 for(uint32_t b=lane;b<shape.batch;b+=workers){
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
  post->pending=(b==lane)?completed:0;post->error=uint32_t(error);post->deferred=0;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
  auto flag=allFlags+b*kFlagCount+8+q;flag->error=error;flag->ready=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),sizeof(*flag));
  if(profile){auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   ft[9]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ft),24*8);}
 }
}
__aicore__ inline void FinalizeStreaming(__gm__ Args* a,uint32_t b){
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;uint64_t error=0;
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+96;
 if(profile)ft[0]=GetSystemCycle();
 // This request's resolver core owns its prefetch CQ; Q workers only wait
 // on this request's flag when a conflicting range exists.
 uint64_t polls=0,drain=0,arrival=0;
 CompletePrefetch(ctx,b,true,profile?&polls:nullptr,profile?&drain:nullptr,profile?&arrival:nullptr);
 if(profile){ft[4]=GetSystemCycle();ft[5]=polls;ft[6]=drain;ft[7]=arrival;}error|=row->error;flags[7].error=error;flags[7].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),sizeof(BatchFlag));
 for(uint32_t f=6;f<12;++f){uint64_t start=GetSystemCycle();while(true){
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+f),sizeof(BatchFlag));
  if(flags[f].ready){error|=flags[f].error;break;}
  if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}}}
 auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*4);
 auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 auto updateStatus=reinterpret_cast<__gm__ uint32_t*>(update[6])+b;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(updateStatus),sizeof(uint32_t));
 uint32_t updateError=*updateStatus,total=0,completed=0,deferred=0;
 uint64_t lastHead=0,lastCq=0;
 error|=row->error;error|=info[0]?kvaf_device::kInvalidArgs:0;
 for(uint32_t q=0;q<shape.finalWorkers;++q){
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
  total+=post->count;completed+=post->pending;error|=post->error;
  if(post->deferred){++deferred;error|=kvaf_device::kInvalidArgs;}
  uint32_t workers=(a->prefetchCount>>16)&0xffU;
  uint32_t pair=2*shape.batch+2*(q*workers+b%workers);
  auto queueArgs=GetArgs(ctx,pair);
  auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(queueArgs.queue_state);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
  lastHead=queue->sq_producer_index;lastCq=queue->cq_consumer_index;
  post->pending=0;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
 }
 if(row->pending || total!=info[1])error|=kvaf_device::kInvalidArgs;
 if(profile)ft[1]=GetSystemCycle();
 if(!error && !updateError){
  Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},shape.resident,shape.finalPostOffset,ctx,b);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 if(profile)ft[2]=GetSystemCycle();
 info[0]|=uint32_t(error);info[3]=deferred?1:0;info[4]=completed;
 if(!error)row->finalPending=0;
 auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+b*2;
 kvaf_device::WriteStatus(status,error,total,completed,lastHead,lastCq,UINT64_MAX,0);
 kvaf_device::WriteStatus(status+1,0,0,0,0,0,UINT64_MAX,0);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),2*sizeof(*status));
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
 if(error || updateError){
  Simt::VF_CALL<CompleteEarlyUpdate<256>>(Simt::Dim3{256,1,1},update,b,error);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 if(profile){ft[3]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ft),32*8);}
}

#endif
extern "C" __global__ __aicore__ void v9_streaming_resolve_update(GM_ADDR arg) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  auto a=reinterpret_cast<__gm__ Args*>(arg);uint32_t block=GetBlockIdx();
  auto shape=LoadRuntimeShape(reinterpret_cast<__gm__ Context*>(a->context));
  uint32_t workers=(a->prefetchCount>>16)&0xffU;
  if(workers!=1 && workers!=2 && workers!=4)return;
  if(block<shape.batch*2){uint32_t b=block/2;if(block%2==0){Producer(a,b);FinalizeStreaming(a,b);}else Consumer(a,b);}
  else {uint32_t lane=block-shape.batch*2,q=lane/workers,part=lane%workers;
   if(q<4 && part<workers){
    FetchQueryGroup(a,q,part,workers);
   }}
 }
#endif
}

// Diagnostic calibration runs separately from all benchmark samples.
extern "C" __global__ __aicore__ void v9_counter_calibration(GM_ADDR output) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  auto values=reinterpret_cast<__gm__ uint64_t*>(output);
  uint64_t start=GetSystemCycle(),now=start;
  while(now-start<20000000ULL)now=GetSystemCycle();
  values[0]=start;values[1]=now;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(values),32);
 }
#endif
}

// Standalone dispatch uses the identical Q worker body as the fused kernel.
// 0..3: one Q across two AIVs; 4: all four Qs, two AIVs each.
extern "C" __global__ __aicore__ void v9_query_fetch_only(GM_ADDR arg,uint32_t query) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  uint32_t block=GetBlockIdx();
  if(block<(query==4?8U:2U) && query<5){
   uint32_t workers=2;
   uint32_t q=query==4?block/2:query;
   uint32_t lane=query==4?block%2:block;
   FetchQueryGroup(reinterpret_cast<__gm__ Args*>(arg),q,lane,workers);
  }
 }
#endif
}

// Native reference: a real KVAF BatchGet dispatch across 2 or 8 AIVs.
// Inputs are compacted before timing, as in native LIDU.
extern "C" __global__ __aicore__ void v9_native_fetch_only(GM_ADDR arg,GM_ADDR reference,uint32_t query) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  uint32_t core=GetBlockIdx(),cores=query==4?8U:2U;
  if(core<cores && query<5){
   auto a=reinterpret_cast<__gm__ Args*>(arg);
   auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
   auto ref=reinterpret_cast<__gm__ uint64_t*>(reference);
   // Queue pairs are contiguous; KVAF splits the compacted descriptors itself.
   auto args=GetArgs(ctx,2*shape.batch);
   args.indices=ref[0];args.offsets=ref[1];args.output_indices=ref[2];
   args.miss_count=ref[3];args.core_status=ref[4];
   args.request_capacity=shape.batch*kSlots;args.block_count=cores;args.jettys_per_core=2;
   if(ctx->fallbackAddresses){args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;args.table_capacity=shape.batch*kSlots;}
   if(!args.batch_get_max_inflight_total)args.batch_get_max_inflight_total=kBatchGetInflight;
   auto queues=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state)+core*2;
   auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status)+core;
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queues),2*sizeof(*queues));
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
   kvaf_device::KvafBatchGetDevice(args,core);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queues),2*sizeof(*queues));
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
  }
 }
#endif
}
