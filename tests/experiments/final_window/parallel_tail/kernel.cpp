#include "window_types.h"
#if defined(__CCE_AICORE__)
#include "shadow_exchange_parallel.h"
using namespace pipeline_ab;
// Reuse dead TopK scratch: the original hash checks and swap plans need 20.5 KiB.
constexpr uint32_t TAIL_SRCSET=0, TAIL_DSTSET=2048, TAIL_PLAN_SRC=4096,
 TAIL_PLAN_DST=4608, TAIL_CONTROL=5120;
#include "helpers.h"
#include "emit.h"
#include "parallel_post.h"
// Per-request fences occupy independent cache lines. No all-device barrier.
__aicore__ inline void Signal(__gm__ uint32_t* flags,uint32_t n,uint32_t generation){flags[n*32]=generation;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+n*32),128);}
__aicore__ inline void Await(__gm__ uint32_t* flags,uint32_t n,uint32_t generation){auto p=reinterpret_cast<__gm__ volatile uint32_t*>(flags+n*32);do{kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+n*32),128);}while(*p<generation);}
// Simulation-only CQ latency. SYS_CNT-to-trace calibration is verified in report.
constexpr uint64_t kSimCqDelayTicks=9000;
__simt_callee__ inline void RetireShared(
 __gm__ rotating::Context* ctx,uint32_t begin,uint32_t count,uint32_t history){
 auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes);
 auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping);
 for(uint32_t i=begin+threadIdx.x;i<begin+count;i+=THREADS){
  uint32_t token=asc_ldcg(changes+i);
  if(token<history)asc_stcg(mapping+token,UINT32_MAX);
  asc_stcg(changes+i,UINT32_MAX);
 }
}

__simt_callee__ inline void RetirePriorJournal(__gm__ rotating::Context* ctx,uint32_t history,uint32_t q,uint32_t count){
 auto journal=reinterpret_cast<__gm__ rotating::MappingJournal*>(ctx->mappingJournal);
 if(!journal)return;
 auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping);
 // Count was snapshotted before the existing all-output-reader rendezvous.
 uint32_t begin=count*q/4,end=count*(q+1)/4;
 for(uint32_t i=begin+threadIdx.x;i<end;i+=THREADS){uint32_t token=asc_ldcg(journal->tokens+i);if(token<history)asc_stcg(mapping+token,UINT32_MAX);}
 if(!q && !threadIdx.x)asc_stcg(&journal->count,0U);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void UpdateIndex(
 __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t count,
 uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,
 __gm__ uint64_t* addresses,__gm__ uint64_t* logical,__ubuf__ uint32_t* work,__gm__ rotating::Context* ctx,uint32_t total,uint32_t q,uint32_t first,uint32_t initialCount,__gm__ uint32_t* header,__gm__ uint32_t* progress,uint32_t priorJournalCount){
 if(count)SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work);
 asc_syncthreads();
 if(count && !work[TAIL_CONTROL])SwapColdMetadata(addresses,logical,count,work);
 asc_syncthreads();
 RetireShared(ctx,first,count,history);
 RetirePriorJournal(ctx,history,q,priorJournalCount);
 if(!q){if(!threadIdx.x){auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows);asc_stcg(&row->requests,total);asc_stcg(&row->slots,total);}}
 if(threadIdx.x<32){asc_stcg(header+threadIdx.x,0U);asc_stcg(progress+threadIdx.x,0U);}
 asc_threadfence();
}

#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 bool profile=(a->prefetchCount&0x80000000U)!=0;uint64_t postedAt=0;
 uint32_t q=GetBlockIdx();auto flags=reinterpret_cast<__gm__ uint32_t*>(a->flags);auto ts=reinterpret_cast<__gm__ uint64_t*>(a->times)+q*32;
 uint32_t generation=flags[q*32+1]+1;
 reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[q*32]=0;
 uint32_t initialCount=reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)->requests;
 uint32_t priorJournalCount=ctx->mappingJournal?reinterpret_cast<__gm__ volatile rotating::MappingJournal*>(ctx->mappingJournal)->count:0;
 if(profile)ts[0]=GetSystemCycle();
 TPipe pipe;
 using Type=QLICommon::QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,QLICommon::LI_LAYOUT::BSND,QLICommon::LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QLIVector<Type> op;QLICommon::ConstInfo ci{};ci.gSize=32;ci.qHeadNum=32;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=64;ci.s1BaseSize=2;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=s.history;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=s.history/128;
 QLITilingData tiling{};op.SetStreamingWorkspace(arg);op.InitParams(ci,&tiling,false);op.InitBuffers(&pipe,false);op.SetRotationContext(reinterpret_cast<GM_ADDR>(ctx));
 GlobalTensor<uint16_t> scores;op.InitVecWorkspaceTensor(scores);GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;GlobalTensor<int32_t> out,blocks;out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctx->selected));op.InitVecInputTensor(weights,qs,ks,out,blocks);
 op.AllocEventID();if(profile)ts[1]=GetSystemCycle();if(mode==5 && q==3){uint64_t start=GetSystemCycle();while(GetSystemCycle()-start<36000){} }op.TerminalQuery(q);if(profile)ts[2]=GetSystemCycle();Signal(flags,q,generation);
 auto postWork=op.TerminalWork(),postIds=op.TerminalCompact(),postScratch=op.TerminalScratch();
 auto compact=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+q*2048;
 auto mapping=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping);
 uint32_t candidates=reinterpret_cast<__gm__ volatile uint32_t*>(rotating::TopKHeader(ctx))[q*32+6];
 if(profile)ts[3]=GetSystemCycle();
 auto tailBounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds);
 uint32_t tailBegin=tailBounds?tailBounds[0]:0,tailEnd=tailBounds?tailBounds[1]:0;
 uint32_t tailCapacity=ctx->outputStride?uint32_t(ctx->outputStride)-s.resident:0;
 uint32_t counts[4]={0,0,0,0};counts[q]=candidates;
 uint32_t width=(candidates+31)/32*32;
 uint32_t bitmapWords=s.history/32;
 // Consume the original compact UB result directly; compact destinations use postWork.
 // MERGED_OUTPUT_BEFORE_WQE
 if(profile)ts[6]=GetSystemCycle();
 Simt::VF_CALL<PackUniqueLocal>(Simt::Dim3{1024,1,1},reinterpret_cast<__ubuf__ uint32_t*>(postIds.GetPhyAddr()),width,counts[0],counts[1],counts[2],counts[3],q,mapping,reinterpret_cast<__ubuf__ uint32_t*>(postWork.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch.GetPhyAddr()),reinterpret_cast<__gm__ rotating::Row*>(ctx->rows),reinterpret_cast<__gm__ uint64_t*>(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)->addresses),reinterpret_cast<__ubuf__ uint32_t*>(postWork.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postWork[2048].GetPhyAddr()),s.resident,reinterpret_cast<__ubuf__ uint32_t*>(postWork[4096].GetPhyAddr()),reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots),reinterpret_cast<__gm__ uint32_t*>(ctx->changes),reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots),(uint64_t(tailEnd)<<32)|tailBegin,tailCapacity,reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+q*32,reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+q,reinterpret_cast<__ubuf__ uint32_t*>(op.TerminalIds().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords+8].GetPhyAddr()),reinterpret_cast<__gm__ uint64_t*>(a->logical),flags,generation,s.history);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(profile)ts[11]=GetSystemCycle();
 uint32_t own=postScratch.GetValue(bitmapWords);
 uint32_t slotBase=postScratch.GetValue(bitmapWords+1),all=own,offset=0;
 ts[20]=own;ts[21]=all;ts[22]=offset;
 if(profile)ts[12]=GetSystemCycle();
 auto ids=op.TerminalIds();
 GlobalTensor<uint32_t> output,slots;
 output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+q*2048);slots.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+q*2048);
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
 DataCopy(slots,postScratch,2048);DataCopy(output,ids,2048);
 SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
 reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[q*32]|=postScratch.GetValue(bitmapWords+8);
 if(profile)ts[7]=GetSystemCycle();
 auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+q;
 auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
 uint64_t originalHead=uint64_t(postWork.GetValue(6144+18))|(uint64_t(postWork.GetValue(6144+19))<<32);
 uint64_t remote=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)->addresses;
 if(own){
  if(profile)ts[13]=GetSystemCycle();
  rotating::WriteReservedWqes(s.tokenBytes,args->channel_entity,args->queue_state,args->output,postWork[4096],postScratch,postWork,postWork[2048],postWork[6144],own,256,false,nullptr,offset,all,reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+q*32,originalHead);
 }
 if(profile)ts[14]=GetSystemCycle();
 // Logical IDs are emitted with slots after all admissions are visible.
 if(profile)ts[15]=GetSystemCycle();
 queue->sq_producer_index=originalHead+own;
 auto entity=reinterpret_cast<__gm__ uint64_t*>(args->channel_entity);
 *reinterpret_cast<__gm__ uint64_t*>(entity[8])=originalHead+own;
 // Delay software producer-state writeback until outputs have joined.
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(entity[8]),64);
 postedAt=GetSystemCycle();
 if(profile)ts[4]=GetSystemCycle();

 op.FreeEventID();
 if(profile)ts[5]=GetSystemCycle();
 auto local=op.TerminalScratch();auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
 // Publish output readiness and join in the same full-warp task.
 Simt::VF_CALL<WaitFour>(Simt::Dim3{32,1,1},flags,5,generation,nullptr,q,0,true,false,4);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(profile)ts[8]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),64);
 if(!q)ts[30]=reinterpret_cast<__gm__ volatile rotating::MappingJournal*>(ctx->mappingJournal)->count;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows),128);
 uint32_t total=reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)->slots;
 // Previous admissions are assumed updated already in this fixture.
 uint32_t first=initialCount+(total-initialCount)*q/4,last=initialCount+(total-initialCount)*(q+1)/4,n=last-first;
 auto src=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+first;auto dst=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+first;
 Simt::VF_CALL<UpdateIndex>(Simt::Dim3{512,1,1},src,dst,n,s.resident,s.history,reinterpret_cast<__gm__ uint64_t*>(a->keys),reinterpret_cast<__gm__ uint32_t*>(a->scales),reinterpret_cast<__gm__ uint64_t*>(a->addresses),reinterpret_cast<__gm__ uint64_t*>(a->logical),work,ctx,total,q,first,initialCount,reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+q*32,reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+q*32,priorJournalCount);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 uint32_t indexError=n?local.GetValue(TAIL_CONTROL):0;
 reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[q*32]|=(indexError<<4);
 if(profile)ts[16]=GetSystemCycle();
 while(GetSystemCycle()-postedAt<kSimCqDelayTicks){}
 if(profile)ts[17]=GetSystemCycle();
 if(profile)ts[9]=GetSystemCycle();
 flags[q*32+1]=generation;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q*32),128);
 if(profile)ts[10]=GetSystemCycle();
 pipe.Destroy();if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ts),256);
#endif
}
