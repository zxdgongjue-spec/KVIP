#include "terminal_runtime_abi.h"
#include "window_types.h"
#if defined(__CCE_AICORE__)
#include "shadow_exchange_parallel.h"
#include "stage_trace.h"
#if KVIP_WINDOW_FRONTIER
#include "window_frontier.h"
#endif
using namespace pipeline_ab;
// Reuse dead TopK scratch: the original hash checks and swap plans need 20.5 KiB.
constexpr uint32_t TAIL_SRCSET=0, TAIL_DSTSET=2048, TAIL_PLAN_SRC=4096,
 TAIL_PLAN_DST=6656, TAIL_CONTROL=9216;
#include "helpers.h"
#include "overlap_update.h"
#include "capacity.h"
#include "emit.h"
#include "parallel_post.h"
#include "terminal_output_routes.h"
#include "terminal_finish.h"
#include "staged_fetch_worker.h"
namespace rotating {
#include "terminal_transport.h"
}
// Per-request fences occupy independent cache lines. No all-device barrier.
__aicore__ inline void Signal(__gm__ uint32_t* flags,uint32_t n,uint32_t generation){flags[n*32]=generation;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+n*32),128);}
__aicore__ inline void Await(__gm__ uint32_t* flags,uint32_t n,uint32_t generation){auto p=reinterpret_cast<__gm__ volatile uint32_t*>(flags+n*32);do{kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+n*32),128);}while(*p<generation);}
// Simulation-only CQ latency. SYS_CNT-to-trace calibration is verified in report.
constexpr uint64_t kSimCqDelayTicks=9000;
// Diagnostic pass only: preserve the assignment partition before its owner retires it.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void CaptureTerminalAssignments(
 __gm__ uint32_t* sources,__gm__ uint32_t* destinations,__gm__ uint32_t* savedSources,
 __gm__ uint32_t* savedDestinations,uint32_t first,uint32_t count){
 for(uint32_t i=threadIdx.x;i<count;i+=512){
  asc_stcg(savedSources+first+i,asc_ldcg(sources+i));
  asc_stcg(savedDestinations+first+i,asc_ldcg(destinations+i));
 }
 asc_threadfence();
}
__simt_callee__ inline void RetireShared(
 __gm__ rotating::Context* ctx,uint32_t begin,uint32_t count,uint32_t history,uint32_t b){
 auto changes=(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+uint64_t(b)*rotating::kChangeCapacity);
 auto mapping=(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*history);
 for(uint32_t i=begin+threadIdx.x;i<begin+count;i+=THREADS){
  uint32_t token=asc_ldcg(changes+i);
  if(token<history)asc_stcg(mapping+token,UINT32_MAX);
  asc_stcg(changes+i,UINT32_MAX);
 }
}

__simt_callee__ inline void RetirePriorJournal(__gm__ rotating::Context* ctx,uint32_t history,uint32_t q,uint32_t count,uint32_t b){
 auto journal=(reinterpret_cast<__gm__ rotating::MappingJournal*>(ctx->mappingJournal)+uint64_t(b)*1);
 if(!journal)return;
 auto mapping=(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*history);
 // Count was snapshotted before the existing all-output-reader rendezvous.
 uint32_t begin=count*q/4,end=count*(q+1)/4;
 for(uint32_t i=begin+threadIdx.x;i<end;i+=THREADS){uint32_t token=asc_ldcg(journal->tokens+i);if(token<history)asc_stcg(mapping+token,UINT32_MAX);}
 if(!q && !threadIdx.x)asc_stcg(&journal->count,0U);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void UpdateIndex(
 __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t count,
 uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,
 __gm__ uint64_t* addresses,__gm__ uint64_t* logical,__ubuf__ uint32_t* work,__gm__ rotating::Context* ctx,uint32_t total,uint32_t q,uint32_t first,__gm__ uint32_t* header,__gm__ uint32_t* progress,uint32_t priorJournalCount,uint32_t b,__gm__ uint32_t* diagnostic,bool applied=false){
 if(!applied)SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work,addresses,logical,
  ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident:nullptr,
  total,diagnostic-(12+q)*32);
 if(!applied && (work[TAIL_CONTROL]&0x80000000U))return;
 // Validation has already synchronized all workers. Preserve evidence before retirement.
 if(count && !applied && work[TAIL_CONTROL] && !threadIdx.x){
   uint32_t bad=UINT32_MAX,reason=0,source=UINT32_MAX,target=UINT32_MAX,previous=UINT32_MAX;
   for(uint32_t i=0;i<count && bad==UINT32_MAX;++i){
     uint32_t src=work[TAIL_PLAN_SRC+i],dst=work[TAIL_PLAN_DST+i],why=0;
     if(src>=history)why|=2U;if(dst>=resident)why|=4U;
     for(uint32_t j=0;j<i;++j){
       if(src==work[TAIL_PLAN_SRC+j]){why|=8U;previous=first+j;}
       if(dst==work[TAIL_PLAN_DST+j]){why|=16U;previous=first+j;}
     }
     if(why){bad=first+i;reason=why;source=src;target=dst;}
   }
   asc_stcg(diagnostic+12,bad);asc_stcg(diagnostic+13,reason);
   asc_stcg(diagnostic+14,source);asc_stcg(diagnostic+15,target);
   asc_stcg(diagnostic+16,previous);
 }
 // The immutable plan is already visible; key and metadata stores are disjoint.
 // Metadata swaps run alongside key swaps in the same validated task.
 asc_syncthreads();
 if(!applied && !work[TAIL_CONTROL]){
  for(uint32_t i=threadIdx.x;i<count;i+=THREADS){uint32_t src=work[TAIL_PLAN_SRC+i];if(src<resident){auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(rotating::OwnerRow(ctx,b))*resident;asc_stcg(valid+src,0U);}}
 }
 RetireShared(ctx,first,count,history,b);
 RetirePriorJournal(ctx,history,q,priorJournalCount,b);
 if(!q){if(!threadIdx.x){auto row=(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+uint64_t(b)*1);asc_stcg(&row->slots,total);}}
 if(threadIdx.x<32){asc_stcg(header+threadIdx.x,0U);asc_stcg(progress+threadIdx.x,0U);}
 asc_threadfence();
}

#endif
extern "C" __global__ __aicore__ void v9_parallel_terminal(GM_ADDR binding){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
 auto runtime=reinterpret_cast<__gm__ terminal_runtime::Binding*>(binding);
 GM_ADDR arg=reinterpret_cast<GM_ADDR>(runtime->pipeline);
#if KVIP_STAGE_TRACE
 const uint64_t stageTrace=KVIP_TRACE_WORKSPACE(arg);
#endif
 KVIP_TRACE_BEGIN(kernelScope, stageTrace, Terminal, KernelBody, 0, GetBlockIdx());
 KVIP_TRACE_BEGIN(setupScope, stageTrace, Terminal, Setup, 0, GetBlockIdx());
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 bool profile=(runtime->options&1U)!=0;
 uint32_t b=GetBlockIdx()/4,q=GetBlockIdx()%4,query=b*4+q;
#if KVIP_STAGE_TRACE
 const kvip_trace::ScanSample pipeSample{stageTrace, kvip_trace::Terminal, 0, kvip_trace::PipeDetail(b,q)};
#endif
 auto flags=reinterpret_cast<__gm__ uint32_t*>(runtime->flags)+terminal_runtime::RowFlagOffset(b);
 auto ts=reinterpret_cast<__gm__ uint64_t*>(runtime->times)+terminal_runtime::QueryTraceOffset(b,q);
 const uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
 auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 const uint64_t addressStride=update[7];
 auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(runtime->queueArgs)+query;
 auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(ctx->owners+s.finalPostOffset)+query;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q*32),128);
 uint32_t generation=flags[q*32+1]+1;
 uint32_t priorJournalCount=ctx->mappingJournal?(reinterpret_cast<__gm__ volatile rotating::MappingJournal*>(ctx->mappingJournal)+uint64_t(b)*1)->count:0;
 uint32_t priorSlots=(reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+b)->slots;
 if(profile)ts[0]=GetSystemCycle();
 TPipe pipe;
 using Type=QLICommon::QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,QLICommon::LI_LAYOUT::BSND,QLICommon::LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QLIVector<Type> op;QLICommon::ConstInfo ci{};ci.gSize=32;ci.qHeadNum=32;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=64;ci.s1BaseSize=2;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=s.history;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=s.history/128;
 QLITilingData tiling{};op.SetStreamingWorkspace(arg);op.InitParams(ci,&tiling,false);op.InitBuffers(&pipe,false);op.SetRotationContext(reinterpret_cast<GM_ADDR>(ctx));
 GlobalTensor<uint16_t> scores;op.InitVecWorkspaceTensor(scores);GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;GlobalTensor<int32_t> out,blocks;out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctx->selected));op.InitVecInputTensor(weights,qs,ks,out,blocks);
 op.AllocEventID();if(profile)ts[1]=GetSystemCycle();
 KVIP_TRACE_END(setupScope);
 KVIP_TRACE_BEGIN(topkScope, stageTrace, Terminal, TerminalTopK, 0, query);
 uint32_t topkError=op.TerminalQueryRuntime(query);
 KVIP_TRACE_END(topkScope);
 KVIP_TRACE_BEGIN(resolvePrepareScope, stageTrace, Terminal, TerminalResolvePrepare, 0, query);
 uint32_t ownError=topkError;if(profile)ts[2]=GetSystemCycle();Signal(flags,q,generation);
 auto postWork=op.TerminalWork(),postIds=op.TerminalCompact(),postScratch=op.TerminalScratch();
 auto compact=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048;
 auto mapping=(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*s.history);
 uint32_t candidates=reinterpret_cast<__gm__ volatile uint32_t*>(rotating::TopKHeader(ctx))[query*32+6];
 if(profile)ts[3]=GetSystemCycle();
 auto tailBounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds);
 uint32_t tailBegin=tailBounds?tailBounds[b*2]:0,tailEnd=tailBounds?tailBounds[b*2+1]:0;
 uint32_t tailCapacity=ctx->outputStride?uint32_t(ctx->outputStride)-s.resident:0;
 uint32_t counts[4]={0,0,0,0};counts[q]=candidates;
 uint32_t width=(candidates+31)/32*32;
 uint32_t bitmapWords=s.history/32;

 auto activeLru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*rotating::kLruSlots;
 uint32_t slotCapacity=rotating::kLruSlots;bool capacityCounted=false;
 auto headers=reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+b*128;
 KVIP_TRACE_END(resolvePrepareScope);
 for(uint32_t capacityPass=0;capacityPass<2;++capacityPass){
 if(capacityPass){
 KVIP_TRACE_BEGIN(capacityScope, stageTrace, Terminal, TerminalCapacity, capacityPass, query);
 Simt::VF_CALL<PrepareTerminalCapacity>(Simt::Dim3{32,1,1},
  reinterpret_cast<__ubuf__ uint32_t*>(postIds.GetPhyAddr()),candidates,mapping,
  headers,flags,generation,q,priorSlots,
  reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords+16].GetPhyAddr()));
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 uint32_t upper=postScratch.GetValue(bitmapWords+16);
 capacityCounted=postScratch.GetValue(bitmapWords+18)!=0;
 if(capacityCounted){candidates=postScratch.GetValue(bitmapWords+17);counts[q]=candidates;width=(candidates+31)/32*32;}
 if(upper>rotating::kLruSlots){
  auto extended=reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[0])+uint64_t(b)*(rotating::kChangeCapacity+4*((s.resident+31)/32));
  auto savedIds=reinterpret_cast<__gm__ uint32_t*>(ctx->incremental+uint64_t(s.batch)*512+uint64_t(b)*(4*2048*6+1024)+4*2048*2);
  auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident;
  Simt::VF_CALL<ExtendTerminalLru>(Simt::Dim3{512,1,1},activeLru,extended,savedIds,inverse,
   ctx->queryEnds?reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+b*4:nullptr,
   s.resident,upper<rotating::kChangeCapacity?upper:rotating::kChangeCapacity,q,flags,extended+rotating::kChangeCapacity,
   reinterpret_cast<__ubuf__ uint32_t*>(postIds[2048].GetPhyAddr()));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  slotCapacity=postIds.GetValue(2048);activeLru=extended;
 }
 if(profile)ts[19]=GetSystemCycle();
 }
 // Consume the original compact UB result directly; compact destinations use postWork.
 // MERGED_OUTPUT_BEFORE_WQE
 if(profile)ts[6]=GetSystemCycle();
 KVIP_TRACE_BEGIN(resolveScope, stageTrace, Terminal, TerminalResolve, capacityPass, query);
 KVIP_TRACE_BEGIN(resolveVfScope, stageTrace, Terminal, TerminalResolveVf, capacityPass, kvip_trace::PipeDetail(b,q));
 Simt::VF_CALL<PackUniqueLocal>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(postIds.GetPhyAddr()),width,counts[0],counts[1],counts[2],counts[3],q,mapping,reinterpret_cast<__ubuf__ uint32_t*>(postWork.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch.GetPhyAddr()),(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+uint64_t(b)*1),reinterpret_cast<__gm__ uint64_t*>(rotating::RequestAddressTable(ctx,b,owner,s.history)),reinterpret_cast<__ubuf__ uint32_t*>(postWork.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postWork[2048].GetPhyAddr()),s.resident,reinterpret_cast<__ubuf__ uint32_t*>(postWork[4096].GetPhyAddr()),activeLru,slotCapacity,capacityCounted,capacityPass==0,headers,priorSlots,(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+uint64_t(b)*rotating::kChangeCapacity),(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+uint64_t(b)*rotating::kChangeCapacity),(uint64_t(tailEnd)<<32)|tailBegin,tailCapacity,reinterpret_cast<__gm__ uint32_t*>(runtime->templates)+query*32,args,reinterpret_cast<__ubuf__ uint32_t*>(op.TerminalIds().GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords+8].GetPhyAddr()),(reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history),flags,generation,s.history,reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+query*2048,owner*uint32_t(ctx->outputStride),update,s.batch,query);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 KVIP_TRACE_END(resolveVfScope);
 KVIP_TRACE_END(resolveScope);
 if(!postScratch.GetValue(bitmapWords+14))break;
 }
 if(profile)ts[11]=GetSystemCycle();
 uint32_t own=postScratch.GetValue(bitmapWords);
 uint32_t slotBase=postScratch.GetValue(bitmapWords+1),all=own,offset=0;
 if(profile){ts[20]=own;ts[21]=all;ts[22]=slotBase;}
 if(profile)ts[12]=GetSystemCycle();
 KVIP_TRACE_BEGIN(outputScope, stageTrace, Terminal, TerminalOutput, 0, query);
 auto ids=op.TerminalIds();
 GlobalTensor<uint32_t> output,slots;
 output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(runtime->logical)+query*2048);slots.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048);
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
 KVIP_PIPE_SAMPLE(storeScope, pipeSample, TerminalOutputMte3);
 DataCopy(slots,postScratch,2048);
 PublishTerminalOutputs(update,s.batch,query,ids,postScratch,postScratch[bitmapWords+9],ctx->locations+uint64_t(query)*2048*4);
 KVIP_TRACE_END(storeScope);
 uint32_t packError=postScratch.GetValue(bitmapWords+8);ownError|=packError;
 KVIP_TRACE_END(outputScope);
 if(profile)ts[7]=GetSystemCycle();
 auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
 uint64_t originalHead=uint64_t(postWork.GetValue(6144+18))|(uint64_t(postWork.GetValue(6144+19))<<32);
 uint64_t remote=(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+uint64_t(b)*1)->addresses;
 if(own){
  KVIP_TRACE_BEGIN(wqeScope, stageTrace, Terminal, TerminalWqe, 0, query);
  if(profile)ts[13]=GetSystemCycle();
  rotating::WriteReservedWqes(s.tokenBytes,args->channel_entity,args->queue_state,args->output,postWork[4096],postScratch,postWork,postWork[2048],postWork[6144],own,256,false,nullptr,offset,all,reinterpret_cast<__gm__ uint32_t*>(runtime->templates)+query*32,originalHead KVIP_PIPE_ARG);
 }
 if(profile)ts[14]=GetSystemCycle();
 // Logical IDs are emitted with slots after all admissions are visible.
 if(profile)ts[15]=GetSystemCycle();
 KVIP_TRACE_BEGIN(doorbellScope, stageTrace, Terminal, TerminalDoorbell, 0, query);
 rotating::CommitTerminalPost(args,originalHead,own,post);
 KVIP_TRACE_END(doorbellScope);
 if(profile)ts[4]=GetSystemCycle();

 op.FreeEventID();
 if(profile)ts[5]=GetSystemCycle();
 auto local=op.TerminalScratch();auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
 // Publish output readiness and join in the same full-warp task.
 auto allSources=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+uint64_t(b)*rotating::kChangeCapacity;
 if(profile)ts[8]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),64);
 if(profile && !q)ts[30]=(reinterpret_cast<__gm__ volatile rotating::MappingJournal*>(ctx->mappingJournal)+uint64_t(b)*1)->count;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows)+uint64_t(b)*sizeof(rotating::Row),128);
 uint32_t total=postScratch.GetValue(bitmapWords+13);
 // Include the actual prior-prefetch Index work, after all output readers join.
 uint32_t first=total*q/4,last=total*(q+1)/4,n=total<=rotating::kChangeCapacity?last-first:0;
 auto src=(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+uint64_t(b)*rotating::kChangeCapacity)+first;auto dst=(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+uint64_t(b)*rotating::kChangeCapacity)+first;
 bool captureAssignments=profile;
#if KVIP_WINDOW_FRONTIER
 captureAssignments=captureAssignments ||
  kvip_window_frontier::ActiveMode(reinterpret_cast<GM_ADDR>(a))!=kvip_window_frontier::Off;
#endif
 if(captureAssignments){
  Simt::VF_CALL<CaptureTerminalAssignments>(Simt::Dim3{512,1,1},src,dst,
   reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+uint64_t(b)*rotating::kChangeCapacity,
   reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*rotating::kChangeCapacity,first,n);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 if(profile)ts[26]=GetSystemCycle();
 KVIP_TRACE_BEGIN(indexScope, stageTrace, Terminal, TerminalIndex, 0, query);
 uint32_t indexError=0;
 KVIP_PIPE_SAMPLE(updateScope, pipeSample, TerminalIndexVf);
 Simt::VF_CALL<UpdateIndex>(Simt::Dim3{512,1,1},src,dst,n,s.resident,s.history,(reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16),(reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history),(reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride),(reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history),work,ctx,total,q,first,reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+query*32,reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+query*32,priorJournalCount,b,flags+(12+q)*32,false);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 KVIP_TRACE_END(updateScope);
 indexError=total>rotating::kChangeCapacity?1U:local.GetValue(TAIL_CONTROL);

 bool residentUpdate=(indexError&0x80000000U)!=0;
 if(residentUpdate){
  // No old TopK buffer is read after this point. Reuse its UB allocation for
  // the production planner, which snapshots aliases before writing any row.
  pipe.Reset();
  TBuf<TPosition::VECCALC> parallelScratch;
  pipe.InitBuffer(parallelScratch,kParallelIndexWords*sizeof(uint32_t));
  auto parallelLocal=parallelScratch.Get<uint32_t>();
  work=reinterpret_cast<__ubuf__ uint32_t*>(parallelLocal.GetPhyAddr());
  KVIP_PIPE_SAMPLE(applyScope, pipeSample, TerminalIndexVf);
  if(total>kSmallIndexChanges){
  const uint64_t snapshotChanges=s.resident<rotating::kChangeCapacity?s.resident:rotating::kChangeCapacity;
  const uint64_t snapshotWords=snapshotChanges*2*kSnapshotWordsPerRow;
  const uint64_t snapshotStride=uint64_t(s.history)*3>snapshotWords?uint64_t(s.history)*3:snapshotWords;
  Simt::VF_CALL<ApplyParallelIndex<true>>(Simt::Dim3{512,1,1},allSources,dst-first,total,
   s.resident,s.history,q,1U,
   reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16,
   reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,
   reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*s.resident,
   flags,ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident:nullptr,work,reinterpret_cast<__gm__ uint64_t*>(a->denseShadow)+uint64_t(b)*snapshotStride);
  }else{
  Simt::VF_CALL<ApplyParallelIndex<false>>(Simt::Dim3{512,1,1},allSources,dst-first,total,
   s.resident,s.history,q,1U,
   reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16,
   reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,
   reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*s.resident,
   flags,ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident:nullptr,work,static_cast<__gm__ uint64_t*>(nullptr));
  }
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  KVIP_TRACE_END(applyScope);
  indexError=parallelLocal.GetValue(0);
  // Retirement is shared with the normal path; skip the already applied swap.
  KVIP_PIPE_SAMPLE(retireScope, pipeSample, TerminalIndexVf);
  Simt::VF_CALL<UpdateIndex>(Simt::Dim3{512,1,1},src,dst,n,s.resident,s.history,
   reinterpret_cast<__gm__ uint64_t*>(a->keys),reinterpret_cast<__gm__ uint32_t*>(a->scales),
   reinterpret_cast<__gm__ uint64_t*>(a->addresses),reinterpret_cast<__gm__ uint64_t*>(a->logical),
   work,ctx,total,q,first,reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+query*32,
   reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+query*32,priorJournalCount,b,flags+(12+q)*32,true);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 ownError|=(indexError<<4);
 KVIP_TRACE_END(indexScope);
 if(profile)ts[16]=GetSystemCycle();
 KVIP_TRACE_BEGIN(cqScope, stageTrace, Terminal, TerminalGetCq, 0, query);
 uint64_t cqError=rotating::CompleteTerminalPost(args,post);
 KVIP_TRACE_END(cqScope);
 if(profile)ts[24]=GetSystemCycle();
 if(q==1) {
   KVIP_TRACE_BEGIN(prefetchCqScope, stageTrace, Terminal, TerminalPrefetchCq, 0, query);
   rotating::CompleteTerminalPrefetch(ctx,b);
 }
 KVIP_TRACE_BEGIN(finishScope, stageTrace, Terminal, TerminalFinish, 0, query);
 ownError|=uint32_t(cqError);
 if(profile)ts[17]=GetSystemCycle();
 if(profile)ts[9]=GetSystemCycle();
 uint32_t prefetchError=q==1?(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b)->error:0;
 cqError|=prefetchError;
 if(ownError || cqError){
   auto d=flags+(28+q)*32;
   d[1]=topkError;d[2]=packError;d[3]=indexError;d[4]=post->error;
   d[5]=prefetchError;d[6]=total;d[7]=first;d[8]=n;d[9]=own;
   d[10]=post->pending;d[11]=generation;
   if(!indexError || !n){for(uint32_t i=12;i<=16;++i)d[i]=UINT32_MAX;}
 }
 flags[(28+q)*32]=uint32_t(cqError)|ownError;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+(28+q)*32),4);
 Simt::VF_CALL<FinishTerminalRow>(Simt::Dim3{512,1,1},a,ctx,runtime,b,q,owner,total,first,n,flags,generation,work);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(profile)ts[10]=GetSystemCycle();
 pipe.Destroy();if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ts),256);
 } // ASCEND_IS_AIV
#endif
}
