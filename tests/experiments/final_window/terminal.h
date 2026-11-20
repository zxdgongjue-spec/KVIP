#if defined(__CCE_AICORE__)
// SPDX-License-Identifier: Apache-2.0
// Only orchestration lives here. Production window algorithms stay unchanged.
// Slots 16..26 are outside the window rank records (at most eight windows).
__aicore__ inline void StampTerminal(__gm__ pipeline_ab::Args* a,uint32_t block,uint32_t slot) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 if(block>=s.batch*4)return;
 auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental+
     s.batch*(512+4*2048*6+1024+(s.history+255)/256*32));
 trace[block*32+16+slot]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace+block*32+16+slot),8);
}
__aicore__ inline void RunSharedFinalWindow(__gm__ pipeline_ab::Args* a,uint32_t lane) {
 auto original=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto s=rotating::LoadRuntimeShape(original);
 bool profile=(a->prefetchCount&0x80000000U)!=0;
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(original->events+uint64_t(s.rounds+1)*s.serviceLanes*128+uint64_t(lane)*384);
 static_assert(sizeof(rotating::Context)<=384,"terminal Context allocation");
 auto src=reinterpret_cast<__gm__ uint64_t*>(original),dst=reinterpret_cast<__gm__ uint64_t*>(ctx);
 for(uint32_t i=0;i<sizeof(rotating::Context)/8;++i)dst[i]=src[i];
 ctx->shape.splitQ=0; // service core/sub IDs now map directly to terminal lanes
 ctx->prefetchLimit=rotating::kLruSlots;ctx->windowLimit=8192;ctx->prefetchRankStart=2048;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx),sizeof(rotating::Context));
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  auto e=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
  auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
  for(uint32_t i=0;i<128;++i)e[i]=0;
  e[5]=row->requests;e[7]=row->slots;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(e),512);
  // The preceding cold kernel has completed. Publish its full endpoint to
  // the unchanged window progress reader; query_ends still masks each Q.
  auto progress=reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+b*128;
  for(uint32_t q=0;q<4;++q){progress[q*32]=s.history;progress[q*32+16]=s.history;}
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(progress),512);
 }
 if(profile)StampTerminal(a,lane,0);
 {
 TPipe pipe;
 using Type=QLICommon::QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,QLICommon::LI_LAYOUT::BSND,QLICommon::LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
 QLIKernel::QLIVector<Type> op;
 QLICommon::ConstInfo ci{};ci.gSize=s.indexHeads;ci.qHeadNum=s.indexHeads;ci.kHeadNum=1;ci.headDim=128;ci.mBaseSize=64;ci.s1BaseSize=2;ci.s2BaseSize=128;ci.sparseCount=2048;ci.kSeqSize=s.history;ci.kCacheBlockSize=128;ci.maxBlockNumPerBatch=s.history/128;
 QLITilingData tiling{};op.SetStreamingWorkspace(reinterpret_cast<GM_ADDR>(a));op.InitParams(ci,&tiling,false);op.InitBuffers(&pipe,false);op.SetRotationContext(reinterpret_cast<GM_ADDR>(ctx));
 GlobalTensor<uint16_t> scores;op.InitVecWorkspaceTensor(scores);
 GlobalTensor<bfloat16_t> weights;GlobalTensor<float> qs,ks;GlobalTensor<int32_t> out,blocks;
 out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ctx->selected));op.InitVecInputTensor(weights,qs,ks,out,blocks);
 op.AllocEventID();
 op.RotateFetch(s.rounds,rotating::SequenceHalfWindow(s));
 op.FreeEventID();pipe.Destroy();
 }
 if(profile)StampTerminal(a,lane,3);
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  auto e=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
  auto meta=reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(ctx))+b*64;
  auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
  uint32_t n=0;for(uint32_t q=0;q<4;++q){e[8+q]=uint32_t(meta[q*16+3]);n+=e[8+q];}
  e[2]=1;e[3]=n;e[4]=row->requests-e[5];
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(e),128);
 }
}

// Adapt the saved TopK to the existing public output protocol. No TopK,
// miss selection or Get is run here; use the existing location helpers.
__aicore__ inline void PublishSavedTopK(__gm__ pipeline_ab::Args* a,uint32_t query) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 uint32_t b=query/4,q=query%4;bool profile=(a->prefetchCount&0x80000000U)!=0;
 if(profile)StampTerminal(a,query,1);
 TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,40960);
 auto ids=storage.Get<uint32_t>(),loc=ids[2048],bitmap=ids[4096],err=ids[8192];
 GlobalTensor<uint32_t> saved,selected,output;
 saved.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incremental+uint64_t(s.batch)*512+uint64_t(b)*(4*2048*6+1024)+4*2048*2)+q*2048);
 selected.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+query*2048);
 output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+query*2048);
 bool active=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]!=0;
 if(active){
  DataCopy(ids,saved,2048);Duplicate(bitmap,uint32_t(0),4104);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  kvip_prefetch_rank::RestorePublicIds(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(loc.GetPhyAddr()),s.resident);
  PipeBarrier<PIPE_V>();
  auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds);
  uint32_t begin=bounds?bounds[b*2]:0,end=bounds?bounds[b*2+1]:0;
  uint32_t capacity=ctx->outputStride>s.resident?ctx->outputStride-s.resident:0;
  uint32_t size=end>begin?end-begin:0;if(size>capacity)size=capacity;
  uint32_t shards=4096/((s.resident+31)/32);if(shards>8)shards=8;
  Simt::VF_CALL<rotating::ClassifyPreparedLocationsFlex>(Simt::Dim3{512,1,1},s.history,s.resident,b,ctx,
      begin,size,size?begin%capacity:0,capacity,reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
      reinterpret_cast<__ubuf__ uint32_t*>(loc.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(err.GetPhyAddr()),shards);
  PipeBarrier<PIPE_V>();
  Simt::VF_CALL<rotating::MergeClassifyBitmapFlex>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),s.resident,shards,
      reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(loc.GetPhyAddr()),
      reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+query*2048,reinterpret_cast<__ubuf__ uint32_t*>(err[1].GetPhyAddr()));
 }else{Duplicate(ids,UINT32_MAX,2048);Duplicate(loc,UINT32_MAX,2048);Duplicate(err,uint32_t(0),8);}
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+query*32;
 errors[0]=err.GetValue(0);errors[1]=active?err.GetValue(1):1;
 // A residual miss is a capacity/error failure, never a second Fetch path.
 if(errors[1]!=1)errors[0]|=8;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors),8);
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
 DataCopy(selected,ids,2048);DataCopy(output,loc,2048);
 SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
 pipe.Destroy();if(profile)StampTerminal(a,query,2);
}

#define KVIP_TERMINAL_HANDOFF 1
#include "terminal_prefetch_publish.h"
// The terminal window already assigned every location and posted every miss.
// Only publish the existing Index/output protocol; do not run Producer again.
__aicore__ inline bool PublishTerminalReady(__gm__ pipeline_ab::Args* a,uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto e=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(e),512);
 auto ends=reinterpret_cast<__gm__ int32_t*>(ctx->queryEnds)+b*4;
 bool active=false;
 for(uint32_t q=0;q<4;++q)active=active || ends[q]>0;
 uint32_t error=0;
 if(active){
  if(!e[2] && (e[6] || e[3]))error=16;
  for(uint32_t q=0;q<4;++q)if(e[q*32] || e[q*32+1]!=1)error|=16;
 }
 auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
 error|=row->error;
 bool profile=(a->prefetchCount&0x80000000U)!=0;
 if(profile)StampTerminal(a,b,9);
 auto flags=reinterpret_cast<__gm__ pipeline_ab::BatchFlag*>(a->flags)+b*pipeline_ab::kFlagCount;
 Simt::VF_CALL<PublishPrefetch>(Simt::Dim3{512,1,1},
  reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*10240,
  reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*2048,
  reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents,
  reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents,row->slots);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
 for(uint32_t i=0;i<32;++i)status[i]=0;
 status[0]=error;status[2]=b*8192;
 row->finalBegin=0;row->finalRequests=0;row->finalHead=0;row->finalPending=0;row->finalEnd=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 flags[0].count=row->slots;flags[0].error=error;flags[0].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags),pipeline_ab::kBatchFlagPayloadBytes);
 for(uint32_t q=0;q<4;++q){auto f=flags+pipeline_ab::kRawQueryFlag+q;
  f->count=0;f->error=error;f->reserved[0]=0;f->ready=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),16);
 }
 flags[5].count=0;flags[5].error=error;flags[5].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+5),pipeline_ab::kBatchFlagPayloadBytes);
 e[12]=active && !error;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(e),128);
 if(profile)StampTerminal(a,b,10);
 return !error;
}

#endif
