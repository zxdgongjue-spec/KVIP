#if defined(__CCE_AICORE__)
// SPDX-License-Identifier: Apache-2.0
// Experimental bounded final window. The original tail owns completion/Index updates.
__aicore__ inline void PrepareTerminal(__gm__ pipeline_ab::Args* a,uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto errors=reinterpret_cast<__gm__ volatile uint32_t*>(a->queryErrors)+b*128;
 auto row=reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+b;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(a->queryErrors)+b*512,512);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows)+b*sizeof(rotating::Row),64);
 uint32_t total=0,reason=0,counts[4];
 for(uint32_t q=0;q<4;++q){uint32_t n=errors[q*32+1];
  if(errors[q*32])reason|=1;
  if(!n || n>2049)reason|=2;
  if(n && n<=2049)total+=n-1;
  counts[q]=n?n-1:UINT32_MAX;errors[8+q]=counts[q];
 }
 uint32_t used=row->slots;
 if(total>rotating::kCandidateCapacity)reason|=4;
 if(used>rotating::kLruSlots || total>rotating::kLruSlots-used)reason|=8;
 bool eligible=!reason && total>0;
 errors[2]=uint32_t(eligible);errors[3]=total;errors[4]=0;errors[5]=row->requests;
 errors[6]=reason;errors[7]=used;errors[12]=0;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(a->queryErrors)+b*512,128);
 if(!eligible)return;
 auto metadata=reinterpret_cast<__gm__ volatile uint32_t*>(rotating::TopKHeader(ctx))+b*128;
 for(uint32_t q=0;q<4;++q)metadata[q*32+6]=counts[q];
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(rotating::TopKHeader(ctx))+b*512,512);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RestoreTerminal(uint32_t history,uint32_t resident,
    __gm__ uint32_t* errors,__gm__ uint32_t* loc,__gm__ uint32_t* packed,
    __gm__ uint32_t* map,__gm__ uint32_t* lru,__gm__ rotating::Row* row,
    uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3) {
 uint32_t total=n0+n1+n2+n3;
 for(uint32_t i=threadIdx.x;i<total;i+=256){
  uint32_t q=i<n0?0:i<n0+n1?1:i<n0+n1+n2?2:3;
  uint32_t offset=q==0?0:q==1?n0:q==2?n0+n1:n0+n1+n2;
  uint32_t entry=asc_ldcg(packed+q*2048+i-offset),token=entry>>11;
  uint32_t slot=token<history?rotating::ResolvePhysicalSlot(resident,lru,asc_ldcg(map+token)):UINT32_MAX;
  if(slot==UINT32_MAX)asc_atomic_or(errors+q*32,8U);
  asc_stcg(loc+q*2048+(entry&2047),slot);
 }
 // Locations and counters have disjoint writers; consumers run only after
 // VF completion and the caller's all-core publication barrier.
 for(uint32_t q=threadIdx.x;q<4;q+=256)asc_stcg(errors+q*32+1,1U);
 if(!threadIdx.x){
  asc_stcg(errors+4,asc_ldcg(&row->requests)-asc_ldcg(errors+5));}
 asc_threadfence();
}
// Slots 16..26 are outside the window rank records (at most eight windows).
__aicore__ inline void StampTerminal(__gm__ pipeline_ab::Args* a,uint32_t block,uint32_t slot) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 if(block>=s.batch*4)return;
 auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->incremental+
     s.batch*(512+4*2048*6+1024+(s.history+255)/256*32));
 trace[block*32+16+slot]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace+block*32+16+slot),8);
}
template<bool profile=false>
__aicore__ inline void RunTerminalService(__gm__ pipeline_ab::Args* a,uint32_t lane) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 if constexpr(profile) StampTerminal(a,lane,5);
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  PrepareTerminal(a,b);
 }
 if constexpr(profile) StampTerminal(a,lane,6);
 TPipe pipe;TBuf<TPosition::VECCALC> scratch,flags,ids;
 pipe.InitBuffer(scratch,65536);pipe.InitBuffer(flags,32768);pipe.InitBuffer(ids,8192);
 if constexpr(profile){
 rotating::FastService<true,true,true>(s.batch,s.history,s.resident,s.rounds,s.serviceLanes,s.tokenBytes,
  s.postThreads,s.postBank,s.batchedRows,s.scanTokens,reinterpret_cast<GM_ADDR>(ctx),s.rounds,lane/2,lane,
  scratch.Get<uint32_t>(),flags.Get<uint16_t>(),ids.Get<uint32_t>(),0,a->queryErrors);
 }else{
 rotating::FastService<false,true,true>(s.batch,s.history,s.resident,s.rounds,s.serviceLanes,s.tokenBytes,
  s.postThreads,s.postBank,s.batchedRows,s.scanTokens,reinterpret_cast<GM_ADDR>(ctx),s.rounds,lane/2,lane,
  scratch.Get<uint32_t>(),flags.Get<uint16_t>(),ids.Get<uint32_t>(),0,a->queryErrors);
 }
 pipe.Destroy();
 if constexpr(profile) StampTerminal(a,lane,7);
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  auto error=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(error),128);
  if(!error[2])continue;
  Simt::VF_CALL<RestoreTerminal>(Simt::Dim3{256,1,1},s.history,s.resident,
      reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128,
      reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192,
      reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+b*8192,
      reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*s.history,
      reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*2048,
      reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b,
      error[8],error[9],error[10],error[11]);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 if constexpr(profile) StampTerminal(a,lane,8);
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
