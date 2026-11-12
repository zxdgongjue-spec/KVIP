// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ubmem/fused_batch_get_device.h"
// The preceding LI launch has drained all window MTE copies. Final copies use
// KVAF's mapped-region validation and MTE pipeline, with immutable Q descriptors.
// Different Qs may select adjacent 656-byte rows sharing a 32-byte sector.
// Order their copies per request; independent requests still run concurrently.
__aicore__ inline void FetchQueryGroup(__gm__ Args* a,uint32_t q,uint32_t lane,
    uint32_t workers,bool = true) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 if(lane>=shape.batch)return;
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags);
 auto region=UbmemRegionAt(ctx);
 bool profile=(a->prefetchCount&0x80000000U)!=0;
 uint32_t error=UbmemRegionValid(region,shape.resident)?0:kUbmemInvalidRegion;
 uint32_t completed=0;
 auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+shape.batch*256+
   (shape.batch*kResolveCores+q*workers+lane)*32;
 if(profile)trace[2]=GetSystemCycle();
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,21504);
 for(uint32_t b=lane;b<shape.batch;b+=workers){
  auto f=flags+b*kFlagCount;
  auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+shape.batch*128+b*128+q*24;
  uint64_t started=GetSystemCycle();if(profile)ticks[0]=started;
  while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f+q+1),128);
   if(f[q+1].ready){error|=f[q+1].error;break;}
   if(GetSystemCycle()-started>5000000000ULL){error|=kUbmemIncomplete;break;}}
  if(profile)ticks[1]=GetSystemCycle();
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*4+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);
  error|=post->error;
  if(post->count>kTop || post->descriptor!=b*kSlots+q*kTop)error|=kUbmemInvalidAddress;
  if(q){
   auto previous=FinalPostAt(shape.finalPostOffset,ctx,b*4+q-1);started=GetSystemCycle();
   while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(previous),128);
    if(previous->head){error|=previous->error;break;}
    if(GetSystemCycle()-started>5000000000ULL){error|=kUbmemIncomplete;break;}}
  }
  if(profile)ticks[2]=GetSystemCycle();
  kvaf_device::UbmemBatchGetDeviceResult result{};
  uint64_t start=GetSystemCycle();
  if(profile && b==lane){trace[3]=trace[4]=trace[5]=start;}
  if(!error && post->count){
   SyncFetchDescriptors(ctx,post->descriptor,post->count);
   kvaf_device::UbmemBatchGetDeviceArgs args{
    reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses),
    reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
    region->peerBase,region->mappedBase,region->bytes,region->outputBase,region->outputBytes,
    post->descriptor,post->count,(ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b)*
      (ctx->outputStride?uint32_t(ctx->outputStride):shape.resident),shape.resident,
    0,1,shape.tokenBytes};
   result=kvaf_device::KvafBatchGetDevice(args,scratch.Get<uint8_t>());
   error|=result.error;
   if(result.scheduled!=post->count || result.completed!=post->count)error|=kUbmemIncomplete;
  }
  uint64_t end=GetSystemCycle();completed+=result.completed;
  // Release copy completion before waiting for Index metadata, including on
  // failure and empty Qs, so successors cannot deadlock on missing publication.
  post->pending=result.completed;post->error=error;post->head=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);
  if(profile){ticks[3]=start;ticks[4]=end;ticks[5]=ticks[6]=ticks[7]=ticks[8]=end;
   ticks[12]=end-start;ticks[13]=ticks[14]=ticks[15]=0;ticks[16]=post->count?1:0;
   ticks[17]=0;ticks[18]=ticks[2]-ticks[0];ticks[19]=0;ticks[20]=4;}
 }
 if(profile)trace[6]=trace[12]=GetSystemCycle();
 pipe.Destroy();
 for(uint32_t b=lane;b<shape.batch;b+=workers){
  auto f=flags+b*kFlagCount+6;uint64_t start=GetSystemCycle();
  while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),128);
   if(f->ready){error|=f->error;break;}
   if(GetSystemCycle()-start>5000000000ULL){error|=kUbmemIncomplete;break;}}
 }
 if(profile)trace[13]=GetSystemCycle();
 Simt::VF_CALL<PublishQueryCompletion>(Simt::Dim3{256,1,1},ctx,flags,shape.batch,
   shape.resident,shape.finalPostOffset,q,lane,workers,error,completed);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(profile)trace[14]=trace[15]=GetSystemCycle();
 if(profile)for(uint32_t b=lane;b<shape.batch;b+=workers){
  auto t=reinterpret_cast<__gm__ uint64_t*>(a->times)+shape.batch*128+b*128+q*24;
  t[9]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(t),24*8);
 }
}
