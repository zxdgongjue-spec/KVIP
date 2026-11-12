// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ubmem/fused_batch_get_device.h"
// KVAF owns the MTE pipeline. Only copies sharing a destination write sector
// need ordering across Qs; disjoint copies can overlap on separate Q workers.
constexpr uint32_t kUbmemOverlapFetch = 0x04000000U;
constexpr uint32_t kUbmemSectorCapacity = 16384;
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ClassifyUbmemDependencies(__gm__ Context* ctx,uint32_t offset,uint32_t b,
 uint32_t q,uint32_t resident,uint32_t width,uint32_t ownerBase,
 __ubuf__ uint32_t* sectors,__ubuf__ uint32_t* masks,__ubuf__ uint32_t* control) {
 uint32_t tid=threadIdx.x;
 for(uint32_t i=tid;i<resident;i+=blockDim.x)sectors[i]=0;
 if(!tid){control[0]=0;control[1]=0;control[2]=0;}
 asc_syncthreads();
 auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
 for(uint32_t prior=0;prior<q;++prior){
  auto post=(reinterpret_cast<__gm__ FinalPost*>(ctx->owners+offset)+b*4+prior);
  uint32_t count=asc_ldcg(&post->count),begin=asc_ldcg(&post->descriptor);
  if(count>kTop || begin!=b*kSlots+prior*kTop){if(!tid)asc_atomic_or(control+2,kUbmemInvalidAddress);continue;}
  for(uint32_t i=tid;i<count;i+=blockDim.x){
   uint32_t slot=uint32_t(asc_ldcg(destinations+begin+i))-ownerBase;
   if(slot>=resident){asc_atomic_or(control+2,kUbmemInvalidAddress);continue;}
   uint32_t sector=(width%32U)?slot/2:slot;
   asc_atomic_or(sectors+sector,1U<<prior);
  }
 }
 asc_syncthreads();
 auto post=(reinterpret_cast<__gm__ FinalPost*>(ctx->owners+offset)+b*4+q);
 for(uint32_t i=tid;i<post->count;i+=blockDim.x){
  uint32_t slot=uint32_t(asc_ldcg(destinations+post->descriptor+i))-ownerBase;
  uint32_t dependencies=0;
  if(slot>=resident)asc_atomic_or(control+2,kUbmemInvalidAddress);
  else dependencies=sectors[(width%32U)?slot/2:slot];
  masks[i]=dependencies;
  if(dependencies){asc_atomic_or(control,dependencies);asc_atomic_add(control+1,1U);}
 }
 asc_syncthreads();
}
__aicore__ inline void FetchQueryGroup(__gm__ Args* a,uint32_t q,uint32_t lane,
    uint32_t workers,bool = true) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 if(lane>=shape.batch)return;
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags);
 auto region=UbmemRegionAt(ctx);
 bool profile=(a->prefetchCount&0x80000000U)!=0;
 uint32_t error=(UbmemRegionValid(region,shape.resident) && shape.resident<=kUbmemSectorCapacity)?0:kUbmemInvalidRegion;
 uint32_t completed=0;
 auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+shape.batch*256+
   (shape.batch*kResolveCores+q*workers+lane)*32;
 if(profile)trace[2]=GetSystemCycle();
 bool overlap=(a->prefetchCount&kUbmemOverlapFetch)!=0;
 TPipe pipe;TBuf<TPosition::VECCALC> scratch,dependencyBuffer;pipe.InitBuffer(scratch,21504);
 LocalTensor<uint32_t> dependency;
 if(overlap){pipe.InitBuffer(dependencyBuffer,(kUbmemSectorCapacity+kTop+8)*sizeof(uint32_t));dependency=dependencyBuffer.Get<uint32_t>();}
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
  uint32_t dependencies=overlap?0U:(q?(1U<<(q-1)):0U),conflicts=overlap?0U:post->count;
  if(overlap && q && post->count && !error){
   Simt::VF_CALL<ClassifyUbmemDependencies>(Simt::Dim3{256,1,1},ctx,shape.finalPostOffset,b,q,
     shape.resident,shape.tokenBytes,
     (ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b)*
       (ctx->outputStride?uint32_t(ctx->outputStride):shape.resident),
     reinterpret_cast<__ubuf__ uint32_t*>(dependency.GetPhyAddr()),
     reinterpret_cast<__ubuf__ uint32_t*>(dependency[kUbmemSectorCapacity].GetPhyAddr()),
     reinterpret_cast<__ubuf__ uint32_t*>(dependency[kUbmemSectorCapacity+kTop].GetPhyAddr()));
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   dependencies=dependency.GetValue(kUbmemSectorCapacity+kTop);
   conflicts=dependency.GetValue(kUbmemSectorCapacity+kTop+1);
   error|=dependency.GetValue(kUbmemSectorCapacity+kTop+2);
  }
  if(profile)ticks[2]=GetSystemCycle();
  kvaf_device::UbmemBatchGetDeviceResult result{};
  uint64_t start=GetSystemCycle(),safeEnd=start,waitEnd=start,end=start;
  uint32_t safeGroups=0,conflictGroups=0;
  if(profile && b==lane){trace[3]=trace[4]=trace[5]=start;}
  if(!error && post->count)SyncFetchDescriptors(ctx,post->descriptor,post->count);
  for(uint32_t phase=0;phase<2;++phase){
   if(phase){
    safeEnd=GetSystemCycle();
    for(uint32_t prior=0;prior<q;++prior)if(dependencies&(1U<<prior)){
     auto previous=FinalPostAt(shape.finalPostOffset,ctx,b*4+prior);started=GetSystemCycle();
     while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(previous),128);
      if(previous->head){error|=previous->error;break;}
      if(GetSystemCycle()-started>5000000000ULL){error|=kUbmemIncomplete;break;}}
    }
    waitEnd=GetSystemCycle();
   }
   if(error || !post->count)continue;
   // Submit contiguous safe/conflicting runs directly from the immutable
   // resolver arrays; no descriptor packing and no new transport implementation.
   for(uint32_t begin=0;begin<post->count;){
    bool conflict=!overlap || (q && dependencies && dependency.GetValue(kUbmemSectorCapacity+begin)!=0);uint32_t finish=(!overlap || !dependencies)?post->count:begin+1;
    while(finish<post->count && (!overlap || (q && dependencies && dependency.GetValue(kUbmemSectorCapacity+finish)!=0))==conflict)++finish;
    if(conflict==bool(phase)){
     kvaf_device::UbmemBatchGetDeviceArgs args{
      reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses),
      reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
      region->peerBase,region->mappedBase,region->bytes,region->outputBase,region->outputBytes,
      post->descriptor+begin,finish-begin,(ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b)*
        (ctx->outputStride?uint32_t(ctx->outputStride):shape.resident),shape.resident,
      0,1,shape.tokenBytes};
     auto part=kvaf_device::KvafBatchGetDevice(args,scratch.Get<uint8_t>());
     result.scheduled+=part.scheduled;result.completed+=part.completed;error|=part.error;
     if(phase)++conflictGroups;else ++safeGroups;
     if(part.scheduled!=finish-begin || part.completed!=finish-begin)error|=kUbmemIncomplete;
     if(error)break;
    }
    begin=finish;
   }
  }
  end=GetSystemCycle();completed+=result.completed;
  // Release copy completion before waiting for Index metadata, including on
  // failure and empty Qs, so successors cannot deadlock on missing publication.
  post->pending=result.completed;post->error=error;post->head=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);
  if(profile){ticks[3]=start;ticks[4]=safeEnd;ticks[5]=safeEnd;ticks[6]=waitEnd;ticks[7]=waitEnd;ticks[8]=end;
   ticks[12]=safeEnd-start;ticks[13]=waitEnd-safeEnd;ticks[14]=end-waitEnd;ticks[15]=conflicts;
   ticks[16]=safeGroups;ticks[17]=conflictGroups;ticks[18]=ticks[2]-ticks[1];
   ticks[19]=dependencies;ticks[20]=overlap?5:4;ticks[21]=post->count-conflicts;}

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
