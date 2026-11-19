// SPDX-License-Identifier: Apache-2.0
// Per-query resolver, Index update and transport workers.
#include "kernel_operator.h"
#include <stddef.h>
#if defined(__CCE_AICORE__)
#include "resolver.h"
#include "compact_misses.h"
#include "resolver_staged_dma2.h"
#include "staged_urma_pack.h"
#include "staged_outputs.h"
#include "pipeline_types.h"
#include "shadow_exchange_parallel.h"
#include "index_cache_update.h"
using namespace AscendC;
using namespace rotating;
using namespace pipeline_ab;
// Transport layouts reserve 128 bytes; only the fields through head are live.
constexpr uint32_t kFinalPostPayloadBytes = offsetof(FinalPost, head) + sizeof(uint64_t);
#include "staged_fetch_worker.h"
__simt_callee__
inline void PublishQ(
 __ubuf__ uint32_t* selected,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
 __gm__ uint32_t* source,__gm__ uint32_t* dest,uint32_t begin,uint32_t end) {
 for(uint32_t i=begin+threadIdx.x;i<end;i+=512){asc_stcg(source+kPrefetch+i,selected[ids[i%kTop]]);asc_stcg(dest+kPrefetch+i,slots[i]);}
 asc_threadfence();asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishPrefetch(
 __gm__ uint32_t* source,__gm__ uint32_t* dest,__gm__ uint32_t* outSource,__gm__ uint32_t* outDest,uint32_t count) {
 for(uint32_t i=threadIdx.x;i<count;i+=512){asc_stcg(outSource+i,asc_ldcg(source+i));asc_stcg(outDest+i,asc_ldcg(dest+i));}
#if !V9_UBMEM
 // The admitted prefix has been overwritten by LI. Clear only the remaining
 // ordinals before resolver appends changes, keeping completed rows observable.
 for(uint32_t i=count+threadIdx.x;i<kPrefetch;i+=512)asc_stcg(source+i,UINT32_MAX);
#endif
 asc_threadfence();asc_syncthreads();
}
// ctl[32:64] is outside resolver counts, prefix scratch and protection bitmap.
constexpr uint32_t kOutputMasksOffset=32;
constexpr uint32_t kOutputMasksPerQuery=8;
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void MergeQueryBitmaps(__ubuf__ uint32_t* output,__ubuf__ uint32_t* other,uint32_t words
 ,__gm__ int64_t* unpack,__gm__ int64_t* valid,uint32_t batchCount,uint32_t batch,
 __ubuf__ uint32_t* masks,bool outputEnabled
 ) {
 for(uint32_t i=threadIdx.x;i<words;i+=512)output[i]|=other[i]|other[words+i]|other[2*words+i];
 if(threadIdx.x<4*kOutputMasksPerQuery)masks[threadIdx.x]=0;
 asc_syncthreads();
 if(outputEnabled)for(uint32_t row=threadIdx.x;row<batchCount*4;row+=512){
  bool active=asc_ldcg(valid+row)!=0;
  uint64_t source=active?uint64_t(asc_ldcg(unpack+row)):0;
  if(active && source/4==batch)
   asc_atomic_or(masks+(source%4)*kOutputMasksPerQuery+row/32,1U<<(row%32));
  else if(!active && row/4==batch)asc_atomic_or(masks+2+row/32,1U<<(row%32));
 }
 asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void PublishStreamingUpdateStatus(__gm__ uint32_t* status,uint32_t b,uint32_t error) {
 if(!threadIdx.x)asc_stcg(status+b,error);
 asc_threadfence();
}
// Same operations and buffers as the separate VF calls, with one dispatch.
// Q-ready is still published by the scalar caller after output DMA completion.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ResolveQueryWork(
 StagedContextPointer ctx,uint32_t history,uint32_t resident,uint32_t tokenBytes,uint32_t used,
 uint32_t batchCount,uint32_t b,uint32_t q,
 __gm__ int64_t* logicalRow,__gm__ int64_t* unpack,__gm__ int64_t* valid,
 __gm__ uint32_t* eventSource,__gm__ uint32_t* eventDest,__ubuf__ uint32_t* selected,
 __ubuf__ uint32_t* loc,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,
 __ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* work,
 __ubuf__ uint32_t* ctl,__gm__ int32_t* direct,uint32_t stride,__gm__ uint32_t* deferredProtection) {
 StagedQuery(q,history,resident,tokenBytes,b,used,ctx,
             selected,loc,keys,ranks,ids,slots,work,ctl,deferredProtection);
 if(ctl[1])return;
 PublishQ(selected,ids,slots,eventSource,eventDest,ctl[0]-ctl[2+q],ctl[0]);
}
template<class ContextPointer>
__simt_callee__ inline void PublishPreparedQueryBody(ContextPointer,__gm__ BatchFlag*,uint32_t,uint32_t,
 uint32_t,uint32_t,uint32_t,uint32_t,__gm__ uint32_t*);
// One warp pair handles sparse rows. Preserve the dense resolver's hash ABI
// and first-position ordering so a later query can take either path.
__simt_callee__ inline void ResolveCompactBody(
    StagedContextPointer ctx,uint32_t b,uint32_t q,uint32_t resident,uint32_t used,uint32_t count,
    __gm__ uint32_t* entries,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,
    __ubuf__ uint32_t* slots,__ubuf__ uint32_t* lru,__ubuf__ uint32_t* ctl,
    __gm__ uint32_t* source,__gm__ uint32_t* dest,__gm__ int32_t* direct,
    __gm__ uint64_t* addressList,uint32_t stride,__gm__ uint32_t* liveCount){
    uint32_t tid=threadIdx.x,old=ctl[0],entry=0,h=0;
    if(!count){
        if(!tid){
            ctl[2+q]=0;
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots+old);
        }
        if(direct)PublishPreparedQueryBody(ctx,static_cast<__gm__ BatchFlag*>(nullptr),b,q,
            ctx->shape.finalPostOffset,0,0,0,reinterpret_cast<__gm__ uint32_t*>(direct+6*stride));
        return;
    }
    auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*resident;
    // Check the whole candidate suffix before mutating the cross-Q hash.
    if(tid<count){
        uint32_t ordinal=used+old+tid,slot=lru[ordinal];
        if(slot>=resident || asc_ldcg(inverse+slot)!=int32_t(ordinal))asc_atomic_or(ctl+16,1U);
    }
    asc_syncthreads();
    if(ctl[16])return;
    if(tid<count){
        entry=asc_ldcg(entries+tid);auto token=entry>>kPositionBits;
        h=SparseHash(token)&(ctl[7]-1);
        while(true){auto prev=asc_atomic_cas(keys+h,UINT32_MAX,token);
            if(prev==UINT32_MAX || prev==token)break;
            h=(h+1)&(ctl[7]-1);}
        uint32_t previous=ranks[h];
        if(previous==UINT32_MAX || !(previous&kStagedRankTag))asc_atomic_min(ranks+h,tid);
    }
    asc_syncthreads();
    bool fresh=tid<count && ranks[h]==tid;
    uint32_t mask=asc_ballot(fresh);
    if(!(tid%32))ctl[17+tid/32]=__popc(mask);
    asc_syncthreads();
    uint32_t added=ctl[17]+ctl[18];
    auto counters=direct?reinterpret_cast<__gm__ uint32_t*>(direct+6*stride):nullptr;
    if(direct){
        if(!tid)ctl[19]=added?asc_atomic_add(counters,added):0;
        asc_syncthreads();
    }
    if(fresh){
        uint32_t rank=old+(tid>=32?ctl[17]:0)+__popc(mask&((1U<<(tid%32))-1));
        uint32_t ordinal=used+rank,slot=lru[ordinal],token=entry>>kPositionBits;
        slots[rank]=slot;ranks[h]=kStagedRankTag|rank;
        auto changes=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity;
        auto changedSlots=reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity;
        asc_stcg(changes+kLruSlots+rank,UINT32_MAX);asc_stcg(changes+ordinal,token);asc_stcg(changedSlots+ordinal,slot);
        if(direct){
            // Every destination was verified in the unused LRU suffix above.
            uint32_t inQ=rank-old,pos=ctl[19]+inQ,output=(b*4+q)*kTop+inQ;
            uint32_t dstSlot=OwnerRow(ctx,b)*OutputStride(resident,ctx)+slot;
            auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
            auto addr=asc_ldcg(reinterpret_cast<__gm__ uint64_t*>(row->addresses)+token);
            StoreFallbackSource(ctx->shape.history,ctx->shape.tokenBytes,ctx,b,output,token,addr,
                reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets));
            asc_stcg(reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+output,int32_t(dstSlot));
            asc_stcg(addressList+pos,addr);asc_stcg(direct+pos,int32_t(pos));
            asc_stcg(direct+stride+pos,0);asc_stcg(direct+2*stride+pos,int32_t(dstSlot));
        }
        asc_stcg(source+kPrefetch+rank,token);asc_stcg(dest+kPrefetch+rank,slot);
    }
    asc_syncthreads();
    if(tid<count)asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+(b*4+q)*kTop+(entry&kPositionMask),slots[ranks[h]&~kStagedRankTag]);
    if(!tid){if(direct && added)asc_atomic_add(liveCount,added);
        ctl[2+q]=added;ctl[0]=old+added;
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots+old+added);}
    asc_threadfence();asc_syncthreads();
    if(direct)PublishPreparedQueryBody(ctx,static_cast<__gm__ BatchFlag*>(nullptr),b,q,
        ctx->shape.finalPostOffset,added,added,0,counters);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(64) inline void ResolveCompact(
    StagedContextPointer ctx,uint32_t b,uint32_t q,uint32_t resident,uint32_t used,uint32_t count,
    __gm__ uint32_t* entries,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,
    __ubuf__ uint32_t* slots,__ubuf__ uint32_t* lru,__ubuf__ uint32_t* ctl,
    __gm__ uint32_t* source,__gm__ uint32_t* dest,__gm__ int32_t* direct,
    __gm__ uint64_t* addressList,uint32_t stride,__gm__ uint32_t* liveCount){ResolveCompactBody(ctx,b,q,resident,used,count,entries,keys,ranks,slots,lru,ctl,source,dest,direct,addressList,stride,liveCount);}
// Publish Q1 before starting the remaining queries. Its tagged ranks survive
// both parallel suffix resolution and the protected allocator fallback.
__simt_vf__ __aicore__ LAUNCH_BOUND(64) inline void ResolveFirstCompactQuery(
 __ubuf__ Args* a,StagedContextPointer ctx,uint32_t b,uint32_t resident,uint32_t used,
 uint32_t workers,uint32_t stride,uint32_t meta,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,
 __ubuf__ uint32_t* slots,__ubuf__ uint32_t* lru,__ubuf__ uint32_t* ctl){
 uint32_t tid=threadIdx.x;
 auto hdr=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
 auto base=reinterpret_cast<__gm__ int32_t*>(a->packedFetch);
 constexpr uint32_t q=0;
  uint32_t count=asc_ldcg(hdr+q*32+1)-1;
  if(count>kCompactMissLimit || used+ctl[0]+count>kLruSlots)return;
  if(!tid)ctl[16]=0;
  asc_syncthreads();
  uint32_t w=q*workers+b%workers;
  ResolveCompactBody(ctx,b,q,resident,used,count,
   reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+(b*4+q)*kTop,keys,ranks,slots,lru,ctl,
   reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents,
   reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents,
   base+4*workers*(3*stride+meta)+w*(6*stride+64),
   reinterpret_cast<__gm__ uint64_t*>(base+w*(3*stride+meta)),stride,
   reinterpret_cast<__gm__ uint32_t*>(a->liveCount?a->liveCount:ctx->count));
  if(ctl[16])return;
  if(!tid){
   auto f=flags+kRawQueryFlag+q;
   asc_stcg(&f->reserved[0],1U);asc_stcg(&f->count,ctl[2+q]);asc_stcg(&f->error,0U);
   asc_stcg(status+4+q,ctl[2+q]);asc_stcg(status+8+q,ctl[0]);
   asc_threadfence();asc_stcg(&f->ready,1U);ctl[20]=q+1;
  }
  asc_threadfence();asc_syncthreads();
}
#if !V9_UBMEM && KVIP_OUTPUT_WORKER
#include "compact_parallel.h"
#endif
__aicore__ inline void PublishQueryOutputs(__gm__ uint64_t* update,uint32_t batchCount,uint32_t batch,uint32_t query,
 LocalTensor<uint32_t> values,LocalTensor<uint32_t> loc,LocalTensor<uint32_t> work) {
 if(!update[20]||!update[21]||!update[22]||!update[23])return;
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
#include "output_worker.h"
__aicore__ inline void Producer(__gm__ Args* a, uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
#if !V9_UBMEM
 if(ctx->rowAddressBase){
  uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
  row->addresses=RequestAddressTable(ctx,b,owner,shape.history);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
 }
#endif
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+b*64;
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 if(profile)ticks[1]=GetSystemCycle();
 uint32_t inputError=0,upperMisses=0;bool bounded=true;
 if(!a->queryBitmaps || !a->queryErrors || !a->updateBindings || !a->packedFetch || !ctx->hotValid || !ctx->changeCount || row->slots>kPrefetch ||
    shape.history!=a->history || shape.resident!=a->resident || shape.resident>131072 || shape.finalWorkers!=4 || a->maxEvents<kPrefetch+kSlots)inputError=1;
 if(!inputError){auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+(b*4)*32;
  for(uint32_t q=0;q<4;++q){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+q*32),128);inputError|=errors[q*32];uint32_t n=errors[q*32+1]-1;
   if(n>kCompactMissLimit)bounded=false;else upperMisses+=n;}}
 if(inputError){auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;status[0]=inputError;status[1]=0;
  if(ctx->changeCount){
   auto count=reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b;
   count[0]=kLruSlots;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(count),4);
  }
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  for(uint32_t q=0;q<kFlagCount;++q){if(q==6 || (q>=7 && q<12))continue;flags[q].count=0;flags[q].error=inputError;flags[q].ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q),kBatchFlagPayloadBytes);}
  return;}
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
 Simt::VF_CALL<PublishPrefetch>(Simt::Dim3{512,1,1},reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*10240,
  reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots,reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents,
  reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents,row->slots);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 flags[0].count=row->slots;flags[0].ready=1;if(profile)ticks[0]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[0]),kBatchFlagPayloadBytes);
 auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
 auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
 auto slots=local[kStagedSlotsOffset],work=local[kStagedPositionsOffset],ctl=local[kStagedControlOffset];
 // Bind immutable metadata once on scalar control instead of reloading the
 // pointer chain in every SIMT warp and query. Official maps remain unchanged
 // until the consumer has observed the final Q publication.
 auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 bool outputEnabled=u[20] && u[21] && u[22] && u[23];
 auto logicalRow=outputEnabled?reinterpret_cast<__gm__ int64_t*>(u[3])+
    uint64_t(reinterpret_cast<__gm__ uint32_t*>(u[9])[b])*uint32_t(u[24]):nullptr;
 auto unpack=reinterpret_cast<__gm__ int64_t*>(u[20]);
 auto outputValid=reinterpret_cast<__gm__ int64_t*>(u[21]);
 auto eventSource=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
 auto eventDest=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
 GlobalTensor<uint32_t> input;
 // Context fields are immutable for this invocation; pointer targets remain
 // GM and retain their original loads/stores and publication protocol.
 constexpr uint32_t contextWords=sizeof(Context)/sizeof(uint32_t),contextOffset=128;
 static_assert(contextOffset+contextWords<544,"Context snapshot overlaps prefix scratch");
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx));
 DataCopy(ctl[contextOffset],input,contextWords);
 auto stagedContext=reinterpret_cast<__ubuf__ Context*>(ctl[contextOffset].GetPhyAddr());
 constexpr uint32_t argsOffset=contextOffset+((contextWords+31)/32)*32;
 static_assert(argsOffset+sizeof(Args)/4<544,"Args overlap prefix scratch");
 auto stagedArgs=reinterpret_cast<__ubuf__ Args*>(ctl[argsOffset].GetPhyAddr());
 input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a));
 DataCopy(ctl[argsOffset],input,sizeof(Args)/4);
#if !V9_UBMEM && KVIP_OUTPUT_WORKER
 const bool streamQueries=ctx->reservedHot!=0;
#else
 constexpr bool streamQueries=false;
#endif
 const uint32_t bitmapWords=(shape.resident+31)/32;
 auto protection=reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(b)*4*bitmapWords;
 GlobalTensor<uint32_t> selectedGm,locationsGm;
 selectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+uint64_t(b)*kSlots);
 locationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+uint64_t(b)*kSlots);
 bool shortLru=bounded && row->slots+upperMisses<=kLruSlots;
 uint32_t buckets=bounded?512U:kSlots;
 if(streamQueries){
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
  if(!shortLru)DataCopy(work[4096],input,kLruSlots);
  else if(upperMisses){
   uint32_t lo=row->slots&~7U,hi=(row->slots+upperMisses+7U)&~7U;
   DataCopy(work[4096+lo],input[lo],hi-lo);
  }
  if(bounded)ranks=keys[buckets];
  if(profile)ticks[2]=GetSystemCycle();
  Duplicate(keys,UINT32_MAX,2*buckets); // keys and ranks occupy adjoining UB banks.
  Duplicate(ctl,uint32_t(0),16);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  ctl.SetValue(7,buckets);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  if(profile)ticks[3]=GetSystemCycle();
 }else{
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);DataCopy(selected,input,kSlots);
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);DataCopy(loc,input,kSlots);
  uint32_t words=(shape.resident+31)/32;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(b)*4*words);
  DataCopy(ctl[1024],input,words);
  for(uint32_t q=1;q<4;++q)DataCopy(keys[(q-1)*words],input[q*words],words);
  input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);DataCopy(work[4096],input,kLruSlots);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<MergeQueryBitmaps>(Simt::Dim3{512,1,1},FinalPtr(ctl[1024]),FinalPtr(keys),words
   ,unpack,outputValid,shape.batch,b,FinalPtr(ctl[kOutputMasksOffset]),outputEnabled && !KVIP_OUTPUT_WORKER
  );
  PipeBarrier<PIPE_V>();
  if(profile)ticks[2]=GetSystemCycle();
  Simt::VF_CALL<StagedInitializeReady>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),FinalPtr(ctl));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  if(profile)ticks[3]=GetSystemCycle();
 }
 GlobalTensor<uint32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
 auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;

 // A dense Q3/Q4 can reuse the full result row of an earlier sparse Q.
 bool sparsePrior[4]={};
 uint32_t firstQ=0;
#if !V9_UBMEM && KVIP_OUTPUT_WORKER
 if(streamQueries && ctx->fallbackAddresses && !(a->prefetchCount&0x20000000U) && !profile){
  uint32_t workers=(a->prefetchCount>>16)&0xffU;
  ctl.SetValue(20,0);SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
  Simt::VF_CALL<ResolveFirstCompactQuery>(Simt::Dim3{64,1,1},stagedArgs,stagedContext,b,shape.resident,row->slots,workers,shape.batch*kTop,
   (3*shape.batch+31)&~31U,FinalPtr(keys),FinalPtr(ranks),FinalPtr(slots),FinalPtr(work[4096]),FinalPtr(ctl));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  if(ctl.GetValue(20)==1){
  Simt::VF_CALL<ResolveCompactParallel>(Simt::Dim3{256,1,1},stagedArgs,stagedContext,b,shape.resident,row->slots,workers,shape.batch*kTop,
   (3*shape.batch+31)&~31U,FinalPtr(keys),FinalPtr(ranks),FinalPtr(slots),FinalPtr(work[4096]),FinalPtr(ctl));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  firstQ=ctl.GetValue(20);
  for(uint32_t q=0;q<firstQ;q++)sparsePrior[q]=true;
 }
#endif
 uint32_t publishedQueries=firstQ;
 for(uint32_t q=firstQ;q<4 && !ctl.GetValue(1);++q){
  uint32_t n=streamQueries?reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[(b*4+q)*32+1]:0;
  bool compact=n && n<=kCompactMissLimit+1 && row->slots+ctl.GetValue(0)+n-1<=kLruSlots && !ctl.GetValue(kProtectionState);
  uint32_t fw=(a->prefetchCount>>16)&0xffU,ds=shape.batch*kTop,meta=(3*shape.batch+31)&~31U;
  auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*fw*(3*ds+meta)+(q*fw+b%fw)*(6*ds+64);
  bool earlyFetch=ctx->fallbackAddresses && !(a->prefetchCount&0x20000000U);
  auto addressList=reinterpret_cast<__gm__ uint64_t*>(reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+(q*fw+b%fw)*(3*ds+meta));
  if(profile && compact)ticks[10+q*5]=GetSystemCycle();
  if(compact){
   ctl.SetValue(16,0);SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
   Simt::VF_CALL<ResolveCompact>(Simt::Dim3{64,1,1},stagedContext,b,q,shape.resident,row->slots,n-1,
    reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+(b*4+q)*kTop,FinalPtr(keys),FinalPtr(ranks),FinalPtr(slots),FinalPtr(work[4096]),FinalPtr(ctl),eventSource,eventDest,earlyFetch?direct:nullptr,addressList,ds,
    reinterpret_cast<__gm__ uint32_t*>(a->liveCount?a->liveCount:ctx->count));
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   compact=!ctl.GetValue(16);
  }
  sparsePrior[q]=compact;
  if(streamQueries && !compact){
   if(shortLru){
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
    DataCopy(work[4096],input,kLruSlots);shortLru=false;
   }
   if(q>=2 && sparsePrior[q-2]){
    DataCopy(selected[(q-2)*kTop],selectedGm[(q-2)*kTop],kTop);
    DataCopy(loc[(q-2)*kTop],locationsGm[(q-2)*kTop],kTop);
   }
   DataCopy(selected[q*kTop],selectedGm[q*kTop],kTop);
   DataCopy(loc[q*kTop],locationsGm[q*kTop],kTop);
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }
  if(profile && !compact)ticks[10+q*5]=GetSystemCycle();
  if(!compact){
   Simt::VF_CALL<ResolveQueryWork>(Simt::Dim3{512,1,1},stagedContext,shape.history,shape.resident,shape.tokenBytes,row->slots,
    shape.batch,b,q,logicalRow,unpack,outputValid,eventSource,eventDest,
    FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),
    reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(work),FinalPtr(ctl),
    (a->prefetchCount & 0x20000000U)?nullptr:direct,ds,streamQueries?protection:nullptr);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  if(profile)ticks[11+q*5]=GetSystemCycle();
  uint32_t total=ctl.GetValue(0),added=ctl.GetValue(2+q),error=ctl.GetValue(1);
  if(profile)ticks[12+q*5]=ticks[11+q*5];
  if(profile)ticks[13+q*5]=GetSystemCycle();
  if(!compact){
   SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(output[q*kTop],loc[q*kTop],kTop);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
  if(profile)ticks[14+q*5]=GetSystemCycle();
  uint32_t published=KVIP_OUTPUT_WORKER?kRawQueryFlag+q:q+1;
  flags[published].reserved[0]=compact && earlyFetch;flags[published].count=added;flags[published].error=error;flags[published].ready=1;
  if(profile)ticks[4+q]=GetSystemCycle();status[4+q]=added;status[8+q]=total;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[published]),kBatchFlagPayloadBytes);
  publishedQueries=q+1;
 }
 // Wake every publication reader on error, then join before final transport state.
 for(uint32_t q=publishedQueries;q<4;++q){auto f=flags+kRawQueryFlag+q;f->count=0;f->error=ctl.GetValue(1)?ctl.GetValue(1):1;f->ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),kBatchFlagPayloadBytes);}
#if V9_UBMEM || !KVIP_OUTPUT_WORKER
 uint64_t publishWait=GetSystemCycle();
 while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+4),kBatchFlagPayloadBytes);if(flags[4].ready)break;if(GetSystemCycle()-publishWait>5000000000ULL){ctl.SetValue(1,8);break;}}
 if(flags[4].error)ctl.SetValue(1,ctl.GetValue(1)|flags[4].error);
#endif
 status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);status[2]=b*kSlots;status[3]=0;status[4]=0;
 row->finalBegin=0;row->finalRequests=status[1];row->finalHead=0;row->finalPending=status[1]?1:0;row->finalEnd=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
 // Publish failure to every remaining reader, never leave the consumer waiting.
#if V9_UBMEM || !KVIP_OUTPUT_WORKER
 for(uint32_t q=0;q<4;++q)if(!flags[q+1].ready){flags[q+1].count=0;flags[q+1].error=status[0]?status[0]:1;flags[q+1].ready=1;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[q+1]),kBatchFlagPayloadBytes);}
#endif
 if(profile)ticks[8]=GetSystemCycle();
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);
 pipe.Destroy();
 flags[5].error=status[0];flags[5].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[5]),kBatchFlagPayloadBytes);
}
// Cold sources and distinct hot destinations form independent pairwise swaps.
// Validate the complete plan before changing Index storage. Otherwise use the
// permutation planner, which also handles chains and repeated physical rows.
template<bool Restore=false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void SwapColdIndexRows(
 __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t begin,uint32_t count,
 uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,
 __ubuf__ uint32_t* m){
 uint32_t t=threadIdx.x;
 if constexpr(!Restore){
 if(!t)m[CONTROL]=0;
 if(!begin)for(uint32_t i=t;i<2*SET;i+=THREADS)m[SRCSET+i]=EMPTY;
 asc_syncthreads();
 if(t<count){
  uint32_t i=t;
  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(dest+i);
  m[PLAN_SRC+begin+t]=src;m[PLAN_DST+begin+t]=dst;
  if(src<resident || src>=history || dst>=resident ||
     !insert(m+SRCSET,src) || !insert(m+DSTSET,dst))asc_atomic_or(m+CONTROL,1U);
 }
 asc_syncthreads();
 if(m[CONTROL])return;
 }
 for(uint32_t i=t;i<count*16;i+=THREADS){
  uint32_t src=m[PLAN_SRC+begin+i/16],dst=m[PLAN_DST+begin+i/16],j=i%16;
  auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);
  asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);
 }
 if(t<count){
  uint32_t src=m[PLAN_SRC+begin+t],dst=m[PLAN_DST+begin+t];
  auto x=asc_ldcg(scales+src),y=asc_ldcg(scales+dst);
  asc_stcg(scales+src,y);asc_stcg(scales+dst,x);
 }
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void SwapColdMetadata(
 __gm__ uint64_t* addr,__gm__ uint64_t* logical,uint32_t n,__ubuf__ uint32_t* m){
 uint32_t t=threadIdx.x;
 if(t<n){
  uint32_t src=m[PLAN_SRC+t],dst=m[PLAN_DST+t];
  auto x=asc_ldcg(addr+src),y=asc_ldcg(addr+dst);
  auto a=asc_ldcg(logical+src),b=asc_ldcg(logical+dst);
  asc_stcg(addr+src,y);asc_stcg(addr+dst,x);asc_stcg(logical+src,b);asc_stcg(logical+dst,a);
 }
 asc_threadfence();
}
#if !V9_UBMEM
// Producer has consumed Q4. Fetch and Index update use the published descriptors.
__simt_vf__ __aicore__ LAUNCH_BOUND(128)
inline void ClearSnapshottedMappings(__gm__ Args* a,uint32_t b,uint32_t batch){
 uint32_t t=threadIdx.x;
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
 auto header=reinterpret_cast<__gm__ uint32_t*>(ctx->rows+uint64_t(batch)*sizeof(Row))+b*128+t;
 asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->incremental)+b*128+t,asc_ldcg(header));asc_stcg(header,0U);
 auto journal=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal)+b;
 uint32_t n=asc_ldcg(&journal->count),h=a->history;
 // All four warps must snapshot count before lane zero retires the journal.
 asc_syncthreads();
 auto live=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*h;
 for(uint32_t i=t;i<n;i+=128){auto token=asc_ldcg(journal->tokens+i);if(token<h)asc_stcg(live+token,UINT32_MAX);}
 if(!t){asc_stcg(&journal->count,0U);asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a->liveCount)+3+b,0U);}
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(128)
inline void RetireSelection(__gm__ Args* a,uint32_t batch,uint32_t b){
  uint32_t t=threadIdx.x;
  auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
  auto header=reinterpret_cast<__gm__ uint32_t*>(ctx->rows+uint64_t(batch)*sizeof(Row))+b*128+t;
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->incremental)+b*128+t,asc_ldcg(header));
  asc_stcg(header,0U);
  auto journal=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal)+b;
  auto previous=reinterpret_cast<__gm__ MappingJournal*>(a->completedJournal)+b;
  uint32_t history=a->history;
  auto live=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*history;
  auto published=reinterpret_cast<__gm__ uint32_t*>(a->completedMapping)+uint64_t(b)*history;
  uint32_t oldCount=asc_ldcg(&previous->count),count=asc_ldcg(&journal->count);
  uint32_t first=asc_ldcg(&previous->tailBegin),end=asc_ldcg(&previous->tailEnd);
  for(uint32_t i=t;i<oldCount;i+=128){
   uint32_t token=asc_ldcg(previous->tokens+i);
   if(token<history)asc_stcg(published+token,UINT32_MAX);
  }
  for(uint32_t i=first+t;i<end;i+=128)asc_stcg(published+i,UINT32_MAX);
  // Old and new snapshots may contain the same token. Only this request's AIV participates.
  asc_threadfence();asc_syncthreads();
  for(uint32_t i=t;i<count;i+=128){
   uint32_t token=asc_ldcg(journal->tokens+i);
   if(token>=history)continue;
   asc_stcg(published+token,asc_ldcg(live+token));
   asc_stcg(previous->tokens+i,token);
   asc_stcg(live+token,UINT32_MAX);
  }
  if(!t){
   asc_stcg(&journal->count,0U);asc_stcg(&previous->count,count);
   asc_stcg(&previous->tailBegin,0U);asc_stcg(&previous->tailEnd,0U);
  }
  asc_threadfence();
}
#endif
__aicore__ inline void Consumer(__gm__ Args* a, uint32_t b) {
 auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;auto status=reinterpret_cast<__gm__ uint32_t*>(a->workerStatus)+b*32;
 // Producer and consumer must not write the same cache lines.
 auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+b*64+32;auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,SCRATCH_WORDS*4);auto local=scratch.Get<uint32_t>();
 auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
 uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
 if(profile)ticks[10]=GetSystemCycle();
 uint32_t total=0,error=0,seen=0;bool denseMode=false;
 auto dense=reinterpret_cast<__gm__ uint32_t*>(a->denseShadow)+uint64_t(b)*shape.history*6;
 bool prepareMetadata=(a->prefetchCount & 0x08000000U)!=0;
 auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 auto addresses=reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*update[7];
 auto logical=reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history;
 auto prepared=reinterpret_cast<__gm__ uint64_t*>(dense+2*shape.history);
 bool disjoint=false;
#if !V9_UBMEM && KVIP_OUTPUT_WORKER
 if(ctx->reservedHot && a->queryErrors && prepareMetadata){
  auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
  auto headers=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*4*32;
  uint32_t prefix=row->slots,upper=0;
  bool eligible=prefix<=CHUNK;
  for(uint32_t q=0;q<4 && eligible;q++){
   uint32_t h=headers[q*32+1];
   eligible=h && h<=kCompactMissLimit+1;
   if(eligible)upper+=h-1;
  }
  if(eligible && prefix+upper<=CHUNK){
   auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16;
   auto scales=reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*shape.history;
   auto src=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
   auto dst=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
   error=AwaitPublication(flags);
   bool prefixApplied=false;
   if(!error && prefix){
    Simt::VF_CALL<SwapColdIndexRows<false>>(Simt::Dim3{THREADS,1,1},src,dst,0,prefix,
     shape.resident,shape.history,keys,scales,work);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    prefixApplied=!local.GetValue(CONTROL);
   }
   if(!error && (!prefix || prefixApplied)){
    error=AwaitPublication(flags+kRawQueryFlag+3);
    uint32_t final=0;
    for(uint32_t q=0;q<4;q++){
     auto f=flags+kRawQueryFlag+q;
     kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),kBatchFlagPayloadBytes);
     error|=f->error;
     final+=f->count;
    }
    if(!error && final<=upper){
     if(final){
      Simt::VF_CALL<SwapColdIndexRows<false>>(Simt::Dim3{THREADS,1,1},src+kPrefetch,dst+kPrefetch,
       prefix,final,shape.resident,shape.history,keys,scales,work);
      SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
      disjoint=!local.GetValue(CONTROL);
     }else disjoint=true;
     if(disjoint){total=final;seen=prefix+final;}
    }
    if(prefixApplied && !disjoint){
     // The suffix can alias a prefetched row. Reversing the disjoint prefix
     // restores the input permutation for the general planner.
     Simt::VF_CALL<SwapColdIndexRows<true>>(Simt::Dim3{THREADS,1,1},src,dst,0,prefix,
      shape.resident,shape.history,keys,scales,work);
     SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
   }
  }
 }
#endif
 for(uint32_t phase=0;phase<5 && !error && !disjoint;++phase){
  uint32_t publication=KVIP_OUTPUT_WORKER && phase?kRawQueryFlag+phase-1:phase;
  {uint64_t wait=GetSystemCycle();while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[publication]),kBatchFlagPayloadBytes);if(flags[publication].ready)break;
   if(GetSystemCycle()-wait>5000000000ULL){error=8;break;}}}
  if(error)break;
  uint32_t count=flags[publication].count;
  if(flags[publication].error){error=flags[publication].error;break;}
  uint32_t begin=phase?kPrefetch+total:0;
  auto source=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+b*a->maxEvents;
  auto dest=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+b*a->maxEvents;
  if(profile)ticks[12+2*phase]=GetSystemCycle();
  if(count==0){
   // Publication and error were consumed above. An empty phase has no swaps.
   if(profile)ticks[13+2*phase]=GetSystemCycle();
   continue;
  }
  if(!seen){
   Simt::VF_CALL<InitShadow>(Simt::Dim3{THREADS,1,1},work);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  // Each exchange introduces at most two new positions into either map.
  // Bound linear-probe cost as well as capacity. Near-full maps can make
  // one Q1 exchange hundreds of microseconds slower than dense lookup.
  if(!denseMode && uint64_t(seen+count)*2>=MAP*3/4){
   if(!a->denseShadow){error=1;break;}
   if(prepareMetadata){
    Simt::VF_CALL<ExpandPreparedMetadata>(Simt::Dim3{THREADS,1,1},prepared,shape.history,work);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   }
   Simt::VF_CALL<InitDenseShadow>(Simt::Dim3{THREADS,1,1},dense,shape.history,work);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);denseMode=true;
  }
  if(prepareMetadata){
  if(denseMode)Simt::VF_CALL<ApplyDenseBatch<true>>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,dense,shape.history,work,addresses,logical,prepared);
  else Simt::VF_CALL<ApplyBatch<true>>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,work,addresses,logical,prepared,shape.history);
  }else{
  if(denseMode)Simt::VF_CALL<ApplyDenseBatch<false>>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,dense,shape.history,work,nullptr,nullptr,nullptr);
  else Simt::VF_CALL<ApplyBatch<false>>(Simt::Dim3{THREADS,1,1},source,dest,begin,count,reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*a->history,reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident,shape.resident,work,nullptr,nullptr,nullptr,shape.history);
  }
  seen+=count;
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  error=local.GetValue(CONTROL);if(profile)ticks[13+2*phase]=GetSystemCycle();if(phase)total+=count;
 }
 if(!error){uint64_t start=GetSystemCycle();while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+4),kBatchFlagPayloadBytes);if(flags[4].ready){error=flags[4].error;break;}if(GetSystemCycle()-start>5000000000ULL){error=8;break;}}}
 if(!error){auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);uint64_t addressStride=update[7];if(profile)ticks[22]=GetSystemCycle();
  if(seen && !disjoint && prepareMetadata && !denseMode){
   static_assert(2*MAP*sizeof(uint64_t)<=4*CHUNK*16*sizeof(uint32_t),"Metadata staging exceeds dead key/scale staging");
   GlobalTensor<uint64_t> source;source.SetGlobalBuffer(prepared);
   auto values=local[STAGING].ReinterpretCast<uint64_t>();
   DataCopy(values,source,MAP);DataCopy(values[MAP],source[shape.history],MAP);
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }
  if(seen){
  if(disjoint)Simt::VF_CALL<SwapColdMetadata>(Simt::Dim3{THREADS,1,1},addresses,logical,seen,work);
  else if(prepareMetadata)Simt::VF_CALL<CommitPreparedMetadata>(Simt::Dim3{THREADS,1,1},addresses,logical,prepared,shape.history,work,dense,denseMode);
  else if(denseMode)Simt::VF_CALL<CommitDenseMetadata>(Simt::Dim3{THREADS,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,dense,reinterpret_cast<__gm__ uint64_t*>(dense+2*shape.history),shape.history);
  else Simt::VF_CALL<CommitMetadata>(Simt::Dim3{THREADS,1,1},reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*a->history,work);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  if(profile)ticks[23]=GetSystemCycle();}
 status[0]=error;status[1]=total;status[2]=disjoint?uint32_t(seen!=0):(seen?local.GetValue(CONTROL+4):0);if(profile)ticks[24]=GetSystemCycle();
 if(a->updateBindings){auto u=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},reinterpret_cast<__gm__ uint32_t*>(u[6]),b,error);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
 if(profile)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ticks),256);pipe.Destroy();
#if !V9_UBMEM
 if(a->completedMapping){
  // Producer publishes flag 5 after its last live mapping lookup, including
  // input-error exits. Finalize joins flag 6 before recycling either flag.
  auto producer=flags+5;uint64_t start=GetSystemCycle();
  while(true){
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(producer),kBatchFlagPayloadBytes);
   if(producer->ready){
    if(a->liveCount && shape.batch<=16){
     auto ready=reinterpret_cast<__gm__ uint32_t*>(a->liveCount)+3+b;
     uint64_t start=GetSystemCycle();
     while(true){
      kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),4);
      if(*ready){Simt::VF_CALL<ClearSnapshottedMappings>(Simt::Dim3{128,1,1},a,b,shape.batch);break;}
      if(GetSystemCycle()-start>5000000000ULL){error|=8;break;}
     }
    }else Simt::VF_CALL<RetireSelection>(Simt::Dim3{128,1,1},a,shape.batch,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    break;
   }
   if(GetSystemCycle()-start>5000000000ULL){error|=8;break;}
  }
 }
#endif
 flags[6].error=error;flags[6].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&flags[6]),kBatchFlagPayloadBytes);
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
// One VF handles every request assigned to this Q worker. All slot stores
// complete before any completion flag is released. The scalar caller has joined
// Index update; failure is reported without publishing successful residency.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishQueryCompletion(__gm__ Context* ctx,__gm__ BatchFlag* flags,
 uint32_t batch,uint32_t resident,uint32_t postOffset,uint32_t q,uint32_t lane,
 uint32_t workers,uint64_t error,uint32_t completed,bool deferredResidency) {
 // Each warp owns a request; eight requests no longer run serially in one VF.
 constexpr uint32_t lanesPerRequest=32,requestsPerWave=256/lanesPerRequest;
 for(uint32_t b=lane+(threadIdx.x/lanesPerRequest)*workers;b<batch;b+=requestsPerWave*workers){
  auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+b*4+q;
  if(!error && !deferredResidency && ctx->hotValid){
   auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*resident;
   uint32_t base=OwnerRow(ctx,b)*OutputStride(resident,ctx);
   auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
   uint32_t count=asc_ldcg(&post->count),begin=asc_ldcg(&post->descriptor);
   for(uint32_t i=threadIdx.x%lanesPerRequest;i<count;i+=lanesPerRequest){
    uint32_t slot=uint32_t(asc_ldcg(destinations+begin+i))-base;
    if(slot<resident)asc_stcg(valid+slot,uint32_t(1));
   }
  }
  if(threadIdx.x%lanesPerRequest==0){
#if !V9_UBMEM
   asc_stcg(&post->pending,b==lane?completed:0U);
#endif
   asc_stcg(&post->error,uint32_t(error));asc_stcg(&post->deferred,0U);}
 }
 asc_threadfence();asc_syncthreads();
 for(uint32_t b=lane+threadIdx.x*workers;b<batch;b+=256*workers){
  auto flag=flags+b*kFlagCount+8+q;
  asc_stcg(&flag->reserved[0],uint32_t(deferredResidency));asc_stcg(&flag->error,uint32_t(error));asc_threadfence();asc_stcg(&flag->ready,1U);
 }
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishPrefetchResidency(__gm__ Context* ctx,uint32_t b,uint32_t resident,uint32_t count){
 if(ctx->hotValid && ctx->changes && ctx->lruSlots){
  auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*resident;
  auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
  auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
  for(uint32_t i=threadIdx.x;i<count;i+=256)
   if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),1U);
 }
 asc_threadfence();
}
// Small Gets release their completion as soon as the CQ succeeds. The
// request owner can publish all four Qs in one VF after its own Index update.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PublishDeferredResidency(
 __gm__ Context* ctx,__gm__ BatchFlag* flags,uint32_t b,uint32_t resident,uint32_t postOffset,uint32_t prefix=0,bool publishFinal=true){
 auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid);
 if(!valid)return;
 uint32_t owner=OwnerRow(ctx,b),base=owner*OutputStride(resident,ctx);
 valid+=owner*resident;
 if(prefix && ctx->changes && ctx->lruSlots){
  auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
  auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
  for(uint32_t i=threadIdx.x;i<prefix;i+=256)
   if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),1U);
  asc_threadfence();
 }
 if(!publishFinal)return;
 uint32_t q=threadIdx.x/64;
 if(!asc_ldcg(&flags[8+q].reserved[0]))return;
 auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+b*4+q;
 auto dst=reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+asc_ldcg(&post->descriptor);
 uint32_t count=asc_ldcg(&post->count);
 for(uint32_t i=threadIdx.x%64;i<count;i+=64){
  uint32_t slot=uint32_t(asc_ldcg(dst+i))-base;
  if(slot<resident)asc_stcg(valid+slot,1U);
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
#if V9_UBMEM
#include "staged_ubmem_fetch.h"
#else
constexpr uint32_t kBatchGetInflight = 2048;
// Each worker owns one KVAF queue. It gathers its shard of requests for one Q,
// calls the native BatchGet once per safe/conflicting phase, then acknowledges
// all rows covered by that call. The native scheduler owns WQE and CQ handling.
// Previous LI has finished writing the journals. Copy the observational
// snapshot while the Resolve AIV still owns the live token mapping.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void SnapshotMappings(
 __gm__ Args* a,uint32_t batch,uint32_t first,uint32_t groups){
 uint32_t t=threadIdx.x%128,b=first+(threadIdx.x/128)*groups;
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
 auto previous=reinterpret_cast<__gm__ MappingJournal*>(a->completedJournal);
 auto current=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal);
 uint32_t h=a->history,n=0;
 auto saved=reinterpret_cast<__gm__ uint32_t*>(a->completedMapping);
 auto live=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping);
 if(b<batch){
  uint32_t old=asc_ldcg(&previous[b].count),lo=asc_ldcg(&previous[b].tailBegin),hi=asc_ldcg(&previous[b].tailEnd);
  n=asc_ldcg(&current[b].count);
  for(uint32_t i=t;i<old;i+=128){auto token=asc_ldcg(previous[b].tokens+i);if(token<h)asc_stcg(saved+uint64_t(b)*h+token,UINT32_MAX);}
  for(uint32_t i=lo+t;i<hi;i+=128)asc_stcg(saved+uint64_t(b)*h+i,UINT32_MAX);
 }
 asc_threadfence();asc_syncthreads();
 if(b<batch){
  for(uint32_t i=t;i<n;i+=128){
   auto token=asc_ldcg(current[b].tokens+i);if(token>=h)continue;
   asc_stcg(saved+uint64_t(b)*h+token,asc_ldcg(live+uint64_t(b)*h+token));
   asc_stcg(previous[b].tokens+i,token);
  }
  if(!t){asc_stcg(&previous[b].count,n);asc_stcg(&previous[b].tailBegin,0U);asc_stcg(&previous[b].tailEnd,0U);}
 }
 asc_threadfence();asc_syncthreads();
 if(b<batch && !t)asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a->liveCount)+3+b,1U);
 asc_threadfence();
}
__aicore__ inline void FetchQueryGroup(__gm__ Args* a,uint32_t q,uint32_t lane,uint32_t workers,bool skipEmptyConflict=true,bool standalone=false){
 auto ctx=reinterpret_cast<__gm__ Context*>(a->context);auto shape=LoadRuntimeShape(ctx);
 if(lane>=shape.batch)return;
 bool directLayout=skipEmptyConflict && !(a->prefetchCount & 0x20000000U);
 auto allFlags=reinterpret_cast<__gm__ BatchFlag*>(standalone && a->completedFlags?a->completedFlags:a->flags);
 bool profile=(a->prefetchCount & 0x80000000U)!=0;
 bool groupLayout=directLayout && ctx->fallbackAddresses;
 uint32_t stride=shape.batch*kTop;
 uint32_t metadataWords=(3*shape.batch+31)&~31U;
 auto packed=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+uint64_t(q*workers+lane)*(3*stride+metadataWords);
 auto ranges=reinterpret_cast<__gm__ uint32_t*>(packed+3*stride);
 auto direct=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+4*workers*(3*stride+metadataWords)+(q*workers+lane)*(6*stride+64);
 // The first 2*batch views include the independent prefetch queues. The
 // following views are paired, exclusively owned by this Q worker.
 uint32_t pair=2*shape.batch+2*(q*workers+lane);
 if(!standalone && a->completedMapping && a->liveCount && shape.batch<=16){
  uint32_t groups=4*workers;
  for(uint32_t first=q*workers+lane;first<shape.batch;first+=groups*2){
   Simt::VF_CALL<SnapshotMappings>(Simt::Dim3{256,1,1},a,shape.batch,first,groups);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
 }
 auto args=GetArgs(ctx,pair);args.jettys_per_core=2;
 args.indices=uint64_t(packed);args.offsets=uint64_t(packed+stride);args.output_indices=uint64_t(packed+2*stride);
 args.miss_count=(!standalone && a->liveCount)?a->liveCount:ctx->count;args.request_capacity=stride;
 if(ctx->fallbackAddresses){args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;args.table_capacity=shape.batch*kSlots;}
 if(!args.batch_get_max_inflight_total)args.batch_get_max_inflight_total=kBatchGetInflight;
 auto compactAddresses=reinterpret_cast<__gm__ uint64_t*>(packed);
 if(groupLayout){args.address_table=uint64_t(compactAddresses);args.table_capacity=stride;}
 uint64_t error=0;uint32_t completed=0,requested=0;bool hasConflict=false;
 for(uint32_t phase=0;phase<2 && !error;++phase){
  if(phase && skipEmptyConflict && !hasConflict)break;
  uint64_t prepareStart=profile?GetSystemCycle():0;
  uint32_t size=0,rangeCount=0;
  if(groupLayout && !phase){
   auto counters=reinterpret_cast<__gm__ uint32_t*>(direct+6*stride)+(standalone && a->completedFlags?32:0);
   uint32_t expected=(shape.batch-1-lane)/workers+1;uint64_t started=GetSystemCycle();
   while(true){
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(counters+kGroupReadyWord),64);
    if(counters[kGroupReadyWord]==expected)break;
    if(counters[kGroupReadyWord]>expected){error|=kvaf_device::kInvalidArgs;break;}
    if(GetSystemCycle()-started>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}
   }
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(counters),64);
   error|=counters[kGroupErrorWord];size=counters[kGroupSafeCountWord];hasConflict=counters[kGroupConflictCountWord]!=0;
   if(size>stride || counters[kGroupConflictCountWord]>stride-size)error|=kvaf_device::kInvalidArgs;
   if(!error){
    if(size)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(compactAddresses),uint64_t(size)*8);
    if(counters[kGroupConflictCountWord])kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(compactAddresses+stride-counters[kGroupConflictCountWord]),uint64_t(counters[kGroupConflictCountWord])*8);
   }
   if(profile){uint64_t ready=GetSystemCycle();for(uint32_t b=lane;b<shape.batch;b+=workers){
    auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
    ft[0]=prepareStart;ft[1]=ready;ft[2]=ready;}}
  }else for(uint32_t b=lane;b<shape.batch;b+=workers){
   auto flags=allFlags+b*kFlagCount;auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   if(profile && !phase)ft[0]=GetSystemCycle();
   uint64_t start=GetSystemCycle();
   while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+q+1),kBatchFlagPayloadBytes);
    if(flags[q+1].ready)break;if(GetSystemCycle()-start>5000000000ULL){error=kvaf_device::kCqTimeout;break;}}
   if(profile && !phase)ft[1]=GetSystemCycle();
   auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),kFinalPostPayloadBytes);
   error|=flags[q+1].error|post->error;
   if(post->count>kTop || post->descriptor!=b*kSlots+q*kTop || post->nonconflicting>post->count ||
      post->descriptor>shape.batch*kSlots || post->count>shape.batch*kSlots-post->descriptor)error|=kvaf_device::kInvalidArgs;
   hasConflict|=post->count>post->nonconflicting;
   uint32_t count=phase?post->count-post->nonconflicting:post->nonconflicting;
   uint32_t begin=post->descriptor+(phase?post->nonconflicting:0);
   if(phase && count && !error){
    if(profile)ft[5]=GetSystemCycle();start=GetSystemCycle();
    while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),kBatchFlagPayloadBytes);
     if(flags[7].ready){error|=flags[7].error;break;}if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}}
    if(profile)ft[6]=GetSystemCycle();
   }
   if(size>stride || count>stride-size)error|=kvaf_device::kInvalidArgs;
   if(error)break;
   if(count){
    if(!directLayout)SyncFetchDescriptors(ctx,begin,count);
    else if(ctx->fallbackAddresses)
     kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->fallbackAddresses)+uint64_t(begin)*8,uint64_t(count)*8);

    ranges[3*rangeCount]=begin;ranges[3*rangeCount+1]=count;ranges[3*rangeCount+2]=size;
    ++rangeCount;size+=count;}
   if(profile && !phase)ft[2]=GetSystemCycle();
  }
  if(error)break;
  uint64_t packStart=profile?GetSystemCycle():0;
  if(rangeCount && !directLayout){
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ranges),uint64_t(rangeCount)*12);
   Simt::VF_CALL<PackFetchRanges>(Simt::Dim3{256,1,1},ctx,packed,stride,ranges,rangeCount);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  if(directLayout){
   auto counters=reinterpret_cast<__gm__ uint32_t*>(direct+6*stride)+(standalone && a->completedFlags?32:0);
   kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(counters),64);
   if(counters[phase*8]!=size)error|=kvaf_device::kInvalidArgs;
   packed=direct+phase*3*stride;
   args.indices=uint64_t(packed);args.offsets=uint64_t(packed+stride);args.output_indices=uint64_t(packed+2*stride);
  }
  if(error)break;
  requested+=size;
  uint64_t packEnd=profile?GetSystemCycle():0;
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
   // KVAF returns an enum. A failed Get can legitimately process fewer
   // descriptors; preserve the transport result instead of OR-ing two enums.
   if(status->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC)error=kvaf_device::kInvalidArgs;
   else if(!error && status->processed_request_count!=size)error=kvaf_device::kInvalidArgs;
  }
#else
  if(size)error|=kvaf_device::kInvalidArgs;
#endif
  if(profile){
   auto coreTrace=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+
       uint64_t(kResolveCores*shape.batch+q*workers+lane)*32;
   coreTrace[2+phase*5]=prepareStart;coreTrace[3+phase*5]=packStart;
   coreTrace[4+phase*5]=packEnd;coreTrace[5+phase*5]=start;coreTrace[6+phase*5]=end;
   for(uint32_t b=lane;b<shape.batch;b+=workers){
   auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   ft[phase?7:3]=start;ft[phase?8:4]=end;ft[12+2*phase]=postCycles;ft[13+2*phase]=waitCycles;
   if(!phase){ft[18]=packStart-prepareStart;ft[19]=packEnd-packStart;}
   ft[16+phase]=groups;ft[20]=3; // shared Q-worker group, not per-request time
  }
  }
 }
 if(a->prefetchCount & 0x10000000U){
  auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+
      uint64_t(kResolveCores*shape.batch+q*workers+lane)*32;
  if(profile)trace[12]=GetSystemCycle();
  // Update may invalidate displaced hot slots. Publish only after it finishes.
  bool deferredResidency=requested<=CHUNK;
  for(uint32_t b=lane;!deferredResidency && b<shape.batch;b+=workers){
   auto flag=allFlags+b*kFlagCount+6;uint64_t start=GetSystemCycle();
   while(true){kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),kBatchFlagPayloadBytes);
    if(flag->ready){error|=flag->error;break;}
    if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}}
  }
  if(profile)trace[13]=GetSystemCycle();
  Simt::VF_CALL<PublishQueryCompletion>(Simt::Dim3{256,1,1},ctx,allFlags,shape.batch,
      shape.resident,shape.finalPostOffset,q,lane,workers,error,completed,deferredResidency);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  uint64_t published=profile?GetSystemCycle():0;
  if(profile){trace[14]=published;
   for(uint32_t b=lane;b<shape.batch;b+=workers){
    auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
    ft[9]=published;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ft),24*8);}
   trace[15]=GetSystemCycle();}
  return;
 }
 for(uint32_t b=lane;b<shape.batch;b+=workers){
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),kFinalPostPayloadBytes);
  post->pending=(b==lane)?completed:0;post->error=uint32_t(error);post->deferred=0;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),kFinalPostPayloadBytes);
  auto flag=allFlags+b*kFlagCount+8+q;flag->error=error;flag->ready=1;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),kBatchFlagPayloadBytes);
  if(profile){auto ft=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*128+uint64_t(b)*128+q*24;
   ft[9]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ft),24*8);}
 }
}
#endif
#if !V9_UBMEM
__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void CollectFinalPosts(__gm__ FinalPost* posts,__ubuf__ uint32_t* output){
 uint32_t q=threadIdx.x,count=0,pending=0,error=0,deferred=0;
 if(q<4){
  auto post=posts+q;
  count=asc_ldcg(&post->count);pending=asc_ldcg(&post->pending);
  error=asc_ldcg(&post->error);deferred=asc_ldcg(&post->deferred)!=0;
  asc_stcg(&post->pending,0U);
 }
 for(uint32_t d=2;d;d>>=1){
  count+=asc_shfl_down(count,d);pending+=asc_shfl_down(pending,d);
  error|=asc_shfl_down(error,d);deferred+=asc_shfl_down(deferred,d);
 }
 if(!q){output[0]=count;output[1]=pending;output[2]=error;output[3]=deferred;}
 asc_threadfence();
}
#endif
#if !V9_UBMEM
#include "joined_final.h"
#endif
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
#if V9_UBMEM
 auto prefetchStatus=UbmemStatusAt(ctx,b);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(prefetchStatus),128);
 if(row->pending || prefetchStatus[2]!=row->requests)row->error|=kUbmemIncomplete;
 arrival=GetSystemCycle();
#else
 CompletePrefetch(ctx,b,true,profile?&polls:nullptr,profile?&drain:nullptr,profile?&arrival:nullptr);
#endif
 if(profile){ft[4]=GetSystemCycle();ft[5]=polls;ft[6]=drain;ft[7]=arrival;}error|=row->error;flags[7].error=error;flags[7].ready=1;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+7),kBatchFlagPayloadBytes);
 uint32_t deferredPrefix=0;
 for(uint32_t f=6;f<12;++f){uint64_t start=GetSystemCycle();if(profile)ft[8+2*(f-6)]=start;while(true){
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flags+f),kBatchFlagPayloadBytes);
  if(flags[f].ready){error|=flags[f].error;break;}
  if(GetSystemCycle()-start>5000000000ULL){error|=kvaf_device::kCqTimeout;break;}}
  if(profile)ft[9+2*(f-6)]=GetSystemCycle();
  if(f==6 && (a->prefetchCount & 0x10000000U)){
   if(profile)ft[20]=GetSystemCycle();
#if !V9_UBMEM
   if(!profile){
    if(!error && row->slots<=kLruSlots)deferredPrefix=row->slots;
   }else
#endif
   if(!error && row->slots && row->slots<=kLruSlots){Simt::VF_CALL<PublishPrefetchResidency>(Simt::Dim3{256,1,1},ctx,b,shape.resident,row->slots);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
   if(profile)ft[21]=GetSystemCycle();
  }
 }
#if !V9_UBMEM && KVIP_OUTPUT_WORKER
 // Output owns flags 1..4. Join after CQ and update work so the output row
 // can finish concurrently; retain its error even if the updater exited early.
 error|=AwaitPublication(flags+4);
#endif
#if !V9_UBMEM
 if(!profile && !error && (a->prefetchCount&0x10000000U) && shape.finalWorkers==4 && ctx->hotValid){
  TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,64);
  auto result=storage.Get<uint32_t>();
  Simt::VF_CALL<FinalizeJoinedRow>(Simt::Dim3{256,1,1},a,ctx,flags,b,shape.batch,shape.resident,
   shape.finalPostOffset,(a->prefetchCount>>16)&0xffU,deferredPrefix,FinalPtr(result));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  uint32_t done=result.GetValue(12);pipe.Destroy();if(done)return;
 }
#endif
 auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*4);
 auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
 auto updateStatus=reinterpret_cast<__gm__ uint32_t*>(update[6])+b;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(updateStatus),sizeof(uint32_t));
 uint32_t updateError=*updateStatus,total=0,completed=0,deferred=0;
 uint64_t lastHead=0,lastCq=0;
 error|=row->error;error|=info[0]?kvaf_device::kInvalidArgs:0;
#if !V9_UBMEM
 if(shape.finalWorkers==4){
  TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,32);
  auto result=storage.Get<uint32_t>();
  Simt::VF_CALL<CollectFinalPosts>(Simt::Dim3{32,1,1},FinalPostAt(shape.finalPostOffset,ctx,b*4),FinalPtr(result));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  total=result.GetValue(0);completed=result.GetValue(1);
  error|=result.GetValue(2);deferred=result.GetValue(3);
  if(deferred)error|=kvaf_device::kInvalidArgs;
  pipe.Destroy();
 }else
#endif
 for(uint32_t q=0;q<shape.finalWorkers;++q){
  auto post=FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+q);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),kFinalPostPayloadBytes);
  total+=post->count;completed+=post->pending;error|=post->error;
  if(post->deferred){++deferred;error|=kvaf_device::kInvalidArgs;}
  post->pending=0;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),kFinalPostPayloadBytes);
 }
#if !V9_UBMEM
 // Status exposes only the last Q queue; earlier snapshots were overwritten.
 if(shape.finalWorkers){
  uint32_t workers=(a->prefetchCount>>16)&0xffU;
  uint32_t pair=2*shape.batch+2*((shape.finalWorkers-1)*workers+b%workers);
  auto queueArgs=reinterpret_cast<__gm__ const KvafExecuteBatchGetKernelArgs*>(ctx->args)+pair;
  auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(queueArgs->queue_state);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
  lastHead=queue->sq_producer_index;lastCq=queue->cq_consumer_index;
 }
#endif
 if(row->pending || total!=info[1])error|=kvaf_device::kInvalidArgs;
 if(profile)ft[1]=GetSystemCycle();
 if(!error && !updateError && !(a->prefetchCount & 0x10000000U)){
  Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},shape.resident,shape.finalPostOffset,ctx,b);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
#if !V9_UBMEM
 if((deferredPrefix || (!error && !updateError && total)) && (a->prefetchCount & 0x10000000U)){
  Simt::VF_CALL<PublishDeferredResidency>(Simt::Dim3{256,1,1},ctx,flags,b,shape.resident,shape.finalPostOffset,deferredPrefix,!error && !updateError && total);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
#endif
 if(profile)ft[2]=GetSystemCycle();
 info[0]|=uint32_t(error);info[3]=deferred?1:0;info[4]=completed;
 if(!error)row->finalPending=0;
#if !V9_UBMEM
 if(error){
  // Flags 8..11 have joined. Keep each Q's raw failure in the existing
  // feedback copy, before retirement clears the live flags.
  uint32_t codes=0;
  for(uint32_t q=0;q<4;++q){
   uint32_t code=flags[8+q].error;
   codes|=(code<255U?code:255U)<<(8*q);
  }
  row->finalQueryErrors=codes;
 }
#endif
#if V9_UBMEM
 auto status=UbmemStatusAt(ctx,b);status[0]=error|updateError;status[1]=row->requests;
 status[3]=total;status[4]=completed;status[5]=(status[2]+completed)*shape.tokenBytes;
 status[6]=UbmemRegionAt(ctx)->generation;
 if(total!=completed){error|=kUbmemIncomplete;status[0]|=error;info[0]|=uint32_t(error);}
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
#else
 auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+b*2;
 kvaf_device::WriteStatus(status,error,total,completed,lastHead,lastCq,UINT64_MAX,0);
 kvaf_device::WriteStatus(status+1,0,0,0,0,0,UINT64_MAX,0);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),2*sizeof(*status));
#endif
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
 if(error || updateError){
  Simt::VF_CALL<CompleteEarlyUpdate<256>>(Simt::Dim3{256,1,1},update,b,error);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
 if(profile){ft[3]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ft),32*8);}
}

#endif
#if defined(__CCE_AICORE__) && !V9_UBMEM
__simt_vf__ __aicore__ LAUNCH_BOUND(128)
inline void RetireRoundControl(__gm__ Args* a,uint32_t batch,uint32_t workers,uint32_t block){
 uint32_t t=threadIdx.x;
 if(block<batch && a->completedRows){
  auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
  auto live=reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+block*(sizeof(Row)/4);
  auto snapshot=reinterpret_cast<__gm__ uint32_t*>(a->completedRows)+block*(sizeof(Row)/4);
  for(uint32_t i=t;i<sizeof(Row)/4;i+=128){
   asc_stcg(snapshot+i,asc_ldcg(live+i));asc_stcg(live+i,0U);
  }
 }
 if(block<batch && a->completedEvents){
  auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
  uint32_t words=ctx->shape.rounds*ctx->shape.serviceLanes*(sizeof(Event)/4);
  auto events=reinterpret_cast<__gm__ uint32_t*>(ctx->events);
  auto saved=reinterpret_cast<__gm__ uint32_t*>(a->completedEvents);
  for(uint32_t i=block*128+t;i<words;i+=batch*128){
   asc_stcg(saved+i,asc_ldcg(events+i));asc_stcg(events+i,0U);
  }
 }
 if(!block && !t && a->liveCount){
  auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
  auto live=reinterpret_cast<__gm__ uint32_t*>(a->liveCount);
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->count),asc_ldcg(live));
  asc_stcg(live,0U);
 }
 if(block<batch && t<kFlagCount*3){
  uint32_t word=(block*kFlagCount+t/3)*kFlagWords+t%3;
  auto live=reinterpret_cast<__gm__ uint32_t*>(a->flags)+word;
  asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a->completedFlags)+word,asc_ldcg(live));
  asc_stcg(live,0U);
 }
 if(block<4*workers && t<32){
  uint32_t stride=batch*kTop,meta=(3*batch+31)&~31U;
  auto control=reinterpret_cast<__gm__ uint32_t*>(a->packedFetch)+4*workers*(3*stride+meta)+block*(6*stride+64)+6*stride+t;
  asc_stcg(control+32,asc_ldcg(control));
  asc_stcg(control,0U);
 }
 asc_threadfence();
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
  bool profile=(a->prefetchCount & 0x80000000U)!=0;
  auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(shape.batch)*256+uint64_t(block)*32;
  if(profile)trace[0]=GetSystemCycle();
  if(block<shape.batch*kResolveCores){uint32_t b=block/kResolveCores,role=block%kResolveCores;
   if(role==0){
    Producer(a,b);
    FinalizeStreaming(a,b);
   }else if(role==1)Consumer(a,b);
   else OutputWorker(a,b);
#if !V9_UBMEM
   if(role==kResolveCores-1){
    // All score producers have joined and the final Fetch uses separate flags.
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    const uint32_t words=(shape.history+255)/256*8;
    constexpr uint32_t chunk=4096;
    const uint32_t size=words<chunk?words:chunk;
    constexpr uint32_t progressWords=4*sizeof(Progress)/sizeof(uint32_t);
    const uint32_t clearWords=size>progressWords?size:progressWords;
    TPipe pipe;TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage,clearWords*sizeof(uint32_t));
    auto zero=storage.Get<uint32_t>();Duplicate(zero,0U,clearWords);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<uint32_t> covered;
    covered.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(ctx->incremental,shape.batch,shape.history,b)));
    for(uint32_t offset=0;offset<words;offset+=size)
        DataCopy(covered[offset],zero,words-offset<size?words-offset:size);
    if(ctx->progress){
        GlobalTensor<uint32_t> progress;
        progress.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+b*progressWords);
        DataCopy(progress,zero,progressWords);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
   }
#endif
  }
  else {uint32_t lane=block-shape.batch*kResolveCores,q=lane/workers,part=lane%workers;
   if(q<4 && part<workers){
    FetchQueryGroup(a,q,part,workers);
   }}
#if !V9_UBMEM
  if(a->completedFlags){
   AscendC::SyncAll<true>();
   if(block<shape.batch || block<4*workers)
    Simt::VF_CALL<RetireRoundControl>(Simt::Dim3{128,1,1},a,shape.batch,workers,block);
  }
#endif
  if(profile){trace[1]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),32*8);}
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
  bool skipEmptyConflict=(query & 256U)==0;query &= 255U;
  uint32_t block=GetBlockIdx();
  if(block<(query==4?8U:2U) && query<5){
   uint32_t workers=2;
   uint32_t q=query==4?block/2:query;
   uint32_t lane=query==4?block%2:block;
   #if V9_UBMEM
   FetchQueryGroup(reinterpret_cast<__gm__ Args*>(arg),q,lane,workers,skipEmptyConflict);
#else
   FetchQueryGroup(reinterpret_cast<__gm__ Args*>(arg),q,lane,workers,skipEmptyConflict,true);
#endif
  }
 }
#endif
}

// Native reference: a real KVAF BatchGet dispatch across 2 or 8 AIVs.
// Inputs are compacted before timing, as in native LIDU.
#if !V9_UBMEM
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

#endif
