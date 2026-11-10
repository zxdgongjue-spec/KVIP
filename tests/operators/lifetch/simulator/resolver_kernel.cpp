// SPDX-License-Identifier: Apache-2.0
// Execute production resolver and conflict partition without network work.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "resolver.h"
using namespace AscendC;
using namespace rotating;
#endif
extern "C" __global__ __aicore__ void test_hot_resolver(GM_ADDR context) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    auto ctx=reinterpret_cast<__gm__ Context*>(context);
    auto shape=LoadRuntimeShape(ctx);
    uint32_t core=GetBlockIdx(),h=shape.history,r=shape.resident,workers=shape.finalWorkers;
    uint32_t b=core/workers,w=core%workers; if(b>=shape.batch)return;
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    DataCopy(selected,input,kSlots);
    if(r<=2*kSlots) {
      // Match the service resolver: bulk metadata staging before classification.
      GlobalTensor<uint32_t> metadata;
      uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
      if(ctx->hotValid){metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*r);
        DataCopy(keys,metadata,r);}
      if(ctx->reservedHot){metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*r);
        DataCopy(local[4*kSlots],metadata,r);}
      uint32_t begin=0,end=0;
      if(ctx->tailBounds){auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;begin=bounds[0];end=bounds[1];}
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{512,1,1},r,FinalPtr(selected),FinalPtr(loc),
        FinalPtr(keys),reinterpret_cast<__ubuf__ int32_t*>(local[4*kSlots].GetPhyAddr()),
        ctx->hotValid!=0,ctx->reservedHot!=0,begin,end);
      PipeBarrier<PIPE_V>();
      ColdPositions(h,r,FinalPtr(loc),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    } else {
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},r,ctx,b,FinalPtr(selected),FinalPtr(keys));
      PipeBarrier<PIPE_V>();
      ColdPositions(h,r,FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},h,r,workers,shape.tokenBytes,
      FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),ctl.GetValue(2048),FinalPtr(keys),FinalPtr(ranks),
      reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
      reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*h,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
      reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
      reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/workers)],loc[w*(kSlots/workers)],kSlots/workers);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+core*32;
    status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);status[2]=ctl.GetValue(5);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  }
#endif
}
