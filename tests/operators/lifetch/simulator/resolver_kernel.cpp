// SPDX-License-Identifier: Apache-2.0
// Execute the production resolver without posting network work.
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
    uint32_t w=GetBlockIdx(),h=shape.history,r=shape.resident,workers=shape.finalWorkers;
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations));
    DataCopy(selected,input,kSlots);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},r,ctx,0,FinalPtr(selected),FinalPtr(keys));
    PipeBarrier<PIPE_V>();
    ColdPositions(h,r,FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows);
    Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},h,r,workers,shape.tokenBytes,
      FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),ctl.GetValue(2048),FinalPtr(keys),FinalPtr(ranks),
      reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
      reinterpret_cast<__gm__ uint32_t*>(ctx->mapping),reinterpret_cast<__gm__ uint64_t*>(row->addresses),
      reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
      reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,0,ctx);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/workers)],loc[w*(kSlots/workers)],kSlots/workers);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+w*32;
    status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);status[2]=ctl.GetValue(5);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  }
#endif
}
