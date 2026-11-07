// SPDX-License-Identifier: Apache-2.0
// Execute the production resolver and WQE encoder without posting network work.
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
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t own=ctl.GetValue(3), prefix=ctl.GetValue(2);
    auto cfg=ctl[1024];
    for(uint32_t j=0;j<32;++j)cfg.SetValue(j,0);
    cfg.SetValue(1,6U<<8);cfg.SetValue(2,0x1000123);cfg.SetValue(3,0x456);
    cfg.SetValue(12,656);cfg.SetValue(13,0x789);cfg.SetValue(14,0x80000000);cfg.SetValue(15,0x1200);
    cfg.SetValue(16,2091);cfg.SetValue(17,32768);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> wqes;wqes.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->args)+w*8192*16);
    for(uint32_t first=0;first<own;first+=1024) {
      uint32_t n=own-first<1024?own-first:1024;
      FinalBuild(656,FinalPtr(keys),FinalPtr(ranks),FinalPtr(positions),FinalPtr(local),FinalPtr(cfg),first,n,own);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(wqes[first*16],local,n*16);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+w*32;
    status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);status[2]=ctl.GetValue(5);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
  }
#endif
}
