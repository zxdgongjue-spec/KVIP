// SPDX-License-Identifier: Apache-2.0
// Isolated experimental Q-by-Q resolver; no Fetch and no production launch.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "resolver.h"
#include "resolver_staged_dma2.h"
using namespace AscendC;
using namespace rotating;
#endif
extern "C" __global__ __aicore__ void test_staged_resolver(GM_ADDR context) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        auto ctx=reinterpret_cast<__gm__ Context*>(context);
        auto shape=LoadRuntimeShape(ctx);
        uint32_t b=GetBlockIdx();
        if(b>=shape.batch)return;
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        TPipe pipe;TBuf<TPosition::VECCALC> scratch;
        pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
        auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
        auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
        auto slots=local[kFinalSlotsOffset],work=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
        GlobalTensor<uint32_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
        DataCopy(selected,input,kSlots);
        // Same bulk metadata staging used by the production resolver. It is
        // released immediately after the global protection prepass.
        GlobalTensor<uint32_t> metadata;
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if(ctx->hotValid){metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*shape.resident);
            DataCopy(keys,metadata,shape.resident);}
        if(ctx->reservedHot){metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*shape.resident);
            DataCopy(local[4*kSlots],metadata,shape.resident);}
        metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
        DataCopy(work[4096],metadata,kLruSlots);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<StagedInitialize>(Simt::Dim3{512,1,1},shape.history,shape.resident,b,row->slots,
            ctx,FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),
            reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(ctl));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        GlobalTensor<uint32_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
        auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*32;
        for(uint32_t q=0;q<4 && !ctl.GetValue(1);++q){
            Simt::VF_CALL<StagedQuery>(Simt::Dim3{512,1,1},q,shape.history,shape.resident,shape.tokenBytes,
                b,row->slots,ctx,FinalPtr(selected),FinalPtr(loc),FinalPtr(keys),FinalPtr(ranks),
                reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(work),FinalPtr(ctl));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if(ctl.GetValue(1))break;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(output[q*kTop],loc[q*kTop],kTop);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            status[4+q]=ctl.GetValue(2+q);status[8+q]=ctl.GetValue(0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
        }
        status[0]=ctl.GetValue(1);status[1]=ctl.GetValue(0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
    }
#endif
}
