#if !V9_UBMEM && KVIP_STREAMING_LI
#define KVIP_URMA_WINDOW_LI 1
#endif
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "pipeline_types.h"

struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer/indexer.h"

namespace {
using namespace AscendC;
using namespace QLICommon;
using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,
    LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;

__aicore__ inline void RunLiFinalize(GM_ADDR bindings) {
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    if (!P(9) || !P(12) || !P(14) || !P(15)) return;
    auto shape=rotating::LoadRuntimeShape(
        reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    const uint32_t query=GetBlockIdx();
    if (query>=shape.batch*4) return;

    TPipe pipe;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_=shape;
    op.rotationContext_=P(14);
    op.sequenceSplit_=false;
    op.sequenceTopKOnly_=true;

    QLITilingData tiling{
        shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,
        2048,shape.liCores,128,shape.history/128,0,
        16384,128,shape.resident,shape.splitQ,shape.aicCount
    };

    // Set before Init: topkOnly InitBuffers allocates streaming metadata
    // scratch and does not initialize the Cube/Score execution path.
    op.vectorService.SetStreamingWorkspace(P(15));
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
    op.vectorService.SetPrefixStorage(P(13));
    op.vectorService.SetRotationContext(P(14));

    op.vectorService.AllocEventID();
    op.vectorService.FinalizeExternalTopK(query);
    op.vectorService.FreeEventID();
    pipe.Destroy();
#undef P
}
}
#endif

extern "C" __global__ __aicore__ void v9_nextgen_finalize_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLiFinalize(bindings); }
#endif
}
