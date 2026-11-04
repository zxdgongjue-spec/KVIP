#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize=SIM_BATCH,tSize=SIM_BATCH*4,gSize=INDEX_HEADS,s1Size=4,s2Size=SIM_HISTORY;
    uint32_t sparseCount=2048,usedCoreNum=rotating::kLiCores,blockSize=128,maxBlockNumPerBatch=SIM_HISTORY/128,sparseMode=0;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
};
#define SIM_STORAGE_HISTORY SIM_HISTORY
#include "indexer.h"
#endif
extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC; using namespace QLICommon;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    QLITilingData tiling;
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
    if ASCEND_IS_AIV {
        op.vectorService.SetPrefixStorage(P(13));
        op.vectorService.SetRotationContext(P(14));
    }
    op.Process(); pipe.Destroy();
#undef P
#endif
}
