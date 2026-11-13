#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "indexer.h"
#endif

// P0-C5a: fixed-B4 Q1/K128K diagnostic ONLY. No TopK, Fetch or Epoch.
// The default v9_li_kernel and v9_bridge launch path remain unchanged.
extern "C" __global__ __aicore__ void v9_s2_score_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace AscendC;
    using namespace QLICommon;
    auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(a[14]);
    auto shape = rotating::LoadRuntimeShape(ctx);
    constexpr uint32_t groups = V9_S2_SCORE_CORES;
    // Diagnostic is fail-closed if called outside the admitted ABI.
    if (shape.batch != 4 || shape.history != 131072 || shape.resident != 8192 ||
        shape.indexHeads != 32 || shape.splitQ != 0 || shape.aicCount < groups ||
        ctx->prefetchLimit != 0) return;
    TPipe pipe;
    using Type = QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,
        LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_ = shape;
    op.SetScoreOnlyS2(true);
    // Do not split Query: Q1 is mapped to AIV lane 0 of each AIC group.
    QLITilingData tiling{shape.batch,shape.batch * 4,shape.indexHeads,4,
        shape.history,2048,groups,128,shape.history / 128,0,
        16384,128,shape.resident,0,shape.aicCount};
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
    // ProcessVec1 uses runtimeShape_.resident. Set the context even though
    // this mode suppresses Hit and Progress publication.
    if ASCEND_IS_AIV {
        op.vectorService.SetRotationContext(P(14));
        op.vectorService.SetPrefixStorage(P(13));
    }
    op.ConfigureSingleQueryScoreShard(groups);
    op.Process();
#undef P
    pipe.Destroy();
#endif
}
