#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#if !V9_UBMEM
#include "round_prepare.h"
#endif
struct QLITilingData {
    uint32_t bSize,tSize,gSize,s1Size,s2Size,sparseCount,usedCoreNum,blockSize,maxBlockNumPerBatch,sparseMode;
    uint32_t keyStride0=16384,keyDequantScaleStride0=128;
    uint32_t resident,splitQ,aicCount;
};
#include "../../lightning_indexer_with_fetch/op_kernel/indexer/indexer.h"
namespace {
using QLIKernel::SequenceLiPhase;

__aicore__ inline bool HasUniformLengths(const rotating::RuntimeShape& shape, GM_ADDR q, GM_ADDR k) {
    auto query = reinterpret_cast<__gm__ const uint32_t*>(q);
    auto key = reinterpret_cast<__gm__ const uint32_t*>(k);
    for (uint32_t batch = 0; batch < shape.batch; ++batch) {
        if ((query && query[batch] != 4) || (key && key[batch] != shape.history)) return false;
    }
    return true;
}

template <SequenceLiPhase Phase, bool ScoreOnly = false, bool Precomputed = false>
__aicore__ inline void RunLi(GM_ADDR bindings, uint32_t scoreParts = 1,
    uint32_t begin = 0, uint32_t end = 0, bool predicate = false) {
    using namespace AscendC; using namespace QLICommon;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    bool splitSequence = !ScoreOnly && rotating::UseSequenceSplit(shape) && HasUniformLengths(shape, P(5), P(6));
    if constexpr (Phase != SequenceLiPhase::Hot) {
        if (!splitSequence) return;
    }
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_ = shape;
#if !V9_UBMEM
    op.rotationContext_ = P(14);
    op.scoreOnly_ = ScoreOnly;
    op.scoreParts_ = scoreParts;
    op.precomputedScore_ = Precomputed;
    op.scoreBegin_ = begin; op.scoreEnd_ = end;
    op.producerPredicate_ = predicate;
#endif
    op.sequenceSplit_ = splitSequence;
    op.sequenceTopKOnly_ = Phase == SequenceLiPhase::Threshold || Phase == SequenceLiPhase::Final;
    QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,
        2048,ScoreOnly ? shape.batch * scoreParts : shape.liCores,128,shape.history/128,0,16384,128,shape.resident,
        (splitSequence || ScoreOnly) ? 0U : shape.splitQ,shape.aicCount};
    if ASCEND_IS_AIV { op.vectorService.SetStreamingWorkspace(P(15)); }
    op.Init(P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(9),P(12),&tiling,&pipe);
    if ASCEND_IS_AIV {
        op.vectorService.SetPrefixStorage(P(13));
        op.vectorService.SetRotationContext(P(14));
    }
    if constexpr (Phase == SequenceLiPhase::Hot) {
        if (splitSequence) op.template ProcessSequenceSplit<Phase>();
        else op.Process();
    } else {
        op.template ProcessSequenceSplit<Phase>();
    }
    pipe.Destroy();
#undef P
}
}
#endif

extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Hot>(bindings);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_1(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::Threshold>(bindings); }
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_2(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Cold>(bindings);
#endif
}

extern "C" __global__ __aicore__ void v9_li_kernel_3(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::Final>(bindings); }
#endif
}

#if !V9_UBMEM
// Keep the ordinary entry/defaults intact. These two opt-in kernels replace
// Hot with complete canonical scores and Cold with the original T2 consumer.
extern "C" __global__ __aicore__ void v9_li_kernel_score_t2(GM_ADDR bindings, uint32_t parts) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Hot, true>(bindings, parts);
#endif
}
extern "C" __global__ __aicore__ void v9_li_kernel_cold_t2(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV { RunLi<SequenceLiPhase::Cold, false, true>(bindings); }
#endif
}
// MR14 direct producer mechanics on real MR23 Q/owner inputs: score and Cold
// predicate are written by the same history shard; Service publishes readiness.
extern "C" __global__ __aicore__ void v9_li_kernel_score_window_t2(
    GM_ADDR bindings, uint32_t parts, uint32_t begin, uint32_t end, uint32_t predicate) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunLi<SequenceLiPhase::Hot, true>(bindings, parts, begin, end, predicate != 0);
#endif
}
#endif
