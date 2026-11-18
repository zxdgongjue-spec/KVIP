#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "low_miss_mode.h"
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

__aicore__ inline bool LiNoPrefetch(__gm__ rotating::Context* ctx,
                                    const __gm__ uint64_t* bindings, bool afterInit) {
#if KVIP_LOW_MISS_POST_INDEX
    // Only the combined graph launch has an init -> LI edge for this invocation.
    if (afterInit) {
        auto args = reinterpret_cast<const __gm__ pipeline_ab::Args*>(bindings[15]);
        if (args && args->postIndex) {
            auto word = reinterpret_cast<__gm__ uint64_t*>(args->postIndex) + pipeline_ab::kPostIndexRouteWord;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(*word));
            const uint64_t route = *word;
            if (route <= 1) return route == 1;
        }
    }
#endif
#if KVIP_AUTO_ROUTE
    if (ctx->autoRoute && ctx->admissionControls)
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->admissionControls),ctx->shape.batch*16);
#endif
    return rotating::NoPrefetchRound(ctx);
}

template <SequenceLiPhase Phase>
__aicore__ inline void RunLi(GM_ADDR bindings, bool afterInit=false) {
    using namespace AscendC; using namespace QLICommon;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(a[14]);
    auto shape = rotating::LoadRuntimeShape(ctx);
    const bool noPrefetch = LiNoPrefetch(ctx, a, afterInit);
    bool splitSequence = rotating::UseSequenceSplit(shape) && HasUniformLengths(shape, P(5), P(6));
    if constexpr (Phase != SequenceLiPhase::Hot) {
        if (!splitSequence) return;
    }
    if constexpr (Phase == SequenceLiPhase::Threshold) {
        // There is no speculative threshold consumer in an exact-only round.
        // Return before allocating buffers or issuing any cross-core signals.
        if (noPrefetch) return;
    }
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_ = shape;
    op.rotationContext_ = P(14);
    op.sequenceSplit_ = splitSequence;
    op.noPrefetchRound_ = noPrefetch;
    op.sequenceTopKOnly_ = Phase == SequenceLiPhase::Threshold || Phase == SequenceLiPhase::Final;
    QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,
        2048,shape.liCores,128,shape.history/128,0,16384,128,shape.resident,
        splitSequence ? 0U : shape.splitQ,shape.aicCount};
    if ASCEND_IS_AIV {
        op.vectorService.SetStreamingWorkspace(P(15));
        op.vectorService.SetNoPrefetchRound(noPrefetch);
#if KVIP_PREFETCH_GATE
        op.vectorService.SetAdmissionControls(
            reinterpret_cast<__gm__ rotating::Context*>(a[14])->admissionControls);
#endif
    }
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

#if KVIP_LOW_MISS_POST_INDEX
extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings, uint32_t afterInit) {
#else
extern "C" __global__ __aicore__ void v9_li_kernel(GM_ADDR bindings) {
#endif
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if KVIP_LOW_MISS_POST_INDEX
    RunLi<SequenceLiPhase::Hot>(bindings, afterInit != 0);
#else
    RunLi<SequenceLiPhase::Hot>(bindings);
#endif
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
