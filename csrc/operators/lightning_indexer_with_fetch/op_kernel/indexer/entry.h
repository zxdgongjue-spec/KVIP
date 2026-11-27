#pragma once
#if !V9_UBMEM && KVIP_STREAMING_LI
// Only the URMA LI kernel uses the shared window rank selection.
#define KVIP_URMA_WINDOW_LI 1
#endif
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
#include "indexer.h"
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

template <SequenceLiPhase Phase>
__aicore__ inline void RunLi(GM_ADDR bindings
#if KVIP_FIXED_SCORE_SHARDS
                             , GM_ADDR ready
#endif
) {
    using namespace AscendC; using namespace QLICommon;
    auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
#define P(i) reinterpret_cast<GM_ADDR>(a[i])
    auto shape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(a[14]));
    bool splitSequence = rotating::UseSequenceSplit(shape) && HasUniformLengths(shape, P(5), P(6));
    if constexpr (Phase != SequenceLiPhase::Hot) {
        if (!splitSequence) return;
    }
#if KVIP_URMA_WINDOW_LI && !KVIP_LARGE_BATCH_LI
    if constexpr (Phase == SequenceLiPhase::Threshold) {
        if ASCEND_IS_AIV { fixed_window_q::Reset(P(15), shape, GetBlockIdx()); }
    }
#endif
#if KVIP_FIXED_SCORE_SHARDS
    uint32_t physical = GetBlockIdx();
    if ASCEND_IS_AIV { physical /= 2; }
    if (!splitSequence && physical >= shape.liCores) return;
#endif
    TPipe pipe;
    using Type=QLIType<fp8_e4m3fn_t,fp8_e4m3fn_t,int32_t,true,LI_LAYOUT::BSND,LI_LAYOUT::PA_BSND,
        bfloat16_t,float,float,uint16_t>;
    QLIKernel::QuantLightningIndexerKernel<Type> op;
    op.runtimeShape_ = shape;
#if KVIP_FIXED_SCORE_SHARDS
    op.fixedReady_ = reinterpret_cast<__gm__ uint32_t*>(ready);
    if constexpr (Phase == SequenceLiPhase::Cold) {
        op.fixedDeferCredit_ = splitSequence && fixed_window_q::DeferCredits(P(15), shape);
    }
    op.vectorService.fixedHelper_ = splitSequence && physical >= shape.liCores;
#endif
#if !V9_UBMEM
    op.rotationContext_ = P(14);
#endif
    op.sequenceSplit_ = splitSequence;
    op.sequenceTopKOnly_ = Phase == SequenceLiPhase::Threshold || Phase == SequenceLiPhase::Final;
    QLITilingData tiling{shape.batch,shape.batch*4,shape.indexHeads,4,shape.history,
        2048,shape.liCores,128,shape.history/128,0,16384,128,shape.resident,
        splitSequence ? 0U : shape.splitQ,shape.aicCount};
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
