// Extra K-page producers for small fixed batches. Original PR30 entry points
// are compiled independently and use the original scoring and service layout.
#define KVIP_FIXED_SCORE_SHARDS 1
#define QLIKernel FixedScoreQLIKernel
#include "kernel_operator.h"
#include "fixed_score_groups.h"
#include "terminal/terminal_runtime_abi.h"
#include "indexer/entry.h"

#if defined(__CCE_AICORE__)
namespace {
template <SequenceLiPhase Phase>
__aicore__ inline void FixedLi(GM_ADDR bindings, GM_ADDR terminal) {
    auto args = reinterpret_cast<__gm__ terminal_runtime::Binding*>(terminal);
    auto ready = reinterpret_cast<__gm__ uint32_t*>(args->reserved[3]);
#if KVIP_STAGE_TRACE
    const uint32_t traceKernel = Phase == SequenceLiPhase::Hot ? kvip_trace::Hot : kvip_trace::Cold;
    auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
    const uint64_t trace = KVIP_TRACE_WORKSPACE(reinterpret_cast<GM_ADDR>(a[15]));
#endif
    KVIP_TRACE_RECORD(kernelScope, trace, traceKernel, kvip_trace::KernelBody, 0, AscendC::GetBlockIdx());
    if constexpr (Phase == SequenceLiPhase::Hot) {
        if ASCEND_IS_AIV {
            KVIP_TRACE_BEGIN(resetScope, trace, Hot, ColdProgressReset, 0, AscendC::GetBlockIdx());
            auto line = ready + AscendC::GetBlockIdx() * 32;
            *line = 0;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(line), 128);
        }
    }
    RunLi<Phase>(bindings, reinterpret_cast<GM_ADDR>(ready));
}
}
#endif
extern "C" __global__ __aicore__ void v9_li_fixed_hot(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FixedLi<SequenceLiPhase::Hot>(bindings, terminal);
#endif
}

extern "C" __global__ __aicore__ void v9_li_fixed_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FixedLi<SequenceLiPhase::Cold>(bindings, terminal);
#endif
}
