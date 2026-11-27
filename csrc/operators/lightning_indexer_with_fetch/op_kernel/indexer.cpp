#define KVIP_WINDOW_Q1 1
#include "indexer/entry.h"
#if defined(__CCE_AICORE__)
#include "terminal/terminal_runtime_abi.h"
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

extern "C" __global__ __aicore__ void v9_li_window_q1_threshold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        auto args = reinterpret_cast<__gm__ terminal_runtime::Binding*>(terminal);
        auto words = reinterpret_cast<__gm__ const uint64_t*>(bindings);
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(words[14]);
        auto scratch = reinterpret_cast<GM_ADDR>(args->reserved[5]);
        fixed_window_q1::Reset(scratch, rotating::LoadRuntimeShape(ctx).batch, AscendC::GetBlockIdx());
        RunLi<SequenceLiPhase::Threshold>(bindings);
    }
#endif
}

extern "C" __global__ __aicore__ void v9_li_window_q1_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    auto args = reinterpret_cast<__gm__ terminal_runtime::Binding*>(terminal);
    RunLi<SequenceLiPhase::Cold>(bindings, reinterpret_cast<GM_ADDR>(args->reserved[5]));
#endif
}
