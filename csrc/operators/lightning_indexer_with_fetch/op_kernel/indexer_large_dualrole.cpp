// Isolated device entry. The regular hot and terminal launches keep their original images.
#define KVIP_BS32_SCORE_WINDOW_ROLES 1
#if KVIP_BS32_SHARE_FINAL
#if KVIP_TEST_SPLIT_ROLE_TRACE
#define QLIKernel QLIKernelBS32SharedFinalTrace
#define v9_li_large_bs32_split_role v9_li_large_bs32_shared_final_trace
#else
#define QLIKernel QLIKernelBS32SplitRole
#define v9_li_large_bs32_split_role v9_li_large_bs32_shared_final
#endif
#elif KVIP_TEST_SPLIT_ROLE_TRACE
#define QLIKernel QLIKernelBS32SplitRoleTrace
#define v9_li_large_bs32_split_role v9_li_large_bs32_split_role_trace
#else
#define QLIKernel QLIKernelBS32SplitRole
#endif
#include "indexer_large.cpp"

extern "C" __global__ __aicore__ void v9_li_large_bs32_split_role(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Inputs input(bindings, terminal);
    const uint32_t row = input.core;
    Threshold(input, row, false);
    auto plan = RowCost<true>(input, row);
    ColdOwner(input, plan);
#endif
}
