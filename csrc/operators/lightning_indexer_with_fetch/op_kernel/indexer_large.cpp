#include "indexer/large_pipeline.h"

extern "C" __global__ __aicore__ void v9_li_large_hot(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Inputs input(bindings, terminal);
    bool helper = input.core >= input.shape.batch;
    RowPlan plans[7];
    uint32_t own;
    uint32_t planCount = Plan<false>(input, helper, plans, own);
    if (helper) {
        for (uint32_t i = 0; i < planCount; ++i) {
            Hot(input, plans[i], true);
        }
    } else {
        Hot(input, plans[own], false);
    }
#endif
}

extern "C" __global__ __aicore__ void v9_li_large_threshold_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ThresholdCold(bindings, terminal);
#endif
}
