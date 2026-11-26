// The specialized cold path shares hot scoring and the final terminal with the default launch.
#define KVIP_LARGE_SPLIT_MIDDLE 1
#include "indexer/large_pipeline.h"

extern "C" __global__ __aicore__ void v9_li_large_middle_cold(GM_ADDR bindings, GM_ADDR terminal) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ThresholdCold(bindings, terminal);
#endif
}
