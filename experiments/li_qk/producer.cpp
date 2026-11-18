// New entrypoint only: existing li_score target/default path is unchanged.
#define LI_PRODUCER_CONTROLS 1
#include "score.cpp"
extern "C" __global__ __aicore__ void li_qk_score_producer(
    GM_ADDR bindings,GM_ADDR output,uint32_t rows,uint32_t stride,uint32_t heads,uint32_t group) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunScoreRow(bindings,rows,stride,heads,group,false,output);
#endif
}
