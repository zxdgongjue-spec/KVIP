#define KVIP_FIXED_SMALL_TERMINAL 1
#include "row_kernel.h"

extern "C" __global__ __aicore__ void v9_parallel_terminal_fixed(GM_ADDR binding) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  RunTerminalRow<false>(binding, GetBlockIdx()/4, GetBlockIdx()%4);
 }
#endif
}
