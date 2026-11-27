// SPDX-License-Identifier: Apache-2.0
#include "row_kernel.h"
extern "C" __global__ __aicore__ void v9_terminal_stagger(GM_ADDR binding) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { RunTerminalTeams(binding, GetBlockIdx() / 4, GetBlockIdx() % 4); }
#endif
}
