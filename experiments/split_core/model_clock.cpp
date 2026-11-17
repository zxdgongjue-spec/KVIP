#include "kernel_operator.h"
// Model-only probes bracket the forward; fixture upload and oracle readback
// stay outside. These two AIV kernels are common to every comparison case.
extern "C" __global__ __aicore__ void kvip_model_clock(GM_ADDR output,uint32_t index) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        reinterpret_cast<__gm__ uint64_t*>(output)[index]=AscendC::GetSystemCycle();
    }
}
