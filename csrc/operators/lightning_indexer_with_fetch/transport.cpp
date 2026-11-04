#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "kvaf_batch_get_device.h"
#endif

// Prefill and auxiliary restores use the unchanged standalone BatchGet
// algorithm, sharing the lease stream and persistent cursors with LIFetch.
extern "C" __global__ __aicore__ void v9_get_kernel(GM_ADDR arguments) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        KvafExecuteBatchGetKernelArgs args{};
        auto input = reinterpret_cast<__gm__ uint64_t*>(arguments);
        auto words = reinterpret_cast<uint64_t*>(&args);
        static_assert(sizeof(args) == 14 * sizeof(uint64_t), "KVAF argument ABI changed");
        for (uint32_t i = 0; i < 14; ++i) words[i] = input[i];
        kvaf_device::KvafBatchGetDevice(args, AscendC::GetBlockIdx());
    }
#endif
}
