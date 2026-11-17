#include "kernel_operator.h"
// Model-only probes bracket the forward; fixture upload and oracle readback
// stay outside. These two AIV kernels are common to every comparison case.
extern "C" __global__ __aicore__ void kvip_model_clock(GM_ADDR output,uint32_t index) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        reinterpret_cast<__gm__ uint64_t*>(output)[index]=AscendC::GetSystemCycle();
        // Scalar GM stores remain in the AIV data cache on Model. Publish
        // the timestamp before a later kernel/Host consumes the shared line.
        AscendC::GlobalTensor<uint8_t> bytes;
        bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(bytes[(index*sizeof(uint64_t))&~uint32_t(63)]);
    }
}
