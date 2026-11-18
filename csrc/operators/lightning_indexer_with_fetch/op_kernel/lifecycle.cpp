#include "round_prepare.h"
extern "C" __global__ __aicore__ void v9_init_kernel(
    GM_ADDR bindings,uint32_t fullReset,GM_ADDR workspace,
    uint32_t rt_batch, uint32_t rt_history, uint32_t rt_resident, uint32_t rt_rounds,
    uint32_t rt_serviceLanes, uint32_t rt_finalWorkers, uint32_t rt_finalPostOffset, uint32_t rt_tokenBytes) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    PrepareRequestPart(bindings,fullReset,workspace,rt_batch,rt_history,rt_resident,rt_rounds,
                       rt_serviceLanes,rt_finalWorkers,rt_finalPostOffset,rt_tokenBytes,GetBlockIdx());
#endif
}
// Failure-only diagnostic. Never called by the forward pipeline; queue reads
// must execute on device because driver queue VAs are not ACL allocations.
extern "C" __global__ __aicore__ void kvip_transport_snapshot_kernel(
    GM_ADDR source, GM_ADDR destination, uint32_t bytes) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        constexpr uint32_t kChunkBytes = 4096;
        pipe.InitBuffer(storage, kChunkBytes);
        auto local = storage.Get<uint32_t>();
        GlobalTensor<uint32_t> input, output;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(source));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destination));
        for (uint32_t offset = 0; offset < bytes; offset += kChunkBytes) {
            uint32_t count = bytes - offset;
            if (count > kChunkBytes) count = kChunkBytes;
            DataCopy(local, input[offset / 4], count / 4);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(output[offset / 4], local, count / 4);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        pipe.Destroy();
    }
#endif
}
