#pragma once
// TEST ONLY: a synchronous GM-to-GM payload copy in place of URMA/KVAF. This
// models functional results, never network timing or native WQE/DMA behavior.
namespace fusion3_test {
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void CopyPayload(
    KvafExecuteBatchGetKernelArgs a, uint32_t begin, uint32_t count) {
    auto indices = reinterpret_cast<__gm__ int32_t*>(a.indices);
    auto offsets = reinterpret_cast<__gm__ int32_t*>(a.offsets);
    auto destinations = reinterpret_cast<__gm__ int32_t*>(a.output_indices);
    auto addresses = reinterpret_cast<__gm__ uint64_t*>(a.address_table);
    uint32_t fields = a.token_bytes / 8;
    // One warp owns a token. Cache its address metadata across field chunks;
    // keep adjacent lanes on adjacent words and copy every payload byte.
    uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    for (uint32_t token = warp; token < count; token += 16) {
        uint32_t record = begin + token;
        uint64_t address = asc_ldcg(addresses + asc_ldcg(indices + record)) + uint32_t(asc_ldcg(offsets + record));
        auto src = reinterpret_cast<__gm__ uint64_t*>(address);
        auto dst = reinterpret_cast<__gm__ uint64_t*>(a.output) + uint64_t(asc_ldcg(destinations + record)) * fields;
        for (uint32_t field = lane; field < fields; field += 32)
            asc_stcg(dst + field, asc_ldcg(src + field));
    }
    asc_threadfence();
}
__aicore__ inline void DeviceGet(const KvafExecuteBatchGetKernelArgs& a, uint32_t begin, uint32_t count) {
    // Test-only fault injection. Production builds never include this header.
    if (a.batch_get_max_inflight_total == 0xfffffff1U) return;
    auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(a.queue_state);
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(a.core_status);
    if (a.batch_get_max_inflight_total == 0xfffffff2U) {
        queue->last_error = kvaf_device::kCqError; queue->poisoned = 1;
        kvaf_device::WriteStatus(status, kvaf_device::kCqError, count, queue->completed_batch_count, 0, 0, UINT64_MAX, 0);
        return;
    }
    AscendC::Simt::VF_CALL<CopyPayload>(AscendC::Simt::Dim3{512,1,1}, a, begin, count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    ++queue->sq_producer_index; ++queue->cq_consumer_index; ++queue->completed_batch_count;
    kvaf_device::WriteStatus(status, 0, count, queue->completed_batch_count,
        queue->sq_producer_index, queue->cq_consumer_index, UINT64_MAX, 0);
}
} // namespace fusion3_test
