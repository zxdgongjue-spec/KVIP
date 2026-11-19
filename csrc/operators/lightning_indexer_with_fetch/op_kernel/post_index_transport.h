// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kvaf_batch_get_device.h"

// This experiment reuses the existing fused-context KVAF header at 1c131038.
// Its PostDirectRequests signature includes first_jetty and its blocking API
// returns void. A different native implementation requires a separate review.
#ifndef KVAF_AIV_BATCH_GET_SCHEDULE_H_
#error "Post/Index/CQ requires the original KVAF fused-context schedule header"
#endif

namespace pipeline_ab {
namespace post_index_transport {

// Only scalar ownership survives Post. All WQEs have reached the SQ before
// return, so the same AIV may reuse its UB/events for Index before reading CQ.
struct State {
    uint64_t entity = 0, queue = 0, status = 0, lastWqe = 0;
    uint64_t error = kvaf_device::kSuccess;
    uint32_t requests = 0, completed = 0;
    bool pending = false;
};

// All cores finish preflight before ANY Post. SQ slots count WQEs; CQ slots
// count signalled checkpoints, so their producer/consumer indices cannot be
// subtracted. This invocation exclusively owns each drained single-Jetty view.
__aicore__ inline uint64_t Preflight(
    const KvafExecuteBatchGetKernelArgs& args, uint32_t begin, uint32_t count,
    bool& needsBlocking) {
    needsBlocking = false;
    if (!count) return kvaf_device::kSuccess;
    if (args.block_count != 1 || args.jettys_per_core != 1 ||
        !args.channel_entity || !args.queue_state || !args.core_status ||
        !args.address_table || !args.valid_table || !args.indices || !args.offsets ||
        !args.output_indices || !args.output || !args.miss_count ||
        !args.table_capacity || !args.request_capacity || !args.output_rows ||
        !args.token_bytes || !args.tokens_per_block ||
        begin > args.request_capacity || count > args.request_capacity - begin ||
        (args.batch_get_max_inflight_total && args.batch_get_max_inflight_total < 2))
        return kvaf_device::kInvalidArgs;
    auto entity = reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
    auto queue = reinterpret_cast<const __gm__ KvafBatchGetQueueState*>(args.queue_state);
    if (entity->magic != kvaf_device::kEntityMagic ||
        entity->version != kvaf_device::kEntityVersion ||
        entity->local_hbm_address != args.output ||
        entity->local_hbm_size < args.token_bytes || !entity->local_token_id ||
        !entity->remote_host_ddr_address || entity->remote_host_ddr_size < args.token_bytes ||
        !entity->sq_va || !entity->sq_db_va || entity->sq_depth != kvaf_device::kSqDepthWqebbs ||
        entity->wqe_size != kvaf_device::kWqeBytes ||
        !entity->cq_va || !entity->cq_db_va || entity->cq_depth < 2 ||
        entity->cqe_size < kvaf_device::kCqeProbeBytes)
        return kvaf_device::kInvalidEntity;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(args.queue_state), sizeof(*queue));
    if (queue->last_error || queue->poisoned ||
        (queue->initialized && queue->magic != kvaf_device::kQueueMagic))
        return kvaf_device::kInvalidQueue;
    // One whole-range Post signals its final WQE exactly once. If the original
    // scheduler must split the range, choose its blocking fallback collectively.
    // The original bounded scheduler can have two signalled groups live.
    // Reject depth-one CQ above, before posting even the first group.
    needsBlocking = count >= entity->sq_depth ||
        (args.batch_get_max_inflight_total && count > args.batch_get_max_inflight_total);
    // Descriptor validation is fused into the caller's existing Pack VF.
    return kvaf_device::kSuccess;
}

// Capacity fallback uses the ORIGINAL blocking Get with its existing flow
// control. The native implementation, WQE format and serial binary are intact.
__aicore__ inline uint64_t GetBlocking(
    const KvafExecuteBatchGetKernelArgs& args, uint32_t begin, uint32_t count,
    State& state) {
    if (state.error) return state.error;
    if (state.pending || state.requests != state.completed)
        return state.error = kvaf_device::kInvalidQueue;
    if (!count) return kvaf_device::kSuccess;
    state.requests = count; state.completed = 0;
    kvaf_device::KvafBatchGetDevice(args, 0, begin, count);
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(*status));
    state.error = status->result;
    if (!state.error) state.completed = count;
    return state.error;
}

// Call after collective Preflight success with no needsBlocking flag. Reuse
// KVAF's existing Post helper directly; it never consumes a CQ checkpoint.
__aicore__ inline uint64_t Post(
    const KvafExecuteBatchGetKernelArgs& args, uint32_t begin, uint32_t count,
    State& state) {
    if (state.error) return state.error;
    if (state.pending || state.requests != state.completed)
        return state.error = kvaf_device::kInvalidQueue;
    if (!count) return kvaf_device::kSuccess;
    auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
    auto entity = reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
    if (!queue->initialized) {
        queue->magic = kvaf_device::kQueueMagic; queue->initialized = 1;
        queue->sq_producer_index = queue->cq_consumer_index = 0;
        queue->completed_batch_count = queue->last_error = queue->poisoned = 0;
    }
    uint64_t firstDoorbell = 0;
    state.requests = count; state.completed = 0;
    state.entity = args.channel_entity; state.queue = args.queue_state; state.status = args.core_status;
    kvaf_device::ConfigureUbToGmCopy();
    kvaf_device::PostDirectRequests(
        reinterpret_cast<__gm__ const int32_t*>(args.indices),
        reinterpret_cast<__gm__ const int32_t*>(args.output_indices),
        reinterpret_cast<__gm__ const int32_t*>(args.offsets),
        reinterpret_cast<__gm__ const uint64_t*>(args.address_table),
        begin, count, 0, args.token_bytes, 1, entity, queue, 0,
        &firstDoorbell, &state.lastWqe);
    state.pending = true;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), sizeof(*queue));
    return kvaf_device::kSuccess;
}

// KVAF exposes a blocking CQ poll but no try-poll. This is its CQE protocol
// for the version pinned above. Only the time between unsuccessful probes
// changes. The AIV owns this queue and is its sole CQ consumer.
struct FinalPoll {
    uint32_t probes = 0;
    uint64_t pauseCycles = 0;
    static constexpr uint32_t maxPauseCycles = 512000;
};

__aicore__ inline uint64_t WaitFinal(
    __gm__ const KvafBatchGetDirectUrmaEntity* entity,
    __gm__ KvafBatchGetQueueState* queue, FinalPoll& trace) {
    const uint64_t cursor = queue->cq_consumer_index;
    const uint32_t depth = entity->cq_depth;
    const uint32_t expected = (cursor / depth) & 1U;
    auto address = reinterpret_cast<__gm__ uint8_t*>(entity->cq_va) +
                   (cursor % depth) * entity->cqe_size;
    for (uint32_t poll = 1; poll <= kvaf_device::kMaxCqPolls; ++poll) {
        kvaf_device::CleanCacheLines(address, kvaf_device::kCqeProbeBytes);
        __asm__ __volatile__("");
        const uint32_t word = *reinterpret_cast<__gm__ volatile uint32_t*>(address);
        __asm__ __volatile__("");
        if (((word >> 2U) & 1U) != expected) {
            queue->cq_consumer_index = cursor + 1;
            st_dev(static_cast<uint32_t>(cursor + 1) & kvaf_device::kCqConsumerMask,
                   reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va), 0);
            trace.probes += poll;
            return (word >> 16U) ? kvaf_device::kCqError : kvaf_device::kSuccess;
        }
        // Completed CQEs take the fast branch above without reading a clock.
        // Only the scalar system register is queried during a pause. The
        // budget is shared by both checkpoints and any timeout retries.
        if (poll >= 4 && poll != kvaf_device::kMaxCqPolls &&
            trace.pauseCycles < FinalPoll::maxPauseCycles) {
            uint32_t pause = poll < 8 ? 128 : (poll < 16 ? 256 : 512);
            const uint32_t remaining = FinalPoll::maxPauseCycles - trace.pauseCycles;
            if (pause > remaining) pause = remaining;
            const uint64_t start = AscendC::GetSystemCycle();
            uint64_t elapsed;
            do {
                elapsed = static_cast<uint64_t>(AscendC::GetSystemCycle()) - start;
            } while (elapsed < pause);
            trace.pauseCycles += elapsed;
        }
    }
    trace.probes += kvaf_device::kMaxCqPolls;
    return kvaf_device::kCqTimeout;
}

// Only a timeout leaves the native CQ consumer index unchanged. An error CQE
// has still consumed its slot, so the next FIFO checkpoint may then be polled.
template<bool RetryTimeout>
__aicore__ inline uint64_t Retire(State& state, uint64_t previousError, FinalPoll* trace = nullptr) {
    if (!state.pending) return state.error;
    auto entity = reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(state.entity);
    auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(state.queue);
    auto status = reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(state.status);
    uint32_t polls = 0;
    uint64_t error;
    if constexpr (RetryTimeout) error = WaitFinal(entity, queue, *trace);
    else error = kvaf_device::WaitForCompletion(entity, queue, &polls);
    uint64_t completion = error;
    if constexpr (RetryTimeout) {
        if (error == kvaf_device::kCqTimeout)
            completion = WaitFinal(entity, queue, *trace);
    }
    if (!completion) { ++queue->completed_batch_count; state.completed = state.requests; }
    if (error) queue->poisoned = 1;
    if constexpr (RetryTimeout) {
        if (error && !queue->last_error) queue->last_error = error;
    } else {
        queue->last_error = error;  // Original single-checkpoint path.
    }
    if (!state.error) state.error = error;
    kvaf_device::WriteStatus(status, previousError ? previousError : state.error,
        state.requests, queue->completed_batch_count, queue->sq_producer_index,
        queue->cq_consumer_index, UINT64_MAX, state.lastWqe);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue), sizeof(*queue));
    // The two-checkpoint HIGH path writes only the public status header.
    // The blocking native path can write a full profile and keeps its flush.
    constexpr auto written=offsetof(KvafBatchGetCoreStatus,profile);
    static_assert(written==8*sizeof(uint64_t),"unexpected KVAF status ABI");
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),
                                 RetryTimeout ? written : sizeof(*status));
    state.pending = RetryTimeout && completion == kvaf_device::kCqTimeout;
    return state.error;
}

__aicore__ inline uint64_t Wait(State& state) {
    return Retire<false>(state, 0);
}

// Both states belong to one private FIFO. Retire an accepted suffix after a
// consumed prefix error, but never charge a late prefix CQE to that suffix.
// After two timeouts the queue must be torn down: its pending DMA is unknown.
__aicore__ inline uint64_t RetirePair(State& first, State& second,
                                     __gm__ uint64_t* statistics = nullptr) {
    FinalPoll trace{};
    const uint64_t prefixError = Retire<true>(first, 0, &trace);
    const uint64_t suffixError = first.pending ? 0 : Retire<true>(second, prefixError, &trace);
    if (statistics) {
        const uint64_t ticks = trace.pauseCycles > UINT32_MAX ? UINT32_MAX : trace.pauseCycles;
        *statistics = (ticks << 32U) | trace.probes;
    }
    return prefixError ? prefixError : suffixError;
}

}  // namespace post_index_transport
}  // namespace pipeline_ab
