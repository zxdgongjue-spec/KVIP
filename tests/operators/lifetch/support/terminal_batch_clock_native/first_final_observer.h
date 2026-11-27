// SPDX-License-Identifier: Apache-2.0
// Included in rotating namespace. The observer belongs to the sole consumer of this private CQ.
#pragma once
__aicore__ inline CqAccess FinalCqAccess(const __gm__ KvafBatchGetDirectUrmaEntity* entity,
                                          __gm__ KvafBatchGetQueueState* queue) {
    return {reinterpret_cast<__gm__ uint8_t*>(entity->cq_va), entity->cqe_size,
            reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va), queue};
}

__aicore__ inline void ObserveAfterFirstPost(const __gm__ KvafExecuteBatchGetKernelArgs* args,
                                              __gm__ uint64_t* out) {
    auto entity = reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    auto access = FinalCqAccess(entity, queue);
    const uint32_t depth = entity->cq_depth;
    const uint64_t consumer = queue->cq_consumer_index;
    if (depth < 2) return;
    const uint64_t before = GetSystemCycle();
    const uint32_t word = access.Read(uint32_t(consumer % depth));
    const uint64_t after = GetSystemCycle();
    out[0] = before; out[1] = after; out[2] = consumer; out[3] = depth; out[4] = word;
}

struct FirstFinalCqAccess {
    CqAccess access;
    uint32_t oldOwner;
    uint64_t firstBegin, firstEnd, visible, commitBegin, commitEnd;
    uint32_t firstWord, visibleWord;
    __aicore__ inline uint32_t Read(uint32_t slot) {
        const bool first = !firstBegin;
        if (first) firstBegin = GetSystemCycle();
        const uint32_t word = access.Read(slot);
        if (first) {
            firstEnd = GetSystemCycle();
            firstWord = word;
        }
        if (((word >> 2) & 1U) != oldOwner && !visible) {
            visible = first ? firstEnd : GetSystemCycle();
            visibleWord = word;
        }
        return word;
    }
    __aicore__ inline void Commit(uint64_t next) {
        commitBegin = GetSystemCycle();
        access.Commit(next);
        commitEnd = GetSystemCycle();
    }
};

__aicore__ inline CqDrainResult ObserveFirstFinalDrain(const __gm__ KvafBatchGetDirectUrmaEntity* entity,
        __gm__ KvafBatchGetQueueState* queue, uint32_t pending, __gm__ uint64_t* out) {
    if (!pending) return {0,0,0};
    const uint32_t depth = entity->cq_depth;
    const uint64_t consumer = queue->cq_consumer_index;
    FirstFinalCqAccess observer{};
    observer.access = FinalCqAccess(entity, queue);
    observer.oldOwner = depth ? uint32_t((consumer / depth) & 1U) : 0;
    auto result = DrainCq(observer, consumer, depth, pending,
        kvaf_device::kMaxCqPolls * kFusedCqPollMultiplier,
        kvaf_device::kInvalidArgs, kvaf_device::kCqError, kvaf_device::kCqTimeout);
    queue->completed_batch_count += result.completed;
    out[5] = observer.firstBegin; out[6] = observer.firstEnd; out[7] = consumer; out[8] = pending;
    out[9] = observer.firstWord; out[10] = observer.visible; out[11] = observer.visibleWord;
    out[12] = observer.commitBegin; out[13] = observer.commitEnd; out[14] = queue->cq_consumer_index;
    out[15] = result.polls; out[16] = result.completed; out[17] = result.error;
    return result;
}
