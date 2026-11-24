#pragma once
// Included in rotating namespace after the global staged_fetch_worker.h. The existing
// KVAF post and CQ routines remain authoritative for the real transport ABI.
// This adapter is terminal-only; no intermediate service source is modified.
#if defined(__CCE_AICORE__)
__aicore__ inline void CompleteTerminalPrefetch(__gm__ Context* ctx,uint32_t row) {
    // The legacy initializeSpare=true clears the odd view's status, but that
    // view now belongs to Q1 and may already have published its CQ result.
    CompletePrefetch(ctx,row,false);
}

struct TerminalBatches {
    uint32_t ends[4];  // Exclusive positions in this Q's prepared SQ descriptors.
    uint32_t count;
};

__aicore__ inline TerminalBatches PlanTerminalBatches(uint32_t own, uint32_t slots, uint32_t priorSlots) {
    constexpr uint32_t kPacingThreshold = 256;
    TerminalBatches plan{};
    if (own > 1 && slots >= priorSlots && slots - priorSlots >= kPacingThreshold) {
        // Start a long row with 1/16 of each Q's work, then release the rest in thirds.
        for (uint32_t numerator = 1; numerator < 16; numerator += 5) {
            const uint32_t end = (own * numerator + 15) / 16;
            if (end < own && (plan.count == 0 || end > plan.ends[plan.count - 1])) {
                plan.ends[plan.count++] = end;
            }
        }
    }
    plan.ends[plan.count++] = own;
    return plan;
}

// WriteReservedWqes already requested completion on the final descriptor. Mark
// the other batch boundaries after its SQ writes and before the first doorbell.
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void MarkTerminalPrefixes(
    __gm__ uint32_t* sq, uint32_t depth, uint64_t head, uint32_t first, uint32_t second,
    uint32_t third, uint32_t cuts) {
    const uint32_t lane = threadIdx.x;
    if (lane < cuts) {
        const uint32_t end = lane == 0 ? first : (lane == 1 ? second : third);
        const uint64_t producer = head + end - 1;
        const uint32_t slot = uint32_t(producer) & (depth - 1);
        const uint32_t owner = (producer & depth) == 0;
        asc_stcg(sq + slot * (kvaf_device::kWqeBytes / sizeof(uint32_t)), kvaf_device::WqeControl(true, owner));
    }
    asc_threadfence();
}

__aicore__ inline void PublishTerminalPrefix(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, const __gm__ KvafBatchGetDirectUrmaEntity* entity,
    uint64_t head) {
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    kvaf_device::PendingPost pending{entity->sq_db_va, args->queue_state, head, 0U};
    kvaf_device::CommitPostScalarStage(&pending);
}

__aicore__ inline void CommitTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, uint64_t head, uint32_t first,
    __gm__ FinalPost* post) {
    post->head = head;
    post->count = first;
    post->error = 0;
    post->pending = first != 0;
    if (first != 0) {
        auto entity = reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
        PublishTerminalPrefix(args, entity, head + first);
    }
}

__aicore__ inline uint64_t CompleteTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post,
    const TerminalBatches& batches, uint64_t head) {
    auto entity = reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue = reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    auto result = DrainQueue(entity, queue, post->pending);
    uint32_t completed = result.completed;
    for (uint32_t batch = 1; batch < batches.count && !result.error; ++batch) {
        const uint32_t end = batches.ends[batch];
        PublishTerminalPrefix(args, entity, head + end);
        post->count = end;
        result = DrainQueue(entity, queue, 1);
        completed += result.completed;
    }
    post->error=uint32_t(result.error);
    if(!result.error)post->pending=0;
    if(result.error)queue->last_error=result.error;
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args->core_status);
    kvaf_device::WriteStatus(status,result.error,post->count,completed,
        queue->sq_producer_index,queue->cq_consumer_index,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
    return result.error;
}
#endif
