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

__aicore__ inline void CommitTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args,
    uint64_t initialHead, uint32_t count, __gm__ FinalPost* post) {
    post->count=count;
    post->pending=0;
    post->error=0;
    post->head=initialHead;
    if(!count)return;
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    // The last WQE in each published prefix requests its one completion.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    kvaf_device::PendingPost pending{entity->sq_db_va,args->queue_state,initialHead+count,0U};
    kvaf_device::CommitPostScalarStage(&pending);
    post->pending=1;
}

__aicore__ inline uint64_t FinishTerminalCompletion(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post,
    __gm__ KvafBatchGetQueueState* queue, CqDrainResult result) {
    post->error=uint32_t(result.error);
    if(!result.error)post->pending=0;
    if(result.error)queue->last_error=result.error;
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args->core_status);
    kvaf_device::WriteStatus(status,result.error,post->count,result.completed,
        queue->sq_producer_index,queue->cq_consumer_index,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
    return result.error;
}

__aicore__ inline uint64_t CompleteTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post) {
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    auto result=DrainQueue(entity,queue,post->pending);
    return FinishTerminalCompletion(args,post,queue,result);
}

// Every Q is the sole producer and consumer of its private queues. Grant the
// next prefix only after a successful CQE for the preceding prefix.
__aicore__ inline uint64_t CompleteTerminalBatches(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post,
    const terminal_batches::Plan& plan, uint64_t initialHead) {
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    uint32_t completed=0;
    for(uint32_t batch=0;batch<plan.count;++batch){
        auto result=DrainQueue(entity,queue,1);
        completed+=result.completed;
        if(result.error || batch+1==plan.count){
            result.completed=completed;
            return FinishTerminalCompletion(args,post,queue,result);
        }
        uint32_t next=plan.ends[batch+1];
        // SQ descriptors and checkpoint bits were written before the first post.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        kvaf_device::PendingPost pending{entity->sq_db_va,args->queue_state,initialHead+next,0U};
        kvaf_device::CommitPostScalarStage(&pending);
        post->count=next;
    }
    return kvaf_device::kInvalidArgs;
}
#endif
