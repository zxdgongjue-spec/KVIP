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
    // FinalBuild requests a completion only on the last WQE. One posted batch
    // therefore consumes one CQE, not count CQEs.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    kvaf_device::PendingPost pending{entity->sq_db_va,args->queue_state,initialHead+count,0U};
    kvaf_device::CommitPostScalarStage(&pending);
    post->pending=1;
}

__aicore__ inline uint64_t CompleteTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post) {
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    auto result=DrainQueue(entity,queue,post->pending);
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
#endif
