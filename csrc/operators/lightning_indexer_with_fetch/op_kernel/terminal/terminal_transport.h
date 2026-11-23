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

// Diagnostic owner reads. Negative and positive observations bound CQ visibility
// on the device clock. Only DrainCq is allowed to reclaim a slot.
struct CreditStamp {
    uint64_t postBegin,postEnd,lastAbsent,firstVisible,drainBegin,drainEnd;
};
struct CreditAccess {
    CqAccess cq;
    CreditStamp* stamp;
    uint32_t expectedOwner;
    __aicore__ inline uint32_t Read(uint32_t slot) {
        uint64_t before=GetSystemCycle();
        uint32_t word=cq.Read(slot);
        uint64_t after=GetSystemCycle();
        if(!stamp->firstVisible){
            if(((word>>2)&1U)!=expectedOwner)stamp->firstVisible=after;
            else stamp->lastAbsent=before;
        }
        return word;
    }
    __aicore__ inline void Commit(uint64_t consumer){cq.Commit(consumer);}
};
__aicore__ inline CreditAccess CreditView(
    __gm__ const KvafBatchGetDirectUrmaEntity* entity,__gm__ KvafBatchGetQueueState* queue,
    uint64_t consumer,CreditStamp* stamp) {
    return {{reinterpret_cast<__gm__ uint8_t*>(entity->cq_va),entity->cqe_size,
        reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va),queue},
        stamp,uint32_t((consumer/entity->cq_depth)&1U)};
}
__aicore__ inline void SampleCredit(const __gm__ KvafExecuteBatchGetKernelArgs* args,CreditStamp* stamp){
    if(stamp->firstVisible)return;
    auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    if(entity->cq_depth<2)return;
    uint64_t consumer=queue->cq_consumer_index;
    auto access=CreditView(entity,queue,consumer,stamp);
    access.Read(uint32_t(consumer%entity->cq_depth));
}
__aicore__ inline CqDrainResult DrainCredit(
    __gm__ const KvafBatchGetDirectUrmaEntity* entity,__gm__ KvafBatchGetQueueState* queue,
    uint32_t pending,CreditStamp* stamp){
    if(!stamp || !pending || entity->cq_depth<2)return DrainQueue(entity,queue,pending);
    uint64_t consumer=queue->cq_consumer_index;
    auto access=CreditView(entity,queue,consumer,stamp);
    stamp->drainBegin=GetSystemCycle();
    auto result=DrainCq(access,consumer,entity->cq_depth,pending,
        kvaf_device::kMaxCqPolls*kFusedCqPollMultiplier,kvaf_device::kInvalidArgs,
        kvaf_device::kCqError,kvaf_device::kCqTimeout);
    stamp->drainEnd=GetSystemCycle();
    queue->completed_batch_count+=result.completed;
    return result;
}
__aicore__ inline uint64_t CompleteTerminalPost(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post,CreditStamp* stamp=nullptr) {
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    auto result=DrainCredit(entity,queue,post->pending,stamp);
    return FinishTerminalCompletion(args,post,queue,result);
}

// Every Q is the sole producer and consumer of its private queues. Grant the
// next prefix only after a successful CQE for the preceding prefix.
__aicore__ inline uint64_t CompleteTerminalBatches(
    const __gm__ KvafExecuteBatchGetKernelArgs* args, __gm__ FinalPost* post,
    const terminal_batches::Plan& plan, uint64_t initialHead,CreditStamp* stamps=nullptr) {
    auto entity=reinterpret_cast<const __gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    uint32_t completed=0;
    for(uint32_t batch=0;batch<plan.count;++batch){
        auto result=DrainCredit(entity,queue,1,stamps?stamps+batch:nullptr);
        completed+=result.completed;
        if(result.error || batch+1==plan.count){
            result.completed=completed;
            return FinishTerminalCompletion(args,post,queue,result);
        }
        uint32_t next=plan.ends[batch+1];
        // SQ descriptors and checkpoint bits were written before the first post.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        kvaf_device::PendingPost pending{entity->sq_db_va,args->queue_state,initialHead+next,0U};
        if(stamps)stamps[batch+1].postBegin=GetSystemCycle();
        kvaf_device::CommitPostScalarStage(&pending);
        if(stamps)stamps[batch+1].postEnd=GetSystemCycle();
        post->count=next;
    }
    return kvaf_device::kInvalidArgs;
}
#endif
