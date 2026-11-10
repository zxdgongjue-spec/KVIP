#pragma once

// One resolver worker owns one physical Jetty for the whole replay. Submit the
// safe prefix after releasing resolver UB; its READs can run during Index work.
// No payload destination in that prefix aliases an outstanding prefetch write.
__aicore__ inline uint64_t AsyncFinalView(__gm__ rotating::Context* ctx,
    const rotating::RuntimeShape& shape,uint32_t b,uint32_t w,
    __gm__ rotating::FinalPost* post,KvafExecuteBatchGetKernelArgs& args) {
    args=GetArgs(ctx,rotating::FinalQueue(shape.batch,shape.finalWorkers,b,w));
    args.indices=ctx->sourceBlocks;args.offsets=ctx->sourceOffsets;
    args.output_indices=ctx->destinations;args.miss_count=ctx->count;
    args.request_capacity=shape.batch*rotating::kSlots;
    if(ctx->fallbackAddresses) {
        args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;
        args.table_capacity=shape.batch*rotating::kSlots;
    }
    if(args.block_count!=1 || args.jettys_per_core!=1 || args.batch_get_max_inflight_total ||
        !args.address_table || !args.valid_table || !args.indices || !args.offsets ||
        !args.output_indices || !args.output || !args.channel_entity || !args.queue_state ||
        !args.core_status || !args.table_capacity || !args.output_rows || !args.token_bytes ||
        !args.tokens_per_block || post->count>rotating::kSlots ||
        post->count>args.request_capacity || post->descriptor>args.request_capacity-post->count ||
        post->nonconflicting>post->count)return kvaf_device::kInvalidArgs;
    auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
    // Same one-Jetty transport contract as native KvafBatchGetDevice. The two
    // submitted ranges fit in one SQ and need at most two completion entries.
    if(entity->magic!=kvaf_device::kEntityMagic || entity->version!=kvaf_device::kEntityVersion ||
        entity->local_hbm_address!=args.output || entity->local_hbm_size<args.token_bytes ||
        !entity->local_token_id || !entity->remote_host_ddr_address ||
        entity->remote_host_ddr_size<args.token_bytes || !entity->sq_va || !entity->sq_db_va ||
        entity->sq_depth!=kvaf_device::kSqDepthWqebbs || entity->wqe_size!=kvaf_device::kWqeBytes ||
        !entity->cq_va || !entity->cq_db_va || !entity->cq_depth ||
        entity->cqe_size<kvaf_device::kCqeProbeBytes)return kvaf_device::kInvalidEntity;
    uint32_t checkpoints=uint32_t(post->nonconflicting!=0)+uint32_t(post->nonconflicting!=post->count);
    if(post->count>=entity->sq_depth || checkpoints>entity->cq_depth)return kvaf_device::kQueueFull;
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    if(queue->initialized && (queue->magic!=kvaf_device::kQueueMagic || queue->poisoned))
        return kvaf_device::kInvalidQueue;
    if(!queue->initialized) {
        if(post->pending)return kvaf_device::kInvalidQueue;
        queue->magic=kvaf_device::kQueueMagic;queue->initialized=1;
        queue->sq_producer_index=0;queue->cq_consumer_index=0;
        queue->completed_batch_count=0;queue->last_error=0;queue->poisoned=0;
    }
    return 0;
}

__aicore__ inline void AsyncFinalPost(KvafExecuteBatchGetKernelArgs& args,
    __gm__ rotating::FinalPost* post,uint32_t begin,uint32_t count) {
    if(!count)return;
    uint64_t firstDoorbell=0,lastWqe=0;
    kvaf_device::ConfigureUbToGmCopy();
    kvaf_device::PostDirectRequests(
        reinterpret_cast<__gm__ const int32_t*>(args.indices),
        reinterpret_cast<__gm__ const int32_t*>(args.output_indices),
        reinterpret_cast<__gm__ const int32_t*>(args.offsets),
        reinterpret_cast<__gm__ const uint64_t*>(args.address_table),begin,count,0,args.token_bytes,1,
        reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity),
        reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state),0,&firstDoorbell,&lastWqe);
    // Native submission drains its UB-to-SQ pipeline, not the NIC READs. One
    // range signals only its final WQE, regardless of its number of WQE posts.
    ++post->pending;
}

__aicore__ inline void SubmitSafeFinal(__gm__ rotating::Context* ctx,uint32_t core) {
    const auto shape=rotating::LoadRuntimeShape(ctx);
    if(core>=shape.finalCores)return;
    const uint32_t b=core/shape.finalWorkers,w=core%shape.finalWorkers;
    auto post=rotating::FinalPostAt(shape.finalPostOffset,ctx,core);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
    // Only this worker's record is published yet. In particular, do not read
    // worker zero's rowStatus while sibling resolver invocations are running.
    uint64_t error=post->error;
    KvafExecuteBatchGetKernelArgs args{};
    if(!error && post->count) {
        error=AsyncFinalView(ctx,shape,b,w,post,args);
        if(!error && (post->pending || !post->deferred))error=kvaf_device::kInvalidArgs;
        if(!error)AsyncFinalPost(args,post,post->descriptor,post->nonconflicting);
        if(args.queue_state) {
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            queue->last_error=error;
            if(error && post->pending)queue->poisoned=1;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        }
    }
    post->error=uint32_t(error);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}

__aicore__ inline void CompleteAsyncFinal(__gm__ rotating::Context* ctx,uint32_t core) {
    const auto shape=rotating::LoadRuntimeShape(ctx);
    if(core>=shape.finalCores)return;
    const uint32_t b=core/shape.finalWorkers,w=core%shape.finalWorkers;
    auto post=rotating::FinalPostAt(shape.finalPostOffset,ctx,core);
    auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*rotating::kRowStatusWords;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),rotating::kRowStatusWords*sizeof(uint32_t));
    uint64_t error=post->error|row->error;
    if(info[0] || row->pending || post->pending>1 ||
        post->nonconflicting>post->count)error|=kvaf_device::kInvalidArgs;
    KvafExecuteBatchGetKernelArgs args{};
    if(post->count) {
        const uint64_t viewError=AsyncFinalView(ctx,shape,b,w,post,args);
        error|=viewError;
        if(!viewError) {
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
            const bool pendingValid=post->pending==uint32_t(post->nonconflicting!=0);
            if(!post->deferred || !pendingValid)
                error|=kvaf_device::kInvalidArgs;
            if(!error)AsyncFinalPost(args,post,post->descriptor+post->nonconflicting,
                post->count-post->nonconflicting);
            // Even when another worker/prefetch failed, retire this queue's
            // already submitted safe READs. Never submit the conflicting tail.
            while(pendingValid && post->pending) {
                uint32_t polls=0;
                uint64_t result=kvaf_device::WaitForCompletion(entity,queue,&polls);
                if(result) {error|=result;break;}
                --post->pending;
            }
            if(error && post->pending)queue->poisoned=1;
            if(!error) {
                ++queue->completed_batch_count;post->head=queue->completed_batch_count;
                post->deferred=0;
            }
            queue->last_error=error;
            kvaf_device::WriteStatus(reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status),
                error,post->count,queue->completed_batch_count,queue->sq_producer_index,
                queue->cq_consumer_index,UINT64_MAX,0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(args.core_status),sizeof(KvafBatchGetCoreStatus));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        } else if(args.queue_state && post->pending) {
            // Validation failed after the safe phase may have submitted work.
            // Preserve that ambiguity across graph replay; never reset/reuse it.
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            queue->poisoned=1;queue->last_error=error;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        }
    }
    post->error=uint32_t(error);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
