#pragma once

// One row owns its prefetch CQ. Join it before submitting any final write,
// including destinations that overlapped prefetch. Native submission releases
// UB-to-SQ staging before returning, so this AIV can reuse UB for the next Jetty.
// Index runs in a later kernel while these payload READs remain outstanding.
__aicore__ inline void SubmitFinalBeforeIndex(__gm__ rotating::Context* ctx,uint32_t b) {
    const auto shape=rotating::LoadRuntimeShape(ctx);
    if(b>=shape.batch)return;
    CompletePrefetch(ctx,b);
    auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*rotating::kRowStatusWords;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),rotating::kRowStatusWords*sizeof(uint32_t));
    const uint64_t rowError=row->error|((info[0] || row->pending)?kvaf_device::kInvalidArgs:0);
    for(uint32_t w=0;w<shape.finalWorkers;++w) {
        auto post=rotating::FinalPostAt(shape.finalPostOffset,ctx,b*shape.finalWorkers+w);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        uint64_t error=rowError|post->error;
        KvafExecuteBatchGetKernelArgs args{};
        post->head=0;
        if(!error && post->count) {
            // The entire range produces exactly one tail CQE, regardless of
            // the retained safe-prefix metadata or number of WQE posts.
            error=AsyncFinalView(ctx,shape,b,w,post,args,1);
            if(!error && (post->pending || !post->deferred))error=kvaf_device::kInvalidArgs;
            if(!error)AsyncFinalPost(args,post,post->descriptor,post->count);
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
}

// Every final worker remains the exclusive consumer of its original CQ. No
// submissions happen here. Even when Index/prefetch/another worker failed,
// retire this queue's known outstanding READ before reporting the error.
__aicore__ inline void CompletePostedFinal(__gm__ rotating::Context* ctx,uint32_t core) {
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
    if(info[0] || row->pending)error|=kvaf_device::kInvalidArgs;
    KvafExecuteBatchGetKernelArgs args{};
    if(post->count || post->pending) {
        const uint64_t viewError=AsyncFinalView(ctx,shape,b,w,post,args,1);
        error|=viewError;
        if(!viewError) {
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
            const bool submitted=post->count && post->pending==1 && post->deferred;
            if(!submitted && !post->error)error|=kvaf_device::kInvalidArgs;
            if(post->pending && !submitted)error|=kvaf_device::kInvalidArgs;
            if(submitted) {
                uint32_t polls=0;
                const uint64_t result=kvaf_device::WaitForCompletion(entity,queue,&polls);
                if(result)error|=result;
                else post->pending=0;
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
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            queue->poisoned=1;queue->last_error=error;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        }
    } else if(!error)post->deferred=0;
    post->error=uint32_t(error);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
