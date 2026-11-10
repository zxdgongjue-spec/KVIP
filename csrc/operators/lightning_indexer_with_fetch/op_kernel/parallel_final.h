#pragma once

// Called only after a separate prefetch-completion kernel. Every invocation
// owns one final queue/status view and one immutable resolver descriptor range.
// This is parallel synchronous Get, not asynchronous Get/Index overlap.
__aicore__ inline void ParallelFinalGet(__gm__ rotating::Context* ctx,uint32_t core) {
    const auto shape=rotating::LoadRuntimeShape(ctx);
    if(core>=shape.finalCores)return;
    const uint32_t b=core/shape.finalWorkers,w=core%shape.finalWorkers;
    auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*rotating::kRowStatusWords;
    auto post=rotating::FinalPostAt(shape.finalPostOffset,ctx,core);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),rotating::kRowStatusWords*sizeof(uint32_t));
    uint64_t error=post->error|row->error;
    if(info[0] || row->pending || post->nonconflicting>post->count)error|=kvaf_device::kInvalidArgs;
    post->head=0;
    if(!error && post->count) {
        auto args=GetArgs(ctx,rotating::FinalQueue(shape.batch,shape.finalWorkers,b,w));
        // KvafQueueView slices one physical jetty per view. A wider view would
        // let sibling AIVs touch the same persistent queue and CQ consumer.
        if(args.block_count!=1 || args.jettys_per_core!=1 || !args.queue_state || !args.core_status ||
            !post->deferred)error=kvaf_device::kInvalidArgs;
        if(!error) {
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            args.indices=ctx->sourceBlocks;args.offsets=ctx->sourceOffsets;
            args.output_indices=ctx->destinations;args.miss_count=ctx->count;
            args.request_capacity=shape.batch*rotating::kSlots;
            if(ctx->fallbackAddresses) {
                args.address_table=ctx->fallbackAddresses;args.valid_table=ctx->fallbackValid;
                args.table_capacity=shape.batch*rotating::kSlots;
            }
            auto done=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
            // A native argument rejection can return without writing status.
            // Never accept a successful status left by the previous replay.
            done->magic=0;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
#if !V9_SIMULATION
            kvaf_device::KvafBatchGetDevice(args,0,post->descriptor,post->count);
            // Native Get mutates persistent SQ/CQ cursors and may poison the
            // queue on failure. Publish them before a different AIV reduces
            // this round or acquires the same view on the next graph replay.
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
            error|=done->result;
            if(done->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || done->processed_request_count!=post->count)
                error|=kvaf_device::kInvalidArgs;
            if(!error)post->head=done->completed_batch_count;
#else
            error|=kvaf_device::kInvalidArgs;
#endif
        }
    }
    post->error=uint32_t(error);
    post->pending=error && post->count ? 1U:0U;
    post->deferred=error && post->count ? 1U:0U;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
