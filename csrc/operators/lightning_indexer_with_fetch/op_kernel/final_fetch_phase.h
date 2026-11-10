// URMA: join early posts and submit the compact ranges left to Fetch.
        for(uint32_t w=0;w<rt_finalWorkers;++w){
            auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
            auto workerArgs=GetArgs(ctx,FinalQueue(rt_batch,rt_finalWorkers,b,w));
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(workerArgs.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            if(!phase)total+=post->count;
            error|=post->error;
            if(post->nonconflicting>post->count) {
                error|=kvaf_device::kInvalidArgs;
                continue;
            }
#if !V9_SIMULATION
            // Even if another queue failed, retire this queue's early post
            // before reusing it for the conflicting suffix or a later round.
            if(phase && post->pending) {
                auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(workerArgs.channel_entity);
                auto drained=DrainQueue(entity,queue,post->pending);
                error|=drained.error;
                completed+=drained.completed;
                queue->last_error=drained.error;
                if(drained.error)queue->poisoned=1;
                else {
                    post->pending=0;
                    if(post->nonconflicting==post->count)post->deferred=0;
                }
            }
#endif
            uint32_t begin=post->descriptor+(phase?post->nonconflicting:0);
            uint32_t count=phase?post->count-post->nonconflicting:post->nonconflicting;
            if(post->deferred && count && !post->pending){
                if(phase)++deferred;
#if !V9_SIMULATION
                if(!error && (!phase || !row->pending)){
                    workerArgs.indices=ctx->sourceBlocks;workerArgs.offsets=ctx->sourceOffsets;
                    workerArgs.output_indices=ctx->destinations;workerArgs.miss_count=ctx->count;
                    workerArgs.request_capacity=rt_batch*kSlots;
                    if(ctx->fallbackAddresses) {
                        workerArgs.address_table=ctx->fallbackAddresses;
                        workerArgs.valid_table=ctx->fallbackValid;
                        workerArgs.table_capacity=rt_batch*kSlots;
                    }
                    kvaf_device::KvafBatchGetDevice(workerArgs,0,begin,count);
                    auto done=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(workerArgs.core_status);
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
                    error|=done->result;completed+=done->completed_batch_count;
                    if(done->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || done->processed_request_count!=count)
                        error|=kvaf_device::kInvalidArgs;
                    if(!done->result && (phase || post->nonconflicting==post->count))post->deferred=0;
                }
#else
                error|=kvaf_device::kInvalidArgs; // DMA fallback needs real hardware.
#endif
            }
            lastHead=queue->sq_producer_index;lastCq=queue->cq_consumer_index;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        }
