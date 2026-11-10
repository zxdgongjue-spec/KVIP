// URMA phase body: KVAF submits this compact descriptor range.
        for(uint32_t w=0;w<rt_finalWorkers;++w){
            auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
            auto workerArgs=GetArgs(ctx,FinalQueue(rt_batch,rt_finalWorkers,b,w));
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(workerArgs.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            if(!phase)total+=post->count;
            error|=post->error;
            if(post->nonconflicting>post->count)error|=kvaf_device::kInvalidArgs;
            uint32_t begin=post->descriptor+(phase?post->nonconflicting:0);
            uint32_t count=phase?post->count-post->nonconflicting:post->nonconflicting;
            if(post->deferred && count){
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
                    auto done=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(workerArgs.core_status);
                    done->profile.magic=KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&done->profile),sizeof(done->profile));
                    kvaf_device::KvafBatchGetDevice(workerArgs,0,begin,count);
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
                    error|=done->result;completed+=done->completed_batch_count;
                    submitCycles+=done->profile.post_active_cycles;
                    completionCycles+=done->profile.drain_wait_cycles;
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
