#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "rotation_abi.h"
#include "cq_drain.h"
#include "rotation_device.h"
#include "final_resolve.h"
#include "final_submit.h"
#include "final_dense.h"
#include "final_sparse.h"
#include "final_worker_submit.h"
#include "final_parallel.h"
#include "final_replicated.h"
#include "resident_state.h"
#include "index_cache_device.h"
#include "contract_final.h"
#include "index_cache_writer.h"
using namespace AscendC;
using namespace rotating;
__aicore__ inline __gm__ Context* GetContext(GM_ADDR bindings) {
    return reinterpret_cast<__gm__ Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
}
__aicore__ inline KvafExecuteBatchGetKernelArgs GetArgs(__gm__ Context* ctx,uint32_t owner) {
    KvafExecuteBatchGetKernelArgs args{};
    static_assert(sizeof(args)==14*sizeof(uint64_t),"KVAF argument ABI changed");
    auto src=reinterpret_cast<__gm__ uint64_t*>(ctx->args)+owner*14;
    auto dst=reinterpret_cast<uint64_t*>(&args);
    for(uint32_t i=0;i<14;++i) dst[i]=src[i];
    return args;
}
struct CqAccess {
    __gm__ uint8_t* base;
    uint32_t stride;
    __gm__ uint32_t* doorbell;
    __gm__ KvafBatchGetQueueState* queue;
    __aicore__ inline uint32_t Read(uint32_t slot) {
        auto address=base+uint64_t(slot)*stride;
        kvaf_device::CleanCacheLines(address,kvaf_device::kCqeProbeBytes);
        __asm__ __volatile__("");
        uint32_t word=*reinterpret_cast<__gm__ uint32_t*>(address);
        __asm__ __volatile__("");
        return word;
    }
    __aicore__ inline void Commit(uint64_t consumer) {
        queue->cq_consumer_index=consumer;
        st_dev(uint32_t(consumer)&kvaf_device::kCqConsumerMask,doorbell,0);
    }
};
__aicore__ inline CqDrainResult DrainQueue(
    __gm__ const KvafBatchGetDirectUrmaEntity* entity,
    __gm__ KvafBatchGetQueueState* queue,uint32_t pending) {
    if (!pending) return {0,0,0};
    CqAccess access{reinterpret_cast<__gm__ uint8_t*>(entity->cq_va),entity->cqe_size,
        reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va),queue};
    auto result=DrainCq(access,queue->cq_consumer_index,entity->cq_depth,pending,
        kvaf_device::kMaxCqPolls,kvaf_device::kInvalidArgs,kvaf_device::kCqError,kvaf_device::kCqTimeout);
    queue->completed_batch_count+=result.completed;
    return result;
}
#endif
extern "C" __global__ __aicore__ void v9_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx(); if(b>=SIM_BATCH) return;
        uint64_t kernelStarted=GetSystemCycle();
        auto ctx=GetContext(bindings);
        TPipe pipe; TBuf<TPosition::VECCALC> scratch;
        if(ctx->contract)pipe.InitBuffer(scratch,kIndexScratchWords*sizeof(uint32_t));
        uint64_t initStarted=GetSystemCycle(),stateDone=initStarted;
        if(ctx->contract) {
            Simt::VF_CALL<InitResidentState<0>>(Simt::Dim3{256,1,1},ctx,b,FinalPtr(scratch.Get<uint32_t>()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            RecordContractTicks(ctx,b,7,GetSystemCycle()-initStarted);
            Simt::VF_CALL<InitResidentState<1>>(Simt::Dim3{256,1,1},ctx,b,FinalPtr(scratch.Get<uint32_t>()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ContractOf(ctx)->status)+b*4,4);
            stateDone=GetSystemCycle();
            RecordContractTicks(ctx,b,0,stateDone-initStarted);
            if(ContractPtr<int32_t>(ContractOf(ctx)->status)[b]!=kPending) {
                pipe.Destroy();RecordContractTicks(ctx,b,12,GetSystemCycle()-kernelStarted);return;
            }
        }
        auto args=GetArgs(ctx,b*2);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        row->entity=args.channel_entity; row->queue=args.queue_state;
        constexpr uint64_t prefix=4*2048*6+1024+4*uint64_t(SIM_HISTORY)*2;
        constexpr uint64_t tail=uint64_t(SIM_HISTORY)*8+8192*64+512;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        row->addresses=a[13]+SIM_BATCH*prefix+b*tail;
        row->output=ctx->contract?args.output:args.output+uint64_t(b)*(SIM_RESIDENT+kSlots)*kTokenBytes;
        row->initialHead=queue->sq_producer_index;
        row->limit=ctx->contract?kTop:kSlots;
        row->prefetchLimit=ctx->prefetchLimit;
        row->queueLimit=entity->sq_depth>1 ? entity->sq_depth-1 : 0;
        row->error=queue->last_error;
        if(entity->sq_depth<2 || (entity->sq_depth&(entity->sq_depth-1)) || entity->cq_depth<2 ||
           !entity->sq_va || !entity->sq_db_va || ctx->prefetchLimit>kSlots ||
           ctx->windowLimit==0 || ctx->windowLimit>480)
            row->error=kvaf_device::kInvalidArgs;
#if !V9_SIMULATION
        if(queue->initialized && queue->magic!=kvaf_device::kQueueMagic)
            row->error=kvaf_device::kInvalidArgs;
        if(!queue->initialized) {
            queue->magic=kvaf_device::kQueueMagic; queue->initialized=1;
        }
        if(entity->sq_depth>(1U<<30) || row->initialHead>UINT64_MAX-ctx->prefetchLimit ||
           args.token_bytes!=kTokenBytes || args.tokens_per_block!=128 ||
           args.block_count!=1 || args.jettys_per_core!=1)
            row->error=kvaf_device::kInvalidArgs;
        if(entity->magic!=kvaf_device::kEntityMagic || entity->version!=kvaf_device::kEntityVersion ||
           entity->wqe_size!=64 || !entity->cq_va || !entity->cq_db_va ||
           entity->cqe_size<kvaf_device::kCqeProbeBytes || !entity->local_token_id ||
           entity->local_hbm_address!=args.output ||
           entity->local_hbm_size<(ctx->contract?uint64_t(ContractOf(ctx)->output_tokens):uint64_t(SIM_BATCH)*(SIM_RESIDENT+kSlots))*kTokenBytes ||
           !entity->remote_host_ddr_address || entity->remote_host_ddr_size<kTokenBytes)
            row->error=kvaf_device::kInvalidEntity;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
#endif
        if(ctx->contract) {
            auto c=ContractOf(ctx);
            if(args.output!=c->resident_output)row->error=kvaf_device::kInvalidArgs;
            // Validate every queue and directory address before LI can issue WQEs.
            for(uint32_t w=0;w<kFinalWorkers;++w) {
                auto finalArgs=GetArgs(ctx,FinalQueue(b,w));
                auto e=ContractPtr<KvafBatchGetDirectUrmaEntity>(finalArgs.channel_entity);
                auto q=ContractPtr<KvafBatchGetQueueState>(finalArgs.queue_state);
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(q),sizeof(*q));
                if(finalArgs.token_bytes!=kTokenBytes || finalArgs.tokens_per_block!=128 ||
                    finalArgs.block_count!=1 || finalArgs.jettys_per_core!=1 ||
                    e->magic!=kvaf_device::kEntityMagic || e->version!=kvaf_device::kEntityVersion ||
                    e->wqe_size!=64 || e->cqe_size<kvaf_device::kCqeProbeBytes || !e->local_token_id ||
                    e->local_hbm_address!=args.output || e->local_hbm_size<uint64_t(c->output_tokens)*kTokenBytes ||
                    e->remote_host_ddr_address!=entity->remote_host_ddr_address ||
                    e->remote_host_ddr_size!=entity->remote_host_ddr_size ||
                    e->sq_depth>(1U<<30) || q->sq_producer_index>UINT64_MAX-kSlots ||
                    finalArgs.output!=args.output || e->sq_depth<2 || (e->sq_depth&(e->sq_depth-1)) ||
                    e->cq_depth<2 || !e->sq_va || !e->sq_db_va || !e->cq_va || !e->cq_db_va ||
                    q->last_error || (q->initialized && q->magic!=kvaf_device::kQueueMagic))
                    row->error=kvaf_device::kInvalidArgs;
            }
            Simt::VF_CALL<ValidateContractRemote>(Simt::Dim3{256,1,1},ctx,b,
                entity->remote_host_ddr_address,entity->remote_host_ddr_size);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->status)+b*4,4);
            if(ContractPtr<int32_t>(c->status)[b]!=kPending)row->error=kvaf_device::kInvalidEntity;
            if(row->error) {
                Simt::VF_CALL<FailContractPreflight>(Simt::Dim3{256,1,1},ctx,b,int32_t(kInvalid));
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(*row));
        if(ctx->contract) {
            uint64_t validationDone=GetSystemCycle();
            RecordContractTicks(ctx,b,1,validationDone-stateDone);
            uint64_t cacheStarted=GetSystemCycle();
            Simt::VF_CALL<ReconcileIndexCache>(Simt::Dim3{256,1,1},ctx,b,false,FinalPtr(scratch.Get<uint32_t>()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            RecordContractTicks(ctx,b,2,GetSystemCycle()-cacheStarted);
            RecordIndexWork(ctx,b,false);
            RecordContractTicks(ctx,b,13,initStarted-kernelStarted);
        }
        pipe.Destroy();
        if(ctx->contract){RecordContractTicks(ctx,b,12,GetSystemCycle()-kernelStarted);
            RecordKernelSpan(ctx,b,kernelStarted,GetSystemCycle());}
    }
#endif
}
#if defined(__CCE_AICORE__)
__aicore__ inline uint64_t CompletePrefetch(__gm__ Context* ctx,uint32_t b) {
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        // Only the first cache line and the pending field are read/written here.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
        auto args=GetArgs(ctx,b*2);
        auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(row->entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(row->queue);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        uint64_t error=row->error;
        if(row->pending>=entity->cq_depth) error=kvaf_device::kInvalidArgs;
        uint32_t completed=0;
        uint64_t cqTicks=0;
#if !V9_SIMULATION
        if(!error) {
            uint64_t waitStarted=GetSystemCycle();
            auto result=DrainQueue(entity,queue,row->pending);
            cqTicks+=GetSystemCycle()-waitStarted;
            error=result.error;completed=result.completed;
        }
#endif
        row->error=uint32_t(error);
        if(!error) row->pending=0;
        if(error) queue->last_error=error;
        auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
        kvaf_device::WriteStatus(status,error,row->requests,completed,queue->sq_producer_index,
            queue->cq_consumer_index,UINT64_MAX,0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
        // The odd view is final-owned; its status is not prefetch completion.
        auto spare=GetArgs(ctx,b*2+1);
        auto unused=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(spare.core_status);
        kvaf_device::WriteStatus(unused,0,0,0,0,0,UINT64_MAX,0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(unused),sizeof(*unused));
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        // Only the first cache line and the pending field are read/written here.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
        return cqTicks;
}
#endif
// Retained for the compute-only diagnostic Graph, which has no final submission.
extern "C" __global__ __aicore__ void v9_wait_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx();if(b<SIM_BATCH)CompletePrefetch(GetContext(bindings),b);
    }
#endif
}
extern "C" __global__ __aicore__ void v9_resolve_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        if(GetContext(bindings)->contract) {
            ContractFinal(GetContext(bindings));
        }
        else if constexpr(kFinalReplicated || kFinalWorkers==1)ReplicatedFinal(GetContext(bindings));
        else ParallelFinal(GetContext(bindings));
    }
#endif
}
// Completion is a separate launch: final resolver + complete SIMD WQE/SQ/post
// can be timed without CQ latency. Small queues retain KVAF's bounded fallback.
extern "C" __global__ __aicore__ void v9_fetch_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx(); if(b>=SIM_BATCH) return;
        uint64_t kernelStarted=GetSystemCycle();
        auto ctx=GetContext(bindings);
        TPipe pipe;TBuf<TPosition::VECCALC> scratch;
        if(ctx->contract)pipe.InitBuffer(scratch,32);
        uint64_t fetchStarted=GetSystemCycle();
        if(ctx->contract) {
            auto c=ContractOf(ctx);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->status)+b*4,4);
            if(ContractPtr<int32_t>(c->status)[b]==kSkipped) {pipe.Destroy();return;}
            if(!ContractPtr<Row>(ctx->rows)[b].entity) {
                Simt::VF_CALL<PublishResidentState>(Simt::Dim3{256,1,1},ctx,b,kInvalid,0,FinalPtr(scratch.Get<uint32_t>()));
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                pipe.Destroy();return;
            }
        }
        uint64_t cqTicks=CompletePrefetch(ctx,b);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*sizeof(uint32_t));
        auto args=GetArgs(ctx,b*2);
        args.core_status=ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus);
        auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
        auto spare=status+1;
        kvaf_device::WriteStatus(spare,0,0,0,0,0,UINT64_MAX,0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(spare),sizeof(*spare));
        uint64_t error=info[0] ? kvaf_device::kInvalidArgs : row->error;
        uint32_t completed=0,total=0,deferred=0,submitted=0;
        uint64_t lastHead=0,lastCq=0;
        for(uint32_t w=0;w<kFinalWorkers;++w){
            auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
            auto workerArgs=GetArgs(ctx,FinalQueue(b,w));
            auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(workerArgs.channel_entity);
            auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(workerArgs.queue_state);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            total+=post->count;error|=post->error;
            if(post->deferred){
                ++deferred;
#if !V9_SIMULATION
                if(!error && !row->pending){
                    workerArgs.indices=ctx->sourceBlocks;workerArgs.offsets=ctx->sourceOffsets;
                    workerArgs.output_indices=ctx->destinations;workerArgs.miss_count=ctx->count;
                    workerArgs.request_capacity=SIM_BATCH*kSlots;
                    uint64_t waitStarted=GetSystemCycle();
                    kvaf_device::KvafBatchGetDevice(workerArgs,0,post->descriptor,post->count);
                    cqTicks+=GetSystemCycle()-waitStarted;
                    auto done=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(workerArgs.core_status);
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done),sizeof(*done));
                    error|=done->result;completed+=done->completed_batch_count;
                    post->reserved=done->processed_request_count;post->error|=done->result;
                    if(done->magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || done->processed_request_count!=post->count) {
                        error|=kvaf_device::kInvalidArgs;post->error|=kvaf_device::kInvalidArgs;
                    }
                    if(!post->error)post->deferred=0;
                }
#else
                error|=kvaf_device::kInvalidArgs; // DMA fallback needs real hardware.
#endif
            }
            if(post->pending){
#if !V9_SIMULATION
                uint64_t waitStarted=GetSystemCycle();
                auto drained=DrainQueue(entity,queue,post->pending);
                cqTicks+=GetSystemCycle()-waitStarted;
                uint64_t waitError=drained.error;
                error|=waitError;
                completed+=drained.completed;
                if(!waitError){post->pending=0;}
#else
                post->pending=0;++completed;
#endif
            }
            submitted+=post->reserved;
            lastHead=queue->sq_producer_index;lastCq=queue->cq_consumer_index;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        }
        if(row->pending || total!=info[1])error|=kvaf_device::kInvalidArgs;
        info[0]|=uint32_t(error);info[3]=deferred?1:0;info[4]=completed;
        if(!error)row->finalPending=0;
        kvaf_device::WriteStatus(status,error,total,completed,lastHead,lastCq,UINT64_MAX,0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
        if(ctx->contract) {
            RecordContractTicks(ctx,b,4,GetSystemCycle()-fetchStarted);
            RecordContractTicks(ctx,b,6,cqTicks);
            uint64_t publishStarted=GetSystemCycle();
            Simt::VF_CALL<PublishResidentState>(Simt::Dim3{256,1,1},ctx,b,int32_t(error),submitted,FinalPtr(scratch.Get<uint32_t>()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            RecordContractTicks(ctx,b,5,GetSystemCycle()-publishStarted);
        }
        pipe.Destroy();
        if(ctx->contract){RecordContractTicks(ctx,b,15,GetSystemCycle()-kernelStarted);
            RecordKernelSpan(ctx,224+b,kernelStarted,GetSystemCycle());}
    }
#endif
}
extern "C" __global__ __aicore__ void v9_probe_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t lane=GetBlockIdx(); if(lane>=kServiceLanes) return;
        auto ctx=GetContext(bindings);
        TPipe pipe; TBuf<TPosition::VECCALC> buffer; pipe.InitBuffer(buffer,98304);
        Service(reinterpret_cast<GM_ADDR>(ctx),V9_PROBE_EPOCH,ServiceCore(V9_PROBE_EPOCH,lane/2),lane%2,
            buffer.Get<uint32_t>(),buffer.Get<uint16_t>()[24576],buffer.Get<uint32_t>()[20480]);
        pipe.Destroy();
    }
#endif
}
extern "C" __global__ __aicore__ void v9_index_write_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        Simt::VF_CALL<WriteOwnedIndexQ4>(Simt::Dim3{256,1,1},
            reinterpret_cast<__gm__ uint64_t*>(bindings),GetBlockIdx());
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
#endif
}
