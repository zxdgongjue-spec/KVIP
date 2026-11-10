#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"

#include <stdint.h>
#if defined(__CCE_AICORE__)
#define V9_CQ_INLINE __aicore__ inline
#else
#define V9_CQ_INLINE inline
#endif
namespace rotating {
struct CqDrainResult { uint64_t error; uint32_t completed; uint64_t polls; };
// Exclusive CQ consumer. Inspect every status; never infer success from the tail.
// pending < depth ensures no CQ space must be reclaimed before the tail arrives.
template<class Access>
V9_CQ_INLINE CqDrainResult DrainCq(Access& access, uint64_t consumer,
    uint32_t depth, uint32_t pending, uint32_t maxPolls,
    uint64_t invalid, uint64_t cqError, uint64_t timeout) {
    CqDrainResult result{0,0,0};
    if (!pending) return result;
    if (depth < 2 || pending >= depth) { result.error=invalid; return result; }
    const uint64_t initial=consumer;
    for (uint32_t i=0;i<pending;++i) {
        bool ready=false;
        for (uint32_t poll=0;poll<maxPolls;++poll) {
            uint32_t word=access.Read(uint32_t(consumer%depth));
            ++result.polls;
            if (((word>>2)&1U)==((consumer/depth)&1U)) continue;
            ready=true;
            ++consumer;
            if ((word>>16)!=0) result.error=cqError;
            else ++result.completed;
            break;
        }
        if (!ready) result.error=timeout;
        if (result.error) break;
    }
    // Also reclaim an observed error CQE or the successful prefix on timeout.
    if (consumer!=initial) access.Commit(consumer);
    return result;
}
}
#undef V9_CQ_INLINE

#include "prefetch.h"
#include "resolver.h"
#include "index_cache_update.h"
using namespace AscendC;
using namespace rotating;
// Extend fused CQ waits to 10x the native polling budget (not wall-clock time).
constexpr uint32_t kFusedCqPollMultiplier = 10U;
#if V9_RESOLVER_COMPARE
namespace resolver_original { using namespace rotating;
#define rotating original_body
#include "resolver_reference/original.h"
#undef rotating
}
namespace resolver_adapted { using namespace rotating;
#define rotating adapted_body
#include "resolver_reference/adapted.h"
#undef rotating
}
#endif
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
        kvaf_device::kMaxCqPolls*kFusedCqPollMultiplier,kvaf_device::kInvalidArgs,kvaf_device::kCqError,kvaf_device::kCqTimeout);
    queue->completed_batch_count+=result.completed;
    return result;
}
#endif
#if defined(__CCE_AICORE__)
__aicore__ inline void FillRoundWords(uint64_t address,uint32_t words,
    LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=4096) {
        uint32_t count=words-offset;if(count>4096)count=4096;
        DataCopy(target[offset],local,count);
    }
    // Caller joins the whole fill group before reusing the shared UB value.
}
__aicore__ inline uint64_t InitializeRound(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t b) {
    TPipe pipe;TBuf<TPosition::VECCALC> storage;pipe.InitBuffer(storage,4096*4);
    auto local=storage.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillRoundWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    FillRoundWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    FillRoundWords(ctx->owners+b*128,32,local);
    FillRoundWords(ctx->owners+rt_batch*128+b*rt_finalWorkers*kFinalStateWords*4,rt_finalWorkers*kFinalStateWords,local);
    FillRoundWords(ctx->owners+rt_finalPostOffset+b*rt_finalWorkers*sizeof(FinalPost),rt_finalWorkers*sizeof(FinalPost)/4,local);
    FillRoundWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    FillRoundWords(ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus),2*sizeof(KvafBatchGetCoreStatus)/4,local);
    // Both hot and cold predicates are overwritten before progress is
    // published. Original per-query snapshot clipping excludes unready bits.
    for(uint32_t e=b;e<(rt_rounds?rt_rounds:1)*rt_serviceLanes;e+=rt_batch)
        FillRoundWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    FillRoundWords(ctx->mapping+uint64_t(b)*rt_history*4,rt_history,local);
    if(ctx->changes)FillRoundWords(ctx->changes+b*kChangeCapacity*4,kChangeCapacity,local);
    uint64_t reservedRow=0;
#if V9_COMPACT_RESERVED
    reservedRow=ClearReservedHot(ctx->lruSlots,ctx->reservedHot,ctx->queryEnds,b,rt_resident,local);
#endif
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    // Copy only the candidate prefix; resolver writes extended destinations.
    if(ctx->changeSlots) {
        GlobalTensor<uint32_t> source,target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity);
        DataCopy(local,source,kLruSlots);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(target,local,kLruSlots);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    pipe.Destroy();
    return reservedRow;
}
#endif
extern "C" __global__ __aicore__ void v9_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_rounds = runtimeShape.rounds;
    const uint32_t rt_serviceLanes = runtimeShape.serviceLanes;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx(); if(b>=rt_batch) return;
        auto ctx=GetContext(bindings);
        auto reservedRow=InitializeRound(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_finalWorkers,rt_finalPostOffset,ctx,b);
        auto args=GetArgs(ctx,b*2);
        auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
        auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args.queue_state);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        row->entity=args.channel_entity; row->queue=args.queue_state;
        const uint64_t prefix=4*2048*6+1024+4*uint64_t(rt_history)*2;
        const uint64_t tail=uint64_t(rt_history)*8+8192*64+512;
        auto a=reinterpret_cast<__gm__ uint64_t*>(bindings);
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        uint64_t outputStride=ctx->outputStride ? ctx->outputStride : rt_resident;
        row->addresses=ctx->tokenAddresses ? ctx->tokenAddresses+uint64_t(owner)*ctx->addressStride*8 : a[13]+rt_batch*prefix+b*tail;
        row->output=args.output+uint64_t(owner)*outputStride*rt_tokenBytes;
        Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},rt_history,rt_resident,ctx,b,reservedRow);
        row->initialHead=queue->sq_producer_index;
        row->limit=kSlots;
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
           args.token_bytes!=rt_tokenBytes || args.tokens_per_block!=128 ||
           args.block_count!=1 || args.jettys_per_core!=1)
            row->error=kvaf_device::kInvalidArgs;
        if(entity->magic!=kvaf_device::kEntityMagic || entity->version!=kvaf_device::kEntityVersion ||
           entity->wqe_size!=64 || !entity->cq_va || !entity->cq_db_va ||
           entity->cqe_size<kvaf_device::kCqeProbeBytes || !entity->local_token_id ||
           entity->local_hbm_address!=args.output ||
           entity->local_hbm_size<(uint64_t(owner)+1)*outputStride*rt_tokenBytes ||
           !entity->remote_host_ddr_address || entity->remote_host_ddr_size<rt_tokenBytes)
            row->error=kvaf_device::kInvalidEntity;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
#endif
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(*row));
    }
#endif
}
#if defined(__CCE_AICORE__)
__aicore__ inline void CompletePrefetch(__gm__ Context* ctx,uint32_t b) {
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
#if !V9_SIMULATION
        if(!error) {
            auto result=DrainQueue(entity,queue,row->pending);
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
}
#endif
// Retained for the compute-only diagnostic Graph, which has no final submission.
extern "C" __global__ __aicore__ void v9_wait_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t b=GetBlockIdx();if(b<rt_batch)CompletePrefetch(GetContext(bindings),b);
    }
#endif
}
#if defined(__CCE_AICORE__) && V9_URMA_ASYNC_FINAL
#include "async_final.h"
#endif
extern "C" __global__ __aicore__ void v9_resolve_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalCores = runtimeShape.finalCores;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        ReplicatedFinal(rt_batch,rt_history,rt_resident,rt_finalWorkers,rt_finalPostOffset,rt_tokenBytes,GetContext(bindings));
#if V9_URMA_ASYNC_FINAL
        // ReplicatedFinal destroyed its pipe and published this worker's
        // descriptor slice. Native WQE staging may now borrow UB offset zero.
        SubmitSafeFinal(GetContext(bindings),GetBlockIdx());
#endif
    }
#endif
}
// Final transport consumes compact resolver ranges through native KVAF.
// Submit disjoint destinations first; join prefetch before overwriting its slots.
#if defined(__CCE_AICORE__)
#include "parallel_final.h"
extern "C" __global__ __aicore__ void v9_parallel_fetch_kernel(GM_ADDR bindings) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
#if V9_URMA_ASYNC_FINAL
        CompleteAsyncFinal(GetContext(bindings),GetBlockIdx());
#else
        ParallelFinalGet(GetContext(bindings),GetBlockIdx());
#endif
    }
}
__aicore__ inline void FinishFetch(GM_ADDR bindings, GM_ADDR update) {
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;

    uint32_t b=GetBlockIdx(); if(b>=rt_batch) return;
    auto ctx=GetContext(bindings);
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
#if V9_URMA_PARALLEL_FINAL
    // The prefetch join ran on a different AIV invocation. Its row/error and
    // pending writes must be acquired before reducing the final worker posts.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
#endif
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),kRowStatusWords*sizeof(uint32_t));
    uint32_t updateError=0;
    if(update) {
        auto updateBindings=reinterpret_cast<__gm__ uint64_t*>(update);
        auto status=reinterpret_cast<__gm__ uint32_t*>(updateBindings[6]);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status+b),sizeof(uint32_t));
        updateError=status[b];
        if(!updateError)
            Simt::VF_CALL<PublishDestinationResidency>(Simt::Dim3{256,1,1},rt_resident,ctx,b,info[1]);
    }
    auto args=GetArgs(ctx,b*2);
    args.core_status=ctx->finalStatus+b*2*sizeof(KvafBatchGetCoreStatus);
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
    auto spare=status+1;
    kvaf_device::WriteStatus(spare,0,0,0,0,0,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(spare),sizeof(*spare));
    uint64_t error=info[0] ? kvaf_device::kInvalidArgs : row->error;
    uint32_t completed=0,total=0,deferred=0;
    uint64_t lastHead=0,lastCq=0;
#if V9_URMA_PARALLEL_FINAL
    // The separate parallel kernel joined every worker's native Get. This
    // reducer never reuses its queues or submits a second copy of the range.
    for(uint32_t w=0;w<rt_finalWorkers;++w) {
        auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
        auto workerArgs=GetArgs(ctx,FinalQueue(rt_batch,rt_finalWorkers,b,w));
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(workerArgs.queue_state);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        total+=post->count;completed+=post->head;deferred+=post->deferred!=0;
        error|=post->error;
        if(post->pending || post->deferred || post->nonconflicting>post->count)error|=kvaf_device::kInvalidArgs;
        lastHead=queue->sq_producer_index;lastCq=queue->cq_consumer_index;
    }
#else
    for(uint32_t phase=0;phase<2;++phase) {
      if(phase==1) {
          CompletePrefetch(ctx,b);
          error|=row->error;
      }
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
    }
#endif
    if(row->pending || total!=info[1])error|=kvaf_device::kInvalidArgs;
    info[0]|=uint32_t(error);info[3]=deferred?1:0;info[4]=completed;
    if(!error) {
        if(!update)
            Simt::VF_CALL<PublishDestinationResidency>(Simt::Dim3{256,1,1},rt_resident,ctx,b,total);
        row->finalPending=0;
    }
    kvaf_device::WriteStatus(status,error,total,completed,lastHead,lastCq,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),32);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    if(update && (error || updateError))
        Simt::VF_CALL<CompleteEarlyUpdate<256>>(Simt::Dim3{256,1,1},
            reinterpret_cast<__gm__ uint64_t*>(update),b,error);
}
#endif
extern "C" __global__ __aicore__ void v9_fetch_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishFetch(bindings,nullptr); }
#endif
}
extern "C" __global__ __aicore__ void v9_fetch_update_kernel(GM_ADDR bindings, GM_ADDR update) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { FinishFetch(bindings,update); }
#endif
}
extern "C" __global__ __aicore__ void v9_probe_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_splitQ = runtimeShape.splitQ;
    const uint32_t rt_rounds = runtimeShape.rounds;
    const uint32_t rt_serviceAics = runtimeShape.serviceAics;
    const uint32_t rt_serviceLanes = runtimeShape.serviceLanes;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;
    const uint32_t rt_postThreads = runtimeShape.postThreads;
    const uint32_t rt_postBank = runtimeShape.postBank;
    const uint32_t rt_batchedRows = runtimeShape.batchedRows;
    const uint32_t rt_scanTokens = runtimeShape.scanTokens;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        uint32_t lane=GetBlockIdx(); if(lane>=rt_serviceLanes) return;
        auto ctx=GetContext(bindings);
        TPipe pipe; TBuf<TPosition::VECCALC> buffer; pipe.InitBuffer(buffer,98304);
        Service(rt_batch,rt_history,rt_resident,rt_splitQ,rt_rounds,rt_serviceAics,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,reinterpret_cast<GM_ADDR>(ctx),V9_PROBE_EPOCH,ServiceCore(rt_batch,rt_splitQ,rt_serviceAics,V9_PROBE_EPOCH,lane/2),lane%2,
            buffer.Get<uint32_t>(),buffer.Get<uint16_t>()[24576],buffer.Get<uint32_t>()[20480]);
        pipe.Destroy();
    }
#endif
}

#if V9_RESOLVER_COMPARE
extern "C" __global__ __aicore__ void v9_resolve_original_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { resolver_original::original_body::ReplicatedFinal(rt_batch,rt_history,rt_resident,rt_finalWorkers,rt_finalPostOffset,rt_tokenBytes,GetContext(bindings)); }
#endif
}
extern "C" __global__ __aicore__ void v9_resolve_adapted_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    auto runtimeShape = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]));
    const uint32_t rt_batch = runtimeShape.batch;
    const uint32_t rt_history = runtimeShape.history;
    const uint32_t rt_resident = runtimeShape.resident;
    const uint32_t rt_finalWorkers = runtimeShape.finalWorkers;
    const uint32_t rt_finalPostOffset = runtimeShape.finalPostOffset;
    const uint32_t rt_tokenBytes = runtimeShape.tokenBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV { resolver_adapted::adapted_body::ReplicatedFinal(rt_batch,rt_history,rt_resident,rt_finalWorkers,rt_finalPostOffset,rt_tokenBytes,GetContext(bindings)); }
#endif
}
#endif
