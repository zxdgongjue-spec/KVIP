#pragma once
#define V9_CQ_INLINE __aicore__ inline
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
constexpr uint32_t kFusedCqPollMultiplier = 10U;
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
__aicore__ inline void CompletePrefetch(__gm__ Context* ctx,uint32_t b,bool initializeSpare,
    uint64_t* diagnosticPolls=nullptr,uint64_t* diagnosticDrain=nullptr,uint64_t* diagnosticArrival=nullptr) {
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
            uint64_t start=diagnosticDrain?GetSystemCycle():0;
            auto result=DrainQueue(entity,queue,row->pending);
            if(diagnosticArrival)*diagnosticArrival=GetSystemCycle();
            if(diagnosticDrain)*diagnosticDrain=*diagnosticArrival-start;
            if(diagnosticPolls)*diagnosticPolls=result.polls;
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
        if(initializeSpare){
            auto spare=GetArgs(ctx,b*2+1);
            auto unused=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(spare.core_status);
            kvaf_device::WriteStatus(unused,0,0,0,0,0,UINT64_MAX,0);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(unused),sizeof(*unused));
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        // Only the first cache line and the pending field are read/written here.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
}
