#pragma once
#include "prefetch.h"
#include "../../lidu/op_kernel/hot_slot_selection.h"

// Imported from final_resolve.h

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

#include "../../common/lifetch/resolver_candidates.h"

#include "../../common/lifetch/resident_selection.h"

// Imported from final_worker_submit.h

namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

#define LIFETCH_CLEAN_CACHE_LINES kvaf_device::CleanCacheLines
#include "../../common/lifetch/final_post.h"
#undef LIFETCH_CLEAN_CACHE_LINES
}

// Imported from final_scratch.h

#include "../../common/lifetch/final_scratch.h"

namespace rotating {
// Every worker independently builds the same rank/slot plan from immutable
// prefetch mapping. Rank is the first TopK occurrence, NOT atomic arrival order.
// Only disjoint location/descriptor/WQE outputs are partitioned across workers.
#include "../../common/lifetch/resolver.h"

#if !V9_SIMULATION
// Each resolver worker owns one final Jetty. Post one checkpoint for its
// disjoint prefix; larger ranges still use KVAF's scheduled Fetch path.
__aicore__ inline bool PostSafeFinalPrefix(__gm__ Context* ctx,uint32_t owner,
    uint32_t tokenBytes,uint32_t begin,uint32_t count,uint64_t& head) {
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+owner;
    const uint32_t budget=args->batch_get_max_inflight_total;
    const uint64_t addresses=ctx->fallbackAddresses?ctx->fallbackAddresses:args->address_table;
    if(!addresses || !args->channel_entity || !args->queue_state ||
       args->block_count!=1 || args->jettys_per_core>1 || args->token_bytes!=tokenBytes ||
       (budget && (budget<2 || count>budget)))return false;
    auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
    if(entity->magic!=kvaf_device::kEntityMagic || entity->version!=kvaf_device::kEntityVersion ||
       !args->output || entity->local_hbm_address!=args->output ||
       !entity->local_token_id || entity->local_hbm_size<tokenBytes ||
       !entity->remote_host_ddr_address || entity->remote_host_ddr_size<tokenBytes ||
       entity->sq_depth!=kvaf_device::kSqDepthWqebbs || count>=entity->sq_depth ||
       !entity->sq_va || !entity->sq_db_va || entity->wqe_size!=kvaf_device::kWqeBytes ||
       !entity->cq_va || !entity->cq_db_va || entity->cq_depth<2 ||
       entity->cqe_size<kvaf_device::kCqeProbeBytes)return false;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    if(queue->poisoned || queue->last_error ||
       (queue->initialized && queue->magic!=kvaf_device::kQueueMagic) ||
       queue->sq_producer_index>UINT64_MAX-count)return false;
    if(!queue->initialized) {
        queue->magic=kvaf_device::kQueueMagic;
        queue->initialized=1;
        queue->sq_producer_index=0;
        queue->cq_consumer_index=0;
        queue->completed_batch_count=0;
    }
    head=queue->sq_producer_index;
    // Resolver scratch starts at UB offset zero and is dead after location
    // writeback. KVAF reuses it for its two WQE staging banks.
    static_assert(kvaf_device::kWqeBankCount*kvaf_device::kWqeBankBytes<=kFinalScratchBytes,
                  "KVAF WQE staging exceeds resolver scratch");
    kvaf_device::ConfigureUbToGmCopy();
    uint64_t firstDoorbell=0,lastWqe=0;
    kvaf_device::PostDirectRequests(reinterpret_cast<__gm__ const int32_t*>(ctx->sourceBlocks),
        reinterpret_cast<__gm__ const int32_t*>(ctx->destinations),
        reinterpret_cast<__gm__ const int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ const uint64_t*>(addresses),begin,count,0,tokenBytes,1,
        entity,queue,0,&firstDoorbell,&lastWqe);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    return true;
}
#endif

__aicore__ inline void ReplicatedFinal(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,uint32_t rt_tokenBytes,__gm__ Context* ctx) {
#include "../../common/lifetch/resolver_front.h"
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},rt_history,rt_resident,rt_finalWorkers,rt_tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
#include "../../common/lifetch/resolver_output.h"
    const uint32_t descriptor=batchBase+prefix, safe=error?0:ctl.GetValue(5);
    uint64_t head=0;
    bool posted=false;
#if !V9_SIMULATION
    if(safe && safe<=own)
        posted=PostSafeFinalPrefix(ctx,FinalQueue(rt_batch,rt_finalWorkers,b,w),rt_tokenBytes,descriptor,safe,head);
#endif
    PrepareFinalFetch(rt_finalWorkers,rt_finalPostOffset,ctx,b,w,own,descriptor,error,safe,posted,head);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalHead=FinalPostAt(rt_finalPostOffset,ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
