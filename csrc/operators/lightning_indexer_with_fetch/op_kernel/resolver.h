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
__aicore__ inline void ReplicatedFinal(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,uint32_t rt_tokenBytes,__gm__ Context* ctx) {
#include "../../common/lifetch/resolver_front.h"
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},rt_history,rt_resident,rt_finalWorkers,rt_tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
#include "../../common/lifetch/resolver_output.h"
    PrepareFinalFetch(rt_finalWorkers,rt_finalPostOffset,ctx,b,w,own,batchBase+prefix,error,error?0:ctl.GetValue(5));
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
