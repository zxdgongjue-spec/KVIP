#pragma once
#include "prefetch.h"
#include "final_scratch.h"
#include "final_worker_submit.h"
#include "../../../common/lifetch/resolver_candidates.h"
#include "../../../common/lifetch/resident_selection.h"
namespace rotating {
#include "../../../common/lifetch/resolver.h"

__aicore__ inline void ReplicatedFinal(__gm__ Context* ctx) {
    const auto shape=LoadRuntimeShape(ctx);
    const uint32_t rt_history=shape.history,rt_resident=shape.resident,rt_finalWorkers=shape.finalWorkers;
#include "../../../common/lifetch/resolver_front.h"
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},shape.history,shape.resident,shape.finalWorkers,shape.tokenBytes,FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*shape.history,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
#include "../../../common/lifetch/resolver_output.h"
    PrepareFinalFetch(rt_finalWorkers,shape.finalPostOffset,ctx,b,w,own,batchBase+prefix,error,error?0:ctl.GetValue(5));
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalCompleted=FinalPostAt(shape.finalPostOffset,ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
