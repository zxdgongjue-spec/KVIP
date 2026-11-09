#pragma once

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

#include "simt_api/device_functions.h"
// Adapt logical fetch slots at the transport boundary; preserve the original
// hot-slot replacement policy; no KV payload fallback arena is allocated.
namespace rotating {
#include "../../../common/lifetch/cache_metadata.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailMappings(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b) {
    if(!threadIdx.x && ctx->changeCount) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots);
        asc_threadfence();
    }
    if(!ctx->tailBounds)return;
    const uint32_t tailTokens=OutputStride(rt_resident,ctx)-rt_resident;
    if (!tailTokens) return;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
    uint32_t begin=asc_ldcg(bounds),end=asc_ldcg(bounds+1);
    for(uint32_t i=threadIdx.x;i<tailTokens;i+=256) {
        uint32_t token=begin+i;
        if(token<end && token<rt_history)
            asc_stcg(map+token,rt_resident+kSlots+token%tailTokens);
    }
    asc_threadfence();
}

// Publish residency only after all UBmem copy pipelines report completion. Resolver
// workers must all observe the same pre-round validity, not one another's posts.



}

namespace rotating {
using namespace AscendC;
// v9 GatherCandidateMap's indexed-load loop, adapted to packed uint16 scores.
// One thread owns both halves of each UB word; no half-word write sharing.
#include "../../../common/lifetch/victim_ops.h"


}

namespace rotating {
__aicore__ inline uint32_t PhysicalSlotScalar(__gm__ Context* ctx,uint32_t batch,uint32_t slot) {
    uint32_t resident=ctx->shape.resident;
    return ctx->lruSlots && uint32_t(slot-resident)<kLruSlots ?
        reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)[batch*kLruSlots+slot-resident] : slot;
}

}
