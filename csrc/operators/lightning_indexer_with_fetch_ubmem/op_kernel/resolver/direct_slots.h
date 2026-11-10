#pragma once
#include "../device_context.h"

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
// Imported from direct_slots.h

#include "simt_api/device_functions.h"
#include "../../../common/lifetch/reserved_hot.h"
#include "../../../common/lifetch/mapping.h"
// Adapt logical fetch slots at the transport boundary; preserve the original
// hot-slot replacement policy; no KV payload fallback arena is allocated.
namespace rotating {
__simt_callee__ inline uint32_t CurrentMappingTag(__gm__ Context* ctx,uint32_t b) {
#if V9_UBMEM_TAGGED_MAP
    return asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+uint64_t(b)*256)<<kMappingSlotBits;
#else
    return 0;
#endif
}
__simt_callee__ inline uint32_t OwnerRow(__gm__ Context* ctx,uint32_t b) {
    return ctx->rowOwners ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)+b) : b;
}
__simt_callee__ inline uint32_t OutputStride(uint32_t rt_resident,__gm__ Context* ctx) {
    return ctx->outputStride ? uint32_t(ctx->outputStride) : rt_resident;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailMappings(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,
    uint64_t reservedRow=0,uint32_t mappingTag=0) {
#if V9_COMPACT_RESERVED
    if(reservedRow)ScatterReservedHot(reinterpret_cast<__gm__ uint32_t*>(reservedRow),
        reinterpret_cast<const __gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*kLruSlots,rt_resident,kLruSlots);
#endif
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
            asc_stcg(map+token,EncodeMappingSlot(rt_resident+kSlots+token%tailTokens,mappingTag,
                V9_UBMEM_TAGGED_MAP?kMappingSlotMask:UINT32_MAX));
    }
    asc_threadfence();
}
__simt_callee__ inline bool IsTailToken(__gm__ Context* ctx,uint32_t b,uint32_t token) {
    if(!ctx->tailBounds)return false;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    return token>=asc_ldcg(bounds) && token<asc_ldcg(bounds+1);
}
__simt_callee__ inline uint32_t QueryEnd(uint32_t rt_history,__gm__ Context* ctx,uint32_t query) {
    return ctx->queryEnds ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+query) : rt_history;
}
__simt_callee__ inline void StoreFallbackSource(uint32_t rt_history,uint32_t rt_tokenBytes,__gm__ Context* ctx,uint32_t batch,uint32_t dst,
    uint32_t token,uint64_t remote,__gm__ int32_t* blocks,__gm__ int32_t* offsets) {
    if(ctx->fallbackAddresses) {
        asc_stcg(reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses)+dst,remote);
        asc_stcg(blocks+dst,int32_t(dst));asc_stcg(offsets+dst,int32_t(0));
    } else {
        asc_stcg(blocks+dst,int32_t(batch*(rt_history/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*rt_tokenBytes));
    }
}
__simt_callee__ inline bool HotValid(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t token) {
    return !ctx->hotValid || asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*rt_resident+token)!=0;
}
__simt_callee__ inline uint32_t PhysicalSlot(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot) {
    if(ctx->lruSlots && uint32_t(slot-rt_resident)<kLruSlots)
        return asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots+slot-rt_resident);
    return slot;
}
__simt_callee__ inline void RecordChange(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot,uint32_t token) {
    if(ctx->changes && uint32_t(slot-rt_resident)<kLruSlots)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity+slot-rt_resident,token);

}
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicate(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate) {
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*rt_resident;
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident &&
            (IsTailToken(ctx,b,token) || (!HotValid(rt_resident,ctx,b,token)) || (ctx->reservedHot && asc_ldcg(reserved+token)>=0)))?rt_resident:token;
    }
}
// Bulk metadata is staged once in UB by the resolver, not fetched per TopK item.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicateLocal(uint32_t rt_resident,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate,
    __ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,bool hasValid,bool hasReserved,
    uint32_t tailBegin,uint32_t tailEnd) {
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident && ((token>=tailBegin && token<tailEnd) ||
            (hasValid && !valid[token]) || (hasReserved && reserved[token]>=0)))?rt_resident:token;
    }
}
__simt_callee__ inline uint32_t ResolvePhysicalSlot(uint32_t rt_resident,__gm__ uint32_t* lru,uint32_t slot) {
    if(lru && uint32_t(slot-rt_resident)<kLruSlots)return asc_ldcg(lru+slot-rt_resident);
    return slot>=rt_resident+kSlots && slot<0x80000000U ? slot-kSlots : slot;
}

}

namespace rotating {
using namespace AscendC;
// v9 GatherCandidateMap's indexed-load loop, adapted to packed uint16 scores.
// One thread owns both halves of each UB word; no half-word write sharing.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherVictimScores(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* output,
    const __gm__ uint32_t* scores) {
    auto input=const_cast<__gm__ uint32_t*>(scores);
    for(uint32_t i=threadIdx.x;i<1024;i+=256) {
        uint32_t a=ids[2*i],b=ids[2*i+1];
        uint32_t low=(asc_ldcg(input+a/2)>>((a%2)*16))&0xffffU;
        uint32_t high=(asc_ldcg(input+b/2)>>((b%2)*16))&0xffffU;
        output[i]=low|(high<<16);
    }
}


}

namespace rotating {
__aicore__ inline uint32_t PhysicalSlotScalar(__gm__ Context* ctx,uint32_t batch,uint32_t slot) {
    return ctx->lruSlots && uint32_t(slot-SIM_RESIDENT)<kLruSlots ?
        reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)[batch*kLruSlots+slot-SIM_RESIDENT] : slot;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RestoreVictimTokens(__ubuf__ uint32_t* ids,__gm__ uint32_t* victims,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i];
        if(token<kLruSlots)ids[i]=asc_ldcg(victims+token);
    }
}
}
