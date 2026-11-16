// SPDX-License-Identifier: Apache-2.0
#pragma once

// Used by scalar init and SIMT readers: both must choose the same fallback.
// Adaptive admission already rejects totals larger than this inverse prefix.
#define KVIP_INVERSE_MAPPING_ACTIVE(ctx) \
    ((ctx)->changes != 0 && (ctx)->lruSlots != 0 && (ctx)->prefetchLimit <= 2048U)

// Included inside namespace rotating, after cache_metadata.h.
struct PrefetchMappingView {
    __gm__ uint32_t* mapping;
    __gm__ uint32_t* inverse;
    uint32_t history, resident, tailBegin, tailEnd, tailTokens;
};

// Construct once per SIMT thread, outside its token loop. Context/bounds are
// immutable during LI; only mapping and inverse records need per-token loads.
__simt_callee__ inline PrefetchMappingView LoadPrefetchMappingView(uint32_t history,
    uint32_t resident, uint32_t batch, __gm__ Context* ctx, __gm__ uint32_t* mapping) {
    PrefetchMappingView view{mapping, nullptr, history, resident, 0, 0, 0};
    if (!KVIP_INVERSE_MAPPING_ACTIVE(ctx)) return view;
    view.inverse = reinterpret_cast<__gm__ uint32_t*>(ctx->changes) +
        uint64_t(batch) * kChangeCapacity;
    view.tailTokens = TailCapacity(resident, ctx);
    if (ctx->tailBounds && view.tailTokens) {
        auto bounds = reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds) + batch * 2;
        view.tailBegin = asc_ldcg(bounds);
        view.tailEnd = asc_ldcg(bounds + 1);
    }
    return view;
}

// Call only before StagedQuery reuses changes for final repair. Prefetch writes
// map and inverse with stcg, then publishes Event.complete after threadfence;
// callers retain that join (or are the next serialized prefetch epoch).
__simt_callee__ inline uint32_t ReadPrefetchMapping(const PrefetchMappingView& view,
    uint32_t token) {
    if (!view.inverse) return asc_ldcg(view.mapping + token);
    if (token >= view.history) return UINT32_MAX;
    // Match InitializeTailMappings, including its capacity-clipped range.
    if (token >= view.tailBegin && token < view.tailEnd &&
        token - view.tailBegin < view.tailTokens)
        return view.resident + kSlots + token % view.tailTokens;
    const uint32_t slot = asc_ldcg(view.mapping + token);
    const uint32_t ordinal = slot - view.resident;
    if (ordinal >= kLruSlots) return UINT32_MAX;
    return asc_ldcg(view.inverse + ordinal) == token ? slot : UINT32_MAX;
}
