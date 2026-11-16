// SPDX-License-Identifier: Apache-2.0
// Read immutable hot metadata only for the final selected physical Index IDs.
// Mapping publication must be joined by the caller before this function runs.
#pragma once

#ifndef KVIP_SELECTED_METADATA
#define KVIP_SELECTED_METADATA 0
#endif

namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(kClassifyThreads)
inline void ClassifySelectedMetadata(uint32_t history, uint32_t resident,
    uint32_t batch, uint32_t owner, uint32_t used,
    uint32_t tailBegin, uint32_t tailEnd, bool hasTail, __gm__ Context* ctx,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* locations,
    __ubuf__ uint32_t* bitmap, __ubuf__ uint32_t* error, bool preserveHot = false) {
    const uint32_t words = (resident + 31) / 32;
    const uint32_t shards = 4096 / words < 8 ? 4096 / words : 8;
    auto mapping = reinterpret_cast<__gm__ uint32_t*>(ctx->mapping) + uint64_t(batch) * history;
    auto lru = reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots) + batch * kLruSlots;
    auto valid = reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid);
    auto reserved = reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot);
    // One VF serves both routes. Keep the per-invocation predicate uniform;
    // zero speculative writes is required before retaining reserved hot hits.
    preserveHot = preserveHot && !used;
    uint32_t tailTokens = preserveHot ? TailCapacity(resident, ctx) : 0;
#if KVIP_INVERSE_MAPPING && !V9_UBMEM
    PrefetchMappingView view;
    if (!preserveHot)
        view = LoadPrefetchMappingView(history, resident, batch, ctx, mapping);
#endif
    for (uint32_t i = threadIdx.x; i < kTop; i += kClassifyThreads) {
        uint32_t token = indices[i];
        uint32_t slot = UINT32_MAX;
        if (token == UINT32_MAX) {
            locations[i] = slot;
            continue;
        }
        if (token >= history) {
            locations[i] = slot;
            asc_atomic_or(error, 1U);
            continue;
        }
        bool inTail = hasTail && token >= tailBegin && token < tailEnd;
        // hotValid and reservedHot are read-only throughout this LI launch.
        // Invalid/tail/cold tokens must never index resident-only metadata.
        // The speculative path keeps its conservative reservation protection.
        if (token < resident && !inTail &&
            (!valid || asc_ldcg(valid + uint64_t(owner) * resident + token)) &&
            (preserveHot || !reserved || asc_ldcg(reserved + uint64_t(batch) * resident + token) < 0)) {
            slot = token;
        } else if (preserveHot) {
            // No speculative mapping exists in this round. Match the exact
            // capacity-clipped ABC mapping without reading last round's table.
            if (inTail && token - tailBegin < tailTokens)
                slot = resident + token % tailTokens;
        } else if (used || hasTail) {
#if KVIP_INVERSE_MAPPING && !V9_UBMEM
            slot = ResolvePhysicalSlot(resident, lru,
                ReadPrefetchMapping(view, token));
#else
            slot = ResolvePhysicalSlot(resident, lru, asc_ldcg(mapping + token));
#endif
        }
        locations[i] = slot;
        if (slot < resident) {
            asc_atomic_or(bitmap + (threadIdx.x % shards) * words + slot / 32, 1U << (slot % 32));
        }
    }
    asc_syncthreads();
}
}  // namespace rotating
