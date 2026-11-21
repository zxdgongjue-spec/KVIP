// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "query_owner_layout.h"
namespace rotating {
// Classification already has both arrays in UB and runs before the tail kernel.
// Preserve only miss DDR values; emit logical IDs straight into the output.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void SaveOldQueryInputs(
    __gm__ pipeline_ab::Args *a, __gm__ Context *ctx, uint32_t query, __gm__ uint64_t *addresses,
    __ubuf__ uint32_t *indices, __ubuf__ uint32_t *locations, __ubuf__ uint32_t *alias) {
    const uint32_t b = query / 4, q = query % 4;
    auto scratch = reinterpret_cast<__gm__ uint32_t *>(a->queryOwnerScratch) + b * pipeline_ab::kQueryOwnerRowWords;
    auto saved = reinterpret_cast<__gm__ uint64_t *>(scratch + pipeline_ab::kOldAddresses) + q * kTop;
    auto mapping = reinterpret_cast<__gm__ uint32_t *>(ctx->mapping) + uint64_t(b) * a->history;
    // The classifier zeroed this spare UB word with the bitmap.
    bool overlaps = false;
    for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
        uint32_t token = indices[i];
        if (token < a->history && locations[i] == UINT32_MAX) {
            if (addresses) {
                asc_stcg(saved + i, asc_ldcg(addresses + token));
            }
            overlaps |= token < a->resident || asc_ldcg(mapping + token) != UINT32_MAX;
        }
    }
    if (overlaps) {
        asc_atomic_or(alias, 1U);
    }
    auto u = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
    if (!u[20] || !u[21] || !u[22] || !u[23]) {
        asc_threadfence();
        return;
    }
    auto packed=reinterpret_cast<__gm__ int64_t*>(u[20]), valid=reinterpret_cast<__gm__ int64_t*>(u[21]);
    auto logical = reinterpret_cast<__gm__ int64_t *>(u[3]) + uint64_t(OwnerRow(ctx, b)) * u[24];
    // One coalesced ballot per 32 output rows. Support repeated unpack
    // indices and inactive rows with the same semantics as the ordinary output worker.
    for (uint32_t base = 0; base < ctx->shape.batch * 4; base += 32) {
        const uint32_t target = base + threadIdx.x % 32;
        const bool active = target < ctx->shape.batch * 4 && asc_ldcg(valid + target) != 0;
        uint32_t targets = asc_ballot(active ? asc_ldcg(packed + target) == query : target == query);
        uint32_t activeMask = asc_ballot(active);
        while (targets) {
            const uint32_t bit = __builtin_ctz(targets);
            const uint32_t offset = (base + bit) * kTop;
            auto out = reinterpret_cast<__gm__ uint32_t *>(u[22]) + offset;
            const bool copy = (activeMask >> bit) & 1U;
            for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
                uint32_t index = copy ? indices[i] : UINT32_MAX;
                asc_stcg(out + i, index < a->history ? uint32_t(asc_ldcg(logical + index)) : UINT32_MAX);
            }
            targets &= targets - 1;
        }
    }
    asc_threadfence();
}
} // namespace rotating
