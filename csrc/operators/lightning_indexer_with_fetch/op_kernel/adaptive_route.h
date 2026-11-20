// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "pipeline_types.h"

namespace pipeline_ab {
namespace adaptive_route {
enum Mode : uint32_t { High = 0, Zero = 1, Early = 2 };
constexpr uint32_t kRequestWords = 4;
constexpr uint32_t kFeedbackWords = 16;  // one cache line per stable owner

// Request records are overwritten by H2D at the same addresses on each
// replay. Different AIVs may own a feedback record after row reordering.
__aicore__ inline void Refresh(uint64_t address, uint32_t bytes) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(address), bytes);
}

// All cores of the first LI launch read the same immutable previous-round
// feedback. Later graph nodes consume the single published mode.
__aicore__ inline uint32_t Choose(__gm__ const Args* a, __gm__ rotating::Context* ctx,
                                  const rotating::RuntimeShape& s) {
    if (!a || !ctx || !a->routeRequests || !a->routeFeedback || !a->routeDemand ||
        (a->routePolicy != Zero && a->routePolicy != Early) || !ctx->rowOwners || !ctx->queryEnds ||
        !ctx->reservedHot || !ctx->fallbackAddresses || !ctx->sourceBlocks ||
        !a->completedFlags || (a->prefetchCount & 0x20000000U) || !(a->prefetchCount & 0x10000000U) ||
        !s.splitQ || s.batch != 16 || s.indexHeads != 32 || s.resident != 16384) return High;
    auto owners = reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners);
    auto ends = reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds);
    auto requests = reinterpret_cast<__gm__ const uint64_t*>(a->routeRequests);
    auto feedback = reinterpret_cast<__gm__ const uint64_t*>(a->routeFeedback);
    Refresh(a->routeRequests, s.batch*kRequestWords*sizeof(uint64_t));
    Refresh(ctx->rowOwners, s.batch*sizeof(uint32_t));
    Refresh(ctx->queryEnds, s.batch*4*sizeof(uint32_t));
    uint64_t bytes = 0, queryTokens = 0;
    for (uint32_t b = 0; b < s.batch; ++b) {
        auto r = requests + b*kRequestWords;
        uint64_t owner = r[0], generation = r[1], step = r[2];
        if (owner >= a->routeOwners || owner != owners[b] || !generation || step <= 1 || r[3] != 4) return High;
        auto f = feedback + owner*kFeedbackWords;
        Refresh(reinterpret_cast<uint64_t>(f), 4*sizeof(uint64_t));
        if (f[0] != generation || f[1] != step-1 || f[3] != 4 || f[2] > kPrefetch+kMaxFinal) return High;
        bytes += f[2]*s.tokenBytes;
        for (uint32_t q = 0; q < 4; ++q) {
            uint32_t end = ends[4*b+q];
            if (end < s.resident || end > s.history) return High;
            queryTokens += end;
        }
    }
    // At 128K, Q4, 656 bytes/row this admits about 125 transfers/request.
    // Shorter contexts or larger payloads reduce the admissible Get volume.
    return bytes*32 <= queryTokens*5 ? a->routePolicy : High;
}

__aicore__ inline uint32_t Published(__gm__ const Args* a) {
    if (!a || !a->routeResult) return High;
    Refresh(a->routeResult, sizeof(uint32_t));
    return *reinterpret_cast<__gm__ volatile uint32_t*>(a->routeResult);
}

__aicore__ inline void Commit(__gm__ Args* a, uint32_t b, uint64_t error, uint32_t reads) {
    if (error || !a || !a->context || !a->routeRequests || !a->routeFeedback || !a->routeDemand) return;
    auto demand = reinterpret_cast<__gm__ uint32_t*>(a->routeDemand) + b*32;
    *demand = reads;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(demand),128);
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(a->context);
    auto r = reinterpret_cast<__gm__ const uint64_t*>(a->routeRequests) + b*kRequestWords;
    Refresh(reinterpret_cast<uint64_t>(r), kRequestWords*sizeof(uint64_t));
    if (ctx->rowOwners) Refresh(ctx->rowOwners+b*sizeof(uint32_t), sizeof(uint32_t));
    uint64_t owner = r[0], generation = r[1], step = r[2];
    if (!ctx->rowOwners || owner >= a->routeOwners || !generation || !step || r[3] != 4 ||
        owner != reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]) return;
    auto f = reinterpret_cast<__gm__ uint64_t*>(a->routeFeedback) + owner*kFeedbackWords;
    Refresh(reinterpret_cast<uint64_t>(f), 4*sizeof(uint64_t));
    if (generation < f[0] || (generation == f[0] && step <= f[1])) return;
    f[0] = generation;
    f[1] = step;
    f[2] = reads;
    f[3] = r[3];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),128);
}
} // namespace adaptive_route

__aicore__ inline bool AdaptiveLow(__gm__ const Args* a) {
    return adaptive_route::Published(a) == adaptive_route::Zero;
}
} // namespace pipeline_ab
