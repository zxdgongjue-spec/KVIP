// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "adaptive_route.h"

namespace pipeline_ab {
namespace adaptive_route {
#if defined(__CCE_AICORE__)
#define KVIP_ROUTE_VF __simt_callee__
template <typename T>
__simt_callee__ inline T RouteLoad(__gm__ T* p) { return asc_ldcg(p); }
#else
#define KVIP_ROUTE_VF
template <typename T>
inline T RouteLoad(T* p) { return *p; }
#endif

struct Traffic { uint32_t reads, tokens, rejected; };

KVIP_ROUTE_VF inline Traffic RowTraffic(uint32_t b, __gm__ uint64_t* requests,
    __gm__ uint64_t* feedback, __gm__ uint32_t* owners, __gm__ uint32_t* ends,
    uint32_t ownerCount, uint32_t resident, uint32_t history) {
    auto r = requests + b*kRequestWords;
    uint64_t owner = RouteLoad(r), generation = RouteLoad(r+1), step = RouteLoad(r+2);
    if (owner >= ownerCount || owner != RouteLoad(owners+b) ||
        !generation || step <= 1 || RouteLoad(r+3) != 4) return {0, 0, 1};
    auto f = feedback + owner*kFeedbackWords;
    uint64_t reads = RouteLoad(f+2);
    if (RouteLoad(f) != generation || RouteLoad(f+1) != step-1 ||
        RouteLoad(f+3) != 4 || reads > kPrefetch+kMaxFinal) return {0, 0, 1};
    uint32_t tokens = 0;
    for (uint32_t q = 0; q < 4; ++q) {
        uint32_t end = RouteLoad(ends+4*b+q);
        if (end < resident || end > history) return {0, 0, 1};
        tokens += end;
    }
    return {uint32_t(reads), tokens, 0};
}

KVIP_ROUTE_VF inline uint32_t FromTotals(Traffic sum, uint32_t tokenBytes, uint32_t policy) {
    return !sum.rejected && uint64_t(sum.reads)*tokenBytes*32 <= uint64_t(sum.tokens)*5 ? policy : High;
}
#undef KVIP_ROUTE_VF

#if defined(__CCE_AICORE__)
// One warp reads all sixteen immutable records in parallel. The second half
// of the warp contributes zero. The 32-bit sums are safe for the supported
// maximums of 16 * 10,240 reads and 64 * 210,944 query positions.
__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void ReduceRows(__gm__ uint64_t* requests, __gm__ uint64_t* feedback,
    __gm__ uint32_t* owners, __gm__ uint32_t* ends, uint32_t ownerCount, uint32_t resident,
    uint32_t history, uint32_t tokenBytes, uint32_t policy, __gm__ uint32_t* result) {
    const unsigned lane = threadIdx.x;
    Traffic sum{0, 0, 0};
    if (lane < 16) sum = RowTraffic(lane, requests, feedback, owners, ends, ownerCount, resident, history);
    for (unsigned d = 8; d; d >>= 1) {
        sum.reads += asc_shfl_down(sum.reads, d);
        sum.tokens += asc_shfl_down(sum.tokens, d);
        sum.rejected |= asc_shfl_down(sum.rejected, d);
    }
    if (!lane) {
        asc_stcg(result, FromTotals(sum, tokenBytes, policy));
        asc_threadfence();
    }
}

// This is called exclusively from the 1:2 MIX Hot LI entry point. Flag 9
// is local to a cube plus its two vector cores. Only AIV0 signals it.
// The network's first MM uses flags 0..3 and streaming rotation uses 8.
__aicore__ inline uint32_t ChooseMixed(__gm__ Args* a, __gm__ rotating::Context* ctx,
                                      const rotating::RuntimeShape& shape) {
    using namespace AscendC;
    if (!Eligible(a, ctx, shape)) return High;
    const uint32_t block = GetBlockIdx();
    constexpr uint32_t flag = 9;
    // The AIV0 profile line has a single writer and is allocated in normal
    // runs too. Word zero remains available for its independently timed probe.
    const uint64_t vectorLines = a->routeResult + (1 + shape.aicCount)*128U;
    if ASCEND_IS_AIV {
        const uint64_t slot = vectorLines + block*128U + sizeof(uint64_t);
        Simt::VF_CALL<ReduceRows>(Simt::Dim3{32,1,1},
            reinterpret_cast<__gm__ uint64_t*>(a->routeRequests),
            reinterpret_cast<__gm__ uint64_t*>(a->routeFeedback),
            reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners),
            reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds),
            a->routeOwners, shape.resident, shape.history, shape.tokenBytes, a->routePolicy,
            reinterpret_cast<__gm__ uint32_t*>(slot));
        // The VF release-stores its mode. The cube waits on PIPE_S before
        // loading it, and the vector's scalar read waits on PIPE_V.
        if (!(block & 1U)) CrossCoreSetFlag<4, PIPE_V>(flag);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        Refresh(slot, sizeof(uint32_t));
        return *reinterpret_cast<__gm__ volatile uint32_t*>(slot);
    } else {
        CrossCoreWaitFlag<4, PIPE_S>(flag);
        const uint64_t slot = vectorLines + (2*block)*128U + sizeof(uint64_t);
        Refresh(slot, sizeof(uint32_t));
        return *reinterpret_cast<__gm__ volatile uint32_t*>(slot);
    }
}
#else
// CPU oracle consumes the same row predicate and final integer comparison.
inline uint32_t Choose(const Args* a, rotating::Context* ctx, const rotating::RuntimeShape& shape) {
    if (!Eligible(a, ctx, shape)) return High;
    Traffic sum{};
    for (uint32_t b = 0; b < shape.batch; ++b) {
        const Traffic row = RowTraffic(b,
            reinterpret_cast<uint64_t*>(a->routeRequests), reinterpret_cast<uint64_t*>(a->routeFeedback),
            reinterpret_cast<uint32_t*>(ctx->rowOwners), reinterpret_cast<uint32_t*>(ctx->queryEnds),
            a->routeOwners, shape.resident, shape.history);
        sum.reads += row.reads;
        sum.tokens += row.tokens;
        sum.rejected |= row.rejected;
    }
    return FromTotals(sum, shape.tokenBytes, a->routePolicy);
}
#endif
} // namespace adaptive_route
} // namespace pipeline_ab
