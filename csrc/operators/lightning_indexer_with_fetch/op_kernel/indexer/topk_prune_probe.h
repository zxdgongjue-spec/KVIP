// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_TOPK_PRUNE_DIAGNOSTICS
#define KVIP_TOPK_PRUNE_DIAGNOSTICS 0
#endif
#if KVIP_TOPK_PRUNE_DIAGNOSTICS != 0 && KVIP_TOPK_PRUNE_DIAGNOSTICS != 1
#error "KVIP_TOPK_PRUNE_DIAGNOSTICS must be 0 or 1"
#endif

#if KVIP_TOPK_PRUNE_DIAGNOSTICS && KVIP_STREAMING_LI && !(defined(V9_UBMEM) && V9_UBMEM)
namespace kvip_topk_probe {
// Independent diagnostic replay only. A query owns one 128-byte times row;
// words 12..15 are disjoint from the existing classification timings 0..11.
constexpr uint32_t kMagic = 0x54504b31; // TPK1, protocol version 1.
enum Status : uint32_t {
    Disabled = 1, GuardFallback = 2, ZeroThreshold = 3,
    InvalidCompaction = 4, InsufficientCandidates = 5, Success = 6,
    EmptyQuery = 7, Running = 8
};
enum Reason : uint32_t {
    ScoreType = 1U << 0, Prefix = 1U << 1, TopkOnly = 1U << 2,
    LoadDistribution = 1U << 3, QueryTile = 1U << 4, UnsplitQuery = 1U << 5,
    TaskCount = 1U << 6, TopkSize = 1U << 7, TrunkSize = 1U << 8,
    ShortResident = 1U << 9, ShortTopk = 1U << 10, ScratchSize = 1U << 11
};
__aicore__ inline uint32_t GuardReasons(bool score16, bool prefix, bool topkOnly,
    bool ld, uint32_t queryTile, bool splitQ, uint32_t batch, uint32_t aics,
    uint32_t topk, uint32_t trunk, uint32_t valid, uint32_t resident, uint32_t scratchBytes) {
    return (!score16 ? ScoreType : 0U) | (prefix ? Prefix : 0U) |
        (topkOnly ? TopkOnly : 0U) | (ld ? LoadDistribution : 0U) |
        (queryTile != 2 ? QueryTile : 0U) | (!splitQ ? UnsplitQuery : 0U) |
        (batch * 2 != aics ? TaskCount : 0U) | (topk != 2048 ? TopkSize : 0U) |
        (trunk != 16384 ? TrunkSize : 0U) | (valid < resident ? ShortResident : 0U) |
        (valid < topk ? ShortTopk : 0U) | (scratchBytes < trunk * 2 ? ScratchSize : 0U);
}
struct Record {
    uint32_t reason = 0, status = Disabled;
    uint32_t visited = 0, survivors = 0, radixInputs = 0, radixCalls = 0;
    uint32_t validEnd = 0, threshold = 0, blocks = 0;
    bool activated = false;

    __aicore__ inline void Store(__gm__ uint64_t* words) const {
        words[0] = uint64_t(kMagic) | (uint64_t(reason) << 32) |
            (uint64_t(status) << 48) | (uint64_t(activated) << 56);
        words[1] = uint64_t(visited) | (uint64_t(survivors) << 32);
        words[2] = uint64_t(radixInputs) | (uint64_t(radixCalls) << 32);
        words[3] = uint64_t(validEnd) | (uint64_t(threshold) << 32) |
            (uint64_t(blocks) << 48);
    }
};
} // namespace kvip_topk_probe
#endif
