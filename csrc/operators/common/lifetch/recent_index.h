// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifndef KVIP_RECENT_SKIP
#if defined(V9_UBMEM) && V9_UBMEM
#define KVIP_RECENT_SKIP 1
#else
#define KVIP_RECENT_SKIP 0
#endif
#endif
#if KVIP_RECENT_SKIP != 0 && KVIP_RECENT_SKIP != 1
#error "KVIP_RECENT_SKIP must be 0 or 1"
#endif

// Index-only seed geometry. This never changes KV tail bounds or residency.
#if defined(__CCE_AICORE__)
#define KVIP_RECENT_DEVICE __aicore__
#else
#define KVIP_RECENT_DEVICE
#endif
namespace kvip_recent_index {
struct Range { uint32_t begin, end; };

KVIP_RECENT_DEVICE inline Range Clip(int32_t begin, int32_t end,
                                    uint32_t resident, uint32_t history,
                                    uint32_t queryEnd) {
    if (begin < 0 || end < begin || uint32_t(end) > history) return {0, 0};
    uint32_t first = uint32_t(begin) > resident ? uint32_t(begin) : resident;
    uint32_t last = uint32_t(end) < queryEnd ? uint32_t(end) : queryEnd;
    return first < last ? Range{first, last} : Range{0, 0};
}

KVIP_RECENT_DEVICE inline Range Tiles(Range range, uint32_t shard = 0,
                                     uint32_t shards = 1) {
    if (range.begin == range.end) return {0, 0};
    const uint32_t begin = range.begin / 128, end = (range.end + 127) / 128;
    const uint32_t count = end - begin;
    return {begin + count * shard / shards, begin + count * (shard + 1) / shards};
}

// Both early classification and the cold scheduler use this exact coverage,
// including the 128-token halo. Partial-history/causal layouts keep rescoring.
KVIP_RECENT_DEVICE inline Range ReusableTiles(int32_t begin, int32_t end,
                                             uint32_t resident, uint32_t history,
                                             uint32_t availableEnd, uint32_t queryMax) {
#if KVIP_RECENT_SKIP
    if (availableEnd == history && history % 128 == 0 && resident % 128 == 0)
        return Tiles(Clip(begin, end, resident, history, queryMax));
#endif
    return {0, 0};
}

struct Remaining { Range first, second; };
KVIP_RECENT_DEVICE inline Remaining Subtract(Range whole, Range skip) {
    if (whole.begin >= whole.end) return {{0, 0}, {0, 0}};
    if (skip.begin >= skip.end || skip.end <= whole.begin || skip.begin >= whole.end)
        return {whole, {0, 0}};
    uint32_t left = skip.begin > whole.begin ? skip.begin : whole.begin;
    uint32_t right = skip.end < whole.end ? skip.end : whole.end;
    return {{whole.begin, left}, {right, whole.end}};
}
KVIP_RECENT_DEVICE inline uint32_t Last(Remaining ranges) {
    if (ranges.second.begin < ranges.second.end) return ranges.second.end - 1;
    if (ranges.first.begin < ranges.first.end) return ranges.first.end - 1;
    return UINT32_MAX;
}
} // namespace kvip_recent_index
#undef KVIP_RECENT_DEVICE

#if defined(__CCE_AICORE__)
namespace kvip_recent_index {
// The single-pass TopK uses tmpIndex/histograms, leaving its historical-index
// bank available for the old 2048 IDs. Remap before this bank is reused.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Remap(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* previous,
                  uint32_t begin, uint32_t count) {
    for (uint32_t i = threadIdx.x; i < 2048; i += 512) {
        uint32_t index = indices[i];
        indices[i] = index < 2048 ? previous[index] :
            (index - 2048 < count ? begin + index - 2048 : UINT32_MAX);
    }
}

// Resident and recent scores were packed into one existing TopK input bank.
// Restore physical Index positions; the gap between the two GM ranges is not
// part of the candidate set, and neither Index storage nor scoreGm is moved.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void RemapUnified(__ubuf__ uint32_t* indices, uint32_t residentCount,
                         uint32_t recentBegin, uint32_t recentCount) {
    for (uint32_t i = threadIdx.x; i < 2048; i += 512) {
        uint32_t index = indices[i];
        indices[i] = index < residentCount ? index :
            (index - residentCount < recentCount ?
                recentBegin + index - residentCount : UINT32_MAX);
    }
}
} // namespace kvip_recent_index
#endif
