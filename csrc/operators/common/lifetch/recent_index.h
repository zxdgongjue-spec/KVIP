// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

// Index-only seed geometry. This never changes KV tail bounds or residency.
#if defined(__CCE_AICORE__)
#define KVIP_RECENT_DEVICE __aicore__
#else
#define KVIP_RECENT_DEVICE
#endif
namespace kvip_recent_index {
struct Range { uint32_t begin, end; };

// UBmem uses this shared LI implementation but retains its original Context
// layout and resident-only threshold. A null URMA pointer also selects the
// original path, so existing callers need no additional tensor.
template <typename Context>
KVIP_RECENT_DEVICE inline uint64_t BoundsAddress(Context* context) {
#if defined(V9_UBMEM) && V9_UBMEM
    return 0;
#else
    return context->thresholdBounds;
#endif
}


// thresholdReady proves that the exact Kth loader retained its input. Both
// the context and native causal bounds must cover the complete resident row.
KVIP_RECENT_DEVICE inline bool UseLocalVictims(bool thresholdReady,
    uint32_t resident, uint32_t validEnd, uint32_t availableEnd) {
    return thresholdReady && resident >= 2048 && resident <= 16384 &&
           validEnd == resident && availableEnd >= resident;
}

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

} // namespace kvip_recent_index
#endif
