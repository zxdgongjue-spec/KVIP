// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_ROLLING_THRESHOLD
#define KVIP_ROLLING_THRESHOLD 0
#endif
#ifndef KVIP_WINDOW_REFILTER
#define KVIP_WINDOW_REFILTER 0
#endif
#if KVIP_WINDOW_REFILTER && !KVIP_ROLLING_THRESHOLD
#error "Window refilter requires the rolling threshold"
#endif
#if defined(V9_UBMEM) && V9_UBMEM && KVIP_ROLLING_THRESHOLD
#error "Rolling threshold experiment is URMA only"
#endif

namespace kvip_rolling {
constexpr uint32_t kBins = 256, kTop = 2048;
// A capped reverse CDF plus a broadcast copy of the immutable seed key.
constexpr uint32_t kStateWords = kBins + 128;
// Progress.end[0..7] is the normal tile publication. [8..15] is padding
// in the non-sequence schedule and occupies a separate 32-byte DMA stripe.
constexpr uint32_t kProgressThreshold = 8;
#if defined(__CCE_AICORE__)
#define KVIP_ROLLING_DEVICE __aicore__
#else
#define KVIP_ROLLING_DEVICE
#endif
KVIP_ROLLING_DEVICE inline uint32_t TileMask(uint32_t tile, uint32_t end,
    uint32_t seedBegin, uint32_t seedEnd) {
    uint32_t valid = end > tile ? end - tile : 0;
    uint32_t lo = seedBegin > tile ? seedBegin - tile : 0;
    uint32_t hi = seedEnd > tile ? seedEnd - tile : 0;
    valid = valid < 128 ? valid : 128;
    lo = lo < 128 ? lo : 128;
    hi = hi < 128 ? hi : 128;
    return valid | (lo << 8) | (hi << 16);
}
#undef KVIP_ROLLING_DEVICE
} // namespace kvip_rolling
