// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_ROLLING_THRESHOLD
#define KVIP_ROLLING_THRESHOLD 0
#endif
#ifndef KVIP_ROLLING_WINDOWED
#define KVIP_ROLLING_WINDOWED 0
#endif
#if KVIP_ROLLING_WINDOWED && !KVIP_ROLLING_THRESHOLD
#error "Window histogram requires rolling threshold"
#endif
#ifndef KVIP_ROLLING_COMPACT
#define KVIP_ROLLING_COMPACT 0
#endif
#ifndef KVIP_ROLLING_FIRST_WINDOW
#define KVIP_ROLLING_FIRST_WINDOW 0
#endif
#if (KVIP_ROLLING_COMPACT || KVIP_ROLLING_FIRST_WINDOW) && !KVIP_ROLLING_WINDOWED
#error "Compact admission and first-window correction require window accumulation"
#endif
#if KVIP_ROLLING_FIRST_WINDOW && !KVIP_ROLLING_COMPACT
#error "First-window correction requires compact admission"
#endif
#ifndef KVIP_ROLLING_BATCHED
#define KVIP_ROLLING_BATCHED 0
#endif
#if KVIP_ROLLING_BATCHED && !KVIP_ROLLING_THRESHOLD
#error "Batched histogram requires the rolling threshold"
#endif
#ifndef KVIP_WINDOW_REFILTER
#define KVIP_WINDOW_REFILTER 0
#endif
#if KVIP_WINDOW_REFILTER && !KVIP_ROLLING_THRESHOLD
#error "Window refilter requires the rolling threshold"
#endif
#ifndef KVIP_REFILTER_SAFE
#define KVIP_REFILTER_SAFE 0
#endif
// Independent of threshold estimation: only changes compact output location.
#ifndef KVIP_COMPACT_OUT_OF_PLACE
#define KVIP_COMPACT_OUT_OF_PLACE 0
#endif
#ifndef KVIP_REFILTER_REFILL
#define KVIP_REFILTER_REFILL 0
#endif
#if KVIP_REFILTER_SAFE && !KVIP_WINDOW_REFILTER
#error "Visibility-safe admission requires window refilter"
#endif
#if KVIP_REFILTER_REFILL && !KVIP_REFILTER_SAFE
#error "Same-window refill requires visibility-safe admission"
#endif
#if defined(V9_UBMEM) && V9_UBMEM && KVIP_ROLLING_THRESHOLD
#error "Rolling threshold experiment is URMA only"
#endif

#if KVIP_ROLLING_WINDOWED && KVIP_ROLLING_BATCHED
#error "Select either window or eight-tile accumulation"
#endif
namespace kvip_rolling {
constexpr uint32_t kWindowScores = 18432;
constexpr uint32_t kBins = 256, kTop = 2048;
// A capped reverse CDF plus a broadcast copy of the immutable seed key.
constexpr uint32_t kAdmission = kBins + 128;
constexpr uint32_t kCdfWords = kAdmission + (KVIP_ROLLING_COMPACT ? 128 : 0);
constexpr uint32_t kBatchTiles = 8;
// Eight packed 128-byte score-bin vectors, per Q. No GM scratch or extra DMA.
constexpr uint32_t kStateWords = kCdfWords + (KVIP_ROLLING_BATCHED ? kBatchTiles * 64 : 0);
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
// The low 24 bits retain the causal/seed mask; the high byte is the batch slot.
KVIP_ROLLING_DEVICE inline uint32_t BatchMask(uint32_t mask, uint32_t& pending, bool advance) {
    uint32_t packed = mask | (pending << 24);
    pending = (advance || pending + 1 == kBatchTiles) ? 0 : pending + 1;
    return packed;
}
#undef KVIP_ROLLING_DEVICE
} // namespace kvip_rolling
