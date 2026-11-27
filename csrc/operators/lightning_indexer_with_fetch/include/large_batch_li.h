#pragma once
#include <stdint.h>

// Unit is one physical 128-token tile, including the last partially valid tile.
namespace large_li {
#if defined(__CCE_AICORE__)
#define LARGE_LI_DEVICE __aicore__
#else
#define LARGE_LI_DEVICE
#endif
struct Slice {
    uint32_t begin, end;
};

LARGE_LI_DEVICE constexpr uint32_t CeilDiv(uint32_t x, uint32_t n) {
    return x / n + uint32_t(x % n != 0);
}

#if !defined(__CCE_AICORE__) || KVIP_LARGE_FOUR_WINDOW_CREDIT
// Each of the four windows earns 256 GETs for its request. Only unused
// credit carries forward. Callers separately enforce the total row budget.
LARGE_LI_DEVICE constexpr uint32_t WindowGrant(uint32_t rounds, uint32_t epoch, uint32_t step,
                                              uint32_t rowLimit, uint32_t used) {
    if (rounds != 4 || step != 256 || rowLimit != 1024) return step;
    const uint32_t earned = (epoch + 1) * step;
    return used < earned ? earned - used : 0;
}
#endif

// LI2 writes 1 to the query's own readiness line. After the final window a
// producer may publish the exact number of compact IDs (including zero).
constexpr uint32_t kStagedBase = 2;
constexpr uint32_t kTopK = 2048;
LARGE_LI_DEVICE constexpr bool HasStagedMisses(uint32_t marker) {
    return marker >= kStagedBase && marker <= kStagedBase + kTopK;
}

LARGE_LI_DEVICE constexpr bool CanStagePublishedMisses(uint32_t valid, uint32_t scan, uint64_t source,
                                                       uint32_t cursor, uint16_t cutoff) {
    return valid && valid <= scan && scan <= 131072 && source && cutoff && cursor >= scan;
}

// The terminal follows exactly the contiguous extent published by this row's
// producers. Each query's causal bound still masks the last partial tile.
template <class Shape, class Ends>
LARGE_LI_DEVICE inline uint32_t ScanHistory(const Shape& shape, Ends ends, uint32_t row) {
    if (!ends) {
        return shape.history;
    }
    uint32_t end = shape.resident;
    for (uint32_t q = 0; q < 4; ++q) {
        if (ends[row * 4 + q] > end) {
            end = ends[row * 4 + q];
        }
    }
    return end >= shape.history ? shape.history : CeilDiv(end, 128) * 128;
}

// Equal work keeps ascending row order. Round-robin over this rank spreads
// the longest rows across different helpers before any helper takes a second.
LARGE_LI_DEVICE inline void RankRows(const uint32_t* costs, uint32_t batch, uint32_t* rows) {
    for (uint32_t i = 0; i < batch; ++i) {
        uint32_t j = i;
        while (j && costs[rows[j - 1]] < costs[i]) {
            rows[j] = rows[j - 1];
            --j;
        }
        rows[j] = i;
    }
}

// Solve a small load-balancing problem once per LI phase. An owner's ABC
// work stays local, while cold/hot tile prefixes are freely divisible. A
// helper serves at most seven rows at these four supported batch sizes.
LARGE_LI_DEVICE constexpr uint32_t OwnerBudget(const uint32_t* total, const uint32_t* fixed, uint32_t count) {
    uint32_t low = 0, high = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (fixed[i] > low) {
            low = fixed[i];
        }
        if (total[i] > high) {
            high = total[i];
        }
    }
    // Smallest T for which every owner and the shared helper can finish
    // within T tiles. Necessary suffix work is sum(max(0, cost_i - T)).
    while (low < high) {
        uint32_t mid = (low + high) / 2, moved = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (total[i] > mid) {
                moved += total[i] - mid;
            }
        }
        if (moved > mid) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

LARGE_LI_DEVICE constexpr uint32_t SuffixQuota(uint32_t tiles, uint32_t fixed, uint32_t budget) {
    return tiles + fixed > budget ? tiles + fixed - budget : 0;
}

// Apportion a full-row suffix quota over its windows. The difference of
// cumulative integer quotients gives exact total tile ownership, including
// windows too small to hand off even one tile.
LARGE_LI_DEVICE constexpr uint32_t SuffixInWindow(Slice span, uint32_t resident, uint32_t total,
                                                  uint32_t quota) {
    if (!total) {
        return 0;
    }
    return uint32_t(uint64_t(span.end - resident) * quota / total -
                    uint64_t(span.begin - resident) * quota / total);
}

// A window of L >= 4 tiles retains owner work: its proportional suffix
// is <= ceil(L/2) and its later-window adjustment is <= floor(L/16).
// This once-per-row check lets the cube prime the next window without
// recalculating either boundary or suffix. Short rows keep on-demand credit.
LARGE_LI_DEVICE constexpr bool CanOverlapBanks(uint32_t tiles, uint32_t quota, uint32_t rounds) {
    return rounds >= 2 && tiles >= 4 * (rounds - 1) && quota <= tiles / 2;
}

// Give a shared helper a small extra suffix after the initial cold window.
// All terms are in this window's tiles. Groups of four or more keep their original
// owner/helper split. A zero suffix still costs no helper visit.
LARGE_LI_DEVICE constexpr uint32_t LaterSuffix(Slice span, uint32_t resident, uint32_t total,
                                                uint32_t quota, uint32_t rows, uint32_t epoch) {
    const uint32_t base = SuffixInWindow(span, resident, total, quota);
    if (epoch < 2 || !rows || rows > 3 || !base) return base;
    const uint32_t length = span.end - span.begin;
    const uint32_t extraLimit = length / 16;
    const uint32_t extraShare = base / (rows + 1);
    const uint32_t extra = extraLimit < extraShare ? extraLimit : extraShare;
    return base + (extra < length - base ? extra : length - base);
}

LARGE_LI_DEVICE constexpr Slice Window(uint32_t residentTile, uint32_t scanTile, uint32_t rounds,
                                       uint32_t epoch) {
    const uint32_t n = scanTile - residentTile;
    return {residentTile + n * (epoch - 1) / (rounds - 1), residentTile + n * epoch / (rounds - 1)};
}

#undef LARGE_LI_DEVICE
} // namespace large_li
