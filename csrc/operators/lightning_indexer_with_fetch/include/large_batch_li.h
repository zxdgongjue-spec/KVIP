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

// Physical epochs 2 and 3 share the old middle window. Form their boundary
// from the two original integer tile positions, including for ragged rows.
LARGE_LI_DEVICE constexpr Slice MiddleWindow(uint32_t resident, uint32_t scan, uint32_t epoch) {
    const uint32_t original = epoch <= 2 ? epoch : epoch - 1;
    const Slice span = Window(resident, scan, 4, original);
    const uint32_t mid = span.begin + (span.end - span.begin) / 2;
    return epoch == 2 ? Slice{span.begin, mid} : epoch == 3 ? Slice{mid, span.end} : span;
}

LARGE_LI_DEVICE constexpr uint32_t MiddleSuffix(uint32_t resident, uint32_t total, uint32_t quota,
                                                uint32_t rows, uint32_t epoch) {
    const uint32_t original = epoch <= 2 ? epoch : epoch - 1;
    const Slice span = Window(resident, resident + total, 4, original);
    const uint32_t suffix = LaterSuffix(span, resident, total, quota, rows, original);
    if (original != 2 || !suffix) return suffix;
    const uint32_t length = span.end - span.begin;
    const uint32_t early = uint32_t(uint64_t(length / 2) * suffix / length);
    return epoch == 2 ? early : suffix - early;
}

// Use the same rule for a Q's nominations and a row's actual admissions.
// Leftover budget at the new midpoint remains available until the old boundary.
LARGE_LI_DEVICE constexpr uint32_t MiddleAllowance(uint32_t full, uint32_t epoch, uint32_t earlyUsed) {
    return epoch == 2 ? full / 2 : epoch == 3 ? full - (earlyUsed < full ? earlyUsed : full) : full;
}

#undef LARGE_LI_DEVICE
} // namespace large_li
