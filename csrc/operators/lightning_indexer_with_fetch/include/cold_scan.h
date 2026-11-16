// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef KVIP_REVERSE_COLD
#define KVIP_REVERSE_COLD 0
#endif
#if defined(V9_UBMEM) && V9_UBMEM && KVIP_REVERSE_COLD
#error "Reverse cold scan is an URMA-only experiment"
#endif
#if defined(__CCE_AICORE__)
#define KVIP_SCAN_DEVICE __aicore__
#else
#define KVIP_SCAN_DEVICE
#endif

namespace kvip_cold_scan {
constexpr uint32_t kBlock = 2048;
struct Range { uint32_t begin, end; };

// Only complete blocks valid for all four queries are permuted. The resident
// prefix and partial/causal tail keep their original coordinates. Within a
// block, the existing 128-key tiles and 16-tile scale DMA remain contiguous.
KVIP_SCAN_DEVICE inline Range Bounds(uint32_t resident, uint32_t history,
                                     const uint32_t ends[4]) {
    uint32_t end = history;
    for (uint32_t q = 0; q < 4; ++q) if (ends[q] < end) end = ends[q];
    uint32_t begin = (resident + kBlock - 1) / kBlock * kBlock;
    end = end / kBlock * kBlock;
    return end > begin ? Range{begin, end} : Range{0, 0};
}

KVIP_SCAN_DEVICE inline uint32_t Remap(uint32_t token, Range range) {
    if (token < range.begin || token >= range.end) return token;
    return range.begin + range.end - kBlock - (token & ~(kBlock - 1)) + (token & (kBlock - 1));
}
} // namespace kvip_cold_scan
#undef KVIP_SCAN_DEVICE
