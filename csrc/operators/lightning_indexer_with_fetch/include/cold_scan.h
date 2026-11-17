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

struct Runs { Range spans[5]; uint32_t count = 0; };

// A scan window crosses at most two partial reversed blocks. Its complete
// middle blocks form one physical interval; histogram order is irrelevant.
// Keep all returned spans ascending so the existing contiguous score DMA and
// causal/recent masks can be reused without a per-tile address list.
KVIP_SCAN_DEVICE inline Runs WindowRuns(uint32_t begin, uint32_t end, Range reverse) {
    Runs runs{};
    if (begin >= end) return runs;
    if (begin < reverse.begin) {
        uint32_t stop = end < reverse.begin ? end : reverse.begin;
        runs.spans[runs.count++] = {begin, stop};
        begin = stop;
    }
    if (begin < end && begin < reverse.end) {
        uint32_t stop = end < reverse.end ? end : reverse.end;
        if (begin % kBlock) {
            uint32_t next = (begin / kBlock + 1) * kBlock;
            if (next > stop) next = stop;
            uint32_t physical = Remap(begin, reverse);
            runs.spans[runs.count++] = {physical, physical + next - begin};
            begin = next;
        }
        uint32_t fullEnd = stop / kBlock * kBlock;
        if (begin < fullEnd) {
            runs.spans[runs.count++] = {reverse.begin + reverse.end - fullEnd,
                                        reverse.begin + reverse.end - begin};
            begin = fullEnd;
        }
        if (begin < stop) {
            uint32_t physical = Remap(begin, reverse);
            runs.spans[runs.count++] = {physical, physical + stop - begin};
            begin = stop;
        }
    }
    if (begin < end) runs.spans[runs.count++] = {begin, end};
    return runs;
}
} // namespace kvip_cold_scan
#undef KVIP_SCAN_DEVICE
