// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cold_score_partition.h"

#if defined(__CCE_AICORE__)
#define KVIP_DEDICATED_COLD_DEVICE __aicore__
#else
#define KVIP_DEDICATED_COLD_DEVICE
#endif

namespace kvip_dedicated_cold {
constexpr uint32_t kBatch = 16;
constexpr uint32_t kRounds = 4;
constexpr uint32_t kHistory = 131072;
constexpr uint32_t kResident = 16384;
constexpr uint32_t kProducerGroups = 24;
constexpr uint32_t kServiceLanes = kBatch;
constexpr uint32_t kMixGroups = kProducerGroups + kServiceLanes / 2;
constexpr uint32_t kQueries = kvip_cold_score::kQueries;
constexpr uint32_t kFragments = 3;
constexpr uint32_t kProgressWords = kvip_cold_score::kProgressWords;
constexpr uint32_t kTileTokens = kvip_cold_score::kTileTokens;
constexpr uint32_t kProgressWordCount = kBatch * kQueries * (kRounds - 1) * kFragments * kProgressWords;
using TileRange = kvip_cold_score::TileRange;

// Each request has a dedicated producer for the lower half and an upper
// prefix. The third producer in each 2-request group scores both upper tails.
KVIP_DEDICATED_COLD_DEVICE inline TileRange Fragment(
    uint32_t history, uint32_t resident, uint32_t half, uint32_t epoch, uint32_t fragment) {
    const uint32_t windowStart = resident + 2 * (epoch - 1) * half;
    const uint32_t start = windowStart < history ? windowStart : history;
    const uint32_t end = start + 2 * half < history ? start + 2 * half : history;
    const uint32_t halfEnd = start + half < end ? start + half : end;
    const uint32_t pages = (end - start) / 256;
    const uint32_t balancedEnd = start + ((2 * pages + 1) / 3) * 256;
    // Keep one upper page with the request-local producer when possible.
    // Its final fragment releases the query buffer kept over all windows.
    const uint32_t minOwnerEnd = halfEnd + 256 < end ? halfEnd + 256 : end;
    const uint32_t ownerEnd = balancedEnd > minOwnerEnd ? balancedEnd : minOwnerEnd;
    const uint32_t begins[kFragments] = {start, halfEnd, ownerEnd};
    const uint32_t ends[kFragments] = {halfEnd, ownerEnd, end};
    return {begins[fragment] / kTileTokens, ends[fragment] / kTileTokens};
}

KVIP_DEDICATED_COLD_DEVICE inline uint32_t ProgressOffset(
    uint32_t globalQuery, uint32_t epoch, uint32_t fragment) {
    return ((globalQuery * (kRounds - 1) + epoch - 1) * kFragments + fragment) * kProgressWords;
}

KVIP_DEDICATED_COLD_DEVICE inline uint32_t ProgressBytes() {
    return kProgressWordCount * sizeof(uint32_t);
}

// No later fragment or epoch may advance a consumer across a publication hole.
// 'ends' is a compact per-query snapshot in [epoch - 1][fragment] order.
KVIP_DEDICATED_COLD_DEVICE inline uint32_t Prefix(
    uint32_t history, uint32_t resident, uint32_t half, uint32_t epoch, uint32_t shard,
    const uint32_t* ends) {
    // Threshold builds resident TopK without publishing a Cold cursor.
    // Keep the native zero sentinel until this half has produced a tile.
    uint32_t ready = 0;
    const uint32_t first = shard == 0 ? 0 : 1;
    const uint32_t limit = shard == 0 ? 1 : kFragments;
    for (uint32_t e = 1; e <= epoch; ++e) {
        for (uint32_t fragment = first; fragment < limit; ++fragment) {
            const auto tiles = Fragment(history, resident, half, e, fragment);
            if (tiles.begin == tiles.end) continue;
            const uint32_t visible = ends[(e - 1) * kFragments + fragment];
            if (visible <= tiles.begin * kTileTokens) return ready;
            if (visible < tiles.end * kTileTokens) return visible;
            ready = tiles.end * kTileTokens;
        }
    }
    return ready;
}
}

#undef KVIP_DEDICATED_COLD_DEVICE
