// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "hot_score_partition.h"

#if defined(__CCE_AICORE__)
#define KVIP_COLD_SCORE_DEVICE __aicore__
#else
#define KVIP_COLD_SCORE_DEVICE
#endif

namespace kvip_cold_score {
constexpr uint32_t kTileTokens = 128;
constexpr uint32_t kProgressWords = 16;
constexpr uint32_t kProgressBytes = kProgressWords * sizeof(uint32_t);
constexpr uint32_t kQueries = 4;
constexpr uint32_t kMaxParts = 32;
using TileRange = kvip_hot_score::TileRange;

KVIP_COLD_SCORE_DEVICE inline uint64_t ScoreBytes(uint32_t batch, uint32_t history) {
    return uint64_t(batch) * kQueries * history * sizeof(uint16_t);
}

KVIP_COLD_SCORE_DEVICE inline uint64_t ProgressOffset(uint32_t query, uint32_t part, uint32_t parts) {
    return (uint64_t(query) * parts + part) * kProgressWords;
}

// Subdivide each of the existing two logical half-windows independently.
KVIP_COLD_SCORE_DEVICE inline TileRange Split(uint32_t history, uint32_t resident, uint32_t half,
    uint32_t epoch, uint32_t part, uint32_t parts) {
    const uint32_t shardParts = parts / 2;
    const uint32_t begin = resident + (2 * (epoch - 1) + part / shardParts) * half;
    if (begin >= history) return {history / kTileTokens, history / kTileTokens};
    const uint32_t end = begin + half < history ? begin + half : history;
    const auto local = kvip_hot_score::Split(end - begin, part % shardParts, shardParts);
    return {begin / kTileTokens + local.begin, begin / kTileTokens + local.end};
}

// A producer can advance to a later window while a sibling is still behind.
// Fold the whole logical shard in traversal order and stop at the first gap.
KVIP_COLD_SCORE_DEVICE inline uint32_t Prefix(const uint32_t* cursors, uint32_t history,
    uint32_t resident, uint32_t half, uint32_t epoch, uint32_t shard, uint32_t parts) {
    const uint32_t shardParts = parts / 2;
    uint32_t ready = 0;
    for (uint32_t window = 1; window <= epoch; ++window) {
        for (uint32_t part = shard * shardParts; part < (shard + 1) * shardParts; ++part) {
            const auto tiles = Split(history, resident, half, window, part, parts);
            const uint32_t begin = tiles.begin * kTileTokens;
            const uint32_t end = tiles.end * kTileTokens;
            if (cursors[part] <= begin) return ready;
            if (cursors[part] < end) return cursors[part];
            ready = end;
        }
    }
    return ready;
}

// The experimental path uses dense page-aligned windows and even subdivisions.
// Every part must have a final tile, so its normal MIX credit drain remains valid.
inline uint32_t Parts(uint32_t batch, uint32_t history, uint32_t resident, uint32_t half,
    uint32_t rounds, uint32_t aics) {
    if (!batch || aics > kMaxParts || rounds < 2 || rounds > 4 ||
        !half || history <= resident || history % 256 || resident % 256 || half % 256)
        return 0;
    const uint32_t parts = (aics / batch) & ~1U;
    if (parts < 2) return 0;
    for (uint32_t epoch = 1; epoch < rounds; ++epoch) {
        for (uint32_t part = 0; part < parts; ++part) {
            const auto tiles = Split(history, resident, half, epoch, part, parts);
            if (tiles.begin >= tiles.end) return 0;
        }
    }
    return parts;
}
}

#undef KVIP_COLD_SCORE_DEVICE
