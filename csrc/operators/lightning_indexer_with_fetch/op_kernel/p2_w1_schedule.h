// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../include/context.h"

#if defined(__CCE_AICORE__)
#define KVIP_P2_W1_DEVICE __aicore__
#else
#define KVIP_P2_W1_DEVICE
#endif

namespace kvip_p2_w1 {
constexpr uint32_t kAvailableAics = 32;
constexpr uint32_t kRequestedWindows = 4;
constexpr uint32_t kMaxEarlyGroups = 4;
constexpr uint32_t kPrefixTiles = 2;

KVIP_P2_W1_DEVICE inline bool Supports(const rotating::RuntimeShape& shape) {
    const bool length = (shape.history == 7168 && shape.resident == 4096) ||
        (shape.history == 131072 && shape.resident == 16384);
    const auto batch = shape.batch;
    if (!length || batch == 0 || batch > 16 || (batch & (batch - 1))) return false;
    const uint32_t groups = 2 * batch;
    const uint32_t rounds = groups < kRequestedWindows ? groups : kRequestedWindows;
    const uint32_t serviceAics = groups / rounds;
    if (shape.finalWorkers != 4 || shape.indexHeads != 32 || shape.aicCount != kAvailableAics ||
        shape.liCores != groups || shape.rounds != rounds || shape.serviceAics != serviceAics ||
        shape.serviceLanes != 2 * serviceAics || !rotating::UseSequenceSplit(shape)) return false;

    // Both W1 halves must have a native suffix after consuming their prescore prefix.
    constexpr uint32_t prefixTokens = kPrefixTiles * 128;
    const uint32_t half = rotating::SequenceHalfWindow(shape);
    return half > prefixTokens && shape.history - shape.resident > half + prefixTokens;
}

KVIP_P2_W1_DEVICE inline bool EarlyGroup(uint32_t group, uint32_t groups) {
    const uint32_t half = groups / 2;
    const uint32_t early = half < kMaxEarlyGroups ? half : kMaxEarlyGroups;
    return group >= groups - early && group < groups;
}
}

#undef KVIP_P2_W1_DEVICE
