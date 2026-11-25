// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../include/context.h"

#if defined(__CCE_AICORE__)
#define KVIP_P2_W1_DEVICE __aicore__
#else
#define KVIP_P2_W1_DEVICE
#endif

namespace kvip_p2_w1 {
constexpr uint32_t kFirstGroup = 28;
constexpr uint32_t kGroups = 32;
constexpr uint32_t kPrefixTiles = 2;

KVIP_P2_W1_DEVICE inline bool Supports(const rotating::RuntimeShape& shape) {
    const bool length = (shape.history == 7168 && shape.resident == 4096) ||
        (shape.history == 131072 && shape.resident == 16384);
    return length && shape.batch == 16 && shape.finalWorkers == 4 && shape.rounds == 4 &&
        shape.indexHeads == 32 && shape.aicCount == kGroups && shape.liCores == kGroups &&
        shape.serviceLanes == 16 && rotating::UseSequenceSplit(shape);
}

KVIP_P2_W1_DEVICE inline bool EarlyGroup(uint32_t group) {
    return group >= kFirstGroup && group < kGroups;
}
}

#undef KVIP_P2_W1_DEVICE
