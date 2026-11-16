// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace lifetch {
struct CheckpointTopk {
    uint32_t initialHotRank;
    uint32_t windowMask;
    uint32_t reuseFinal;
    uint32_t reserved;
};

constexpr uint32_t kCheckpointEndOffset = 8; // uint16 words in each aligned threshold record
constexpr uint32_t kThresholdRecordWords = 128;
constexpr uint32_t kCheckpointProfileOffset = 8; // uint64 words after the publication block
constexpr uint32_t kMaxTopkCheckpoints = 6;
static_assert((kCheckpointProfileOffset + 4 * kMaxTopkCheckpoints) * 4 == kThresholdRecordWords);
}
