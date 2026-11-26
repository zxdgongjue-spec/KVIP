// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel_operator.h"

namespace kvip_resident_topk {

constexpr uint32_t kTableOffset = 216;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kShards = 2;
constexpr uint32_t kBins = 256;
constexpr uint32_t kResident = 16384;
constexpr uint32_t kHistory = 131072;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kShardTokens = kResident / kShards;

__aicore__ inline GM_ADDR HighTable(GM_ADDR workspace)
{
    if (!workspace) return nullptr;
    return reinterpret_cast<GM_ADDR>(
        *reinterpret_cast<__gm__ const uint64_t*>(workspace + kTableOffset));
}

}  // namespace kvip_resident_topk
