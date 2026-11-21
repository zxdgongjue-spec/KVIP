// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "batch_get_abi.h"
#include "pipeline_types.h"
namespace pipeline_ab {
// Four compact ID lists and readiness flags, Index handoff and four native Get statuses.
constexpr uint32_t kOwnerResolverWords = kMaxFinal + 4 * kFlagWords;
constexpr uint32_t kOwnerIndexWords = 4 * kMapSize + 32;
constexpr uint32_t kOwnerStatusWords = ((sizeof(KvafBatchGetCoreStatus) + 127) / 128) * 32;
constexpr uint32_t kOwnerChunkLimit = 32;
constexpr uint32_t kOldAddresses = kOwnerResolverWords + kOwnerIndexWords + 4 * kOwnerStatusWords;
constexpr uint32_t kChunkFlags = kOldAddresses + 2 * kMaxFinal;
constexpr uint32_t kEdgeDeps = kChunkFlags + kOwnerChunkLimit * kFlagWords;
constexpr uint32_t kQueryOwnerRowWords = kEdgeDeps + 2 * (kPrefetch + kMaxFinal);
constexpr uint32_t kQueryAliasWord = 2, kOwnerPacketCountWord = 3;
constexpr uint32_t kOwnerPlanEnd = 1, kOwnerPlanEdges = 2, kOwnerPlanFirst = 3;
}
