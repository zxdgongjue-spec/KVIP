// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "batch_get_abi.h"
#include "pipeline_types.h"
namespace pipeline_ab {
// Four compact ID lists and readiness flags, Index handoff and four cache-line-isolated native Get statuses.
constexpr uint32_t kOwnerResolverWords=kMaxFinal+4*kFlagWords;
constexpr uint32_t kOwnerIndexWords=4*kMapSize+32;
constexpr uint32_t kOwnerStatusWords=((sizeof(KvafBatchGetCoreStatus)+127)/128)*32;
constexpr uint32_t kQueryOwnerRowWords=kOwnerResolverWords+kOwnerIndexWords+4*kOwnerStatusWords;
}
