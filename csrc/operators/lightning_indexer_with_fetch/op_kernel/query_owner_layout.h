// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "batch_get_abi.h"
#include "pipeline_types.h"
namespace pipeline_ab {
// Resolver handoff, Index handoff and four cache-line-isolated native Get statuses.
constexpr uint32_t kOwnerResolverWords=3*8192+32+4096;
constexpr uint32_t kOwnerIndexWords=4*kMapSize+32;
constexpr uint32_t kOwnerStatusWords=((sizeof(KvafBatchGetCoreStatus)+127)/128)*32;
constexpr uint32_t kOwnerTopkReadyOffset=kOwnerResolverWords+kOwnerIndexWords+4*kOwnerStatusWords;
static_assert(kOwnerTopkReadyOffset%kFlagWords==0, "one writer per aligned classification flag");
constexpr uint32_t kQueryOwnerRowWords=kOwnerTopkReadyOffset+4*kFlagWords;
}
