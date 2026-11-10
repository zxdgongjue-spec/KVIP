// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "context.h"

// Device-only views of the existing ABI; no storage or transport ownership.
namespace rotating {
constexpr uint32_t kLruSlots=2048;
constexpr uint32_t kChangeCapacity=kLruSlots+kSlots;
__aicore__ inline __gm__ UbmemRegion* UbmemRegionAt(__gm__ Context* ctx) {
    return reinterpret_cast<__gm__ UbmemRegion*>(ctx->ubmemRegion);
}
__aicore__ inline __gm__ uint64_t* UbmemStatusAt(__gm__ Context* ctx, uint32_t batch) {
    return reinterpret_cast<__gm__ uint64_t*>(ctx->ubmemStatus) + batch*kUbmemStatusWords;
}
__aicore__ inline bool UbmemRegionValid(__gm__ UbmemRegion* region) {
    return region && region->magic == kUbmemMagic && region->version == kUbmemVersion &&
        region->locality == kUbmemHostLocal && region->generation &&
        UbmemRange(region->peerBase, region->bytes) &&
        UbmemRange(region->mappedBase, region->bytes) &&
        UbmemRange(region->outputBase, region->outputBytes) &&
        !(region->outputBase & 31) &&
        // The registration contains stable-owner rows with hot + optional ABC.
        // InitializeRound validates each selected owner's full stride. Neither
        // packed batch size nor the virtual Q4 tag space adds payload storage.
        region->outputBytes >= uint64_t(SIM_RESIDENT)*kTokenBytes;
}
__aicore__ inline __gm__ FinalPost* FinalPostAt(__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+kFinalPostOffset)+core;
}
}
