// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef V9_UBMEM_TAGGED_MAP
#define V9_UBMEM_TAGGED_MAP 0
#endif
namespace rotating {
constexpr uint32_t kMappingSlotBits=18;
constexpr uint32_t kMappingSlotMask=(1U<<kMappingSlotBits)-1;
constexpr uint32_t kMappingEpochMax=(1U<<(32-kMappingSlotBits))-1;
constexpr uint32_t kMappingEpochMagic=0x4D415031U; // MAP1
constexpr uint32_t kMappingMaxOutputStride=(1U<<kMappingSlotBits)-8192;
struct MappingGeneration {uint64_t state;uint32_t tag;bool clear;};
#if defined(__CCE_AICORE__)
#define KVIP_MAPPING_HOST __aicore__
#else
#define KVIP_MAPPING_HOST
#endif
// The first use, a zero/reset/invalid state, and wrap clear the map inside the
// init kernel. No old epoch can become valid again without that full clear.
KVIP_MAPPING_HOST constexpr MappingGeneration NextMappingGeneration(uint64_t previous) {
    uint32_t epoch=uint32_t(previous);
    bool clear=uint32_t(previous>>32)!=kMappingEpochMagic || !epoch || epoch>=kMappingEpochMax;
    epoch=clear?1:epoch+1;
    return {(uint64_t(kMappingEpochMagic)<<32)|epoch,epoch<<kMappingSlotBits,clear};
}
#undef KVIP_MAPPING_HOST
}
