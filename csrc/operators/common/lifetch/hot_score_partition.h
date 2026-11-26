// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_HOT_SCORE_DEVICE __aicore__
#else
#define KVIP_HOT_SCORE_DEVICE
#endif

namespace kvip_hot_score {
struct TileRange { uint32_t begin, end; };

inline uint32_t Parts(uint32_t batch, uint32_t resident, uint32_t aics) {
    if (!batch || resident % 512 || resident < 512 || aics / batch < 2) return 0;
    const uint32_t budget = aics / batch;
    const uint32_t pages = resident / 256;
    return budget < pages ? budget : pages;
}

// Each page is two 128-token tiles. No two producers write the same page.
KVIP_HOT_SCORE_DEVICE inline TileRange Split(uint32_t resident, uint32_t part, uint32_t parts) {
    const uint32_t pages = resident / 256;
    return {2 * (pages * part / parts), 2 * (pages * (part + 1) / parts)};
}
}

#undef KVIP_HOT_SCORE_DEVICE
