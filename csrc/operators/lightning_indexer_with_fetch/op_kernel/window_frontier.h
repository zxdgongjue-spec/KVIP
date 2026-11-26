// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

namespace kvip_window_frontier {
// Two experiment words in the zero-initialized tail of pipeline_ab::Args.
constexpr uint32_t kAddressOffset = 200;
constexpr uint32_t kModeOffset = 208;
constexpr uint32_t kRowBytes = 128;
constexpr uint32_t kQueries = 4;
enum Mode : uint64_t { Off = 0, Capture = 1, Replay = 2 };

#if defined(__CCE_AICORE__)
__aicore__ inline GM_ADDR Table(GM_ADDR workspace) {
    if (!workspace) return nullptr;
    return reinterpret_cast<GM_ADDR>(*reinterpret_cast<__gm__ const uint64_t*>(workspace + kAddressOffset));
}

__aicore__ inline Mode ActiveMode(GM_ADDR workspace) {
    if (!Table(workspace)) return Off;
    const auto mode = *reinterpret_cast<__gm__ const uint64_t*>(workspace + kModeOffset);
    return mode == Capture || mode == Replay ? static_cast<Mode>(mode) : Off;
}

__aicore__ inline GM_ADDR Row(GM_ADDR table, uint32_t batch, uint32_t epoch, uint32_t row) {
    return table + (epoch * batch + row) * kRowBytes;
}

__aicore__ inline void WaitClassified(__gm__ uint32_t* cursor, uint64_t observed, uint64_t packed, uint32_t cap) {
    uint32_t end0 = uint32_t(packed), end1 = uint32_t(packed >> 32);
    if (end0 > cap) end0 = cap;
    if (end1 > cap) end1 = cap;
    if (uint32_t(observed) >= end0 && uint32_t(observed >> 32) >= end1) return;
    auto live = reinterpret_cast<__gm__ volatile uint32_t*>(cursor);
    // The saved snapshot was reachable at this point under the same P2
    // per-lane scan and service order. Later-window progress is not needed.
    do {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(cursor), kRowBytes);
    } while (live[0] < end0 || live[16] < end1);
}
#endif
}
