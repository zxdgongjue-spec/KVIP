// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_ADMISSION_DEVICE __aicore__
#else
#define KVIP_ADMISSION_DEVICE
#endif

namespace kvip_admission {
constexpr uint32_t kTotalLimit = 2048;
constexpr uint32_t kWindowLimit = 480;
constexpr uint32_t kKeyLimit = 65535;

// One immutable row per request during a graph execution. These controls
// govern speculative admission only, never exact TopK or final repair.
struct Controls {
    int32_t total, window, keySteps, carry;
};
static_assert(sizeof(Controls) == 16, "URMA admission row ABI");

KVIP_ADMISSION_DEVICE inline bool Valid(const Controls& c) {
    return c.total >= 0 && c.total <= int32_t(kTotalLimit) &&
        c.window >= 1 && c.window <= int32_t(kWindowLimit) &&
        c.keySteps >= 0 && c.keySteps <= int32_t(kKeyLimit) &&
#if KVIP_PREFETCH_RANK
        c.keySteps == 0 &&
#endif
        (c.carry == 0 || c.carry == 1);
}

// LI stores monotonically ordered uint16 keys, not raw BF16 bit patterns.
// Raising this admission gate must not overwrite the exact Kth lower bound.
KVIP_ADMISSION_DEVICE inline uint16_t LiftKey(uint16_t lower, uint32_t steps) {
    uint32_t room = kKeyLimit - lower;
    return uint16_t(uint32_t(lower) + (steps < room ? steps : room));
}

KVIP_ADMISSION_DEVICE inline uint32_t WindowAllowance(
    uint32_t window, uint32_t epoch, uint32_t posted, bool carry) {
    if (window > kWindowLimit) window = kWindowLimit;
    if (!carry) return window;
    // Earlier unused quota may be spent now, but a single window still has
    // the 480-descriptor hard cap. Total/slot/queue caps are applied afterward.
    uint64_t cumulative = (uint64_t(epoch) + 1) * window;
    uint64_t remaining = cumulative > posted ? cumulative - posted : 0;
    return remaining < kWindowLimit ? uint32_t(remaining) : kWindowLimit;
}
}

#undef KVIP_ADMISSION_DEVICE
