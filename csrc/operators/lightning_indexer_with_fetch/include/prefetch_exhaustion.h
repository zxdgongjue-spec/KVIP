// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_PREFETCH_DEVICE __aicore__
#else
#define KVIP_PREFETCH_DEVICE
#endif

namespace kvip_prefetch {
// These three capacities cannot recover inside a round: prefetch only grows
// slots/requests, and outstanding requests drain after all service epochs.
// A per-window allowance is deliberately absent: carry may replenish it.
KVIP_PREFETCH_DEVICE inline bool Exhausted(uint32_t slots, uint32_t requests,
    uint32_t slotLimit, uint32_t requestLimit, uint32_t queueLimit) {
    return slots >= slotLimit || requests >= requestLimit || requests >= queueLimit;
}
}

#undef KVIP_PREFETCH_DEVICE
