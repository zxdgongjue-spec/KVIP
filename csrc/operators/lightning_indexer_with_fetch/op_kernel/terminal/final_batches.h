#pragma once
#include <stdint.h>

namespace terminal_batches {
#if defined(__CCE_AICORE__)
#define KVIP_BATCH_FN __aicore__ inline
#else
#define KVIP_BATCH_FN inline
#endif

struct Plan {
    uint32_t ends[4];
    uint32_t count;
};

// Only final admissions contribute to the threshold. Prefetched slots already
// belong to the input of the terminal stage and must not trigger pacing.
KVIP_BATCH_FN bool ShouldPace(uint32_t own, uint32_t totalSlots, uint32_t priorSlots) {
    return own > 1 && totalSlots >= priorSlots && totalSlots - priorSlots >= 256;
}

KVIP_BATCH_FN Plan MakePlan(uint32_t own, uint32_t totalSlots, uint32_t priorSlots) {
    Plan plan{};
    if (ShouldPace(own, totalSlots, priorSlots)) {
        for (uint32_t numerator = 1; numerator < 16; numerator += 5) {
            uint32_t end = (own * numerator + 15) / 16;
            if (end < own && (!plan.count || end > plan.ends[plan.count - 1])) {
                plan.ends[plan.count++] = end;
            }
        }
    }
    plan.ends[plan.count++] = own;
    return plan;
}

KVIP_BATCH_FN uint32_t CompletionControl(uint64_t producer, uint32_t depth) {
    return 0x30360000U | ((producer & depth) ? 0U : 0x80000000U);
}
#undef KVIP_BATCH_FN
}
