#pragma once
// Internal three-kernel contract. No public binding/Context layout changes.
#include <stdint.h>

namespace fusion3 {
#if defined(__CCE_AICORE__)
#define FUSION3_INLINE __aicore__ inline
#else
#define FUSION3_INLINE inline
#endif
constexpr uint32_t kArenaBytes = 212 * 1024;
constexpr uint32_t kControlWord = 208 * 1024 / 4;
constexpr uint32_t kControlWords = 1024;
constexpr uint32_t kColdStatsWord = 640;
constexpr uint32_t kStateWord = 768;
constexpr uint32_t kLogicalWord = 2 * 8192;
constexpr uint32_t kThreads = 512;
constexpr uint32_t kLogicInvalid = 1U, kPlanError = 2U, kTransportError = 4U;

// Subtraction form avoids overflow and accepts an empty range at its end.
FUSION3_INLINE bool RangeValid(uint32_t begin, uint32_t count, uint32_t base,
                       uint32_t capacity, uint32_t safe) {
    return safe <= count && count <= capacity && begin >= base &&
           begin - base <= capacity - count;
}
// Logic failures must not prevent retirement of already posted healthy DMA.
FUSION3_INLINE bool CanSubmit(uint32_t logicError, uint64_t transportError,
                      uint32_t count, bool conflictPhase, uint32_t pending) {
    return !logicError && !transportError && count && (!conflictPhase || !pending);
}
FUSION3_INLINE bool ConsumerReady(uint32_t logicError, uint64_t transportError,
                          uint32_t pending, uint32_t total, uint32_t expected,
                          bool deferred) {
    return !logicError && !transportError && !pending && total == expected && !deferred;
}
#undef FUSION3_INLINE
} // namespace fusion3
