#pragma once
// P0-A1: host-testable, per-row mapping clear bound for a fixed-capacity ABI.
// A missing or inconsistent producer bound conservatively restores full clear.
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define V9_MAP_BOUND_INLINE __aicore__ inline
#else
#define V9_MAP_BOUND_INLINE inline
#endif

namespace rotating {
V9_MAP_BOUND_INLINE uint32_t ActiveKMappingClearTokens(
    uint32_t history, uint32_t resident,
    bool hasQueryEnds, bool validQueryEnds, uint32_t maxQueryEnd,
    bool hasActualKeyLength, uint32_t actualKeyLength,
    bool hasTail, uint32_t tailBegin, uint32_t tailEnd) {
    if (!history || resident > history || !hasQueryEnds ||
        !validQueryEnds || !hasActualKeyLength ||
        maxQueryEnd > history || actualKeyLength > history ||
        maxQueryEnd != actualKeyLength) {
        return history;
    }
    // ABC's committed/write_end interval is allowed to extend past the
    // current decode QueryEnd. Its mapping writes must stay initialized.
    if (hasTail && (tailBegin > tailEnd || tailEnd > history)) {
        return history;
    }
    uint32_t end = maxQueryEnd > resident ? maxQueryEnd : resident;
    if (hasTail && tailEnd > end) end = tailEnd;
    return end >= history ? history : ((end + 127U) & ~127U);
}
}  // namespace rotating

#undef V9_MAP_BOUND_INLINE
