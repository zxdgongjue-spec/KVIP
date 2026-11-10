#pragma once
#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVIP_INIT_DEVICE __aicore__
#define KVIP_INIT_SIMT __simt_callee__
#else
#define KVIP_INIT_DEVICE
#define KVIP_INIT_SIMT
#endif

namespace rotating {
// Each writer owns complete 128-byte cache lines. This is stricter than the
// 32-byte DataCopy requirement and also isolates the scalar ABC stores.
constexpr uint32_t kInitLineWords = 32;
struct InitWordRange { uint32_t begin, end; };

template<uint32_t Words>
KVIP_INIT_DEVICE constexpr InitWordRange PartitionInitWords(uint32_t part, uint32_t parts) {
    static_assert(Words % kInitLineWords == 0, "Init rows must contain complete cache lines");
    const uint32_t lines = Words / kInitLineWords;
    return {uint32_t(uint64_t(lines) * part / parts) * kInitLineWords,
            uint32_t(uint64_t(lines) * (part + 1) / parts) * kInitLineWords};
}

KVIP_INIT_SIMT inline InitWordRange InitTailIntersection(
    InitWordRange owned, uint32_t begin, uint32_t end, uint32_t history) {
    if (end > history) end = history;
    if (begin < owned.begin) begin = owned.begin;
    if (end > owned.end) end = owned.end;
    return {begin, end > begin ? end : begin};
}
}

#undef KVIP_INIT_DEVICE
#undef KVIP_INIT_SIMT
