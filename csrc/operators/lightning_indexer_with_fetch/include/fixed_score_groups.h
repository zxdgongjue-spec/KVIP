#pragma once
#include "runtime_shape.h"

namespace fixed_score {
#if defined(__CCE_AICORE__)
#define KVIP_FIXED_DEVICE __aicore__
#else
#define KVIP_FIXED_DEVICE
#endif
// liCores is deliberately the original two groups per row. It defines window
// timing, progress halves and the queue schedule, even with extra producers.
KVIP_FIXED_DEVICE inline uint32_t GroupsPerRow(uint32_t batch) {
    return batch == 8 ? 4U : 8U;
}
KVIP_FIXED_DEVICE inline uint32_t LaunchGroups(const rotating::RuntimeShape& s) {
    const uint32_t b = s.batch;
    if ((b != 1 && b != 2 && b != 4 && b != 8) || !s.splitQ || s.aicCount != 32 ||
        s.indexHeads != 32 || s.resident != 16384 || (s.history != 131072 && s.history != 131328) ||
        s.rounds != (b == 1 ? 2U : 4U) || s.finalWorkers != 4 || s.liCores != 2 * b ||
        s.tokenBytes != 656) return 0;
    return b * GroupsPerRow(b);
}
struct Slice { uint32_t begin, end; };
KVIP_FIXED_DEVICE inline Slice Split(uint32_t begin, uint32_t end, uint32_t part, uint32_t count) {
    const uint32_t tiles = end - begin;
    return {begin + tiles * part / count, begin + tiles * (part + 1) / count};
}
#undef KVIP_FIXED_DEVICE
} // namespace fixed_score
