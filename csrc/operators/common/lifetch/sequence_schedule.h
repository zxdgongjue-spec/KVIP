#pragma once
#if defined(__CCE_AICORE__)
#define LIFETCH_SCHEDULE_DEVICE __aicore__
#else
#define LIFETCH_SCHEDULE_DEVICE
#endif
// Each shard owns half of the resident range and half of every cold window.
// Align halves to two tiles so neither shard loads its sibling's Key page.
LIFETCH_SCHEDULE_DEVICE inline bool UseSequenceSplit(const RuntimeShape& s) {
    return s.splitQ && s.indexHeads == 32 && s.rounds > 1 && s.resident % 512 == 0 &&
        (s.history - s.resident) % (512 * (s.rounds - 1)) == 0;
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceHalfWindow(const RuntimeShape& s) {
    return (s.history - s.resident) / (2 * (s.rounds - 1));
}

struct SequenceSlice { uint32_t shard, end; };

// Hot predicates belong to the shared prefix; cold halves alternate in S2.
LIFETCH_SCHEDULE_DEVICE inline SequenceSlice SequenceSliceAt(uint32_t token, uint32_t resident, uint32_t half) {
    if (token < resident) return {0, resident};
    uint32_t index = (token - resident) / half;
    return {index % 2, resident + (index + 1) * half};
}
#undef LIFETCH_SCHEDULE_DEVICE
