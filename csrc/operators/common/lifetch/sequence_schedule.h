#pragma once
#if defined(__CCE_AICORE__)
#define LIFETCH_SCHEDULE_DEVICE __aicore__
#else
#define LIFETCH_SCHEDULE_DEVICE
#endif
// Each shard owns half of the resident range and half of every cold window.
// Align halves to two tiles so neither shard loads its sibling's Key page.
LIFETCH_SCHEDULE_DEVICE inline bool UseSequenceSplit(const RuntimeShape& s) {
    if (s.rounds <= 1 || s.history <= s.resident) return false;
    uint32_t blocks=(s.history-s.resident)/512;
    uint32_t halfBlocks=(blocks+s.rounds-2)/(s.rounds-1);
    // Both shards must have a final tile to drain their persistent pipelines.
    return halfBlocks*(2*s.rounds-3)<2*blocks && s.splitQ && s.indexHeads == 32 && s.rounds > 1 && s.resident % 512 == 0 &&
        s.history > s.resident && (s.history - s.resident) % 512 == 0 &&
        (s.history - s.resident) >= 512 * (s.rounds - 1);
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceHalfWindow(const RuntimeShape& s) {
    // Keep all shard starts tile-aligned; the last shard is clipped to history.
    const uint32_t blocks = (s.history - s.resident) / 512;
    return ((blocks + s.rounds - 2) / (s.rounds - 1)) * 256;
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceWindowEnd(uint32_t history,uint32_t resident,uint32_t half,uint32_t epoch) {
    uint32_t end=resident+2*half*epoch;
    return end<history?end:history;
}

// In reverse mode progress and bitmap cursors still increase. Only cold
// coordinates are mirrored; hot victim ordinals retain their original order.
LIFETCH_SCHEDULE_DEVICE inline uint32_t ReverseColdWindowEnd(uint32_t history, uint32_t resident,
    uint32_t half, uint32_t rounds, uint32_t epoch) {
    return history + resident - SequenceWindowEnd(history, resident, half, rounds - 1 - epoch);
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t ReverseColdFirstShard(uint32_t history, uint32_t resident,
    uint32_t half, uint32_t rounds) {
    // The final forward window may have a short high half. It becomes the
    // low scan-coordinate half of the first reverse window.
    return history - resident - half * (2 * rounds - 3);
}

struct SequenceSlice { uint32_t shard, end; };

// Hot predicates belong to the shared prefix; cold halves alternate in S2.
LIFETCH_SCHEDULE_DEVICE inline SequenceSlice SequenceSliceAt(uint32_t token, uint32_t resident, uint32_t half) {
    if (token < resident) return {0, resident};
    uint32_t index = (token - resident) / half;
    return {index % 2, resident + (index + 1) * half};
}

LIFETCH_SCHEDULE_DEVICE inline SequenceSlice ReverseColdSliceAt(uint32_t token, uint32_t resident,
    uint32_t half, uint32_t firstShard) {
    if (token < resident) return {0, resident};
    uint32_t cut = resident + firstShard;
    if (token < cut) return {0, cut};
    uint32_t index = (token - cut) / half;
    return {(index + 1) % 2, cut + (index + 1) * half};
}
#undef LIFETCH_SCHEDULE_DEVICE
