#pragma once
#if defined(__CCE_AICORE__)
#define LIFETCH_SCHEDULE_DEVICE __aicore__
#else
#define LIFETCH_SCHEDULE_DEVICE
#endif
#ifndef KVIP_COLD_FETCH_ONLY
#define KVIP_COLD_FETCH_ONLY 0
#endif
// UBmem retains its original resident publication; this URMA experiment
// schedules Fetch at cold boundaries only.
LIFETCH_SCHEDULE_DEVICE inline uint32_t ColdWindowCount(const RuntimeShape& s) {
    return KVIP_COLD_FETCH_ONLY ? s.rounds : (s.rounds ? s.rounds-1 : 0);
}
// The ranked G7 replay has 449 pairs of tiles. The reference gives the
// final odd pair to one shard and clips the other at the end of history.
LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceBlockPairs(uint32_t cold) {
#if KVIP_RANKED_PORT && KVIP_G7_SCAN_CLOCK != 1 && KVIP_G7_SCAN_CLOCK != 2
    return (cold / 256 + 1) / 2;
#else
    return cold / 512;
#endif
}
// Each shard owns half of the resident range and half of every cold window.
// Align halves to two tiles so neither shard loads its sibling's Key page.
LIFETCH_SCHEDULE_DEVICE inline bool UseSequenceSplit(const RuntimeShape& s) {
    if (s.rounds <= 1 || s.history <= s.resident) return false;
    uint32_t windows=ColdWindowCount(s);
    uint32_t cold=s.history-s.resident;
    uint32_t blocks=SequenceBlockPairs(cold);
    uint32_t halfBlocks=(blocks+windows-1)/windows;
    // Both shards must have a final tile to drain their persistent pipelines.
    return halfBlocks*(2*windows-1)<cold/256 && s.splitQ && s.indexHeads == 32 && s.rounds > 1 && s.resident % 512 == 0 &&
        s.history > s.resident &&
#if KVIP_RANKED_PORT && KVIP_G7_SCAN_CLOCK != 1 && KVIP_G7_SCAN_CLOCK != 2
        cold % 256 == 0 &&
#else
        cold % 512 == 0 &&
#endif
        (s.history - s.resident) >= 512 * windows;
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceHalfWindow(const RuntimeShape& s) {
    // Keep all shard starts tile-aligned; the last shard is clipped to history.
    const uint32_t blocks = SequenceBlockPairs(s.history - s.resident);
    const uint32_t windows=ColdWindowCount(s);
    return ((blocks + windows - 1) / windows) * 256;
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceWindowEnd(uint32_t history,uint32_t resident,uint32_t half,uint32_t epoch) {
    uint32_t end=resident+2*half*(epoch+KVIP_COLD_FETCH_ONLY);
    return end<history?end:history;
}

struct SequenceSlice { uint32_t shard, end; };

// Hot predicates belong to the shared prefix; cold halves alternate in S2.
LIFETCH_SCHEDULE_DEVICE inline SequenceSlice SequenceSliceAt(uint32_t token, uint32_t resident, uint32_t half) {
    if (token < resident) return {0, resident};
    uint32_t index = (token - resident) / half;
    return {index % 2, resident + (index + 1) * half};
}
#undef LIFETCH_SCHEDULE_DEVICE
