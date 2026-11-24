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
    uint32_t halfPages=(s.history-s.resident)/256;
    uint32_t blocks=(halfPages+1)/2;
    uint32_t halfBlocks=(blocks+s.rounds-2)/(s.rounds-1);
    // Both shards must have a final tile to drain their persistent pipelines.
    return halfBlocks*(2*s.rounds-3)<halfPages && s.splitQ && s.indexHeads == 32 && s.rounds > 1 && s.resident % 512 == 0 &&
        s.history > s.resident && (s.history - s.resident) % 256 == 0 &&
        (s.history - s.resident) >= 512 * (s.rounds - 1);
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceHalfWindow(const RuntimeShape& s, uint32_t scanEnd) {
    // Keep all shard starts tile-aligned; the last shard is clipped to history.
    const uint32_t blocks = (scanEnd - s.resident + 511) / 512;
    return ((blocks + s.rounds - 2) / (s.rounds - 1)) * 256;
}
LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceHalfWindow(const RuntimeShape& s) {
    return SequenceHalfWindow(s, s.history);
}

// Only controls which physical score tiles execute. GM row, score, bitmap and
// mapping strides always use s.history. The incremental selector already masks
// every physical position >= its Q's exclusive end before allowing a winner.
// A service lane takes a snapshot of several request rows at once, so all its
// producers and consumers must agree on one half-window for that set of rows.
constexpr uint32_t kShortQ4History = 128U * 1024 + 256;
template<class Ends>
LIFETCH_SCHEDULE_DEVICE inline uint32_t ActiveScanHistory(const RuntimeShape& s, Ends ends, uint32_t row) {
    if (s.history <= kShortQ4History || !ends || row >= s.batch || !s.serviceLanes || s.rounds < 2)
        return s.history;
    uint32_t upper = s.resident + 512 * (s.rounds - 1);
    for (uint32_t b = row % s.serviceLanes; b < s.batch; b += s.serviceLanes) {
        for (uint32_t q = 0; q < 4; ++q) {
            const uint32_t end = ends[4 * b + q];
            if (end > upper) upper = end;
        }
    }
    if (upper >= s.history) return s.history;
    upper = (upper + 255) / 256 * 256;
    const uint32_t half = SequenceHalfWindow(s, upper);
    // The last shard of the last epoch must issue at least one cold tile to
    // drain the persistent AIC/AIV pipelines. This can require a little valid
    // allocated padding when the rounded half-window just increased.
    const uint32_t minimum = s.resident + (2 * s.rounds - 3) * half + 256;
    if (upper < minimum) upper = minimum;
    return upper < s.history ? upper : s.history;
}

LIFETCH_SCHEDULE_DEVICE inline uint32_t SequenceWindowEnd(uint32_t history,uint32_t resident,uint32_t half,uint32_t epoch) {
    uint32_t end=resident+2*half*epoch;
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
