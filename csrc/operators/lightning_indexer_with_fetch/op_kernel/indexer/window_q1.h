// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace fixed_window_q1 {
constexpr uint32_t kQueryCount = 4;
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kLinesPerRequest = kQueryCount + 1;

struct Snapshot {
    uint64_t frontier[kQueryCount];
    uint32_t coveredEmpty;
};

struct Completion {
    uint32_t ready;
    uint32_t calls;
    uint32_t scanned;
    uint32_t candidates;
};

static_assert(sizeof(Snapshot) <= kLineBytes && sizeof(Completion) <= kLineBytes);

__aicore__ inline GM_ADDR Line(GM_ADDR scratch, uint32_t row, uint32_t line) {
    return scratch + (row * kLinesPerRequest + line) * kLineBytes;
}

__aicore__ inline __gm__ volatile Snapshot* Frontier(GM_ADDR scratch, uint32_t row) {
    return reinterpret_cast<__gm__ volatile Snapshot*>(Line(scratch, row, 0));
}

__aicore__ inline __gm__ volatile Completion* Signal(GM_ADDR scratch, uint32_t row, uint32_t query) {
    // The original Service owner uses Query 0's line to release the snapshot.
    return reinterpret_cast<__gm__ volatile Completion*>(Line(scratch, row, query + 1));
}

__aicore__ inline void Publish(GM_ADDR scratch, uint32_t row, uint32_t query, uint32_t ready) {
    Signal(scratch, row, query)->ready = ready;
    kvaf_device::CleanCacheLines(Line(scratch, row, query + 1), kLineBytes);
}

__aicore__ inline void Wait(GM_ADDR scratch, uint32_t row, uint32_t query) {
    auto signal = Signal(scratch, row, query);
    do {
        kvaf_device::CleanCacheLines(Line(scratch, row, query + 1), kLineBytes);
    } while (!signal->ready);
}

__aicore__ inline void Reset(GM_ADDR scratch, uint32_t batch, uint32_t aiv) {
    if (scratch && aiv < batch * kQueryCount) {
        Publish(scratch, aiv / kQueryCount, aiv % kQueryCount, 0);
    }
}
}
