// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace kvip_window_q {
constexpr uint32_t kOptionOffset = 232;
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kRequests = 2;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kLinesPerRequest = 5;
constexpr uint32_t kScratchBytes = kRequests * kLinesPerRequest * kLineBytes;

struct Snapshot {
    uint64_t frontier[kQueries];
    uint32_t coveredEmpty;
};

struct Completion {
    uint32_t ready;
    uint32_t calls;
    uint32_t scanned;
    uint32_t candidates;
};

static_assert(sizeof(Snapshot) <= kLineBytes && sizeof(Completion) <= kLineBytes);

struct Trial {
    GM_ADDR scratch = nullptr;
    uint32_t workers = 0;
};

__aicore__ inline Trial LoadTrial(GM_ADDR workspace, const rotating::RuntimeShape& shape, uint32_t parts)
{
    if (!workspace || parts != 4 || shape.batch != kRequests || shape.rounds != 2 ||
        shape.history != 131072 || shape.resident != 16384 || shape.indexHeads != 32 ||
        shape.aicCount != 32 || shape.splitQ != 1 || shape.liCores != 4 ||
        shape.serviceAics != 1 || shape.serviceLanes != kRequests) {
        return {};
    }
    const uint64_t option = *reinterpret_cast<__gm__ const uint64_t*>(workspace + kOptionOffset);
    const uint32_t workers = option % kLineBytes;
    const uint64_t address = option - workers;
    if (!address || (workers != 2 && workers != 4)) return {};
    return {reinterpret_cast<GM_ADDR>(address), workers};
}

__aicore__ inline GM_ADDR Line(GM_ADDR scratch, uint32_t row, uint32_t line)
{
    return scratch + (row * kLinesPerRequest + line) * kLineBytes;
}

__aicore__ inline __gm__ volatile Snapshot* Frontier(GM_ADDR scratch, uint32_t row)
{
    return reinterpret_cast<__gm__ volatile Snapshot*>(Line(scratch, row, 0));
}

__aicore__ inline __gm__ volatile Completion* Signal(GM_ADDR scratch, uint32_t row, uint32_t worker)
{
    // Worker 0 is the request owner. Its signal releases the snapshot.
    return reinterpret_cast<__gm__ volatile Completion*>(Line(scratch, row, worker + 1));
}

__aicore__ inline void Wait(GM_ADDR scratch, uint32_t row, uint32_t worker)
{
    auto signal = Signal(scratch, row, worker);
    do {
        kvaf_device::CleanCacheLines(Line(scratch, row, worker + 1), kLineBytes);
    } while (!signal->ready);
}

__aicore__ inline void Publish(GM_ADDR scratch, uint32_t row, uint32_t worker, uint32_t ready)
{
    Signal(scratch, row, worker)->ready = ready;
    kvaf_device::CleanCacheLines(Line(scratch, row, worker + 1), kLineBytes);
}

__aicore__ inline void Reset(Trial trial, uint32_t lane)
{
    // The partitioned threshold launch has eight AIVs, one per signal.
    if (lane < kRequests * kQueries) {
        Publish(trial.scratch, lane / kQueries, lane % kQueries, 0);
    }
}
}
