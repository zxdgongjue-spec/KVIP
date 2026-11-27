// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace fixed_window_q {
constexpr uint32_t kScratchOffset = 232;
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kLinesPerRow = 5;

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
static_assert(kScratchOffset + sizeof(uint64_t) <= sizeof(pipeline_ab::Args));

__aicore__ inline GM_ADDR Scratch(GM_ADDR workspace, const rotating::RuntimeShape& shape)
{
    const auto batch = shape.batch;
    if (!workspace || !shape.splitQ || shape.aicCount != 32 || shape.indexHeads != 32 ||
        shape.finalWorkers != 4 || shape.liCores != 2 * batch ||
        shape.serviceAics != (batch + 1) / 2 || shape.serviceLanes != 2 * shape.serviceAics ||
        (batch != 1 && batch != 2 && batch != 4 && batch != 8) ||
        shape.rounds != (batch == 1 ? 2U : 4U) || shape.resident != 16384 ||
        (shape.history != 131072 && shape.history != 131328) || shape.tokenBytes != 656) return nullptr;
    return reinterpret_cast<GM_ADDR>(*reinterpret_cast<__gm__ const uint64_t*>(workspace + kScratchOffset));
}

struct Assignment {
    uint32_t row, query;
};

__aicore__ inline Assignment Helper(const rotating::RuntimeShape& shape, uint32_t aiv)
{
    // Both score variants use the original logical AIVs for window work.
    // Additional physical K producers never join the window rendezvous.
    if (aiv >= shape.liCores * 2) return {0, 0};
    if (shape.batch == 1) return {0, aiv < 2 ? aiv + 1 : aiv == 3 ? 3U : 0U};
    const auto group = aiv / 2;
    const auto epoch = rotating::ServiceRank(shape.batch, shape.splitQ, group) / shape.serviceAics;
    const auto row = rotating::ServiceLane(shape.batch, shape.splitQ, shape.serviceAics, group, aiv % 2);
    return {row, epoch == shape.rounds - 1 ? 0U : epoch + 1};
}

__aicore__ inline GM_ADDR Line(GM_ADDR scratch, uint32_t row, uint32_t line)
{
    return scratch + (row * kLinesPerRow + line) * kLineBytes;
}

__aicore__ inline __gm__ volatile Snapshot* Frontier(GM_ADDR scratch, uint32_t row)
{
    return reinterpret_cast<__gm__ volatile Snapshot*>(Line(scratch, row, 0));
}

// Q0 is the owner. Its ready flag publishes the snapshot.
__aicore__ inline __gm__ volatile Completion* Signal(GM_ADDR scratch, uint32_t row, uint32_t q)
{
    return reinterpret_cast<__gm__ volatile Completion*>(Line(scratch, row, q + 1));
}

__aicore__ inline void Wait(GM_ADDR scratch, uint32_t row, uint32_t q)
{
    auto flag = Signal(scratch, row, q);
    do {
        kvaf_device::CleanCacheLines(Line(scratch, row, q + 1), kLineBytes);
    } while (!flag->ready);
}

__aicore__ inline void Publish(GM_ADDR scratch, uint32_t row, uint32_t q, uint32_t ready)
{
    Signal(scratch, row, q)->ready = ready;
    kvaf_device::CleanCacheLines(Line(scratch, row, q + 1), kLineBytes);
}

__aicore__ inline void Reset(GM_ADDR workspace, const rotating::RuntimeShape& shape, uint32_t aiv)
{
    auto scratch = Scratch(workspace, shape);
    if (scratch && aiv < shape.batch * kQueries) Publish(scratch, aiv / kQueries, aiv % kQueries, 0);
}
}
