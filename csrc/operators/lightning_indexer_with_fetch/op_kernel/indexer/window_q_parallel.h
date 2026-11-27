// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace kvip_window_q {
constexpr uint32_t kOptionOffset = 232;
constexpr uint32_t kLineBytes = 128;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kLinesPerRequest = 5;

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
    uint32_t parts = 0;
};

__aicore__ inline Trial LoadTrial(GM_ADDR workspace, const rotating::RuntimeShape& shape, uint32_t parts)
{
    const uint32_t batch = shape.batch;
    const bool p4 = parts == 4 && shape.rounds == 2 && (batch == 1 || batch == 2 || batch == 4 || batch == 8);
    const bool p2 = parts == 2 && (batch == 1 || batch == 2 || batch == 4 || batch == 8 || batch == 16)
        && (shape.rounds == 2 || (shape.rounds == 4 && batch >= 2));
    if (!workspace || (!p4 && !p2) ||
        shape.history != 131072 || shape.resident != 16384 || shape.indexHeads != 32 ||
        shape.aicCount != 32 || shape.splitQ != 1 || shape.liCores != 2 * batch ||
        shape.serviceAics != (batch + 1) / 2 || shape.serviceLanes != 2 * shape.serviceAics) {
        return {};
    }
    const uint64_t option = *reinterpret_cast<__gm__ const uint64_t*>(workspace + kOptionOffset);
    const uint32_t workers = option % kLineBytes;
    const uint64_t address = option - workers;
    if (!address || (workers != 4 && (shape.rounds != 2 || workers != 2))) return {};
    return {reinterpret_cast<GM_ADDR>(address), workers, parts};
}

struct Helper {
    uint32_t row;
    uint32_t worker;
};

__aicore__ inline Helper ScoreHelper(const rotating::RuntimeShape& shape, uint32_t aiv, uint32_t parts)
{
    const uint32_t batch = shape.batch;
    if (batch == 1) {
        if (parts == 2) {
            // AIV2 owns W1. The other three lanes have finished Score and
            // any W0 Service or inactive W1 lane completion at the helper hook.
            return {0, aiv < 2 ? aiv + 1 : aiv == 3 ? 3U : 0U};
        }
        // G0 serves W0 and G2 serves W1. Reuse both peers of G1 and one of G3.
        return {0, aiv == 2 ? 1U : aiv == 3 ? 2U : aiv == 6 ? 3U : 0U};
    }
    const uint32_t group = aiv / 2;
    if (parts == 2) {
        if (shape.rounds == 4) {
            const uint32_t epoch = rotating::ServiceRank(batch, shape.splitQ, group) / shape.serviceAics;
            const uint32_t row = rotating::ServiceLane(batch, shape.splitQ, shape.serviceAics, group, aiv % 2);
            // W3 retains its original owner. Former W0, W1 and W2 owners
            // help only after the score loop and their earlier Service.
            return {row, epoch == 3 ? 0 : epoch + 1};
        }
        // Even groups in the two halves own W0 and W1, respectively.
        // Reuse the two odd groups first. The W0 owner can help only at the
        // post-Score hook, after completing its original W0 Service.
        const uint32_t half = group / batch;
        const uint32_t row = (group % batch / 2) * 2 + aiv % 2;
        const uint32_t worker = group % 2 ? 1 + half : (half == 0 ? 3 : 0);
        return {row, worker};
    }
    // W0 owns G0, G4, ... in the first half of the Score grid. W1 has
    // its owners in the second half. All other first-half groups have no
    // Service work in W2 and can contribute one helper to each paired row.
    const uint32_t rowPair = group / 4;
    return {2 * rowPair + aiv % 2, rowPair < batch / 2 ? group % 4 : 0};
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

__aicore__ inline void Reset(Trial trial, uint32_t batch, uint32_t lane)
{
    // The partitioned threshold launch has four AIVs per request.
    if (lane < batch * kQueries) {
        Publish(trial.scratch, lane / kQueries, lane % kQueries, 0);
    }
}
}
