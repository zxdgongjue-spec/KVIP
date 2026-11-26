// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../include/context.h"

#if defined(__CCE_AICORE__)
#define KVIP_COLD_GM_DEVICE __aicore__
#else
#define KVIP_COLD_GM_DEVICE
#endif

namespace kvip_cold_gm {
constexpr uint32_t kQueries = 4;
constexpr uint32_t kReadyLineBytes = 128;

KVIP_COLD_GM_DEVICE inline uint64_t ScoreBytes(const rotating::RuntimeShape& shape) {
    return uint64_t(shape.batch) * kQueries * shape.history * sizeof(uint16_t);
}

KVIP_COLD_GM_DEVICE inline bool Supports(const rotating::RuntimeShape& shape) {
    const auto batch = shape.batch;
    if (!batch || batch > 16 || (batch & (batch - 1))) return false;
    const auto groups = 2 * batch;
    const auto rounds = groups < 4 ? groups : 4;
    return shape.history == 131072 && shape.resident == 16384 && shape.indexHeads == 32 &&
        shape.finalWorkers == kQueries && shape.aicCount == 32 && shape.liCores == groups &&
        shape.rounds == rounds && shape.serviceAics == groups / rounds &&
        shape.serviceLanes == 2 * shape.serviceAics && rotating::UseSequenceSplit(shape);
}

KVIP_COLD_GM_DEVICE inline bool ThresholdFirst(uint32_t group, uint32_t groups) {
    return group >= groups / 2;
}

#if defined(__CCE_AICORE__)
enum ReadyWord : uint32_t { RowReady = 0, ThresholdReady = 1 };
enum TimeWord : uint32_t {
    RawBegin = 1, RawEnd, ThresholdBegin, ThresholdEnd, ConsumeBegin, ConsumeEnd
};

__aicore__ inline void PublishReady(GM_ADDR line, ReadyWord word) {
    reinterpret_cast<__gm__ volatile uint32_t*>(line)[word] = 1;
    kvaf_device::CleanCacheLines(line, kReadyLineBytes);
}

__aicore__ inline void WaitRow(GM_ADDR ready, uint32_t row, ReadyWord word) {
    for (uint32_t q = 0; q < kQueries; ++q) {
        auto line = ready + (row * kQueries + q) * kReadyLineBytes;
        do {
            kvaf_device::CleanCacheLines(line, kReadyLineBytes);
        } while (!reinterpret_cast<__gm__ volatile uint32_t*>(line)[word]);
    }
}
#endif
}

#undef KVIP_COLD_GM_DEVICE
