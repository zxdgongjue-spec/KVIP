#pragma once
#include "context.h"

namespace kvip_window_probe {
// The trace, frontier replay and resident histogram use bytes 192–223.
// Zero selects the native route. One selects P8 bulk cursor reads.
// An aligned address is a BS2 read-ahead flag buffer.
constexpr uint32_t kOptionOffset = 224;
constexpr uint64_t kBulk = 1;
constexpr uint32_t kReadyBytes = 128;
constexpr uint32_t kReadyWords = kReadyBytes / sizeof(uint32_t);

__aicore__ inline uint64_t Option(GM_ADDR workspace) {
    return workspace ? *reinterpret_cast<__gm__ const uint64_t*>(workspace + kOptionOffset) : 0;
}

__aicore__ inline bool EarlyShape(const rotating::RuntimeShape& shape) {
    return shape.batch == 2 && shape.rounds == 3 && shape.serviceLanes == 2 &&
        shape.history == 131072 && shape.resident == 16384;
}

__aicore__ inline __gm__ uint32_t* Ready(uint64_t option) {
    return option && option % kReadyBytes == 0 ? reinterpret_cast<__gm__ uint32_t*>(option) : nullptr;
}
}
