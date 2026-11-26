// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../../common/lifetch/cold_score_partition.h"

namespace kvip_retired_service {
constexpr uint32_t kGroups = 32;
constexpr uint32_t kLineBytes = 128;

__aicore__ inline GM_ADDR Scratch(GM_ADDR compute, uint32_t batch, uint32_t history, uint32_t parts) {
    return compute + kvip_cold_score::ScoreBytes(batch, history) +
        batch * kvip_cold_score::kQueries * parts * kvip_cold_score::kProgressBytes;
}

__aicore__ inline __gm__ uint32_t* Word(GM_ADDR scratch, uint32_t line) {
    return reinterpret_cast<__gm__ uint32_t*>(scratch + line * kLineBytes);
}

__aicore__ inline uint32_t Load(__gm__ uint32_t* word) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), kLineBytes);
    return *reinterpret_cast<__gm__ volatile uint32_t*>(word);
}

__aicore__ inline void Store(__gm__ uint32_t* word, uint32_t value) {
    *reinterpret_cast<__gm__ volatile uint32_t*>(word) = value;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), kLineBytes);
}

__aicore__ inline bool EventComplete(__gm__ rotating::Event* event) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event), sizeof(rotating::Event));
    return reinterpret_cast<__gm__ volatile rotating::Event*>(event)->complete != 0;
}

__simt_vf__ inline void Claim(__gm__ uint32_t* word, __ubuf__ uint32_t* old) {
    if (!threadIdx.x) {
        *old = asc_atomic_add(word, 1U);
    }
}
}
