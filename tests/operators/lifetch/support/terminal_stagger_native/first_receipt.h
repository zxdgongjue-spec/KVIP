// SPDX-License-Identifier: Apache-2.0
#pragma once
// Two private, separately aligned, single-writer permits per row. Zero is set
// in PrepareTerminalTeams before the ordinary schedule becomes visible.
__aicore__ inline void PublishFirstReceipt(__gm__ uint32_t* word, bool ok) {
    *reinterpret_cast<__gm__ volatile uint32_t*>(word) = ok ? 1U : 2U;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(*word));
}
__aicore__ inline uint32_t LoadPermit(__gm__ uint32_t* word) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(*word));
    return *reinterpret_cast<__gm__ volatile uint32_t*>(word);
}
__aicore__ inline uint64_t WaitFirstReceipt(__gm__ uint32_t* word) {
    const uint64_t maxPolls = uint64_t(kvaf_device::kMaxCqPolls) * kFusedCqPollMultiplier * 4;
    for (uint64_t i = 0; i < maxPolls; ++i) {
        const uint32_t value = LoadPermit(word);
        if (value) return value == 1 ? 0 : kvaf_device::kCqError;
    }
    return kvaf_device::kCqTimeout;
}
