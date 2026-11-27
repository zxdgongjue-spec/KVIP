// SPDX-License-Identifier: Apache-2.0
#pragma once
// A quarter-period for each row/Q class. No other queue controls this wait.
constexpr uint32_t kGateWidth = 4096;
constexpr uint32_t kGateDisabled = 4;
__aicore__ inline uint32_t DelayUntilQuarter(uint64_t now, uint32_t group) {
    const uint32_t phase = uint32_t(now) & (4 * kGateWidth - 1);
    const uint32_t start = group * kGateWidth;
    if (phase >= start && phase < start + kGateWidth - 1024) return 0;
    return (start - phase) & (4 * kGateWidth - 1);
}
__aicore__ inline void WaitDoorbellQuarter(uint32_t group) {
    if (group == kGateDisabled) return;
    const uint64_t now = GetSystemCycle();
    const uint32_t delay = DelayUntilQuarter(now, group);
    // Elapsed subtraction handles counter wrap. A delayed AIV never waits a second period.
    if (delay) while (GetSystemCycle() - now < delay) {}
}
