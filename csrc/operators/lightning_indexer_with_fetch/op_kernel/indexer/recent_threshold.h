// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../include/rolling_threshold.h"

namespace kvip_recent_threshold {
using namespace AscendC::MicroAPI;

// Histograms(ACCUMULATE) produces the cumulative count at each byte value.
// ReduceMin/Max only promise the first lane; explicitly broadcast that lane
// before another vector operation consumes the selected bin or its count.
__simd_callee__ inline void FirstCdfBin(RegTensor<uint16_t>& result,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper,
    RegTensor<uint16_t>& rank) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<int16_t> sequence;
    Arange(sequence, int16_t(0));
    auto& index = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
    RegTensor<uint16_t> upperIndex, absent, left, right, reduced;
    Adds(upperIndex, index, uint16_t(128), all);
    Duplicate(absent, uint16_t(0xffff));
    MaskReg selected;
    Compare<uint16_t, CMPMODE::GE>(selected, lower, rank, all);
    Select(left, index, absent, selected);
    Compare<uint16_t, CMPMODE::GE>(selected, upper, rank, all);
    Select(right, upperIndex, absent, selected);
    Min(left, left, right, all);
    ReduceMin(reduced, left, all);
    Duplicate<uint16_t, HighLowPart::LOWEST>(result, reduced, all);
}

__simd_callee__ inline void CountBelowBin(RegTensor<uint16_t>& result,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper,
    RegTensor<uint16_t>& bin) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<int16_t> sequence;
    Arange(sequence, int16_t(0));
    auto& index = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
    RegTensor<uint16_t> upperIndex, zero, left, right, reduced;
    Adds(upperIndex, index, uint16_t(128), all);
    Duplicate(zero, uint16_t(0));
    MaskReg before;
    Compare<uint16_t, CMPMODE::LT>(before, index, bin, all);
    Select(left, lower, zero, before);
    Compare<uint16_t, CMPMODE::LT>(before, upperIndex, bin, all);
    Select(right, upper, zero, before);
    Max(left, left, right, all);
    ReduceMax(reduced, left, all);
    Duplicate<uint16_t, HighLowPart::LOWEST>(result, reduced, all);
}

#if KVIP_ROLLING_REUSE_SEED
// Reuse the selected high-byte CDF, already computed by RegisterThreshold.
// Convert count(score<=x) to count(score>=x) in 256 lanes, instead of
// rescanning every resident/recent score. Counts above this byte are exact;
// thresholds that would leave it are conservatively capped at its upper end.
__simd_callee__ inline void ExportRollingSeed(__ubuf__ uint16_t* state,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper,
    RegTensor<uint16_t>& rank, RegTensor<uint16_t>& key) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<int16_t> sequence; Arange(sequence, int16_t(0));
    auto& lane = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
    RegTensor<uint16_t> offsets, cap, last, left, right, lowerLast, zero, above, top;
    Duplicate(cap, uint16_t(126)); Sub(offsets, cap, lane, all);
    Duplicate(last, uint16_t(127)); Duplicate(zero, uint16_t(0));
    Min(offsets, offsets, last, all);
    // Stay in registers: do not round-trip the CDF through mutable UB state.
    // left = CDF[254..127], right = CDF[126..0] followed by zero.
    Gather(left, upper, offsets); Gather(right, lower, offsets);
    Gather(lowerLast, lower, last);
    MaskReg valid; Compare<uint16_t, CMPMODE::LT>(valid, lane, last, all);
    Select(left, left, lowerLast, valid); Select(right, right, zero, valid);
    // rank = N - K + 1 - count(high_byte < selected_byte), hence
    // rank + K - 1 counts all keys at or above the selected byte.
    // This uses the exact residual rank already consumed by the low selector.
    Adds(above, rank, uint16_t(kvip_rolling::kTop - 1), all);
    Sub(left, above, left, all); Sub(right, above, right, all);
    Duplicate(top, uint16_t(kvip_rolling::kTop));
    Min(left, left, top, all); Min(right, right, top, all);
    StoreAlign(state, left, all); StoreAlign(state + 128, right, all);
    StoreAlign(state + kvip_rolling::kBins, key, all);
    StoreAlign(state + kvip_rolling::kAdmission, key, all);
}
#endif

// Exact unsigned-u16 Kth key, using the same two cumulative histograms as
// the ordinary four-VF selector. The caller limits the aligned input to
// 18432 values, so every cumulative count and residual rank fits in u16.
// Input scores are read-only. With seed reuse off, only the original 256-byte
// threshold bank is written. Seed reuse additionally exports the existing CDF.
__simd_vf__ inline void RegisterThreshold(__ubuf__ uint16_t* input,
    __ubuf__ uint16_t* output, uint16_t loops, uint16_t bottomK
#if KVIP_ROLLING_REUSE_SEED
    , __ubuf__ uint16_t* rollingState = nullptr
#endif
    ) {
    auto all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    auto all8 = CreateMask<uint8_t, MaskPattern::ALL>();
    RegTensor<uint16_t> lower, upper, lowBytes, highBytes;
    Duplicate(lower, uint16_t(0));
    Duplicate(upper, uint16_t(0));
    for (uint16_t tile = 0; tile < loops; ++tile) {
        LoadAlign<uint16_t, LoadDist::DIST_DINTLV_B8>(lowBytes, highBytes, input + tile * 256);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(
            lower, reinterpret_cast<RegTensor<uint8_t>&>(highBytes), all8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(
            upper, reinterpret_cast<RegTensor<uint8_t>&>(highBytes), all8);
    }
    RegTensor<uint16_t> rank, high, previous, highBroadcast;
    Duplicate(rank, bottomK);
    FirstCdfBin(high, lower, upper, rank);
    CountBelowBin(previous, lower, upper, high);
    Sub(rank, rank, previous, all16);
    // Each u16 contains two copies of the byte, not alternating high/zero.
    Muls(highBroadcast, high, uint16_t(257), all16);
    Duplicate(lower, uint16_t(0));
    Duplicate(upper, uint16_t(0));
    for (uint16_t tile = 0; tile < loops; ++tile) {
        LoadAlign<uint16_t, LoadDist::DIST_DINTLV_B8>(lowBytes, highBytes, input + tile * 256);
        MaskReg selected;
        Compare<uint8_t, CMPMODE::EQ>(selected,
            reinterpret_cast<RegTensor<uint8_t>&>(highBytes),
            reinterpret_cast<RegTensor<uint8_t>&>(highBroadcast), all8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(
            lower, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(
            upper, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
    }
    RegTensor<uint16_t> low, key;
    FirstCdfBin(low, lower, upper, rank);
    Muls(key, high, uint16_t(256), all16);
    Add(key, key, low, all16);
    StoreAlign(output, key, all16);
#if KVIP_ROLLING_REUSE_SEED
    if (rollingState) ExportRollingSeed(rollingState, lower, upper, rank, key);
#endif
}
} // namespace kvip_recent_threshold
