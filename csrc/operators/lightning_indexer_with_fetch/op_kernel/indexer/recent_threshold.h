// SPDX-License-Identifier: Apache-2.0
#pragma once

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

// Exact unsigned-u16 Kth key, using the same two cumulative histograms as
// the ordinary four-VF selector. The caller limits the aligned input to
// 18432 values, so every cumulative count and residual rank fits in u16.
// Input scores are read-only; only the original 256-byte threshold bank is
// written. No histogram, bin list or intermediate rank is written to UB.
__simd_vf__ inline void RegisterThreshold(__ubuf__ uint16_t* input,
    __ubuf__ uint16_t* output, uint16_t loops, uint16_t bottomK) {
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
}
#if KVIP_PREFETCH_RANK
// Two exact ranks over the same resident/recent scores. Share the entire high
// histogram and load the input only once in the low-byte pass. The safe K2048
// key and speculative gate remain separate, including BF16 ties.
__simd_vf__ inline void RegisterThresholdPair(__ubuf__ uint16_t* input,
    __ubuf__ uint16_t* safeOutput, __ubuf__ uint16_t* gateOutput,
    uint16_t loops, uint16_t safeBottomK, uint16_t gateBottomK) {
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
    RegTensor<uint16_t> safeRank, gateRank, safeHigh, gateHigh, previous;
    Duplicate(safeRank, safeBottomK);
    Duplicate(gateRank, gateBottomK);
    FirstCdfBin(safeHigh, lower, upper, safeRank);
    CountBelowBin(previous, lower, upper, safeHigh);
    Sub(safeRank, safeRank, previous, all16);
    FirstCdfBin(gateHigh, lower, upper, gateRank);
    CountBelowBin(previous, lower, upper, gateHigh);
    Sub(gateRank, gateRank, previous, all16);
    RegTensor<uint16_t> safeBroadcast, gateBroadcast, gateLower, gateUpper;
    Muls(safeBroadcast, safeHigh, uint16_t(257), all16);
    Muls(gateBroadcast, gateHigh, uint16_t(257), all16);
    Duplicate(lower, uint16_t(0));
    Duplicate(upper, uint16_t(0));
    Duplicate(gateLower, uint16_t(0));
    Duplicate(gateUpper, uint16_t(0));
    for (uint16_t tile = 0; tile < loops; ++tile) {
        LoadAlign<uint16_t, LoadDist::DIST_DINTLV_B8>(lowBytes, highBytes, input + tile * 256);
        MaskReg selected;
        Compare<uint8_t, CMPMODE::EQ>(selected,
            reinterpret_cast<RegTensor<uint8_t>&>(highBytes),
            reinterpret_cast<RegTensor<uint8_t>&>(safeBroadcast), all8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(
            lower, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(
            upper, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
        Compare<uint8_t, CMPMODE::EQ>(selected,
            reinterpret_cast<RegTensor<uint8_t>&>(highBytes),
            reinterpret_cast<RegTensor<uint8_t>&>(gateBroadcast), all8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(
            gateLower, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(
            gateUpper, reinterpret_cast<RegTensor<uint8_t>&>(lowBytes), selected);
    }
    RegTensor<uint16_t> low, key;
    FirstCdfBin(low, lower, upper, safeRank);
    Muls(key, safeHigh, uint16_t(256), all16);
    Add(key, key, low, all16);
    StoreAlign(safeOutput, key, all16);
    FirstCdfBin(low, gateLower, gateUpper, gateRank);
    Muls(key, gateHigh, uint16_t(256), all16);
    Add(key, key, low, all16);
    StoreAlign(gateOutput, key, all16);
}
#endif
} // namespace kvip_recent_threshold
