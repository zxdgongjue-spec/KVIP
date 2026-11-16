// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rolling_threshold.h"

#if KVIP_ROLLING_THRESHOLD
#include "recent_threshold.h"
namespace kvip_rolling {
using namespace AscendC::MicroAPI;

// Count descending sortable BF16 keys, saturating every CDF entry at K.
// Each update adds <=128, hence no u16 overflow even for arbitrarily long
// histories. Clamping keys above seed+255 gives a conservative lower bound
// if the true Kth leaves the covered range; it never over-tightens admission.
__simd_callee__ inline void Accumulate(RegTensor<uint16_t>& score,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper,
    RegTensor<uint16_t>& anchor, uint32_t tileMask) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> delta, cap, encoded, rank;
    Duplicate(cap, uint16_t(255)); Duplicate(rank, uint16_t(kTop));
    Sub(delta, score, anchor, all); Min(delta, delta, cap, all);
    Sub(encoded, cap, delta, all);
    RegTensor<int16_t> sequence;
    Arange(sequence, int16_t(0));
    auto& index = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
    MaskReg valid, left, right, outside, admitted;
    CompareScalar<uint16_t, CMPMODE::LT>(valid, index, uint16_t(tileMask & 255), all);
    CompareScalar<uint16_t, CMPMODE::LT>(left, index, uint16_t((tileMask >> 8) & 255), all);
    CompareScalar<uint16_t, CMPMODE::GE>(right, index, uint16_t((tileMask >> 16) & 255), all);
    MaskOr(outside, left, right, all); MaskAnd(valid, valid, outside, all);
    Compare<uint16_t, CMPMODE::GE>(admitted, score, anchor, all);
    MaskAnd(valid, valid, admitted, all);
    // Invalid lanes affect only the floor bin, which cannot raise the seed.
    Select(encoded, encoded, cap, valid);
    RegTensor<uint8_t> bytes, discarded;
    auto& packed = reinterpret_cast<RegTensor<uint8_t>&>(encoded);
    DeInterleave(bytes, discarded, packed, packed);
    uint32_t lanes = 128;
    auto firstHalf = UpdateMask<uint8_t>(lanes);
    Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(lower, bytes, firstHalf);
    Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(upper, bytes, firstHalf);
    Min(lower, lower, rank, all); Min(upper, upper, rank, all);
}

// Runs inside the existing score/classify VF, using its live score register.
__simd_callee__ inline void Observe(RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* state, __ubuf__ uint16_t* threshold,
    uint32_t mask, bool advance) {
    if (!state) return;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> lower, upper, anchor;
    LoadAlign(lower, state); LoadAlign(upper, state + 128);
    LoadAlign(anchor, state + kBins);
    Accumulate(score, lower, upper, anchor, mask);
    StoreAlign(state, lower, all); StoreAlign(state + 128, upper, all);
    if (advance) {
        RegTensor<uint16_t> rank, bin, cap, key;
        Duplicate(rank, uint16_t(kTop)); Duplicate(cap, uint16_t(255));
        kvip_recent_threshold::FirstCdfBin(bin, lower, upper, rank);
        Sub(key, cap, bin, all); Add(key, key, anchor, all);
        StoreAlign(threshold, key, all);
    }
}

// Seed from the resident+recent union already loaded for the exact Kth.
// No additional Index DMA or LI arithmetic. Extra histogram work is timed.
__simd_vf__ inline void Seed(__ubuf__ uint16_t* input, uint32_t count,
    __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* state) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> lower, upper, anchor, score;
    Duplicate(lower, uint16_t(0)); Duplicate(upper, uint16_t(0));
    LoadAlign(anchor, threshold);
    for (uint16_t offset = 0; offset < count; offset += 128) {
        LoadAlign(score, input + offset);
        uint32_t n = count - offset;
        Accumulate(score, lower, upper, anchor, n < 128 ? n : 128);
    }
    StoreAlign(state, lower, all); StoreAlign(state + 128, upper, all);
    StoreAlign(state + kBins, anchor, all);
}
} // namespace kvip_rolling
#endif
