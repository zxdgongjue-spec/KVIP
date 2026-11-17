// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rolling_threshold.h"

#if KVIP_ROLLING_THRESHOLD
#include "recent_threshold.h"
namespace kvip_rolling {
using namespace AscendC::MicroAPI;

// Collapse (score > seed && score >= current) to one strict comparison.
// Compute this once per update, not once per 128-score tile. Clamp before
// subtracting so a zero threshold cannot wrap to 65535.
__simd_callee__ inline void SaveAdmission(__ubuf__ uint16_t* state,
    RegTensor<uint16_t>& key, RegTensor<uint16_t>& anchor) {
#if KVIP_ROLLING_COMPACT
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> one, cutoff;
    Duplicate(one, uint16_t(1));
    Max(cutoff, key, one, all); Sub(cutoff, cutoff, one, all);
    Max(cutoff, cutoff, anchor, all);
    StoreAlign(state + kAdmission, cutoff, all);
#endif
}

// Count descending sortable BF16 keys, saturating every CDF entry at K.
// Each update adds <=128, hence no u16 overflow even for arbitrarily long
// histories. Clamping keys above seed+255 gives a conservative lower bound
// if the true Kth leaves the covered range; it never over-tightens admission.
__simd_callee__ inline void Encode(RegTensor<uint8_t>& bytes, RegTensor<uint16_t>& score,
    RegTensor<uint16_t>& anchor, uint32_t tileMask) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> delta, cap, encoded;
    Duplicate(cap, uint16_t(255));
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
    RegTensor<uint8_t> discarded;
    auto& packed = reinterpret_cast<RegTensor<uint8_t>&>(encoded);
    DeInterleave(bytes, discarded, packed, packed);
}

__simd_callee__ inline void Accumulate(RegTensor<uint16_t>& score,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper,
    RegTensor<uint16_t>& anchor, uint32_t tileMask) {
    RegTensor<uint8_t> bytes;
    RegTensor<uint16_t> rank;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(rank, uint16_t(kTop));
    Encode(bytes, score, anchor, tileMask);
    uint32_t lanes = 128;
    auto firstHalf = UpdateMask<uint8_t>(lanes);
    Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(lower, bytes, firstHalf);
    Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(upper, bytes, firstHalf);
    Min(lower, lower, rank, all); Min(upper, upper, rank, all);
}

#if KVIP_ROLLING_WINDOWED
#if KVIP_ROLLING_FAST_PATH
// The reused seed fixes a byte-aligned range. Decode both bytes directly
// with the selector's existing load pattern, rather than encoding 128 u16
// scores twice and interleaving them. Higher bytes clamp to bin 0, lower
// bytes to bin 255. This is identical to WindowEncode, not sampling.
__simd_callee__ inline void EncodeFullPair(RegTensor<uint8_t>& bytes,
    __ubuf__ uint16_t* input, RegTensor<uint16_t>& seedByte,
    RegTensor<uint16_t>& ones, RegTensor<uint16_t>& zero) {
    auto all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    auto all8 = CreateMask<uint8_t, MaskPattern::ALL>();
    RegTensor<uint16_t> low, high, inverted;
    LoadAlign<uint16_t, LoadDist::DIST_DINTLV_B8>(low, high, input);
    Xor(inverted, low, ones, all16);
    auto& high8 = reinterpret_cast<RegTensor<uint8_t>&>(high);
    auto& seed8 = reinterpret_cast<RegTensor<uint8_t>&>(seedByte);
    MaskReg above, below;
    Compare<uint8_t, CMPMODE::GT>(above, high8, seed8, all8);
    Compare<uint8_t, CMPMODE::LT>(below, high8, seed8, all8);
    Select(bytes, reinterpret_cast<RegTensor<uint8_t>&>(zero),
           reinterpret_cast<RegTensor<uint8_t>&>(inverted), above);
    Select(bytes, reinterpret_cast<RegTensor<uint8_t>&>(ones), bytes, below);
}
#endif
// Common full-cold tiles need neither lane clipping nor seed-range masks.
// Max before unsigned subtraction maps below-seed keys to the floor bin.
__simd_callee__ inline void WindowEncode(RegTensor<uint16_t>& encoded,
    RegTensor<uint16_t>& score, RegTensor<uint16_t>& anchor,
    uint32_t tile, uint32_t end, uint32_t recentBegin, uint32_t recentEnd) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> cap;
    Duplicate(cap, uint16_t(255));
    Max(encoded, score, anchor, all);
    Sub(encoded, encoded, anchor, all); Min(encoded, encoded, cap, all);
    Sub(encoded, cap, encoded, all);
    if (tile + 128 > end || (tile < recentEnd && tile + 128 > recentBegin)) {
        RegTensor<int16_t> sequence; Arange(sequence, int16_t(0));
        auto& index = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
        uint32_t n = end > tile ? end - tile : 0; if (n > 128) n = 128;
        uint32_t lo = recentBegin > tile ? recentBegin - tile : 0; if (lo > 128) lo = 128;
        uint32_t hi = recentEnd > tile ? recentEnd - tile : 0; if (hi > 128) hi = 128;
        MaskReg valid, left, right, outside;
        CompareScalar<uint16_t, CMPMODE::LT>(valid, index, uint16_t(n), all);
        CompareScalar<uint16_t, CMPMODE::LT>(left, index, uint16_t(lo), all);
        CompareScalar<uint16_t, CMPMODE::GE>(right, index, uint16_t(hi), all);
        MaskOr(outside, left, right, all); MaskAnd(valid, valid, outside, all);
        Select(encoded, encoded, cap, valid);
    }
}

__simd_callee__ inline void AccumulateWindow(__ubuf__ uint16_t* input, uint32_t count,
    RegTensor<uint16_t>& lower, RegTensor<uint16_t>& upper, RegTensor<uint16_t>& anchor,
    uint32_t scanBegin, uint32_t end, uint32_t recentBegin, uint32_t recentEnd,
    uint32_t reverseBegin, uint32_t reverseEnd) {
    RegTensor<uint16_t> a, b, encodedA, encodedB;
    RegTensor<uint8_t> bytes, discarded;
#if KVIP_ROLLING_FAST_PATH
    auto full16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> seedByte, ones, zero;
    ShiftRights(seedByte, anchor, int16_t(8), full16);
    Muls(seedByte, seedByte, uint16_t(257), full16);
    Duplicate(ones, uint16_t(65535)); Duplicate(zero, uint16_t(0));
#endif
    for (uint16_t offset = 0; offset < count; offset += 256) {
        uint32_t first = scanBegin + offset, second = first + 128;
#if KVIP_REVERSE_COLD
        if (first >= reverseBegin && first < reverseEnd)
            first = reverseBegin + reverseEnd - 2048 - (first & ~2047U) + (first & 2047U);
        if (second >= reverseBegin && second < reverseEnd)
            second = reverseBegin + reverseEnd - 2048 - (second & ~2047U) + (second & 2047U);
#endif
#if KVIP_ROLLING_FAST_PATH
        const bool firstFull = first + 128 <= end &&
            (first >= recentEnd || first + 128 <= recentBegin);
        const bool secondFull = second + 128 <= end &&
            (second >= recentEnd || second + 128 <= recentBegin);
        if (offset + 128 < count && firstFull && secondFull) {
            EncodeFullPair(bytes, input + offset, seedByte, ones, zero);
        } else {
#endif
        LoadAlign(a, input + offset);
        WindowEncode(encodedA, a, anchor, first, end, recentBegin, recentEnd);
        if (offset + 128 < count) {
            LoadAlign(b, input + offset + 128);
            WindowEncode(encodedB, b, anchor, second, end, recentBegin, recentEnd);
        } else {
            Duplicate(encodedB, uint16_t(255));
        }
        DeInterleave(bytes, discarded, reinterpret_cast<RegTensor<uint8_t>&>(encodedA),
                      reinterpret_cast<RegTensor<uint8_t>&>(encodedB));
#if KVIP_ROLLING_FAST_PATH
        }
#endif
        uint32_t lanes = offset + 128 < count ? 256 : 128;
        auto valid = UpdateMask<uint8_t>(lanes);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(lower, bytes, valid);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(upper, bytes, valid);
    }
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> top;
    Duplicate(top, uint16_t(kTop));
    Min(lower, lower, top, all); Min(upper, upper, top, all);
}

// Producer score writes land directly in this dedicated UB window buffer.
// The normal score DMA reads that same buffer; no extra per-tile copy or GM
// read is introduced. Only the boundary tile's bitmap needs reclassification.
__simd_vf__ inline void ObserveWindow(__ubuf__ uint16_t* input, uint32_t count,
    __ubuf__ uint16_t* state, __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits,
    uint32_t scanBegin, uint32_t end, uint32_t recentBegin, uint32_t recentEnd,
    uint32_t reverseBegin, uint32_t reverseEnd) {
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> lower, upper, anchor, a;
    LoadAlign(lower, state); LoadAlign(upper, state + 128);
    LoadAlign(anchor, state + kBins);
#if KVIP_ROLLING_REUSE_SEED
    RegTensor<uint16_t> histogramAnchor;
    RegTensor<uint16_t> byteMask; Duplicate(byteMask, uint16_t(0xff00));
    And(histogramAnchor, anchor, byteMask, all);
#else
    auto& histogramAnchor = anchor;
#endif
    AccumulateWindow(input, count, lower, upper, histogramAnchor, scanBegin, end,
                     recentBegin, recentEnd, reverseBegin, reverseEnd);
    // A window adds <=18432 to capped counts <=2048: no uint16 overflow.
    RegTensor<uint16_t> rank, bin, cap, key;
    Duplicate(rank, uint16_t(kTop)); Duplicate(cap, uint16_t(255));
    kvip_recent_threshold::FirstCdfBin(bin, lower, upper, rank);
    Sub(key, cap, bin, all); Add(key, key, histogramAnchor, all);
    StoreAlign(state, lower, all); StoreAlign(state + 128, upper, all);
    StoreAlign(threshold, key, all);
    SaveAdmission(state, key, anchor);
    LoadAlign(a, input + count - 128);
    MaskReg current, original;
    Compare<uint16_t, CMPMODE::GE>(current, a, key, all);
    Compare<uint16_t, CMPMODE::GT>(original, a, anchor, all);
    MaskAnd(current, current, original, all);
    StoreAlign(hits, current);
}
#endif

// Runs inside the existing score/classify VF, using its live score register.
__simd_callee__ inline void Observe(RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* state, __ubuf__ uint16_t* threshold,
    uint32_t mask, bool advance) {
    if (!state) return;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> lower, upper, anchor;
#if KVIP_ROLLING_BATCHED
    // Store only 128 encoded bytes while scores are live. Two tiles fit in
    // one histogram instruction; keep its CDF in registers for the batch.
    LoadAlign(anchor, state + kBins);
    RegTensor<uint8_t> bytes;
    Encode(bytes, score, anchor, mask);
    uint32_t slot = mask >> 24;
    auto pending = reinterpret_cast<__ubuf__ uint8_t*>(state + kCdfWords);
    uint32_t lanes = 128;
    auto firstHalf = UpdateMask<uint8_t>(lanes);
    StoreAlign(pending + slot * 128, bytes, firstHalf);
    if (slot + 1 != kBatchTiles && !advance) return;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(lower, state); LoadAlign(upper, state + 128);
    for (uint16_t i = 0; i <= slot; i += 2) {
        LoadAlign(bytes, pending + i * 128);
        uint32_t count = i < slot ? 256 : 128;
        auto valid = UpdateMask<uint8_t>(count);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(lower, bytes, valid);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(upper, bytes, valid);
    }
    // <=1024 additions to a previously capped count: uint16 cannot overflow.
    RegTensor<uint16_t> top;
    Duplicate(top, uint16_t(kTop));
    Min(lower, lower, top, all); Min(upper, upper, top, all);
#else
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(lower, state); LoadAlign(upper, state + 128);
    LoadAlign(anchor, state + kBins);
    Accumulate(score, lower, upper, anchor, mask);
#endif
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
#if KVIP_ROLLING_WINDOWED
    // The seed is an already unique resident+recent union. Full 256-lane
    // histogram updates reuse the window helper and clip only its final tile.
    AccumulateWindow(input, count, lower, upper, anchor, 0, count, count, count, 0, 0);
#else
    for (uint16_t offset = 0; offset < count; offset += 128) {
        LoadAlign(score, input + offset);
        uint32_t n = count - offset;
        Accumulate(score, lower, upper, anchor, n < 128 ? n : 128);
    }
#endif
    StoreAlign(state, lower, all); StoreAlign(state + 128, upper, all);
    StoreAlign(state + kBins, anchor, all);
    SaveAdmission(state, anchor, anchor);
}
} // namespace kvip_rolling
#endif
