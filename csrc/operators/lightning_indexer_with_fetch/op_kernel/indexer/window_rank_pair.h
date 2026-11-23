// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace kvip_window_ranks {
using namespace AscendC;
using namespace AscendC::MicroAPI;
// The final bucket contains every score at least 64 representable values above the old floor.
// If either cutoff reaches it, the caller uses the full-range selector.
__simd_vf__ inline void AddScores(__ubuf__ uint16_t *scores, __ubuf__ uint16_t *state, uint16_t floor,
                                  uint16_t count, bool fresh) {
    auto full = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> cdf, first, second, base;
    RegTensor<uint8_t> low, high;
    Duplicate(base, floor);
    if (fresh)
        Duplicate(cdf, uint16_t(0));
    else
        LoadAlign(cdf, state);
    for (uint16_t i = 0; i < count; i += 256) {
        LoadAlign(first, scores + i);
        LoadAlign(second, scores + i + 128);
        Max(first, first, base, full);
        Max(second, second, base, full);
        Sub(first, first, base, full);
        Sub(second, second, base, full);
        Mins(first, first, uint16_t(64), full);
        Mins(second, second, uint16_t(64), full);
        DeInterleave(low, high, reinterpret_cast<RegTensor<uint8_t> &>(first),
                     reinterpret_cast<RegTensor<uint8_t> &>(second));
        uint32_t remaining = count - i;
        auto valid = UpdateMask<uint8_t>(remaining);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cdf, low, valid);
    }
    StoreAlign(state, cdf, full);
}
__simd_callee__ inline void WriteCut(RegTensor<uint16_t> &cdf, RegTensor<uint16_t> &index,
                                     __ubuf__ uint16_t *output, uint16_t floor, uint16_t bottom) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    auto one = CreateMask<uint16_t, MaskPattern::VL1>();
    RegTensor<uint16_t> none, val, reduced;
    Duplicate(none, uint16_t(65535));
    MaskReg selected;
    CompareScalar<uint16_t, CMPMODE::GE>(selected, cdf, bottom, all);
    Select(val, index, none, selected);
    ReduceMin(reduced, val, all);
    Adds(val, reduced, floor, one);
    StoreAlign(output, val, one);
}
__simd_vf__ inline void ReadCuts(__ubuf__ uint16_t *state, __ubuf__ uint16_t *output, uint16_t floor,
                                 uint16_t total, uint16_t rank) {
    RegTensor<uint16_t> cdf;
    RegTensor<int16_t> seq;
    LoadAlign(cdf, state);
    Arange(seq, int16_t(0));
    auto &ix = reinterpret_cast<RegTensor<uint16_t> &>(seq);
    WriteCut(cdf, ix, output, floor, total - 2048 + 1);
    WriteCut(cdf, ix, output + 128, floor, total - rank + 1);
}

// Strict winners precede cutoff ties, in their previous traversal order.
template <bool tieOnly>
__simd_vf__ inline void ScoreSlots(__ubuf__ uint16_t *keys, uint32_t total, uint32_t cut,
                                   __ubuf__ uint32_t *slots) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> seq;
    RegTensor<uint32_t> packed;
    RegTensor<uint16_t> val;
    MaskReg take;
    UnalignRegForStore store;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t pass = tieOnly ? 1 : 0; pass < 2; ++pass) {
        Arange(seq, int32_t(0));
        for (uint16_t i = 0; i < total; i += 64) {
            uint32_t remain = total - i;
            auto active = UpdateMask<uint32_t>(remain);
            LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(val, keys + i);
            if (!pass)
                CompareScalar<uint32_t, CMPMODE::GT>(take, reinterpret_cast<RegTensor<uint32_t> &>(val), cut,
                                                     all);
            else
                CompareScalar<uint32_t, CMPMODE::EQ>(take, reinterpret_cast<RegTensor<uint32_t> &>(val), cut,
                                                     all);
            MaskAnd(take, take, active, all);
            Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
                packed, reinterpret_cast<RegTensor<uint32_t> &>(seq), take);
            StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(slots, packed, store);
            Adds(seq, seq, int32_t(64), all);
        }
    }
    StoreUnAlignPost(slots, store);
}
template <bool withPosition>
__simd_vf__ inline void GatherPairs(__ubuf__ uint32_t *slots, __ubuf__ uint32_t *ids, __ubuf__ uint16_t *keys,
                                    uint32_t count, __ubuf__ uint32_t *resultIds,
                                    __ubuf__ uint32_t *resultKeys) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> pos, id, val, index, shift, one, mask16, zero;
    Duplicate(one, uint32_t(1));
    Duplicate(mask16, uint32_t(65535));
    Duplicate(zero, uint32_t(0));
    for (uint16_t i = 0; i < count; i += 64) {
        uint32_t n = count - i;
        auto active = UpdateMask<uint32_t>(n);
        LoadAlign(pos, slots + i);
        Select(pos, pos, zero, active);
        Gather(id, ids, pos, all);
        ShiftRights(index, pos, int16_t(1), all);
        Gather(val, reinterpret_cast<__ubuf__ uint32_t *>(keys), index, all);
        And(shift, pos, one, all);
        ShiftLefts(shift, shift, int16_t(4), all);
        ShiftRight(val, val, reinterpret_cast<RegTensor<int32_t> &>(shift), all);
        And(val, val, mask16, all);
        StoreAlign(resultIds + i, id, all);
        if constexpr (withPosition) {
            ShiftLefts(index, pos, int16_t(16), all);
            Or(val, val, index, all);
            StoreAlign(resultKeys + i, val, all);
        } else
            StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(resultKeys + i / 2, val, all);
    }
}
} // namespace kvip_window_ranks
