// SPDX-License-Identifier: Apache-2.0
#pragma once

#if KVIP_INCREMENTAL_SEGMENTS
namespace kvip_incremental {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Keep strict IDs at the front and ties in reverse order at the back.
// Appending strict winners now overwrites only ties that were evicted.
__simd_callee__ inline void RankIds(RegTensor<uint32_t>& result, __ubuf__ uint32_t* ids,
                                    uint32_t start, uint32_t strict) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> positions;
    Arange(positions, int32_t(start));
    auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(positions);
    RegTensor<uint32_t> index, last;
    Duplicate(last, kTop - 1 + strict);
    Sub(index, last, lane, all);
    MaskReg prefix;
    CompareScalar<uint32_t, CMPMODE::LT>(prefix, lane, strict, all);
    Select(index, lane, index, prefix);
    Gather(result, ids, index, all);
}

#if KVIP_SEED_MASK
// Consume the selector's 16-bit ordinals directly. Produce the final internal
// tie order and eligibility tag in the same vector store.
__simd_vf__ inline void MaterializeSeed(__ubuf__ uint16_t* ordinals, __ubuf__ uint32_t* bits,
    __ubuf__ uint32_t* output, uint32_t strict, uint32_t residentCount, uint32_t recentBegin,
    uint32_t resident, uint32_t tailBegin, uint32_t tailEnd) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> last, zero, one, lowBits, fetch;
    Duplicate(last, kTop - 1 + strict); Duplicate(zero, uint32_t(0)); Duplicate(one, uint32_t(1));
    Duplicate(lowBits, uint32_t(7)); Duplicate(fetch, kNeedsFetch);
    for (uint16_t i = 0; i < kTop; i += 64) {
        RegTensor<int32_t> lanes; Arange(lanes, int32_t(i));
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(lanes);
        RegTensor<uint32_t> rank, ordinal, token, index, shift, word, tagged;
        MaskReg keep, prefix, hot, flagged, cold, left, right, outside, needs;
        Sub(rank, last, lane, all);
        CompareScalar<uint32_t, CMPMODE::LT>(keep, lane, strict, all);
        Select(rank, lane, rank, keep);
        GatherU16(ordinal, ordinals, rank);
        Adds(token, ordinal, recentBegin - residentCount, all);
        CompareScalar<uint32_t, CMPMODE::LT>(prefix, ordinal, residentCount, all);
        Select(token, ordinal, token, prefix);
        CompareScalar<uint32_t, CMPMODE::LT>(hot, token, resident, all);
        Select(index, token, zero, hot); // No out-of-range gather on recent tokens.
        And(shift, index, lowBits, all); ShiftLefts(shift, shift, int16_t(2), all);
        ShiftRights(index, index, int16_t(3), all);
        Gather(word, bits, index, all);
        ShiftRight(word, word, reinterpret_cast<RegTensor<int32_t>&>(shift), all);
        And(word, word, one, all);
        CompareScalar<uint32_t, CMPMODE::NE>(flagged, word, uint32_t(0), all);
        CompareScalar<uint32_t, CMPMODE::GE>(cold, token, resident, all);
        MaskOr(needs, flagged, cold, all);
        CompareScalar<uint32_t, CMPMODE::LT>(left, token, tailBegin, all);
        CompareScalar<uint32_t, CMPMODE::GE>(right, token, tailEnd, all);
        MaskOr(outside, left, right, all); MaskAnd(needs, needs, outside, all);
        Or(tagged, token, fetch, all); Select(token, tagged, token, needs);
        StoreAlign(output + i, token, all);
    }
}
#endif

// The permutation is its own inverse. Source and destination never alias.
template<bool stripTag>
__simd_vf__ inline void FlipTies(__ubuf__ uint32_t* src, __ubuf__ uint32_t* dst,
                                 uint32_t strict) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> id, mask;
    Duplicate(mask, kIdMask);
    for (uint16_t offset = 0; offset < kTop; offset += 64) {
        RankIds(id, src, offset, strict);
        if constexpr (stripTag) And(id, id, mask, all);
        StoreAlign(dst + offset, id, all);
    }
}

// Read the completed window once to find its nonzero positions. Gather just
// those scores into disjoint scratch before the original window is reused.
__simd_vf__ inline void GatherAdmitted(__ubuf__ uint16_t* window, __ubuf__ uint16_t* positions,
                                       __ubuf__ uint16_t* scores, __ubuf__ uint16_t* offsets,
                                       uint32_t count) {
    uint32_t left = count;
    for (uint32_t i = 0; i < count; i += 128) {
        auto mask = UpdateMask<uint16_t>(left);
        RegTensor<uint16_t> index, score;
        LoadAlign(index, positions + i);
        Gather(score, window, index, mask);
        StoreAlign(scores + i, score, mask);
        StoreAlign(offsets + i, index, mask);
    }
}

__simd_vf__ inline void AppendStrict(__ubuf__ uint16_t* scores, __ubuf__ uint32_t* ids,
                                     __ubuf__ uint16_t* freshScores, __ubuf__ uint32_t* freshIds,
                                     uint32_t strict, uint32_t count) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero;
    Duplicate(zero, uint32_t(0));
    for (uint32_t offset = strict / 64 * 64; offset < strict + count; offset += 64) {
        RegTensor<int32_t> positions;
        Arange(positions, int32_t(offset));
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(positions);
        RegTensor<uint32_t> src, score, id;
        MaskReg lo, hi, write;
        CompareScalar<uint32_t, CMPMODE::GE>(lo, lane, strict, all);
        CompareScalar<uint32_t, CMPMODE::LT>(hi, lane, strict + count, all);
        MaskAnd(write, lo, hi, all);
        Adds(src, lane, uint32_t(0) - strict, all);
        Select(src, src, zero, write);
        Gather(id, freshIds, src, all);
        GatherU16(score, freshScores, src);
        StoreAlign(ids + offset, id, write);
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(
            scores + offset, reinterpret_cast<RegTensor<uint16_t>&>(score), write);
    }
}

__simd_vf__ inline void PackRankRange(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* dst,
                                      uint32_t begin, uint32_t end, uint32_t strict) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> mask;
    Duplicate(mask, kIdMask);
    UnalignRegForStore stream;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint32_t offset = begin / 64 * 64; offset < end; offset += 64) {
        RegTensor<uint32_t> id, packed;
        RegTensor<int32_t> positions;
        Arange(positions, int32_t(offset));
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(positions);
        MaskReg hit, lo, hi;
        CompareScalar<uint32_t, CMPMODE::GE>(lo, lane, begin, all);
        CompareScalar<uint32_t, CMPMODE::LT>(hi, lane, end, all);
        RankIds(id, ids, offset, strict);
        CompareScalar<uint32_t, CMPMODE::GE>(hit, id, kNeedsFetch, all);
        MaskAnd(hit, hit, lo, all);
        MaskAnd(hit, hit, hi, all);
        And(id, id, mask, all);
        Squeeze<uint32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, id, hit);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(dst, packed, stream);
    }
    StoreUnAlignPost(dst, stream);
}

}  // namespace kvip_incremental
#endif
