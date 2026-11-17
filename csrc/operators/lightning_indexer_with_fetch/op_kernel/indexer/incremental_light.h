// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "incremental_topk.h"

#if KVIP_INCREMENTAL_LIGHT
namespace kvip_incremental {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Convert only the newly admitted offsets. The retained K IDs already carry
// their physical position and fixed hot/ABC eligibility tag.
__simd_vf__ inline void MapNewIds(__ubuf__ uint16_t* offsets,
    __ubuf__ uint32_t* output, uint32_t count, uint32_t scanBegin,
    uint32_t reverseBegin, uint32_t reverseEnd, uint32_t tailBegin, uint32_t tailEnd) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint16_t> packed;
    RegTensor<uint32_t> id, block, lane, reversed, constant, blockMask, lowMask, flag, tagged;
    Duplicate(constant, reverseBegin + reverseEnd - 2048);
    Duplicate(blockMask, uint32_t(~2047U));
    Duplicate(lowMask, uint32_t(2047));
    Duplicate(flag, kNeedsFetch);
    for (uint16_t base = 0; base < count; base += 64) {
        LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(packed, offsets + base);
        Adds(id, reinterpret_cast<RegTensor<uint32_t>&>(packed), scanBegin, all);
#if KVIP_REVERSE_COLD
        And(block, id, blockMask, all);
        And(lane, id, lowMask, all);
        Sub(reversed, constant, block, all);
        Add(reversed, reversed, lane, all);
        MaskReg lower, upper, reverse;
        CompareScalar<uint32_t, CMPMODE::GE>(lower, id, reverseBegin, all);
        CompareScalar<uint32_t, CMPMODE::LT>(upper, id, reverseEnd, all);
        MaskAnd(reverse, lower, upper, all);
        Select(id, reversed, id, reverse);
#endif
        MaskReg left, right, needs;
        CompareScalar<uint32_t, CMPMODE::LT>(left, id, tailBegin, all);
        CompareScalar<uint32_t, CMPMODE::GE>(right, id, tailEnd, all);
        MaskOr(needs, left, right, all);
        Or(tagged, id, flag, all);
        Select(id, tagged, id, needs);
        StoreAlign(output + base, id, all);
    }
}

// Preserve strict-then-tie ordering when all strict improvements fit into the
// old boundary-tie group. The ordinary gather below can consume these indices.
__simd_vf__ inline void TieOrdinals(__ubuf__ uint16_t* output, uint16_t strict, uint16_t admitted) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<int16_t> sequence;
    RegTensor<uint16_t> shifted, inserted, ordinal;
    for (uint16_t base = 0; base < kTop; base += 128) {
        Arange(sequence, int16_t(base));
        auto& lane = reinterpret_cast<RegTensor<uint16_t>&>(sequence);
        Adds(shifted, lane, uint16_t(0 - admitted), all);
        Adds(inserted, lane, uint16_t(kTop - strict), all);
        MaskReg before, after;
        CompareScalar<uint16_t, CMPMODE::LT>(before, lane, strict, all);
        CompareScalar<uint16_t, CMPMODE::GE>(after, lane, uint16_t(strict + admitted), all);
        Select(ordinal, shifted, inserted, after);
        Select(ordinal, lane, ordinal, before);
        StoreAlign(output + base, ordinal, all);
    }
}

__simd_vf__ inline void GatherIds(__ubuf__ uint16_t* ordinals,
    __ubuf__ uint32_t* previous, __ubuf__ uint32_t* fresh,
    __ubuf__ uint32_t* output, __ubuf__ uint32_t* pending) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint16_t> packed;
    RegTensor<uint32_t> zero, oldIndex, newIndex, oldId, newId, id, mask, compact;
    Duplicate(zero, uint32_t(0));
    Duplicate(mask, kIdMask);
    UnalignRegForStore store;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t base = 0; base < kTop; base += 64) {
        LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(packed, ordinals + base);
        auto& ordinal = reinterpret_cast<RegTensor<uint32_t>&>(packed);
        MaskReg old, eligible;
        CompareScalar<uint32_t, CMPMODE::LT>(old, ordinal, kTop, all);
        Select(oldIndex, ordinal, zero, old);
        Adds(newIndex, ordinal, uint32_t(0) - kTop, all);
        Select(newIndex, zero, newIndex, old);
        Gather(oldId, previous, oldIndex, all);
        Gather(newId, fresh, newIndex, all);
        Select(id, oldId, newId, old);
        StoreAlign(output + base, id, all);
        CompareScalar<uint32_t, CMPMODE::GE>(eligible, id, kNeedsFetch, all);
        And(id, id, mask, all);
        Squeeze<uint32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(compact, id, eligible);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(pending, compact, store);
    }
    StoreUnAlignPost(pending, store);
}

// Keep the ready stripe in the already prepared header. The caller still
// waits for the payload DMA to finish before publishing this stripe.
__simd_vf__ inline void ListHeader(__ubuf__ uint32_t* output,
    __ubuf__ uint16_t* threshold, uint32_t count, uint32_t epoch) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> sequence;
    Arange(sequence, int32_t(0));
    auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
    RegTensor<uint32_t> value, field, mask;
    RegTensor<uint16_t> key;
    Duplicate(value, uint32_t(0));
    MaskReg use, store;
    CompareScalar<uint32_t, CMPMODE::LT>(store, lane, uint32_t(32), all);
    CompareScalar<uint32_t, CMPMODE::EQ>(use, lane, uint32_t(0), all);
    Duplicate(field, count);
    Select(value, field, value, use);
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(key, threshold);
    Duplicate(mask, uint32_t(0xffff));
    And(field, reinterpret_cast<RegTensor<uint32_t>&>(key), mask, all);
    CompareScalar<uint32_t, CMPMODE::EQ>(use, lane, uint32_t(1), all);
    Select(value, field, value, use);
    Duplicate(field, epoch);
    CompareScalar<uint32_t, CMPMODE::EQ>(use, lane, uint32_t(2), all);
    Select(value, field, value, use);
    CompareScalar<uint32_t, CMPMODE::EQ>(use, lane, uint32_t(8), all);
    Select(value, field, value, use);
    StoreAlign(output, value, store);
}
} // namespace kvip_incremental
#endif
