// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "incremental_topk.h"

#if KVIP_INCREMENTAL_DIRECT
namespace kvip_incremental {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Both score and offset arrays are u16; gather their enclosing u32 words and
// select the correct half. All indices supplied here are within the allocation.
__simd_callee__ inline void GatherU16(RegTensor<uint32_t>& result,
    __ubuf__ uint16_t* base, RegTensor<uint32_t>& index) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> wordIndex, word, upper, odd, one, lowMask;
    Duplicate(one, uint32_t(1)); Duplicate(lowMask, uint32_t(0xffff));
    ShiftRights(wordIndex, index, uint32_t(1), all);
    Gather(word, reinterpret_cast<__ubuf__ uint32_t*>(base), wordIndex, all);
    ShiftRights(upper, word, uint32_t(16), all);
    And(odd, index, one, all);
    MaskReg high;
    CompareScalar<uint32_t, CMPMODE::NE>(high, odd, uint32_t(0), all);
    Select(result, upper, word, high);
    And(result, result, lowMask, all);
}

// Produce scores and tagged physical IDs together, with disjoint source and
// destination banks. Avoid the u16->u32 ordinal copy and the SIMT remap launch.
// tieUpdate: old strict winners, new improvements, then surviving old ties.
// Otherwise use the ordinary exact selector's strict-then-tie ordinals.
template<bool tieUpdate>
__simd_vf__ inline void Materialize(__ubuf__ uint16_t* ordinals,
    __ubuf__ uint16_t* values, __ubuf__ uint32_t* previous,
    __ubuf__ uint16_t* offsets, __ubuf__ uint16_t* scoresOut,
    __ubuf__ uint32_t* idsOut, __ubuf__ uint32_t* pendingOut, uint32_t strict, uint32_t admitted,
    uint32_t scanBegin, uint32_t reverseBegin, uint32_t reverseEnd,
    uint32_t tailBegin, uint32_t tailEnd) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero, lowMask, blockMask, flag, constant;
    Duplicate(zero, uint32_t(0)); Duplicate(lowMask, uint32_t(2047));
    Duplicate(blockMask, uint32_t(~2047U)); Duplicate(flag, kNeedsFetch);
    Duplicate(constant, reverseBegin + reverseEnd - 2048);
    RegTensor<uint32_t> idMask; Duplicate(idMask, kIdMask);
    UnalignRegForStore pendingStore;
    ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t base = 0; base < kTop; base += 64) {
        RegTensor<uint32_t> ordinal, shifted, inserted;
        if constexpr (tieUpdate) {
            RegTensor<int32_t> sequence; Arange(sequence, int32_t(base));
            auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
            Adds(shifted, lane, uint32_t(0) - admitted, all);
            Adds(inserted, lane, kTop - strict, all);
            MaskReg before, after;
            CompareScalar<uint32_t, CMPMODE::LT>(before, lane, strict, all);
            CompareScalar<uint32_t, CMPMODE::GE>(after, lane, strict + admitted, all);
            Select(ordinal, shifted, inserted, after);
            Select(ordinal, lane, ordinal, before);
        } else {
            RegTensor<int32_t> sequence; Arange(sequence, int32_t(base));
            GatherU16(ordinal, ordinals, reinterpret_cast<RegTensor<uint32_t>&>(sequence));
        }
        RegTensor<uint32_t> score, oldIndex, oldId, index, physical;
        GatherU16(score, values, ordinal);
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(
            scoresOut + base, reinterpret_cast<RegTensor<uint16_t>&>(score), all);
        MaskReg old;
        CompareScalar<uint32_t, CMPMODE::LT>(old, ordinal, kTop, all);
        Select(oldIndex, ordinal, zero, old);
        Gather(oldId, previous, oldIndex, all);
        Adds(index, ordinal, uint32_t(0) - kTop, all);
        Select(index, zero, index, old);
        if (offsets) GatherU16(index, offsets, index);
        Adds(physical, index, scanBegin, all);
#if KVIP_REVERSE_COLD
        RegTensor<uint32_t> block, lane, reversed;
        And(block, physical, blockMask, all); And(lane, physical, lowMask, all);
        Sub(reversed, constant, block, all); Add(reversed, reversed, lane, all);
        MaskReg lower, upper, reverse;
        CompareScalar<uint32_t, CMPMODE::GE>(lower, physical, reverseBegin, all);
        CompareScalar<uint32_t, CMPMODE::LT>(upper, physical, reverseEnd, all);
        MaskAnd(reverse, lower, upper, all);
        Select(physical, reversed, physical, reverse);
#endif
        RegTensor<uint32_t> tagged;
        MaskReg left, right, needs;
        CompareScalar<uint32_t, CMPMODE::LT>(left, physical, tailBegin, all);
        CompareScalar<uint32_t, CMPMODE::GE>(right, physical, tailEnd, all);
        MaskOr(needs, left, right, all);
        Or(tagged, physical, flag, all); Select(physical, tagged, physical, needs);
        Select(physical, oldId, physical, old);
        StoreAlign(idsOut + base, physical, all);
        // Reuse the freshly materialized tags. Do not reload all K IDs in
        // another VF solely to construct the immutable pending list.
        RegTensor<uint32_t> pending; MaskReg eligible;
        CompareScalar<uint32_t, CMPMODE::GE>(eligible, physical, kNeedsFetch, all);
        And(physical, physical, idMask, all);
        Squeeze<uint32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(pending, physical, eligible);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(pendingOut, pending, pendingStore);
    }
    StoreUnAlignPost(pendingOut, pendingStore);
}
} // namespace kvip_incremental
#endif
