// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "incremental_topk.h"
#include "cold_scan.h"

#if KVIP_INCREMENTAL_TOPK
namespace kvip_incremental {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Fixed 128-lane destination, no compaction count or scalar readback. Only
// causally invalid/already-seeded positions become zero. Lower scores need no
// extra cutoff comparison: the retained K precede the window and the exact
// boundary selection removes them. Even at Kth=0, retained valid zeros fill
// all remaining ties before any masked/padded zero could enter the result.
__simd_callee__ inline void StoreWindow(RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* output, uint32_t laneMask) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> zero, value;
    RegTensor<int16_t> lanes;
    Arange(lanes, int16_t(0));
    auto& index = reinterpret_cast<RegTensor<uint16_t>&>(lanes);
    MaskReg valid, left, right, outside;
    CompareScalar<uint16_t, CMPMODE::LT>(valid, index, uint16_t(laneMask & 255), all);
    CompareScalar<uint16_t, CMPMODE::LT>(left, index, uint16_t((laneMask >> 8) & 255), all);
    CompareScalar<uint16_t, CMPMODE::GE>(right, index, uint16_t((laneMask >> 16) & 255), all);
    MaskOr(outside, left, right, all);
    MaskAnd(valid, valid, outside, all);
    Duplicate(zero, uint16_t(0));
    Select(value, score, zero, valid);
    StoreAlign(output, value, all);
}

// Called with the live score register inside the existing scoring VF. No
// second score DMA, histogram, or random score lookup at submission time.
__simd_callee__ inline void Compact(RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* cutoff, __ubuf__ uint16_t* values,
    __ubuf__ uint16_t* offsets, uint16_t scanOffset, uint32_t laneMask) {
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> threshold, packed, position;
    RegTensor<int16_t> lanes;
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(threshold, cutoff);
    Arange(lanes, int16_t(0));
    auto& index = reinterpret_cast<RegTensor<uint16_t>&>(lanes);
    MaskReg admitted, valid, left, right, outside;
    Compare<uint16_t, CMPMODE::GT>(admitted, score, threshold, all);
    CompareScalar<uint16_t, CMPMODE::LT>(valid, index, uint16_t(laneMask & 255), all);
    CompareScalar<uint16_t, CMPMODE::LT>(left, index, uint16_t((laneMask >> 8) & 255), all);
    CompareScalar<uint16_t, CMPMODE::GE>(right, index, uint16_t((laneMask >> 16) & 255), all);
    MaskOr(outside, left, right, all); MaskAnd(valid, valid, outside, all);
    MaskAnd(admitted, admitted, valid, all);
    UnalignRegForStore store;
    ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, score, admitted);
    StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(values, packed, store);
    StoreUnAlignPost(values, store);
    Adds(position, index, scanOffset, all);
    ClearSpr<AscendC::SpecialPurposeReg::AR>();
    UnalignRegForStore idStore;
    Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, position, admitted);
    StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(offsets, packed, idStore);
    StoreUnAlignPost(offsets, idStore);
}

__simd_vf__ inline void ZeroTail(__ubuf__ uint16_t* values,uint32_t begin,uint32_t end) {
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    RegTensor<uint16_t> value,zero;
    RegTensor<int16_t> index;
    Duplicate(zero,uint16_t(0));
    for(uint32_t base=begin/128*128;base<end;base+=128) {
        Arange(index,int16_t(base));LoadAlign(value,values+base);
        MaskReg padding;
        CompareScalar<uint16_t,CMPMODE::GE>(padding,reinterpret_cast<RegTensor<uint16_t>&>(index),uint16_t(begin),all);
        Select(value,zero,value,padding);StoreAlign(values+base,value,all);
    }
}

// Reuse the selector's exact Kth; no second reduction over 2048 scores.
__simd_vf__ inline void BroadcastThreshold(__ubuf__ uint16_t* value,__ubuf__ uint16_t* out) {
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    RegTensor<uint16_t> threshold;
    LoadAlign<uint16_t,LoadDist::DIST_BRC_B16>(threshold,value);
    StoreAlign(out,threshold,all);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void Remap(__ubuf__ uint32_t* selected, __ubuf__ uint32_t* previous,
    __ubuf__ uint16_t* offsets, uint32_t scanBegin, uint32_t reverseBegin,
    uint32_t reverseEnd, uint32_t seedResident, uint32_t seedBegin,
    uint32_t tailBegin, uint32_t tailEnd) {
    for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
        uint32_t position = selected[i];
        if (seedResident) {
            selected[i] = position < seedResident ? position : seedBegin + position - seedResident;
        } else if (position < kTop) {
            selected[i] = previous[position];
        } else {
#if KVIP_INCREMENTAL_WINDOW
            uint32_t token = scanBegin + position - kTop;
#else
            uint32_t token = scanBegin + offsets[position - kTop];
#endif
#if KVIP_REVERSE_COLD
            if (token >= reverseBegin && token < reverseEnd)
                token = reverseBegin + reverseEnd - 2048 - (token & ~2047U) + (token & 2047U);
#endif
            selected[i] = token | ((token < tailBegin || token >= tailEnd) ? kNeedsFetch : 0);
        }
    }
}

// Cache eligibility once for the seed. Hot validity, victim reservation and
// ABC bounds stay fixed throughout LI; the consumer separately deduplicates
// submitted IDs. Preserve this bit through selection, strip it at publication.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void TagSeed(__gm__ rotating::Context* ctx, uint32_t batch, uint32_t resident,
    __ubuf__ uint32_t* ids) {
    for (uint32_t i = threadIdx.x; i < kTop; i += 256) {
        uint32_t token = ids[i];
        bool hot = token < resident && rotating::HotValid(resident, ctx, batch, token);
        bool victim = hot && ctx->reservedHot && asc_ldcg(
            reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot) + batch * resident + token) >= 0;
        if (!rotating::IsTailToken(ctx, batch, token) && (!hot || victim)) ids[i] |= kNeedsFetch;
    }
}

__simd_vf__ inline void UntagIds(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* output) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> id, mask;
    Duplicate(mask,kIdMask);
    for (uint16_t i=0;i<kTop;i+=64) {
        LoadAlign(id,ids+i); And(id,id,mask,all); StoreAlign(output+i,id,all);
    }
}

__simd_vf__ inline void CompactIds(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* output) {
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> id, mask, compact;
    Duplicate(mask,kIdMask);
    MaskReg selected;
    UnalignRegForStore store;
    ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < kTop; i += 64) {
        LoadAlign(id, ids + i);
        CompareScalar<uint32_t, CMPMODE::GE>(selected, id, kNeedsFetch, all);
        And(id,id,mask,all);
        Squeeze<uint32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(compact, id, selected);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(output, compact, store);
    }
    StoreUnAlignPost(output, store);
}
} // namespace kvip_incremental
#endif
