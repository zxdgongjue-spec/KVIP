#pragma once
#include "../../../common/lifetch/recent_index.h"
namespace kvip_recent_index {
__aicore__ inline Range RecentRange(__gm__ rotating::Context* ctx,uint32_t b,uint32_t end) {
    auto address=ctx->thresholdBounds ? ctx->thresholdBounds : ctx->tailBounds;
    if(!address)return {0,0};
    auto bounds=reinterpret_cast<__gm__ int32_t*>(address)+b*2;
    return Clip(bounds[0],bounds[1],ctx->shape.resident,ctx->shape.history,end);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void RemapUnified(__ubuf__ uint32_t* ids,uint32_t resident,uint32_t begin,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<2048;i+=512) {
        uint32_t id=ids[i];
        if(id>=resident)ids[i]=id-resident<count?begin+id-resident:UINT32_MAX;
    }
}
__simd_vf__ inline void ExcludeSeeded(__ubuf__ uint16_t* bitmap,uint32_t base,uint32_t n,uint32_t begin,uint32_t end) {
    using namespace AscendC::MicroAPI;
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    for(uint32_t i=0;i<n;i+=128) {
        if(base+i>=end || base+i+128<=begin)continue;
        RegTensor<int16_t> sequence;RegTensor<uint16_t> offsets;
        Arange(sequence,int16_t(0));offsets=reinterpret_cast<RegTensor<uint16_t>&>(sequence);
        MaskReg selected,before,after,keep;
        uint16_t lo=begin>base+i?uint16_t(begin-base-i):0;
        uint16_t hi=end<base+i+128?uint16_t(end-base-i):128;
        CompareScalar<uint16_t,CMPMODE::LT>(before,offsets,lo,all);
        CompareScalar<uint16_t,CMPMODE::GE>(after,offsets,hi,all);
        MaskOr(keep,before,after,all);LoadAlign(selected,bitmap+i/8);
        MaskAnd(selected,selected,keep,all);StoreAlign(bitmap+i/8,selected);
    }
}
}

namespace lifetch_ranked {
// The native selector emits strict winners followed by boundary ties. New
// strict winners displace the last ties. Shift the retained IDs backwards
// before inserting new IDs, so the reads never observe an overwritten ID.
__simd_vf__ inline void InsertAboveBoundary(__ubuf__ uint32_t* ids, __ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* freshIds, __ubuf__ uint16_t* freshScores,
    uint32_t strict, uint32_t fresh) {
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero;
    Duplicate(zero, uint32_t(0));
    for (int32_t block = 2048 - 64; block >= int32_t(strict & ~63U); block -= 64) {
        RegTensor<int32_t> seq;
        Arange(seq, block);
        auto& lane = reinterpret_cast<RegTensor<uint32_t>&>(seq);
        RegTensor<uint32_t> index, oldId, newId, id;
        MaskReg moved, write, add;
        CompareScalar<uint32_t, CMPMODE::GE>(moved, lane, strict + fresh, all);
        CompareScalar<uint32_t, CMPMODE::GE>(write, lane, strict, all);
        MaskXor(add, write, moved, all);
        Adds(index, lane, uint32_t(0) - fresh, all);
        Select(index, index, zero, moved);
        Gather(oldId, ids, index, moved);
        Adds(index, lane, uint32_t(0) - strict, all);
        Select(index, index, zero, add);
        Gather(newId, freshIds, index, add);
        Select(id, newId, oldId, add);
        LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
        StoreAlign(ids + block, id, write);
    }
    // All surviving ties have the same old score. Copy only the new scores.
    UnalignRegForStore store;
    auto dst = scores + strict;
    uint32_t remaining = fresh;
    for (uint16_t i = 0; i < fresh; i += 128) {
        auto valid = UpdateMask<uint16_t>(remaining);
        RegTensor<uint16_t> values, packed;
        LoadAlign(values, freshScores+i);
        if (!i) ClearSpr<SpecialPurposeReg::AR>();
        Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, values, valid);
        StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(dst, packed, store);
    }
    StoreUnAlignPost(dst, store);
}

// Only padded scores can enter the native histogram. Preserve the actual
// candidates in the head of the last vector and clear at most 255 scores.
__simd_vf__ inline void ClearRankPadding(__ubuf__ uint16_t* scores, uint32_t n) {
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> zero;
    Duplicate(zero, uint16_t(0));
    for (uint32_t i=n & ~127U; i < ((n+255U) & ~255U); i+=128) {
        RegTensor<int16_t> lane;
        Arange(lane, int16_t(i));
        MaskReg padding;
        CompareScalar<uint16_t, CMPMODE::GE>(padding, reinterpret_cast<RegTensor<uint16_t>&>(lane), uint16_t(n), all);
        StoreAlign(scores+i, zero, padding);
    }
}

// Drop safe-hot tags and padding from the transport list, retaining TopK order.
// The posting owner still checks live reservations and prior submissions.
__simd_vf__ inline void PackOffers(__ubuf__ uint32_t* ids, __ubuf__ uint32_t* out,
    __ubuf__ uint32_t* count, uint32_t validEnd) {
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    auto one = CreateMask<uint32_t, MaskPattern::VL1>();
    RegTensor<uint32_t> id, packed, ones, sum, total;
    MaskReg eligible;
    UnalignRegForStore store;
    Duplicate(sum, uint32_t(0));
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < 2048; i += 64) {
        LoadAlign(id, ids+i);
        CompareScalar<uint32_t, CMPMODE::LT>(eligible, id, validEnd, all);
        Duplicate<uint32_t, MaskMergeMode::ZEROING>(ones, uint32_t(1), eligible);
        Add(sum, sum, ones, all);
        Squeeze<uint32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, id, eligible);
        StoreUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(out, packed, store);
    }
    StoreUnAlignPost(out, store);
    ReduceSum(total, sum, all);
    StoreAlign(count, total, one);
}

// As in the ranked reference's RestorePublicIds, keep the known hot slot
// when removing the private tag. This port uses one tag for hot and ABC.
template<bool PrepareSlots = false>
__simd_vf__ inline void DecodePublicIds(__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* slots = nullptr, uint32_t resident = 0) {
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> id, mask, raw;
    Duplicate(mask, uint32_t(0x7fffffff));
    for (uint16_t i = 0; i < 2048; i += 64) {
        MaskReg padding;
        LoadAlign(id, ids+i);
        CompareScalar<uint32_t, CMPMODE::EQ>(padding, id, UINT32_MAX, all);
        And(raw, id, mask, all);
        if constexpr (PrepareSlots) {
            RegTensor<uint32_t> invalid, slot;
            MaskReg tagged, bounded, known;
            Duplicate(invalid, UINT32_MAX);
            Compare<uint32_t, CMPMODE::NE>(tagged, id, raw, all);
            CompareScalar<uint32_t, CMPMODE::LT>(bounded, raw, resident, all);
            MaskAnd(known, tagged, bounded, all);
            Select(slot, raw, invalid, known);
            StoreAlign(slots + i, slot, all);
        }
        Select(raw, id, raw, padding);
        StoreAlign(ids+i, raw, all);
    }
}
}
