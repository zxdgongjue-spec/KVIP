// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef KVIP_FINAL_TOPK_PRUNE
#define KVIP_FINAL_TOPK_PRUNE 0
#endif
#if KVIP_FINAL_TOPK_PRUNE != 0 && KVIP_FINAL_TOPK_PRUNE != 1
#error "KVIP_FINAL_TOPK_PRUNE must be 0 or 1"
#endif

#if KVIP_FINAL_TOPK_PRUNE && !(defined(V9_UBMEM) && V9_UBMEM)
namespace kvip_final_topk {
// Stable block-local offsets. Equal keys must survive: the prefetch predicate
// is strict >, whereas an exact final selection may discard only keys < tau.
__simd_vf__ inline void CompactPositions(__ubuf__ uint16_t* input,
    __ubuf__ uint16_t* output, uint32_t count, uint16_t threshold) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> values, packed;
    RegTensor<int16_t> positions;
    MaskReg selected;
    UnalignRegForStore store;
    ClearSpr<AscendC::SpecialPurposeReg::AR>();
    uint32_t remaining = count;
    for (uint16_t offset = 0; offset < count; offset += 128) {
        auto valid = UpdateMask<uint16_t>(remaining);
        LoadAlign(values, input + offset);
        Arange(positions, int16_t(offset));
        CompareScalar<uint16_t, CMPMODE::GE>(selected, values, threshold, valid);
        Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,
            reinterpret_cast<RegTensor<uint16_t>&>(positions), selected);
        StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, packed, store);
    }
    StoreUnAlignPost(output, store);
}

// The source starts at K; the destination starts at previousCount <= K.
// Stable increasing offsets make the gather safe in place, including the
// all-surviving case. Rounded tail writes remain inside K+trunk capacity.
__simd_vf__ inline void GatherScores(__ubuf__ uint16_t* source,
    __ubuf__ uint16_t* positions, __ubuf__ uint16_t* destination, uint32_t count) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> offsets, values;
    UnalignRegForStore store;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += 128) {
        auto valid = UpdateMask<uint16_t>(remaining);
        LoadAlign(offsets, positions + offset);
        Gather(values, source, offsets, valid);
        StoreUnAlign(destination, values, store, uint32_t(128));
    }
    StoreUnAlignPost(destination, store, 0);
}

// Duplicate(LocalTensor) requires an aligned destination. A compacted count
// is arbitrary, so preserve its final valid lanes and mask only the padding.
__simd_vf__ inline void PadScores(__ubuf__ uint16_t* input,
    uint32_t count, uint32_t aligned) {
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> values, zero;
    RegTensor<int16_t> positions;
    Duplicate(zero,uint16_t(0));
    for (uint32_t offset = count / 128 * 128; offset < aligned; offset += 128) {
        LoadAlign(values,input+offset);
        Arange(positions,int16_t(offset));
        MaskReg padding;
        CompareScalar<uint16_t,CMPMODE::GE>(padding,
            reinterpret_cast<RegTensor<uint16_t>&>(positions),uint16_t(count),all);
        Select(values,zero,values,padding);
        StoreAlign(input+offset,values,all);
    }
}

// The caller copies output to history only after this VF has completed. The
// radix selector supplies GT-then-EQ order; do not sort by score/logical ID.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void RemapSelected(
    __ubuf__ uint16_t* selected, __ubuf__ uint16_t* positions,
    __ubuf__ uint32_t* previous, __ubuf__ uint32_t* output,
    __ubuf__ uint16_t* input, __ubuf__ uint16_t* scores,
    uint32_t previousCount, uint32_t count, uint32_t blockBegin, bool selectedK) {
    for (uint32_t i = threadIdx.x; i < count; i += 256) {
        uint32_t index = selectedK ? selected[i] : i;
        output[i] = index < previousCount ? previous[index] :
            blockBegin + positions[index - previousCount];
        if (!selectedK) scores[i] = input[i];
    }
}
} // namespace kvip_final_topk
#endif
