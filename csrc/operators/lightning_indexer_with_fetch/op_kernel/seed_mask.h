// SPDX-License-Identifier: Apache-2.0
#pragma once
#if KVIP_SEED_MASK
#include "kernel_operator.h"
#include "context.h"

namespace rotating {
constexpr uint32_t kSeedChunk = kSeedMaskSlots;

__simd_vf__ inline void ResidentPredicates(__ubuf__ uint32_t* valid, __ubuf__ int32_t* reserved,
    __ubuf__ uint32_t* words, uint32_t size, bool hasValid, bool hasReserved) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint32_t> v;
    RegTensor<int32_t> r;
    uint32_t left = size;
    for (uint32_t i = 0; i < size; i += 64) {
        auto lanes = UpdateMask<uint32_t>(left);
        MaskReg invalid, victim, eligible;
        if (hasValid) LoadAlign(v, valid + i);
        else Duplicate(v, uint32_t(1));
        if (hasReserved) LoadAlign(r, reserved + i);
        else Duplicate(r, int32_t(-1));
        CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(invalid, v, uint32_t(0), lanes);
        CompareScalar<int32_t, AscendC::CMPMODE::GE>(victim, r, int32_t(0), lanes);
        MaskOr(eligible, invalid, victim, lanes);
        // Store the hardware predicate directly. No bit packing or scalar counts.
        StoreAlign(words + i / 8, eligible);
    }
}

// Stage the two immutable inputs independently of the init kernel's zero fills.
__aicore__ inline void LoadSeedInputs(__gm__ Context* ctx, uint32_t b, uint32_t resident,
    AscendC::LocalTensor<uint32_t> inputs) {
    using namespace AscendC;
    uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
    DataCopyExtParams copy{1, resident * uint32_t(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    GlobalTensor<uint32_t> source;
    if (ctx->hotValid) {
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid) + owner * resident);
        DataCopyPad(inputs, source, copy, pad);
    }
    if (ctx->reservedHot) {
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot) + b * resident);
        DataCopyPad(inputs[kSeedChunk], source, copy, pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
}

__aicore__ inline void PrepareSeedMask(__gm__ Context* ctx, uint32_t b, uint32_t resident,
    AscendC::LocalTensor<uint32_t> inputs, AscendC::LocalTensor<uint32_t> output) {
    using namespace AscendC;
    GlobalTensor<uint32_t> mask;
    mask.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->seedMask) + b * kSeedMaskWords);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    ResidentPredicates(reinterpret_cast<__ubuf__ uint32_t*>(inputs.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(inputs[kSeedChunk].GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr()), resident,
        ctx->hotValid != 0, ctx->reservedHot != 0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(mask, output, ((resident + 63) / 64) * 8);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
} // namespace rotating
#endif
