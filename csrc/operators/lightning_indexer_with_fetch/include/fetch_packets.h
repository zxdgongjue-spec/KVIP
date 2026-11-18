// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace kvip_fetch_packets {
#if KVIP_FETCH_PACKETS
__simd_callee__ inline void PackScore(AscendC::MicroAPI::RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* packet,
    AscendC::MicroAPI::MaskReg& all) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> limit, packed, one, zero, count;
    RegTensor<int16_t> sequence;
    RegTensor<uint32_t> total;
    MaskReg selected;
    auto first=CreateMask<uint32_t, MaskPattern::VL1>();
    auto first16=CreateMask<uint16_t, MaskPattern::VL16>();
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(limit, threshold);
    Compare<uint16_t, CMPMODE::GT>(selected, score, limit, all);
    Duplicate(one, uint16_t(1)); Duplicate(zero, uint16_t(0));
    Select(count, one, zero, selected); ReduceSum(total, count, all);
    StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(packet), total, first);
    Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed, score, selected);
    StoreAlign(packet + 16, packed, first16);
    Arange(sequence, int16_t(0));
    Squeeze<uint16_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,
        reinterpret_cast<RegTensor<uint16_t>&>(sequence), selected);
    StoreAlign(packet + 32, packed, first16);
}
#endif
}
