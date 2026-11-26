/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

#pragma once

#include "kernel_operator.h"

namespace topkb16gather {

template<typename T>
__simd_vf__ void HistogramsHighVFImpl(__ubuf__ uint32_t* histogramsBuf,
                                      __ubuf__ uint16_t* inputBuf,
                                      uint16_t vfLoop, bool init)
{
    using namespace AscendC;
    MicroAPI::MaskReg maskRegB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRegB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRegB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图coutReg0 0-127 coutReg1 128-255
    MicroAPI::RegTensor<uint16_t> coutReg0;
    MicroAPI::RegTensor<uint16_t> coutReg1;
    MicroAPI::Duplicate(coutReg0, 0);
    MicroAPI::Duplicate(coutReg1, 0);

    MicroAPI::RegTensor<uint32_t> coutReg0U32Even;
    MicroAPI::RegTensor<uint32_t> coutReg0U32Odd;
    MicroAPI::RegTensor<uint32_t> coutReg1U32Even;
    MicroAPI::RegTensor<uint32_t> coutReg1U32Odd;

    MicroAPI::RegTensor<uint16_t> vregHigh;
    MicroAPI::RegTensor<uint16_t> vregLow;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TO_UINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TO_UINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh, inputBuf + i * 256);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(coutReg0,
                                                                    (MicroAPI::RegTensor<uint8_t>&)vregHigh,
                                                                    maskRegB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(coutReg1,
                                                                    (MicroAPI::RegTensor<uint8_t>&)vregHigh,
                                                                    maskRegB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TO_UINT32_EVEN>(coutReg0U32Even, coutReg0, maskRegB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TO_UINT32_ODD>(coutReg0U32Odd, coutReg0, maskRegB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TO_UINT32_EVEN>(coutReg1U32Even, coutReg1, maskRegB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TO_UINT32_ODD>(coutReg1U32Odd, coutReg1, maskRegB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf,
                                                            coutReg0U32Even, coutReg0U32Odd, maskRegB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128,
                                                            coutReg1U32Even, coutReg1U32Odd, maskRegB32);
}

}  // namespace topkb16gather
