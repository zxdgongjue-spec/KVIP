#pragma once
#include "../../../common/lifetch/incremental_compact.h"
namespace lifetch_incremental {
constexpr uint32_t Top = 2048;
__simd_vf__ inline void Count(__ubuf__ uint16_t* mask, __ubuf__ uint32_t* output, uint32_t n) {
    using namespace MicroAPI;
    RegTensor<uint16_t> ones; RegTensor<uint32_t> total, sum;
    MaskReg selected;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    auto one = CreateMask<uint32_t, MaskPattern::VL1>();
    Duplicate(sum, uint32_t(0));
    for (uint16_t i=0; i<n; i+=128) {
        LoadAlign(selected, mask+i/8);
        Duplicate<uint16_t, MaskMergeMode::ZEROING>(ones, uint16_t(1), selected);
        ReduceSum(total, ones, all); Add(sum, sum, total, one);
    }
    StoreAlign(output, sum, one);
}
__simd_vf__ inline void Minimum(__ubuf__ uint16_t* scores, __ubuf__ uint32_t* output) {
    using namespace MicroAPI;
    RegTensor<uint16_t> values, minimum, reduced;
    auto all=CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(minimum, uint16_t(65535));
    for (uint16_t i=0;i<Top;i+=128) {
        LoadAlign(values,scores+i); Min(minimum,minimum,values,all);
    }
    ReduceMin(reduced,minimum,all);
    StoreAlign(reinterpret_cast<__ubuf__ uint16_t*>(output),reduced,all);
}
}
