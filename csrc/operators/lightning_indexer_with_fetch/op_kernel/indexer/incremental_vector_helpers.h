#pragma once
#include "../../../common/lifetch/incremental_compact.h"
namespace lifetch_incremental {
constexpr uint32_t Top = 2048;
#if KVIP_LARGE_BATCH_LI
// The general TopK consumes whole 256-score blocks. Only its final, partial
// block needs padding; every prior compact score is already initialized.
__simd_vf__ inline void PadFinalScoreBlock(__ubuf__ uint16_t* block, uint32_t valid) {
    using namespace MicroAPI;
    RegTensor<uint16_t> zero, values;
    Duplicate(zero, uint16_t(0));
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < 256; offset += 128) {
        auto keep = UpdateMask<uint16_t>(valid);
        LoadAlign(values, block + offset);
        Select(values, values, zero, keep);
        StoreAlign(block + offset, values, all);
    }
}
#endif
__simd_vf__ inline void Count(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* mask, __ubuf__ uint32_t* output, uint32_t n, uint16_t cutoff) {
    using namespace MicroAPI;
    RegTensor<uint16_t> ones, values, counts; RegTensor<uint32_t> total;
    MaskReg selected, above;
    auto all = CreateMask<uint16_t, MaskPattern::ALL>();
    auto one = CreateMask<uint32_t, MaskPattern::VL1>();
    Duplicate(counts, uint16_t(0));
    for (uint16_t i=0; i<n; i+=128) {
        LoadAlign(selected, mask+i/8);
        LoadAlign(values, raw+i);
        CompareScalar<uint16_t,CMPMODE::GT>(above,values,cutoff,all);
        MaskAnd(selected,selected,above,all);
        StoreAlign(mask+i/8,selected);
        Duplicate<uint16_t, MaskMergeMode::ZEROING>(ones, uint16_t(1), selected);
        Add(counts, counts, ones, all);
    }
    ReduceSum(total, counts, all);
    StoreAlign(output, total, one);
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
