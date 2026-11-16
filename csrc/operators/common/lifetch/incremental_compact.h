#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
namespace lifetch_incremental {
using namespace AscendC;
constexpr uint32_t K = 2048;
// LI's uint16 predicate storage: 128 scores per 32-byte mask (two bits/score).
// Consume the native mask directly; vector Squeeze packs 128 lanes at a time.
__simd_vf__ inline void Compact(__ubuf__ uint16_t* raw, __ubuf__ uint16_t* bitmap,
    __ubuf__ uint16_t* packed, __ubuf__ uint16_t* mapping, uint32_t added, uint16_t base = 0) {
    using namespace MicroAPI;
    RegTensor<uint16_t> values,result;RegTensor<int16_t> indices;
    MaskReg selected;UnalignRegForStore scoreStore,idStore;
    ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<added;i+=128){
        LoadAlign(selected,bitmap+i/8);
        LoadAlign(values,raw+i);
        Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(result,values,selected);
        StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(packed,result,scoreStore);
    }
    StoreUnAlignPost(packed,scoreStore);
    ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<added;i+=128){
        LoadAlign(selected,bitmap+i/8);Arange(indices,int16_t(base+i));
        Squeeze<uint16_t,MicroAPI::GatherMaskMode::STORE_REG>(result,reinterpret_cast<RegTensor<uint16_t>&>(indices),selected);
        StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(mapping,result,idStore);
    }
    StoreUnAlignPost(mapping,idStore);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void MapBitmap(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* old,__ubuf__ uint16_t* addedIds,uint32_t base){
    for(uint32_t i=threadIdx.x;i<K;i+=256){uint32_t id=selected[i];selected[i]=id<K?old[id]:base+addedIds[id-K];}
}
}
