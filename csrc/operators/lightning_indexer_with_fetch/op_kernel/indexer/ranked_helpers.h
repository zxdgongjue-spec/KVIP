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
