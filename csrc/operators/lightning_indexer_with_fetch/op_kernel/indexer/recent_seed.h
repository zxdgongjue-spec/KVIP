// SPDX-License-Identifier: Apache-2.0
#pragma once
// Seed the exact ABC range supplied by the serving binding.
namespace kvip_recent_index {
struct Range {uint32_t begin,end;};
__aicore__ inline Range RecentRange(__gm__ rotating::Context* ctx,uint32_t batch,uint32_t queryEnd) {
    if(!ctx->tailBounds)return {0,0};
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+batch*2;
    uint32_t begin=bounds[0],end=bounds[1],resident=ctx->shape.resident;
    if(begin<resident)begin=resident;
    uint32_t capacity=ctx->outputStride>resident?ctx->outputStride-resident:0;
    if(end>begin+capacity)end=begin+capacity;
    if(end>ctx->shape.history)end=ctx->shape.history;
    if(end>queryEnd)end=queryEnd;
    return begin<end?Range{begin,end}:Range{0,0};
}
__aicore__ inline Range Tiles(Range range,uint32_t shard=0,uint32_t shards=1) {
    if(range.begin==range.end)return {0,0};
    uint32_t begin=range.begin/128,end=(range.end+127)/128,count=end-begin;
    return {begin+count*shard/shards,begin+count*(shard+1)/shards};
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Remap(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* previous,
                  uint32_t begin, uint32_t count) {
    for (uint32_t i = threadIdx.x; i < 2048; i += 512) {
        uint32_t index = indices[i];
        indices[i] = index < 2048 ? previous[index] :
            (index - 2048 < count ? begin + index - 2048 : UINT32_MAX);
    }
}

// One-pass resident + recent TopK produces compact indices. Restore the
// physical recent index; padding is kept invalid if ever selected.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void RemapJoined(__ubuf__ uint32_t* indices, uint32_t resident,
                        uint32_t begin, uint32_t count) {
    for (uint32_t i = threadIdx.x; i < 2048; i += 512) {
        uint32_t index = indices[i];
        if (index >= resident)
            indices[i] = index - resident < count ? begin + index - resident : UINT32_MAX;
    }
}

// Native LI uses two bitmap bits per score; operate through its predicate format.
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
} // namespace kvip_recent_index
