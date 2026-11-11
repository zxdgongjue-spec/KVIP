// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef V9_PRUNED_TOPK
#define V9_PRUNED_TOPK 0
#endif

#if V9_PRUNED_TOPK
namespace kvip_pruned {
// The previous K entries precede this cold block in the original merge. A
// cold key equal to their minimum cannot displace any of them: TopK emits GT
// in input order followed by EQ in input order. Keep only strict winners,
// retaining their original order and positions for the final index gather.
__simd_vf__ inline void CompactPositions(__ubuf__ uint16_t* positions,
    __ubuf__ uint16_t* values,__ubuf__ uint16_t* scratch,uint32_t topK,uint32_t count) {
    using namespace AscendC::MicroAPI;
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    RegTensor<uint16_t> minimum,input,reduced,threshold,packed;
    Duplicate(minimum,uint16_t(0xffff));
    for(uint16_t i=0;i<topK;i+=128) {
        LoadAlign(input,values+i);Min(minimum,minimum,input,all);
    }
    ReduceMin(reduced,minimum,all);StoreAlign(scratch,reduced,all);
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    LoadAlign<uint16_t,LoadDist::DIST_BRC_B16>(threshold,scratch);
    ClearSpr<AscendC::SpecialPurposeReg::AR>();UnalignRegForStore store;
    uint32_t remaining=count;
    for(uint16_t i=0;i<count;i+=128) {
        auto valid=UpdateMask<uint16_t>(remaining);
        RegTensor<int16_t> offsets;Arange(offsets,int16_t(i));
        LoadAlign(input,values+topK+i);
        MaskReg keep;Compare<uint16_t,CMPMODE::GT>(keep,input,threshold,valid);
        Squeeze<uint16_t,AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,
            reinterpret_cast<RegTensor<uint16_t>&>(offsets),keep);
        StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
    }
    StoreUnAlignPost(positions,store);
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
}

// Positions increase monotonically and position[i]>=i. Loading a complete
// vector before storing it therefore makes this left-compaction in-place.
__simd_vf__ inline void GatherScores(__ubuf__ uint16_t* values,
    __ubuf__ uint16_t* positions,uint32_t count,uint32_t padded) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> offsets,keys,zero;Duplicate(zero,uint16_t(0));
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    uint32_t remaining=count;
    for(uint16_t i=0;i<padded;i+=128) {
        auto valid=UpdateMask<uint16_t>(remaining);
        LoadAlign(offsets,positions+i);Gather(keys,values,offsets,valid);
        Select(keys,keys,zero,valid);
        StoreAlign(values+i,keys,all);
        LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    }
}

__simd_vf__ inline void RestorePositions(__ubuf__ uint16_t* selected,
    __ubuf__ uint16_t* positions,uint32_t topK) {
    using namespace AscendC::MicroAPI;
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    RegTensor<uint16_t> index,offset,mapped;
    for(uint16_t i=0;i<topK;i+=128) {
        LoadAlign(index,selected+i);
        MaskReg cold;Compares<uint16_t,CMPMODE::GE>(cold,index,uint16_t(topK),all);
        Adds(offset,index,uint16_t(0-topK),all);
        Gather(mapped,positions,offset,cold);Adds(mapped,mapped,uint16_t(topK),cold);
        Select(index,mapped,index,cold);StoreAlign(selected+i,index,all);
    }
}

__aicore__ inline void RestoreResident(GM_ADDR prefix,uint32_t query,
    LocalTensor<uint16_t> scores,LocalTensor<uint32_t> indices) {
    constexpr uint32_t entries=4*2048,rowBytes=entries*6+4*256;
    GlobalTensor<uint16_t> values;GlobalTensor<uint32_t> ids;
    values.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(prefix+(query/4)*rowBytes));
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(prefix+(query/4)*rowBytes+entries*2));
    DataCopy(scores,values[(query%4)*2048],2048);
    DataCopy(indices,ids[(query%4)*2048],2048);
    SetFlag<HardEvent::MTE2_V>(5);WaitFlag<HardEvent::MTE2_V>(5);
}
} // namespace kvip_pruned
#endif
