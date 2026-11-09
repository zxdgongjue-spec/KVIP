#pragma once
namespace rotating {
// The demo's SIMD compaction keeps hot TopK entries out of the SIMT worklist.
// Runtime work scales with selected cold entries, not all four 2048-entry rows.
__simd_vf__ inline void ColdPositions(uint32_t rt_history,uint32_t rt_resident,__ubuf__ uint32_t* selected,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* positions,__ubuf__ uint32_t* count) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> ids,flags,total,packed,errors;
    RegTensor<int32_t> signedIndex;
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(signedIndex);
    Arange(signedIndex,int32_t(0));Duplicate(total,uint32_t(0));Duplicate(errors,uint32_t(0));
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<kSlots;i+=64) {
        LoadAlign(ids,selected+i);StoreAlign(locations+i,ids,all);
        MaskReg invalid,notPadding,valid;
        CompareScalar<uint32_t,CMPMODE::NE>(notPadding,ids,UINT32_MAX,all);
        CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(rt_history),all);
        MaskAnd(invalid,invalid,notPadding,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(rt_resident),all);
        CompareScalar<uint32_t,CMPMODE::LT>(valid,ids,uint32_t(rt_history),all);
        MaskAnd(cold,cold,valid,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),cold);
        Add(total,total,flags,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,cold);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(positions,store);ReduceSum(total,total,all);StoreAlign(count,total,lane);
    ReduceSum(errors,errors,all);StoreAlign(count+8,errors,lane);
}
__simt_callee__ inline uint32_t SparseHash(uint32_t token) {
    return (token*0x9e3779b9U)>>19;
}
}

