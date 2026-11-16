#pragma once
#include "rank_threshold.h"
namespace kvip_prefetch_rank {
using namespace AscendC;
// Read-only complete TopK. Compact mode reduces the second rank selection to
// cold scores only; dense mode is the first implementation used for local A/B.
template<bool compact>
__simd_vf__ inline void PackCold(__ubuf__ uint32_t* ids, __ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* packed, __ubuf__ uint32_t* count,
    uint32_t resident, uint32_t valid) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,value,out,zero,sum,ones,total;
    RegTensor<uint16_t> s; MaskReg cold,bounded,take;UnalignRegForStore store;
    Duplicate(sum,uint32_t(0));Duplicate(zero,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<2048;i+=64) {
        LoadAlign(id,ids+i);LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(s,scores+i);
        CompareScalar<uint32_t,CMPMODE::GE>(cold,id,resident,all);
        CompareScalar<uint32_t,CMPMODE::LT>(bounded,id,valid,all);MaskAnd(take,cold,bounded,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
        if(compact) {
            Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,reinterpret_cast<RegTensor<uint32_t>&>(s),take);
            StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(packed,out,store);
        } else {
            Select(value,reinterpret_cast<RegTensor<uint32_t>&>(s),zero,take);StoreAlign(packed+i,value,all);
        }
    }
    if(compact)StoreUnAlignPost(packed,store);
    ReduceSum(total,sum,all);auto one=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(count,total,one);
}
__simd_vf__ inline void Narrow(__ubuf__ uint32_t* input,__ubuf__ uint16_t* output,uint32_t n,uint32_t padded) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> value,zero;Duplicate(zero,uint32_t(0));
    for(uint16_t i=0;i<padded;i+=64){uint32_t left=n>i?n-i:0;auto mask=UpdateMask<uint32_t>(left);
        LoadAlign(value,input+i);Select(value,value,zero,mask);
        StoreAlign<uint32_t,StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ uint32_t*>(output+i),value,all);}
}
// Greater scores precede boundary ties, so truncating to quota never drops a
// better score. Equal scores retain original TopK order, deterministically.
__simd_vf__ inline void Emit(__ubuf__ uint32_t* ids,__ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* output,__ubuf__ uint32_t* count,
    uint32_t resident,uint32_t valid,uint32_t threshold) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,out,sum,ones,total;RegTensor<uint16_t> s;
    MaskReg cold,bounded,take,above;UnalignRegForStore store;
    Duplicate(sum,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t pass=0;pass<2;++pass)for(uint16_t i=0;i<2048;i+=64){
        LoadAlign(id,ids+i);LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(s,scores+i);
        CompareScalar<uint32_t,CMPMODE::GE>(cold,id,resident,all);
        CompareScalar<uint32_t,CMPMODE::LT>(bounded,id,valid,all);MaskAnd(take,cold,bounded,all);
        if(pass==0)CompareScalar<uint32_t,CMPMODE::GT>(above,reinterpret_cast<RegTensor<uint32_t>&>(s),threshold,all);
        else CompareScalar<uint32_t,CMPMODE::EQ>(above,reinterpret_cast<RegTensor<uint32_t>&>(s),threshold,all);
        MaskAnd(take,take,above,all);Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,id,take);StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,out,store);
    }
    StoreUnAlignPost(output,store);ReduceSum(total,sum,all);auto one=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(count,total,one);
}
// scratch >= 14 KiB, output >= 8 KiB (ties are truncated after compaction).
// Scratch holds cold count and threshold banks. Optional clocks
// isolate pack, window rank, quota rank and emit without changing the helper.
__aicore__ inline uint32_t Select(LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,
    LocalTensor<uint32_t> output,LocalTensor<uint32_t> scratch,
    uint32_t resident,uint32_t valid,uint32_t rank,uint32_t quota,bool compact=true,
    uint64_t* clocks=nullptr) {
    auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto full=reinterpret_cast<__ubuf__ uint16_t*>(p+3072);
    auto cold=reinterpret_cast<__ubuf__ uint16_t*>(p+2048);
    auto cap=full+128;auto stats=p+3200;
    auto id=reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
    auto score=reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr());
    auto out=reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr());
    if(clocks)clocks[0]=GetSystemCycle();
    if(compact)PackCold<true>(id,score,p,stats,resident,valid);
    else PackCold<false>(id,score,p,stats,resident,valid);
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t n=scratch.GetValue(3200);
    if(clocks)clocks[1]=GetSystemCycle();
    if(!quota||!n){if(clocks)clocks[2]=clocks[3]=clocks[4]=clocks[1];return 0;}
    uint32_t threshold=0;
    // Last window accepts the full Top2048; its minimum cannot reject any input.
    if(rank<2048) {
        RegisterThreshold(score,full,8,2048-rank+1);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        threshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6144);
    }
    if(clocks)clocks[2]=GetSystemCycle();
    uint32_t capThreshold=0;
    if(!compact||n>quota){
        uint32_t padded=compact?(n+255)/256*256:2048;
        Narrow(p,cold,compact?n:2048,padded);PipeBarrier<PIPE_V>();
        RegisterThreshold(cold,cap,padded/256,padded-quota+1);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        capThreshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6272);
        if(capThreshold>threshold)threshold=capThreshold;
    }
    if(clocks)clocks[3]=GetSystemCycle();
    Emit(id,score,out,stats+8,resident,valid,threshold);
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t count=scratch.GetValue(3208);
    if(clocks)clocks[4]=GetSystemCycle();
    return count<quota?count:quota;
}
} // namespace kvip_prefetch_rank
