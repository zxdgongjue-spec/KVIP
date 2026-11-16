#pragma once
#include "rank_threshold.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_atomic_functions.h"
namespace kvip_prefetch_rank {
using namespace AscendC;
__aicore__ inline uint64_t CoveredAddress(uint64_t incremental,uint32_t batch,uint32_t history,uint32_t b) {
    return incremental+uint64_t(batch)*(512+4*2048*6+1024)+uint64_t(b)*((history+255)/256*32);
}
__simt_callee__ inline uint64_t CoveredAddressSimt(uint64_t incremental,uint32_t batch,uint32_t history,uint32_t b) {
    return incremental+uint64_t(batch)*(512+4*2048*6+1024)+uint64_t(b)*((history+255)/256*32);
}
// Compress hot valid / reserved metadata once per round, not per Q/window.
// Native mask has four bits per uint32 lane; two Pack operations make one bit.
__simd_vf__ inline void HotNativeBits(__ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,
    uint32_t count,bool hasValid,bool hasReserved) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> v;RegTensor<int32_t> r;MaskReg live,free,hit,half,packed;
    for(uint16_t i=0;i<count;i+=64){
        if(hasValid)LoadAlign(v,valid+i);else Duplicate(v,uint32_t(1));
        if(hasReserved)LoadAlign(r,reserved+i);else Duplicate(r,int32_t(-1));
        CompareScalar<uint32_t,CMPMODE::NE>(live,v,uint32_t(0),all);
        CompareScalar<int32_t,CMPMODE::LT>(free,r,int32_t(0),all);MaskAnd(hit,live,free,all);
        Pack(half,hit);Pack(packed,half);
        StoreAlign(reinterpret_cast<__ubuf__ uint8_t*>(valid)+i/2,packed);
    }
}
__simd_vf__ inline void CompactHotBits(__ubuf__ uint32_t* native,__ubuf__ uint32_t* output,uint32_t count) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> value,packed,index,mask7;RegTensor<int32_t> sequence;MaskReg take;
    Arange(sequence,int32_t(0));Duplicate(mask7,uint32_t(7));
    And(index,reinterpret_cast<RegTensor<uint32_t>&>(sequence),mask7,all);
    CompareScalar<uint32_t,CMPMODE::LT>(take,index,uint32_t(2),all);
    UnalignRegForStore store;ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<count/8;i+=64){
        LoadAlign(value,native+i);Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,value,take);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,packed,store);
    }
    StoreUnAlignPost(output,store);
}
// Existing init scratch is 16 KiB. Metadata reads total 128 KiB at resident=16K.
__aicore__ inline void InitializeCovered(LocalTensor<uint32_t> scratch,__gm__ uint32_t* target,
    __gm__ uint32_t* hot,__gm__ int32_t* reserved,uint32_t resident,uint32_t history) {
    GlobalTensor<uint32_t> dst,valid,slots;dst.SetGlobalBuffer(target);
    if(hot)valid.SetGlobalBuffer(hot);if(reserved)slots.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(reserved));
    Duplicate(scratch,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    DataCopy(dst,scratch,(history+255)/256*8);
    // Fence the scalar issuer before reusing scratch for the next MTE2.
    // MTE3_V alone orders V, but allows the next input DMA to overwrite it.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID4);WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
    for(uint32_t base=0;base<resident;base+=2048){
        uint32_t count=resident-base;if(count>2048)count=2048;
        if(hot)DataCopy(scratch,valid[base],count);
        if(reserved)DataCopy(scratch[2048],slots[base],count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
        HotNativeBits(p,reinterpret_cast<__ubuf__ int32_t*>(p+2048),count,hot!=nullptr,reserved!=nullptr);
        PipeBarrier<PIPE_V>();CompactHotBits(p,p+512,count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID4);WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
        DataCopy(dst[base/32],scratch[512],count/32);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID4);WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
    }
}
// Same uncached indexed-load pattern as FastService::GatherCandidateMap.
// Predecessor service publication must complete before this read. The map
// includes in-flight admissions and ABC-local tags, not only completed copies.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void MissIds(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* eligible,
    __gm__ uint32_t* map,__gm__ uint32_t* hotValid,__gm__ int32_t* reserved,
    uint32_t resident,uint32_t valid) {
    for(uint32_t i=threadIdx.x;i<2048;i+=256) {
        uint32_t token=ids[i];bool miss=false;
        if(token<valid) {
            bool local=token<resident && (!hotValid || asc_ldcg(hotValid+token)!=0)
                && (!reserved || asc_ldcg(reserved+token)<0);
            miss=!local && asc_ldcg(map+token)==UINT32_MAX;
        }
        eligible[i]=miss?token:UINT32_MAX;
    }
}
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
// One bit per physical Index position: valid local or already admitted.
// Packed scores and eligibility use disjoint banks; the full TopK stays intact.
__simd_vf__ inline void PackUncovered(__ubuf__ uint32_t* ids,__ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* covered,__ubuf__ uint32_t* eligible,
    __ubuf__ uint32_t* packed,__ubuf__ uint32_t* count,uint32_t valid) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,safe,index,shift,bits,zero,one,mask31,invalid,kept,out,sum,ones,total;
    RegTensor<uint16_t> score;MaskReg bounded,miss,take;UnalignRegForStore store;
    Duplicate(zero,uint32_t(0));Duplicate(one,uint32_t(1));Duplicate(mask31,uint32_t(31));
    Duplicate(invalid,UINT32_MAX);Duplicate(sum,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<2048;i+=64){
        LoadAlign(id,ids+i);LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(score,scores+i);
        CompareScalar<uint32_t,CMPMODE::LT>(bounded,id,valid,all);Select(safe,id,zero,bounded);
        ShiftRights(index,safe,int16_t(5),all);And(shift,safe,mask31,all);
        Gather(bits,covered,index,all);ShiftRight(bits,bits,reinterpret_cast<RegTensor<int32_t>&>(shift),all);And(bits,bits,one,all);
        CompareScalar<uint32_t,CMPMODE::EQ>(miss,bits,uint32_t(0),all);MaskAnd(take,bounded,miss,all);
        Select(kept,id,invalid,take);StoreAlign(eligible+i,kept,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,reinterpret_cast<RegTensor<uint32_t>&>(score),take);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(packed,out,store);
    }
    StoreUnAlignPost(packed,store);ReduceSum(total,sum,all);
    auto first=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(count,total,first);
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
    uint64_t* clocks=nullptr,uint32_t minimum=0,uint32_t packedOffset=0) {
    auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto full=reinterpret_cast<__ubuf__ uint16_t*>(p+3072);
    auto cold=reinterpret_cast<__ubuf__ uint16_t*>(p+2048);
    auto cap=full+128;auto stats=p+3200;
    auto id=reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
    auto score=reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr());
    auto out=reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr());
    if(clocks)clocks[0]=GetSystemCycle();
    if(!packedOffset) {
        if(compact)PackCold<true>(id,score,p,stats,resident,valid);
        else PackCold<false>(id,score,p,stats,resident,valid);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t n=scratch.GetValue(3200);
    if(clocks)clocks[1]=GetSystemCycle();
    if(!quota||!n){if(clocks)clocks[2]=clocks[3]=clocks[4]=clocks[1];return 0;}
    minimum=minimum<quota?minimum:quota;
    uint32_t threshold=0;
    // Last window accepts the full Top2048; its minimum cannot reject any input.
    if(rank<2048 && !(minimum && n<=minimum)) {
        RegisterThreshold(score,full,8,2048-rank+1);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        threshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6144);
    }
    if(clocks)clocks[2]=GetSystemCycle();
    uint32_t capThreshold=0;
    uint32_t padded=compact?(n+255)/256*256:2048;
    bool capNeeded=!compact||n>quota;
    // Reuse the packed cold scores for both ranks; no additional sort or buffer.
    if(capNeeded || (minimum && n>minimum && threshold)) {
        Narrow(p+packedOffset,cold,compact?n:2048,padded);PipeBarrier<PIPE_V>();
    }
    if(capNeeded){
        RegisterThreshold(cold,cap,padded/256,padded-quota+1);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        capThreshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6272);
    }
    // If Top32 already dominates the window threshold, at least 32 candidates
    // qualify. Top16 cannot change the result; avoid that selection entirely.
    if(minimum && n>minimum && threshold>capThreshold) {
        RegisterThreshold(cold,full,padded/256,padded-minimum+1);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        uint32_t floorThreshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6144);
        if(floorThreshold<threshold)threshold=floorThreshold;
    }
    if(capThreshold>threshold)threshold=capThreshold;
    if(clocks)clocks[3]=GetSystemCycle();
    Emit(id,score,out,stats+8,resident,valid,threshold);
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t count=scratch.GetValue(3208);
    if(clocks)clocks[4]=GetSystemCycle();
    return count<quota?count:quota;
}
// Uses 22 KiB of the existing dead TopK scratch (no additional UB allocation).
// Full scores remain unchanged: only quota/floor selection uses miss IDs.
__aicore__ inline uint32_t SelectMiss(LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,
    LocalTensor<uint32_t> output,LocalTensor<uint32_t> scratch,
    __gm__ uint32_t* map,__gm__ uint32_t* hotValid,__gm__ int32_t* reserved,
    uint32_t resident,uint32_t valid,uint32_t rank,uint32_t quota,
    uint64_t* clocks=nullptr,uint32_t minimum=0) {
    uint64_t begin=clocks?GetSystemCycle():0;
    auto eligible=scratch[3584];
    Simt::VF_CALL<MissIds>(Simt::Dim3{256,1,1},
        reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(eligible.GetPhyAddr()),
        map,hotValid,reserved,resident,valid);
    PipeBarrier<PIPE_V>();
    uint32_t count=Select(eligible,scores,output,scratch,0,valid,rank,quota,true,clocks,minimum);
    if(clocks)clocks[0]=begin;
    return count;
}
// scratch >= 25 KiB, output >= 16 KiB. Bitmap bank dies before rank selection.
// Reading the complete 16-KiB bitmap is included in clocks[0..1].
__aicore__ inline uint32_t SelectBitmap(LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,
    LocalTensor<uint32_t> output,LocalTensor<uint32_t> scratch,
    __gm__ uint32_t* covered,uint32_t history,uint32_t valid,uint32_t rank,uint32_t quota,
    uint64_t* clocks=nullptr,uint32_t minimum=0) {
    uint64_t begin=clocks?GetSystemCycle():0;
    GlobalTensor<uint32_t> bits;bits.SetGlobalBuffer(covered);
    DataCopy(scratch,bits,(history+255)/256*8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto eligible=output[2048];
    PackUncovered(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr()),p,
        reinterpret_cast<__ubuf__ uint32_t*>(eligible.GetPhyAddr()),p+4096,p+6144,valid);
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t n=scratch.GetValue(6144);scratch.SetValue(3200,n);
    SetFlag<HardEvent::S_V>(EVENT_ID4);WaitFlag<HardEvent::S_V>(EVENT_ID4);
    uint32_t count=Select(eligible,scores,output,scratch,0,valid,rank,quota,true,clocks,minimum,4096);
    if(clocks)clocks[0]=begin;
    return count;
}

// Private incremental IDs retain stable local residency from the prefix.
// Both tags are outside the supported history range; UINT32_MAX stays padding.
constexpr uint32_t kHotTag=0x80000000U,kTailTag=0x40000000U,kTokenMask=0x3fffffffU;
__simd_vf__ inline void TagResidentHalf(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* hot,
    __ubuf__ int32_t* reserved,uint32_t base,uint32_t size,uint32_t tailBegin,uint32_t tailEnd) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,offset,safe,zero,valid,tagged,hotTag,tailTag;
    RegTensor<int32_t> res;MaskReg ge,lt,inside,live,free,take,tail;
    Duplicate(zero,uint32_t(0));Duplicate(hotTag,kHotTag);Duplicate(tailTag,kTailTag);
    for(uint16_t i=0;i<2048;i+=64) {
        LoadAlign(id,ids+i);
        CompareScalar<uint32_t,CMPMODE::GE>(ge,id,base,all);CompareScalar<uint32_t,CMPMODE::LT>(lt,id,base+size,all);MaskAnd(inside,ge,lt,all);
        Adds(offset,id,uint32_t(0-base),all);Select(safe,offset,zero,inside);
        Gather(valid,hot,safe,all);Gather(res,reserved,safe,all);
        CompareScalar<uint32_t,CMPMODE::NE>(live,valid,uint32_t(0),all);CompareScalar<int32_t,CMPMODE::LT>(free,res,int32_t(0),all);
        MaskAnd(take,live,free,all);MaskAnd(take,take,inside,all);
        Or(tagged,id,hotTag,all);Select(tagged,tagged,id,take);
        CompareScalar<uint32_t,CMPMODE::GE>(ge,id,tailBegin,all);CompareScalar<uint32_t,CMPMODE::LT>(lt,id,tailEnd,all);MaskAnd(tail,ge,lt,all);
        Or(offset,id,tailTag,all);Select(tagged,offset,tagged,tail);StoreAlign(ids+i,tagged,all);
    }
}
__simd_vf__ inline void RestorePublicIds(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* locations,uint32_t resident) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,raw,mask,hot,local,invalid,slot;MaskReg bounded,tagged,padding,known;
    Duplicate(mask,kTokenMask);Duplicate(hot,kHotTag);Duplicate(invalid,UINT32_MAX);
    for(uint16_t i=0;i<2048;i+=64) {
        LoadAlign(id,ids+i);And(raw,id,mask,all);And(local,id,hot,all);
        CompareScalar<uint32_t,CMPMODE::NE>(tagged,local,uint32_t(0),all);CompareScalar<uint32_t,CMPMODE::LT>(bounded,raw,resident,all);MaskAnd(known,tagged,bounded,all);
        Select(slot,raw,invalid,known);StoreAlign(locations+i,slot,all);
        CompareScalar<uint32_t,CMPMODE::EQ>(padding,id,UINT32_MAX,all);Select(raw,id,raw,padding);StoreAlign(ids+i,raw,all);
    }
}

} // namespace kvip_prefetch_rank
