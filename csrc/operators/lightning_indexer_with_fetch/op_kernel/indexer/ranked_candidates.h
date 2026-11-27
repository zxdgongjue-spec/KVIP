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
#if defined(KVIP_TEST_WINDOW_RANK1)
constexpr bool kCompactWindowIds = true;
#else
constexpr bool kCompactWindowIds = false;
#endif
// One bit per physical Index position: valid local or already admitted.
// Packed scores and eligibility use disjoint banks; the full TopK stays intact.
template<bool Gathered=false, bool Empty=false, bool PositiveFinal=false>
__simd_vf__ inline void PackUncovered(__ubuf__ uint32_t* ids,__ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* covered,__ubuf__ uint32_t* eligible,
    __ubuf__ uint32_t* packed,__ubuf__ uint32_t* count,uint32_t valid) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,safe,index,shift,bits,zero,one,mask31,invalid,kept,out,sum,ones,total;
    RegTensor<uint16_t> score;MaskReg bounded,miss,take;UnalignRegForStore store;
    RegTensor<int32_t> slot;RegTensor<uint32_t> indexScore;
    Duplicate(zero,uint32_t(0));Duplicate(one,uint32_t(1));Duplicate(mask31,uint32_t(31));
    Duplicate(invalid,UINT32_MAX);Duplicate(sum,uint32_t(0));ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<2048;i+=64){
        LoadAlign(id,ids+i);
        if constexpr (!PositiveFinal) LoadAlign<uint16_t,LoadDist::DIST_UNPACK_B16>(score,scores+i);
        CompareScalar<uint32_t,CMPMODE::LT>(bounded,id,valid,all);Select(safe,id,zero,bounded);
        ShiftRights(index,safe,int16_t(5),all);And(shift,safe,mask31,all);
        if constexpr(Empty) { Duplicate(bits,uint32_t(0)); }
        else {
            if constexpr(Gathered)LoadAlign(bits,covered+i);else Gather(bits,covered,index,all);
            ShiftRight(bits,bits,reinterpret_cast<RegTensor<int32_t>&>(shift),all);And(bits,bits,one,all);
        }
        CompareScalar<uint32_t,CMPMODE::EQ>(miss,bits,uint32_t(0),all);MaskAnd(take,bounded,miss,all);
        if constexpr (!PositiveFinal) {
            if constexpr (kCompactWindowIds) {
                Arange(slot,int32_t(i));
                ShiftLefts(indexScore,reinterpret_cast<RegTensor<uint32_t>&>(slot),int16_t(16),all);
                Or(indexScore,indexScore,reinterpret_cast<RegTensor<uint32_t>&>(score),all);
            } else {
                Select(kept,id,invalid,take);StoreAlign(eligible+i,kept,all);
            }
        }
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
        if constexpr (PositiveFinal)
            Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,id,take);
        else if constexpr (kCompactWindowIds)
            Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,indexScore,take);
        else
            Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,reinterpret_cast<RegTensor<uint32_t>&>(score),take);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(packed,out,store);
    }
    StoreUnAlignPost(packed,store);ReduceSum(total,sum,all);
    auto first=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(count,total,first);
}
// The full final quota admits every uncovered winner. A positive minimum
// makes the generic selector's zero-score pass empty, so packing IDs directly
// preserves both membership and the exact Post/Index order.
__aicore__ inline uint32_t CollectPositiveFinalMisses(LocalTensor<uint32_t> ids,
    LocalTensor<uint32_t> output, LocalTensor<uint32_t> scratch,
    __gm__ uint32_t* covered, uint32_t valid) {
    GlobalTensor<uint32_t> bits;
    bits.SetGlobalBuffer(covered);
    DataCopy(scratch, bits, (valid + 255) / 256 * 8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    auto bitmap = reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto compact = reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr());
    auto winners = reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
    PackUncovered<false, false, true>(winners, nullptr, bitmap, nullptr, compact, bitmap + 4096, valid);
    SetFlag<HardEvent::V_S>(EVENT_ID4);
    WaitFlag<HardEvent::V_S>(EVENT_ID4);
    return scratch.GetValue(4096);
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
// Walk the uncovered part of the window in its original slot order.
__simd_vf__ inline void EmitUncovered(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* packed,
    __ubuf__ uint32_t* output,__ubuf__ uint32_t* count,uint32_t n,uint32_t threshold) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,entry,score,index,scoreMask,out,sum,ones,total;
    MaskReg take; UnalignRegForStore store;
    Duplicate(sum,uint32_t(0)); Duplicate(scoreMask,uint32_t(65535)); ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t pass=0;pass<2;pass++) for(uint16_t i=0;i<n;i+=64) {
        uint32_t left=n-i;auto live=UpdateMask<uint32_t>(left);
        LoadAlign(entry,packed+i);
        And(score,entry,scoreMask,all);ShiftRights(index,entry,int16_t(16),all);
        // Only live lanes address a real slot; map trailing lanes back to slot zero.
        RegTensor<uint32_t> zero;Duplicate(zero,uint32_t(0));Select(index,index,zero,live);
        Gather(id,ids,index,all);
        if(pass==0)CompareScalar<uint32_t,CMPMODE::GT>(take,score,threshold,live);
        else CompareScalar<uint32_t,CMPMODE::EQ>(take,score,threshold,live);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(ones,uint32_t(1),take);Add(sum,sum,ones,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,id,take);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,out,store);
    }
    StoreUnAlignPost(output,store);ReduceSum(total,sum,all);
    auto first=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(count,total,first);
}
// A list that fits in the per-Q minimum admits all its IDs. Even when
// its scores include zero we must emit positive scores before the zero ties.
__simd_vf__ inline void EmitSmallUncovered(__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* packed,__ubuf__ uint32_t* output,uint32_t n) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    uint32_t left=n;auto live=UpdateMask<uint32_t>(left);
    RegTensor<uint32_t> value,slot,score,scoreMask,zero,id,out;
    MaskReg take;UnalignRegForStore store;
    Duplicate(zero,uint32_t(0));Duplicate(scoreMask,uint32_t(65535));
    LoadAlign(value,packed);And(score,value,scoreMask,all);ShiftRights(slot,value,int16_t(16),all);
    Select(slot,slot,zero,live);Gather(id,ids,slot,all);
    ClearSpr<SpecialPurposeReg::AR>();
    CompareScalar<uint32_t,CMPMODE::GT>(take,score,uint32_t(0),live);
    Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,id,take);
    StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,out,store);
    CompareScalar<uint32_t,CMPMODE::EQ>(take,score,uint32_t(0),live);
    Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(out,id,take);
    StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,out,store);
    StoreUnAlignPost(output,store);
}
// scratch >= 14 KiB, output >= 8 KiB (ties are truncated after compaction).
// Scratch holds cold count and threshold banks. Optional clocks
// isolate pack, window rank, quota rank and emit without changing the helper.
__aicore__ inline uint32_t Select(LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,
    LocalTensor<uint32_t> output,LocalTensor<uint32_t> scratch,
    uint32_t resident,uint32_t valid,uint32_t rank,uint32_t quota,bool compact=true,
    uint64_t* clocks=nullptr,uint32_t minimum=0,uint32_t packedOffset=0,
    uint32_t knownRankCut=UINT32_MAX,uint32_t n=0) {
    auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto full=reinterpret_cast<__ubuf__ uint16_t*>(p+3072);
    auto cold=reinterpret_cast<__ubuf__ uint16_t*>(p+2048);
    auto cap=full+128;auto stats=p+3200;
    auto id=reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
    auto score=reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr());
    auto out=reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr());
    if(clocks)clocks[0]=GetSystemCycle();
    if constexpr (!kCompactWindowIds) {
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        n=scratch.GetValue(3200);
    }
    if(clocks)clocks[1]=GetSystemCycle();
    if(!quota||!n){if(clocks)clocks[2]=clocks[3]=clocks[4]=clocks[1];return 0;}
    minimum=minimum<quota?minimum:quota;
    if(kCompactWindowIds && n<=minimum && n<=64) {
        EmitSmallUncovered(id,p+packedOffset,out,n);
        if(clocks){SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
                   clocks[2]=clocks[3]=clocks[4]=GetSystemCycle();}
        return n;
    }
    uint32_t threshold=0;
    // Last window accepts the full Top2048; its minimum cannot reject any input.
    if(rank<2048 && !(minimum && n<=minimum)) {
        if (knownRankCut != UINT32_MAX) {
            threshold=knownRankCut;
        } else {
            RegisterThreshold(score,full,8,2048-rank+1);
            SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
            threshold=scratch.template ReinterpretCast<uint16_t>().GetValue(6144);
        }
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
    if constexpr (kCompactWindowIds) EmitUncovered(id,p+packedOffset,out,stats+8,n,threshold);
    else Emit(id,score,out,stats+8,resident,valid,threshold);
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t count=scratch.GetValue(3208);
    if(clocks)clocks[4]=GetSystemCycle();
    return count<quota?count:quota;
}
// Long histories keep the same UB budget: gather only the words addressed by
// this Q's TopK, rather than loading a history-sized bitmap into a 16-KiB bank.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherCoveredWords(__ubuf__ uint32_t* ids,__gm__ uint32_t* covered,
    __ubuf__ uint32_t* words,uint32_t valid) {
    for(uint32_t i=threadIdx.x;i<2048;i+=256){
        uint32_t id=ids[i];words[i]=id<valid?asc_ldcg(covered+id/32):0U;
    }
}
__simd_vf__ inline void GatherCoveredChunk(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* bitmap,
    __ubuf__ uint32_t* words,uint32_t base,uint32_t end) {
    using namespace MicroAPI;auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    RegTensor<uint32_t> id,offset,index,safe,zero,bits,previous;MaskReg ge,lt,inside;
    Duplicate(zero,uint32_t(0));
    for(uint16_t i=0;i<2048;i+=64){
        LoadAlign(id,ids+i);CompareScalar<uint32_t,CMPMODE::GE>(ge,id,base,all);
        CompareScalar<uint32_t,CMPMODE::LT>(lt,id,end,all);MaskAnd(inside,ge,lt,all);
        Adds(offset,id,uint32_t(0-base),all);Select(safe,offset,zero,inside);
        ShiftRights(index,safe,int16_t(5),all);Gather(bits,bitmap,index,all);
        if(base==0)Duplicate(previous,uint32_t(0));else LoadAlign(previous,words+i);
        Select(bits,bits,previous,inside);StoreAlign(words+i,bits,all);
    }
}
// scratch >= 25 KiB, output >= 16 KiB. Bitmap bank dies before rank selection.
// Bitmap loading or sparse gathering is included in clocks[0..1].
__aicore__ inline uint32_t SelectBitmap(LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,
    LocalTensor<uint32_t> output,LocalTensor<uint32_t> scratch,
    __gm__ uint32_t* covered,uint32_t history,uint32_t valid,uint32_t rank,uint32_t quota,
    uint64_t* clocks=nullptr,uint32_t minimum=0,bool coveredEmpty=false,
    uint32_t knownRankCut=UINT32_MAX) {
    uint64_t begin=clocks?GetSystemCycle():0;
    auto p=reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto eligible=output[2048];
    auto id=reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr());
    auto score=reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr());
    auto out=reinterpret_cast<__ubuf__ uint32_t*>(eligible.GetPhyAddr());
    if(coveredEmpty){
        PackUncovered<false,true>(id,score,p,out,p+4096,p+6144,valid);
    }else if(history<=131072){
        GlobalTensor<uint32_t> bits;bits.SetGlobalBuffer(covered);
        DataCopy(scratch,bits,(history+255)/256*8);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        PackUncovered<false>(id,score,p,out,p+4096,p+6144,valid);
    }else if(history<=262144){
        // At most two DMA chunks; reuse the eligible bank for gathered words.
        // Longer histories use sparse loads rather than scanning every chunk.
        GlobalTensor<uint32_t> bits;bits.SetGlobalBuffer(covered);
        for(uint32_t base=0;base<history;base+=131072){
            uint32_t size=history-base<131072?history-base:131072;
            DataCopy(scratch,bits[base/32],(size+255)/256*8);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            GatherCoveredChunk(id,p,out,base,base+size);
            // Event 4 is primed by the enclosing TopK pipeline. Do not let
            // the next bitmap DMA consume that signal before this gather ends.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        PackUncovered<true>(id,score,out,out,p+4096,p+6144,valid);
    }else{
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        Simt::VF_CALL<GatherCoveredWords>(Simt::Dim3{256,1,1},id,covered,p,valid);
        SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
        PackUncovered<true>(id,score,p,out,p+4096,p+6144,valid);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);
    uint32_t n=scratch.GetValue(6144);
    uint32_t count;
    if constexpr (kCompactWindowIds) {
        count=Select(ids,scores,output,scratch,0,valid,rank,quota,true,clocks,minimum,4096,knownRankCut,n);
    } else {
        scratch.SetValue(3200,n);
        SetFlag<HardEvent::S_V>(EVENT_ID4);WaitFlag<HardEvent::S_V>(EVENT_ID4);
        count=Select(eligible,scores,output,scratch,0,valid,rank,quota,true,clocks,minimum,4096,knownRankCut);
    }
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
