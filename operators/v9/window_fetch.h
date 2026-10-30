#pragma once
#include "fetch_profiling.h"
#include "kvaf_batch_get_device.h"
#include "final_submit.h"
#include "batch_post.h"
#ifndef V9_BATCH_POST
#define V9_BATCH_POST 2
#endif
namespace rotating {
// The native predicate has 256 bits for 128 BF16 lanes. No per-token address
// or map DMA is performed while scanning these masks.
// Reserve the last 8KiB of the 32000-byte Q4 scratch for candidate IDs
// when indicesOut holds the complete lane's batch rows.
constexpr bool kBatchedRows=V9_BATCH_POST && SIM_BATCH<=16;
constexpr uint32_t kScanTokens=kBatchedRows?23808:28672, kCandidateCapacity=2048;
static_assert(!kBatchedRows || kScanTokens+4*kCandidateCapacity<=32000,"Candidate scratch overflow");
static_assert(!kBatchedRows || ((SIM_BATCH+kServiceLanes-1)/kServiceLanes)*sizeof(Row)<=8448,"Row scratch overflow");
#ifndef V9_PARTIAL_SCAN_SIMD
#define V9_PARTIAL_SCAN_SIMD 1
#endif
// Clip each query before union so publication skew never admits future bits.
__simd_callee__ inline void ClipWords(MicroAPI::RegTensor<uint32_t>& value,
    MicroAPI::RegTensor<uint32_t>& index,__ubuf__ uint32_t* cfg,uint32_t q,
    MicroAPI::MaskReg& all) {
    using namespace MicroAPI;
    RegTensor<uint32_t> low,high,lowMask,highMask,mask,ones,zero;
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(low,cfg+80+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(high,cfg+81+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(lowMask,cfg+82+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(highMask,cfg+83+q*4);
    Duplicate(ones,uint32_t(0x55555555U));Duplicate(zero,uint32_t(0));
    MaskReg lower,upper,valid,edge;
    Compare<uint32_t,CMPMODE::GE>(lower,index,low,all);
    Compare<uint32_t,CMPMODE::LE>(upper,index,high,all);
    MaskAnd(valid,lower,upper,all);Select(value,value,zero,valid);
    Compare<uint32_t,CMPMODE::EQ>(edge,index,low,all);Select(mask,lowMask,ones,edge);And(value,value,mask,all);
    Compare<uint32_t,CMPMODE::EQ>(edge,index,high,all);Select(mask,highMask,ones,edge);And(value,value,mask,all);
}
template<bool Partial=false>
__simd_vf__ inline void UnionWords(__ubuf__ uint32_t* masks,__ubuf__ uint32_t* indices,__ubuf__ uint32_t* values,
    __ubuf__ uint32_t* cfg,uint32_t words,uint32_t wordBase) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto one=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> a,b,c,d,packed,counts,accum,total,bitMask;
    RegTensor<int32_t> indexSigned;Arange(indexSigned,int32_t(wordBase));
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(indexSigned);
    Duplicate(accum,uint32_t(0));Duplicate(bitMask,uint32_t(0x55555555U));uint32_t remaining=words;
    UnalignRegForStore indexStore;ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<words;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);
        LoadAlign(a,masks+i);LoadAlign(b,masks+words+i);LoadAlign(c,masks+2*words+i);LoadAlign(d,masks+3*words+i);
        if constexpr(Partial){
            ClipWords(a,index,cfg,0,all);ClipWords(b,index,cfg,1,all);
            ClipWords(c,index,cfg,2,all);ClipWords(d,index,cfg,3,all);
        }
        Or(a,a,b,all);Or(c,c,d,all);Or(a,a,c,all);
        And(a,a,bitMask,all);
        MaskReg nonzero;CompareScalar<uint32_t,CMPMODE::NE>(nonzero,a,uint32_t(0),valid);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(counts,uint32_t(1),nonzero);Add(accum,accum,counts,all);
        StoreAlign(masks+i,a,valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,nonzero);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(indices,packed,indexStore);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(indices,indexStore);
    ReduceSum(total,accum,all);StoreAlign(cfg+72,total,one);
    // The second pass reloads q0 after the OR stores above. A scalar V_S
    // after this VF cannot order its internal vector store/load pipelines.
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore valueStore;remaining=words;
    for(uint16_t i=0;i<words;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);LoadAlign(a,masks+i);
        MaskReg nonzero;CompareScalar<uint32_t,CMPMODE::NE>(nonzero,a,uint32_t(0),valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,a,nonzero);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(values,packed,valueStore);
    }
    StoreUnAlignPost(values,valueStore);
}
// Each SIMT thread owns consecutive bitmap words. A two-level prefix sum
// assigns stable, token-ordered output positions without per-candidate atomics.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ExpandWords(__ubuf__ uint32_t* input,__ubuf__ uint32_t* wordIndices,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* cfg,uint32_t base,uint32_t span,uint32_t oldCount,uint32_t words) {
    const uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    const uint32_t perThread=(words+255)/256;
    uint32_t count=0;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t word=tid*perThread+i;
        if(word<words)count+=uint32_t(__popc(input[word]));
    }
    uint32_t inclusive=count;
    for(uint32_t delta=1;delta<warpSize;delta*=2) {
        uint32_t prior=asc_shfl_up(inclusive,delta);
        if(lane>=delta)inclusive+=prior;
    }
    if(lane==warpSize-1)cfg[64+warp]=inclusive;
    asc_syncthreads();
    if(!tid) {
        uint32_t total=oldCount;
        for(uint32_t w=0;w<256/warpSize;++w) {
            uint32_t n=cfg[64+w];cfg[64+w]=total;total+=n;
        }
        cfg[40]=total<kCandidateCapacity?total:kCandidateCapacity;cfg[48]=span;
    }
    asc_syncthreads();
    uint32_t rank=cfg[64+warp]+inclusive-count,stopValue=span;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t word=tid*perThread+i;
        uint32_t bits=word<words?input[word]:0;
        while(bits) {
            uint32_t offset=wordIndices[word]*16+uint32_t(__ffs(int32_t(bits))-1)/2;
            if(rank<kCandidateCapacity)ids[rank]=base+offset;
            uint32_t select=0U-uint32_t(rank==kCandidateCapacity);
            stopValue=(stopValue&~select)|(offset&select);
            ++rank;bits&=bits-1;
        }
    }
    if(stopValue<span)asc_atomic_min(cfg+48,stopValue);
}
// Explicit two-bit predicate decoding for unequal query publication bounds.
__simt_callee__ inline uint32_t ReadyWord(__ubuf__ uint32_t* input,__ubuf__ uint32_t* cfg,uint32_t word,uint32_t words) {
    uint32_t bits=0,start=word*16;
    for(uint32_t q=0;q<4;++q) {
        uint32_t low=cfg[28+q],high=cfg[24+q];
        if(start>=high || start+16<=low)continue;
        uint32_t lo=low>start?low-start:0,hi=high-start<16?high-start:16;
        uint32_t mask=0x55555555U<<(lo*2);
        if(hi<16)mask&=(1U<<(hi*2))-1;
        bits|=input[q*words+word]&mask;
    }
    return bits;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ScanPartial(__ubuf__ uint32_t* input,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* cfg,
    uint32_t base,uint32_t span,uint32_t oldCount) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    uint32_t words=span/16,perThread=(words+255)/256,count=0;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t w=tid*perThread+i;
        if(w<words)count+=uint32_t(__popc(ReadyWord(input,cfg,w,words)));
    }
    uint32_t inclusive=count;
    for(uint32_t d=1;d<warpSize;d*=2) {
        uint32_t previous=asc_shfl_up(inclusive,d);if(lane>=d)inclusive+=previous;
    }
    if(lane==warpSize-1)cfg[64+warp]=inclusive;
    asc_syncthreads();
    if(!tid) {
        uint32_t total=oldCount;
        for(uint32_t w=0;w<256/warpSize;++w) {uint32_t n=cfg[64+w];cfg[64+w]=total;total+=n;}
        cfg[40]=total<kCandidateCapacity?total:kCandidateCapacity;cfg[48]=span;
    }
    asc_syncthreads();
    uint32_t rank=cfg[64+warp]+inclusive-count,stop=span;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t w=tid*perThread+i,bits=w<words?ReadyWord(input,cfg,w,words):0;
        while(bits) {
            uint32_t offset=w*16+uint32_t(__ffs(int32_t(bits))-1)/2;
            if(rank<kCandidateCapacity)ids[rank]=base+offset;
            uint32_t select=0U-uint32_t(rank==kCandidateCapacity);
            stop=(stop&~select)|(offset&select);++rank;bits&=bits-1;
        }
    }
    if(stop<span)asc_atomic_min(cfg+48,stop);
}
// LIDU pattern: gather only candidate map entries, then SIMD squeeze misses.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherCandidateMap(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __gm__ uint32_t* map,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=256)slots[i]=asc_ldcg(map+ids[i]);
}
__simd_vf__ inline void CompactMisses(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,uint32_t count,uint32_t available) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> token,old,rank,index,written,flags,total,zeroIndex,broadcast,packed;
    Duplicate(written,uint32_t(0));Duplicate(zeroIndex,uint32_t(0));
    uint32_t remaining=count;auto output=ids;ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<count;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);LoadAlign(token,ids+i);LoadAlign(old,slots+i);
        MaskReg miss,capacity;CompareScalar<uint32_t,CMPMODE::EQ>(miss,old,UINT32_MAX,valid);
        Unsqueeze(rank,miss);Add(index,rank,written,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),miss);
        ReduceSum(total,flags,all);Gather(broadcast,total,zeroIndex);Add(written,written,broadcast,all);
        CompareScalar<uint32_t,CMPMODE::LT>(capacity,index,available,valid);MaskAnd(miss,miss,capacity,valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,token,miss);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,packed,store);
    }
    StoreUnAlignPost(output,store);StoreAlign(cfg+56,written,lane);
}
// V7's direct transport pattern: address lookup/slot assignment only for the
// compact miss list. FinalBuild/SubmitWqes are shared with final completion.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PrepareCandidateWqes(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* lo,
    __ubuf__ uint32_t* hi,__ubuf__ uint32_t* slots,__gm__ uint32_t* map,
    __gm__ uint64_t* addresses,uint32_t count,uint32_t slotBase, __gm__ Context* ctx, uint32_t b) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i],slot=slotBase+i;
        uint64_t remote;
        if(ctx->contract) { remote=contract_simt::ContractRemote(contract_simt::ContractOf(ctx),b,token); slot=contract_simt::ReserveContract(ctx,b,token,slotBase+i); }
        else remote=asc_ldcg(addresses+token);
        lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);slots[i]=slot;if(!ctx->contract)asc_stcg(map+token,slot);
    }
    asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void TranslateColdCandidates(__gm__ Context* ctx,uint32_t b,__ubuf__ uint32_t* ids,uint32_t n) {
    auto c=contract_simt::ContractOf(ctx);
    for(uint32_t i=threadIdx.x;i<n;i+=256)ids[i]=uint32_t(contract_simt::IndexIds(c,b,false)[ids[i]]);
}
__aicore__ inline void FastService(GM_ADDR pointer,uint32_t epoch,uint32_t core,uint32_t lane,
    LocalTensor<uint32_t> scratch,LocalTensor<uint16_t> flags,LocalTensor<uint32_t> ids) {
    auto ctx=reinterpret_cast<__gm__ Context*>(pointer);
    const uint32_t coldBegin=ctx->contract?0:SIM_RESIDENT;
    auto events=reinterpret_cast<__gm__ Event*>(ctx->events);
    auto event=events+epoch*kServiceLanes+lane;event->core=core;event->epoch=epoch;V9_FETCH_PROFILE(event->waitBegin=GetSystemCycle();)
    if(epoch) {
        auto previous=reinterpret_cast<__gm__ volatile Event*>(events)+(epoch-1)*kServiceLanes+lane;
        do { kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+(epoch-1)*kServiceLanes+lane),64); }
        while(!previous->complete);
    }
    V9_FETCH_PROFILE(event->begin=GetSystemCycle();)
    uint64_t postedCount=0; // Business state: controls batched WQE submission.
    V9_FETCH_PROFILE(uint64_t scanTicks=0,candidateTicks=0,postTicks=0,bitmapBytes=0,totalCandidates=0;)
    auto work=flags.ReinterpretCast<uint32_t>();auto cfg=work[4096];
    auto rowStorage=ids;
    if constexpr(kBatchedRows)ids=scratch[23808/4];
    // Freeze every ready snapshot before doing any batch work. Cap to the
    // scheduled boundary so slow service cannot chase newly produced history.
    GlobalTensor<uint32_t> progressGm;
    progressGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    if(lane<SIM_BATCH) {
        DataCopyExtParams gather;gather.blockCount=(SIM_BATCH-lane+kServiceLanes-1)/kServiceLanes;
        gather.blockLen=4*sizeof(Progress);gather.srcStride=(kServiceLanes-1)*4*sizeof(Progress);gather.dstStride=0;
        DataCopyPadExtParams<uint32_t> padding{false,0,0,0};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(work,progressGm[lane*128],gather,padding);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes)for(uint32_t q=0;q<4;++q) {
            uint32_t end=work.GetValue((b/kServiceLanes)*128+q*32),cap=(ctx->contract?ContractTrigger(ContractOf(ctx),epoch):Trigger(epoch))*128;
            cfg.SetValue(128+b*4+q,end<cap?end:cap);
        }
    }
    GlobalTensor<uint32_t> allRows;allRows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows));
    if constexpr(kBatchedRows) {
        if(lane<SIM_BATCH) {
            DataCopyExtParams copy;copy.blockCount=(SIM_BATCH-lane+kServiceLanes-1)/kServiceLanes;
            copy.blockLen=sizeof(Row);copy.srcStride=(kServiceLanes-1)*sizeof(Row);copy.dstStride=0;
            DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
            DataCopyPad(rowStorage,allRows[lane*256],copy,pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes) {
        // Transfer state as a block. Avoid scalar GM read/modify/write and
        // repeated cache maintenance for every counter/cache line.
        auto state=kBatchedRows?rowStorage[(b/kServiceLanes)*256]:work[2048];auto state64=state.ReinterpretCast<uint64_t>();
        GlobalTensor<uint32_t> rowGm;
        rowGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+b*256);
        if constexpr(!kBatchedRows) {
            DataCopy(state,rowGm,256);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
#if V9_BATCH_POST && SIM_BATCH <= 16
        auto postInput=cfg[256+(b/kServiceLanes)*80];postInput.SetValue(64,0);
#endif
        if(state.GetValue(11))continue;
        if(ctx->contract) {
            auto c=ContractOf(ctx);
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->hot_ready)+b*128,128);
            if(!ContractPtr<int32_t>(c->hot_ready)[b*32+1])continue;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->eligible_counts)+b*4,4);
            state.SetValue(208,uint32_t(ContractPtr<int32_t>(c->eligible_counts)[b]));
        }
        uint32_t slots=state.GetValue(8),requests=state.GetValue(9),pending=state.GetValue(211);
        uint32_t sequence=state.GetValue(246)+1;
        state.SetValue(246,sequence);state.SetValue(214+epoch,sequence);
        const uint64_t entityAddress=state64.GetValue(0),queueAddress=state64.GetValue(1);
        const uint64_t addresses=state64.GetValue(2),output=state64.GetValue(3);
        uint32_t end[4],first=SIM_HISTORY,last=coldBegin;
        for(uint32_t q=0;q<4;++q) {
            uint32_t saved=cfg.GetValue(128+b*4+q);
            end[q]=saved>state.GetValue(12+q)?saved:state.GetValue(12+q);
            uint32_t lo=state.GetValue(12+q)>coldBegin?state.GetValue(12+q):coldBegin;
            if(lo<first)first=lo;if(end[q]>last)last=end[q];
        }
        V9_FETCH_PROFILE(uint64_t scanBegin=GetSystemCycle();)
        uint32_t candidateCount=0,stop=last,wordCount=0;
        bool aggregate=first%128==0 && last%128==0;
        for(uint32_t q=0;q<4;++q) {
            uint32_t consumed=state.GetValue(12+q);if(consumed<coldBegin)consumed=coldBegin;
            if(consumed!=first || end[q]!=last)aggregate=false;
        }
        GlobalTensor<uint8_t> bitmap;
        bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(ctx->hits)+b*SIM_HISTORY);
        for(uint32_t base=(first/128)*128;first<last && base<last;) {
            uint32_t span=((last-base+127)/128)*128;
            if(span>kScanTokens)span=kScanTokens;
            if(aggregate) {
                uint32_t room=((1792-wordCount)/8)*128;
                if(!room) {stop=base;break;}
                if(span>room)span=room;
            }
            // ready ends are 128-aligned; a capacity cutoff can leave a
            // half-tile cursor, which is masked on the next service.
            bool full=true;
            for(uint32_t q=0;q<4;++q) {
                uint32_t lo=state.GetValue(12+q)>base?state.GetValue(12+q)-base:0;
                if(base<coldBegin && lo<coldBegin-base)lo=coldBegin-base;
                uint32_t hi=end[q]>base?end[q]-base:0;
                cfg.SetValue(28+q,lo<span?lo:span);cfg.SetValue(24+q,hi<span?hi:span);
                if(lo || hi<span)full=false;
#if V9_PARTIAL_SCAN_SIMD
                uint32_t clippedLo=lo<span?lo:span,clippedHi=hi<span?hi:span;
                cfg.SetValue(80+q*4,clippedLo/16);cfg.SetValue(81+q*4,clippedHi/16);
                cfg.SetValue(82+q*4,0x55555555U<<((clippedLo%16)*2));
                cfg.SetValue(83+q*4,((1U<<((clippedHi%16)*2))-1U)&0x55555555U);
#endif
            }
            DataCopyExtParams copy;copy.blockCount=4;copy.blockLen=span/4;
            copy.srcStride=(SIM_HISTORY-span)/4;copy.dstStride=0;
            V9_FETCH_PROFILE(bitmapBytes+=span;)
            DataCopyPadExtParams<uint8_t> padding{false,0,0,0};
            DataCopyPad(scratch.ReinterpretCast<uint8_t>(),bitmap[base/4],copy,padding);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if(full || V9_PARTIAL_SCAN_SIMD) {
                uint32_t prior=aggregate?wordCount:0;
                if(full)UnionWords<false>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                else UnionWords<true>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint32_t words=cfg.GetValue(72);wordCount+=words;
                if(!aggregate && words)Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount,words);
                else {cfg.SetValue(40,candidateCount);cfg.SetValue(48,span);}
            } else Simt::VF_CALL<ScanPartial>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);uint32_t scanned=cfg.GetValue(48);
            // V_S above joins the scan; forward that dependency to MTE2.
            // Do not consume LI's persistent V_MTE2 ping-pong credits.
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            if(scanned<span) { stop=base+scanned;break; }
            base+=span;
        }
        if(aggregate && wordCount) {
            Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,stop,0,wordCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);stop=cfg.GetValue(48);
        }
        V9_FETCH_PROFILE(scanTicks+=GetSystemCycle()-scanBegin;totalCandidates+=candidateCount;)
        V9_FETCH_PROFILE(uint64_t candidateBegin=GetSystemCycle(),postBegin=0;)
        uint32_t count=0,missing=0;
        if(candidateCount) {
            if(ctx->contract) {
                Simt::VF_CALL<TranslateColdCandidates>(Simt::Dim3{256,1,1},ctx,b,
                    reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),candidateCount);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY;
            uint32_t available=ctx->windowLimit;
            uint32_t room=slots<state.GetValue(208)?state.GetValue(208)-slots:0;if(available>room)available=room;
            room=requests<state.GetValue(209)?state.GetValue(209)-requests:0;if(available>room)available=room;
            room=requests<state.GetValue(210)?state.GetValue(210)-requests:0;if(available>room)available=room;
            if(!slots) {
                // No slot has ever been assigned: the OR bitmap already
                // deduplicated tokens, so every candidate is a new miss.
                missing=candidateCount;count=missing<available?missing:available;
            } else {
                Simt::VF_CALL<GatherCandidateMap>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidateCount);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),candidateCount,available);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                missing=cfg.GetValue(56);count=missing<available?missing:available;
            }
            V9_FETCH_PROFILE(candidateTicks+=GetSystemCycle()-candidateBegin;)
            if(count) {
                V9_FETCH_PROFILE(postBegin=GetSystemCycle();)
#if V9_BATCH_POST && SIM_BATCH <= 16
                auto post64=postInput.ReinterpretCast<uint64_t>();
                postInput.SetValue(64,count);postInput.SetValue(65,coldBegin+slots);
                postInput.SetValue(66,pending);post64.SetValue(34,addresses);
                post64.SetValue(35,output);post64.SetValue(36,queueAddress);
                GlobalTensor<uint32_t> entityGm,queueGm;
                entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(entityAddress));
                queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queueAddress));
                DataCopy(postInput,entityGm,48);DataCopy(postInput[48],queueGm,16);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                GlobalTensor<uint32_t> jobs;
                jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(jobs,ids,((count+7)/8)*8);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                slots+=count;requests+=count;
#else
                {
                    Simt::VF_CALL<PrepareCandidateWqes>(Simt::Dim3{256,1,1},
                        reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(work[512].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[1024].GetPhyAddr()),
                        map,reinterpret_cast<__gm__ uint64_t*>(addresses),count,coldBegin+slots,ctx,b);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    if(SubmitWqesAt(entityAddress,queueAddress,output,work[1024],scratch,work,work[512],cfg,count,256,false,nullptr,pending)){++pending;requests+=count;}
                    else state.SetValue(11,kvaf_device::kInvalidArgs);
                    slots+=count;
                }
#endif
            }
        }
        V9_FETCH_PROFILE(if(postBegin)postTicks+=GetSystemCycle()-postBegin;)
        postedCount+=count;
        state.SetValue(16+epoch,count);state.SetValue(48+epoch,missing-count);
        state.SetValue(10,state.GetValue(10)+missing-count);
        state.SetValue(8,slots);state.SetValue(9,requests);state.SetValue(211,pending);
        for(uint32_t q=0;q<4;++q) {
            uint32_t processed=end[q]<stop?end[q]:stop;
            if(processed<state.GetValue(12+q))processed=state.GetValue(12+q);
            state.SetValue(80+epoch*4+q,processed);state.SetValue(12+q,processed);
        }
        if constexpr(!kBatchedRows) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(rowGm,state,256);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    if constexpr(kBatchedRows) {
        if(lane<SIM_BATCH) {
            DataCopyExtParams copy;copy.blockCount=(SIM_BATCH-lane+kServiceLanes-1)/kServiceLanes;
            copy.blockLen=sizeof(Row);copy.srcStride=0;copy.dstStride=(kServiceLanes-1)*sizeof(Row);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(allRows[lane*256],rowStorage,copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
#if V9_BATCH_POST && SIM_BATCH <= 16
    if(postedCount) {
        V9_FETCH_PROFILE(uint64_t begin=GetSystemCycle();)
        uint32_t maxCount=0;
        for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes) {
            uint32_t n=cfg.GetValue(256+(b/kServiceLanes)*80+64);if(n>maxCount)maxCount=n;
        }
        for(uint32_t start=0;start<maxCount;start+=kPostBank) {
            Simt::VF_CALL<BatchBuild<V9_BATCH_POST==1>>(Simt::Dim3{512,1,1},ctx,epoch,lane,start,
                reinterpret_cast<__ubuf__ uint64_t*>(work[512].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg[256].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes) {
                uint32_t g=b/kServiceLanes;auto input=cfg[256+g*80];auto in64=input.ReinterpretCast<uint64_t>();
                uint32_t count=input.GetValue(64);if(count<=start)continue;
                uint32_t n=count-start;if(n>kPostBank)n=kPostBank;
                uint32_t depth=input.GetValue(19),offset=uint32_t(in64.GetValue(25)+start)&(depth-1);
                uint32_t first=n<depth-offset?n:depth-offset;
#if V9_BATCH_POST == 2
                FinalBuild(reinterpret_cast<__ubuf__ uint32_t*>(work[512+g*kPostBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1024+g*kPostBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1536+g*kPostBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(scratch[g*kPostBank*16].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[g*32].GetPhyAddr()),0,n,count-start);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
                GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(7)));
                DataCopy(sq[offset*16],scratch[g*kPostBank*16],first*16);
                if(first<n)DataCopy(sq,scratch[(g*kPostBank+first)*16],(n-first)*16);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes) {
            auto input=cfg[256+(b/kServiceLanes)*80];auto in64=input.ReinterpretCast<uint64_t>();
            uint32_t count=input.GetValue(64);if(!count)continue;
            in64.SetValue(25,in64.GetValue(25)+count);
            GlobalTensor<uint32_t> queue;queue.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(36)));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(queue,input[48],16);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for(uint32_t b=lane;b<SIM_BATCH;b+=kServiceLanes) {
            auto input=cfg[256+(b/kServiceLanes)*80];if(!input.GetValue(64))continue;
            auto in64=input.ReinterpretCast<uint64_t>();uint64_t db=in64.GetValue(8),head=in64.GetValue(25);
#if V9_SIMULATION
            *reinterpret_cast<__gm__ uint64_t*>(db)=head;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(db),64);
#else
            st_dev(uint32_t(head),reinterpret_cast<__gm__ uint32_t*>(db),0);
#endif
        }
        V9_FETCH_PROFILE(postTicks+=GetSystemCycle()-begin;)
    }
#endif
    V9_FETCH_PROFILE(event->scanTicks=scanTicks;event->candidateTicks=candidateTicks;event->postTicks=postTicks;)
    V9_FETCH_PROFILE(event->bitmapBytes=bitmapBytes;event->candidateCount=totalCandidates;event->postedCount=postedCount;)
    V9_FETCH_PROFILE(event->end=GetSystemCycle();)event->complete=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),V9_INTERMEDIATE_PROFILING?sizeof(Event):64);
}
}
