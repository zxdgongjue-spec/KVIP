#pragma once
static_assert(V9_WINDOW_COPY_ENGINE >= 0 && V9_WINDOW_COPY_ENGINE <= 4,
              "Invalid UBmem window copy engine");
#include "fetch_profiling.h"
#include "device_utils.h"
#include "../../../common/lifetch/prefetch_budget.h"

#include "post_geometry.h"
#include "ubmem_copy.h"
static_assert(V9_WINDOW_SCAN_FUSED==0 || V9_WINDOW_SCAN_FUSED==1,"Invalid window scan fusion mode");
static_assert(V9_WINDOW_PREPARED_GROUPS == 0 || V9_WINDOW_PREPARED_GROUPS == 1,
              "Invalid window group planning mode");
namespace rotating {
// The native predicate has 256 bits for 128 BF16 lanes. No per-token address
// or map DMA is performed while scanning these masks.
// Reserve the last 8KiB of the 32000-byte Q4 scratch for candidate IDs
// when indicesOut holds the complete lane's batch rows.
constexpr bool kBatchedRows=true;
#define V9_UBMEM_LANE_COPY (V9_UBMEM && V9_UBPIPE_DEPTH == 8 && V9_UBPIPE_ROWS == 4 && V9_BATCH_POST && SIM_BATCH <= 16)
constexpr bool kWindowScanFused=false;
constexpr uint32_t kScanTokens=kBatchedRows?23808:28672, kCandidateCapacity=2048;
static_assert(!kWindowScanFused || sizeof(uint32_t)*kCandidateCapacity<=kScanTokens,
              "Fused candidate-map values overlap the candidate ID array");
static_assert(!kBatchedRows || kScanTokens+4*kCandidateCapacity<=32000,"Candidate scratch overflow");
static_assert(!kBatchedRows || kPostGroups*sizeof(Row)<=8448,"Row scratch overflow");
static_assert(UbmemGroupedCopyPipeline::kScratchBytes<=kScanTokens,"Grouped copies overlap live candidates");
static_assert(!kWindowScanFused || V9_PARTIAL_SCAN_SIMD,"Fused scan requires SIMD clipping");
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
    __gm__ uint64_t* addresses,uint32_t count,uint32_t slotBase) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i],slot=slotBase+i;uint64_t remote=asc_ldcg(addresses+token);
        lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);slots[i]=slot;asc_stcg(map+token,slot);
    }
    asc_threadfence();
}
// One batch retains IDs in retired bitmap scratch for its sole peer bank.
// Two batches instead retain complete ID lists after the live Row snapshots:
// [0,2048) holds two Rows, [2048,5632) holds two lists of 448 uint32 IDs.
// This separate 8448-byte tensor is neither scanned nor used by the DMA ring,
// so the second batch and subsequent 240-row peer banks cannot clobber IDs.
constexpr uint32_t kUbmemLocalIdCapacity=448;
constexpr uint32_t kUbmemLocalIdOffsetWords=kPostGroups*sizeof(Row)/sizeof(uint32_t);
constexpr uint32_t kUbmemLocalIdGroups=kPostGroups==1?1:
    ((V9_WINDOW_COPY_ENGINE==0 || V9_WINDOW_COPY_ENGINE==4) && kPostGroups==2?2:0);
static_assert(kUbmemLocalIdGroups!=1 || kPostBank==kUbmemLocalIdCapacity,
              "Single-batch local IDs must fit one peer bank");
static_assert(kUbmemLocalIdGroups!=2 ||
              (kUbmemLocalIdOffsetWords+2*kUbmemLocalIdCapacity)*sizeof(uint32_t)<=8448,
              "Retained IDs overlap the end of rowStorage");
__aicore__ inline bool UbmemWindowUseLocalIds(uint32_t count) {
    return kUbmemLocalIdGroups!=0 && count!=0 && count<=kUbmemLocalIdCapacity;
}
__simd_vf__ inline void StageUbmemWindowIds(__ubuf__ uint32_t* dst,
    __ubuf__ uint32_t* src,uint32_t count) {
    using namespace MicroAPI;
    RegTensor<uint32_t> value;
    uint32_t remaining=count;
    for(uint16_t first=0;first<count;first+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);
        LoadAlign(value,src+first);
        StoreAlign(dst+first,value,valid);
    }
}
// Scanning is finished before this task runs. Peer staging occupies only the
// low work area, disjoint from cfg and the ring's bitmap scratch allocation.
constexpr uint32_t kUbmemPeerBytes=kPostGroups*kPostBank*sizeof(uint64_t);
constexpr uint32_t kUbmemPlanCount=kPostGroups*kPostBank/8;
static_assert(kPostBank%8==0,"Prepared groups must not cross peer banks");
static_assert(kUbmemPeerBytes+kUbmemPlanCount*sizeof(uint32_t)<=4096,
              "Peer bank and group plans overlap copy overflow");
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PrepareUbmemWindow(__gm__ Context* ctx,uint32_t firstBatch,uint32_t begin,
    __ubuf__ uint32_t* inputs,__ubuf__ uint32_t* localIds,
    __ubuf__ uint64_t* peers,__ubuf__ uint32_t* plans,
    uint64_t peerBase,uint64_t maxPeerOffset
    ) {
    const uint32_t g=threadIdx.x/kPostThreads,t=threadIdx.x%kPostThreads,b=firstBatch+g*kServiceLanes;
    uint32_t n=0;
    if(b<SIM_BATCH) {
        auto input=inputs+g*80;auto in64=reinterpret_cast<__ubuf__ uint64_t*>(input);
        uint32_t count=input[64];n=count>begin?count-begin:0;
        if(n>kPostBank)n=kPostBank;
        if(input[67])n=0;
        auto jobs=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
        auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY;
        auto addresses=reinterpret_cast<__gm__ uint64_t*>(in64[34]);
        for(uint32_t i=t;i<n;i+=kPostThreads) {
            uint32_t token=localIds && kUbmemLocalIdGroups && count<=kUbmemLocalIdCapacity?
                localIds[g*kUbmemLocalIdCapacity+begin+i]:asc_ldcg(jobs+begin+i);
            uint64_t peer=asc_ldcg(addresses+token);
            peers[g*kPostBank+i]=peer;
            uint32_t logicalSlot=input[65]+begin+i;
            asc_stcg(map+token,logicalSlot);
            RecordChange(SIM_RESIDENT,ctx,b,logicalSlot,token);
        }
    }
    asc_threadfence();
}
__aicore__ inline void CopyUbmemWindow(__gm__ Context* ctx,uint32_t firstBatch,
    LocalTensor<uint32_t> scratch,LocalTensor<uint32_t> work,LocalTensor<uint32_t> cfg,
    LocalTensor<uint32_t> rowStorage,__gm__ Event* event
    ) {
    uint32_t maxCount=0,acceptedTotal=0,lastGroup=0;
    for(uint32_t b=firstBatch;b<SIM_BATCH && (b-firstBatch)/kServiceLanes<kPostGroups;b+=kServiceLanes) {
        auto input=cfg[256+((b-firstBatch)/kServiceLanes)*80];input.SetValue(66,0);input.SetValue(67,0);
        uint32_t n=input.GetValue(64);if(n>maxCount)maxCount=n;
    }
    if(!maxCount)return;
    bool usedSimt=false;
    __ubuf__ uint32_t* localIds=nullptr;
    if constexpr(kUbmemLocalIdGroups==1) {
        if(UbmemWindowUseLocalIds(maxCount))
            localIds=reinterpret_cast<__ubuf__ uint32_t*>(scratch[kScanTokens/4].GetPhyAddr());
    } else if constexpr(kUbmemLocalIdGroups==2) {
        localIds=reinterpret_cast<__ubuf__ uint32_t*>(rowStorage[kUbmemLocalIdOffsetWords].GetPhyAddr());
    }
    // FastService has finished consuming candidate IDs for every batch before
    // calling here. Depth eight borrows scratch[0,31488) plus the dead
    // work[4096,14592) byte range; depth four uses scratch[0,20992) only.
    // Peer staging ends by byte 3840 and cfg starts
    // at byte 16384, so even next-bank SIMT preparation cannot reuse in-flight
    // stages. rowStorage remains a separate buffer until the final drain.
    constexpr uint32_t overflowOffset=4096,cfgOffset=16384;
    static_assert(kPostGroups*kPostBank*sizeof(uint64_t)<=overflowOffset,"Peer bank overlaps copy overflow");
    static_assert(UbmemWindowCopyPipeline::kScratchBytes<=32000,"Window ring exceeds retired candidates");
    static_assert(overflowOffset+UbmemWindowCopyPipeline::kOverflowBytes<=cfgOffset,"Copy overflow overlaps cfg");
    UbmemWindowCopyPipeline copyPipe(ctx,0,scratch.ReinterpretCast<uint8_t>(),
        work.ReinterpretCast<uint8_t>()[overflowOffset]);
    for(uint32_t begin=0;begin<maxCount;begin+=kPostBank) {
        // Publish scalar metadata/error changes and finish reading the prior
        // peer bank before SIMT reuses it. The ring itself uses other scratch.
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Simt::VF_CALL<PrepareUbmemWindow>(Simt::Dim3{512,1,1},ctx,firstBatch,begin,
            reinterpret_cast<__ubuf__ uint32_t*>(cfg[256].GetPhyAddr()),
            localIds,
            reinterpret_cast<__ubuf__ uint64_t*>(work.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(work[kUbmemPeerBytes/4].GetPhyAddr()),
            copyPipe.PlanPeerBase(),copyPipe.PlanMaxPeerOffset()
            );
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto peers=work.ReinterpretCast<uint64_t>();auto plans=work[kUbmemPeerBytes/4];
        if(!usedSimt)
        for(uint32_t b=firstBatch;b<SIM_BATCH && (b-firstBatch)/kServiceLanes<kPostGroups;b+=kServiceLanes) {
            uint32_t g=(b-firstBatch)/kServiceLanes;auto input=cfg[256+g*80];auto in64=input.ReinterpretCast<uint64_t>();
            uint32_t count=input.GetValue(64);if(count<=begin || input.GetValue(67))continue;
            uint32_t n=count-begin;if(n>kPostBank)n=kPostBank;
            uint32_t accepted=input.GetValue(66),slotBase=input.GetValue(65);
            uint64_t output=in64.GetValue(35);
            {
                bool contiguous=!ctx->lruSlots && copyPipe.BeginRunAt(output,slotBase+begin,n);
                for(uint32_t i=0;i<n;) {
                    uint32_t added=1,index=g*kPostBank+i;
                    uint64_t peer=peers.GetValue(index);
                    if(contiguous && n-i>=8 && copyPipe.TryAddNext8(peer,peers.GetValue(index+1),
                        peers.GetValue(index+2),peers.GetValue(index+3),peers.GetValue(index+4),peers.GetValue(index+5),
                        peers.GetValue(index+6),peers.GetValue(index+7)))added=8;
                    else if(contiguous && n-i>=6 && copyPipe.TryAddNext6(peer,peers.GetValue(index+1),
                        peers.GetValue(index+2),peers.GetValue(index+3),peers.GetValue(index+4),peers.GetValue(index+5)))added=6;
                    else if(contiguous && n-i>=4 && copyPipe.TryAddNext4(peer,peers.GetValue(index+1),
                        peers.GetValue(index+2),peers.GetValue(index+3)))added=4;
                    else if(!(contiguous?copyPipe.AddNext(peer):copyPipe.AddAt(peer,output,PhysicalSlotScalar(ctx,b,slotBase+begin+i)))) {
                        input.SetValue(67,kUbmemInvalidAddress);break;
                    }
                    i+=added;accepted+=added;acceptedTotal+=added;lastGroup=g;
                }
            }
            input.SetValue(66,accepted);
        }
    }
    // A released plan exposes the Row and mapping to readers. Completion
    // status and Event state 1 must still wait for every shared ring credit.
    uint32_t completed=usedSimt?acceptedTotal:copyPipe.Finish();
    for(uint32_t b=firstBatch;b<SIM_BATCH && (b-firstBatch)/kServiceLanes<kPostGroups;b+=kServiceLanes) {
        uint32_t g=(b-firstBatch)/kServiceLanes;auto input=cfg[256+g*80];auto state=rowStorage[g*256];
        uint32_t error=input.GetValue(67);if(completed!=acceptedTotal)error|=kUbmemIncomplete;
        if(error)state.SetValue(11,state.GetValue(11)|error);
        if(!input.GetValue(64))continue;
        uint32_t done=completed==acceptedTotal?input.GetValue(66):0;
        auto status=UbmemStatusAt(ctx,b);
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
        status[2]+=done;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),128);
    }
}
__aicore__ inline void FastService(GM_ADDR pointer,uint32_t epoch,uint32_t core,uint32_t lane,
    LocalTensor<uint32_t> scratch,LocalTensor<uint16_t> flags,LocalTensor<uint32_t> ids) {
    auto ctx=reinterpret_cast<__gm__ Context*>(pointer);auto events=reinterpret_cast<__gm__ Event*>(ctx->events);
    auto event=events+epoch*kServiceLanes+lane;event->core=core;event->epoch=epoch;V9_FETCH_PROFILE(event->waitBegin=GetSystemCycle();)
    if(epoch) {
        auto previous=reinterpret_cast<__gm__ volatile Event*>(events)+(epoch-1)*kServiceLanes+lane;
        do { ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+(epoch-1)*kServiceLanes+lane),64); }
        while(!previous->complete);
    }
    V9_FETCH_PROFILE(event->begin=GetSystemCycle();)
    V9_FETCH_PROFILE(uint64_t scanTicks=0,candidateTicks=0,postTicks=0,bitmapBytes=0,totalCandidates=0,totalPosted=0;)
    auto work=flags.ReinterpretCast<uint32_t>();auto cfg=work[4096];
    auto rowStorage=ids;
    if constexpr(kBatchedRows)ids=scratch[23808/4];
    // A service lane owns at most eight Row snapshots in UB. For larger
    // batches/window counts process consecutive groups sequentially; each
    // chunk completes its copy before the same UB storage is reused.
    for(uint32_t groupBase=0;groupBase<kTotalPostGroups;groupBase+=kPostGroups) {
    uint64_t postedCount=0;
    const uint32_t firstBatch=ServiceChunkFirst(lane,groupBase,kServiceLanes);
    const uint32_t groupCount=ServiceChunkCount(SIM_BATCH,firstBatch,kServiceLanes,kPostGroups);
    // Freeze every ready snapshot before scanning any batch. An early snapshot
    // is usable only when every end is at the ceiling; otherwise resample
    // after completion so early planning cannot shorten the captured history.
    GlobalTensor<uint32_t> progressGm;
    progressGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    if(firstBatch<SIM_BATCH) {
        DataCopyExtParams gather;gather.blockCount=groupCount;
        gather.blockLen=4*sizeof(Progress);gather.srcStride=(kServiceLanes-1)*4*sizeof(Progress);gather.dstStride=0;
        DataCopyPadExtParams<uint32_t> padding{false,0,0,0};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(work,progressGm[firstBatch*128],gather,padding);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(uint32_t b=firstBatch;b<SIM_BATCH && (b-firstBatch)/kServiceLanes<kPostGroups;b+=kServiceLanes)for(uint32_t q=0;q<4;++q) {
            uint32_t end=work.GetValue(((b-firstBatch)/kServiceLanes)*128+q*32),cap=Trigger(epoch)*128;
            if(ctx->queryEnds) {
                uint32_t bound=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[b*4+q];
                // Compact victim predicates are clipped before publication.
                if(ctx->lruSlots && bound<SIM_RESIDENT)bound=SIM_RESIDENT;
                if(cap>bound)cap=bound;
            }
            cfg.SetValue(128+b*4+q,end<cap?end:cap);
        }
    }
    GlobalTensor<uint32_t> allRows;allRows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows));
    if constexpr(kBatchedRows) {
        if(firstBatch<SIM_BATCH) {
            DataCopyExtParams copy;copy.blockCount=groupCount;
            copy.blockLen=sizeof(Row);copy.srcStride=(kServiceLanes-1)*sizeof(Row);copy.dstStride=0;
            DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
            DataCopyPad(rowStorage,allRows[firstBatch*256],copy,pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    for(uint32_t b=firstBatch;b<SIM_BATCH && (b-firstBatch)/kServiceLanes<kPostGroups;b+=kServiceLanes) {
        // Transfer state as a block. Avoid scalar GM read/modify/write and
        // repeated cache maintenance for every counter/cache line.
        auto state=kBatchedRows?rowStorage[((b-firstBatch)/kServiceLanes)*256]:work[2048];auto state64=state.ReinterpretCast<uint64_t>();
        GlobalTensor<uint32_t> rowGm;
        rowGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+b*256);
        if constexpr(!kBatchedRows) {
            DataCopy(state,rowGm,256);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        auto postInput=cfg[256+((b-firstBatch)/kServiceLanes)*80];postInput.SetValue(64,0);
        if(state.GetValue(11))continue;
        uint32_t slots=state.GetValue(8),requests=state.GetValue(9),pending=state.GetValue(211);
        uint32_t sequence=state.GetValue(246)+1;
        state.SetValue(246,sequence);state.SetValue(214+epoch,sequence);
        const uint64_t addresses=state64.GetValue(2),output=state64.GetValue(3);
        const uint32_t lower=ctx->lruSlots?0:SIM_RESIDENT;
        uint32_t end[4],first=SIM_HISTORY,last=lower;
        for(uint32_t q=0;q<4;++q) {
            uint32_t saved=cfg.GetValue(128+b*4+q);
            end[q]=saved>state.GetValue(12+q)?saved:state.GetValue(12+q);
            uint32_t lo=state.GetValue(12+q)>lower?state.GetValue(12+q):lower;
            if(lo<first)first=lo;if(end[q]>last)last=end[q];
        }
        V9_FETCH_PROFILE(uint64_t scanBegin=GetSystemCycle();)
        uint32_t candidateCount=0,stop=last,wordCount=0;
        bool aggregate=first%128==0 && last%128==0;
        for(uint32_t q=0;q<4;++q) {
            uint32_t consumed=state.GetValue(12+q);if(consumed<lower)consumed=lower;
            if(consumed!=first || end[q]!=last)aggregate=false;
        }
        GlobalTensor<uint8_t> bitmap;
        bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(ctx->hits)+b*SIM_HISTORY);
        for(uint32_t base=(first/128)*128;first<last && base<last;) {
            if(ctx->lruSlots && base>=kLruSlots && base<SIM_RESIDENT) {base=SIM_RESIDENT;continue;}
            uint32_t span=((last-base+127)/128)*128;
            if(span>kScanTokens)span=kScanTokens;
            if(ctx->lruSlots && base<kLruSlots && span>kLruSlots-base)span=kLruSlots-base;
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
                if(base<lower && lo<lower-base)lo=lower-base;
                uint32_t hi=end[q]>base?end[q]-base:0;
                cfg.SetValue(28+q,lo<span?lo:span);cfg.SetValue(24+q,hi<span?hi:span);
                if(lo || hi<span)full=false;
                uint32_t clippedLo=lo<span?lo:span,clippedHi=hi<span?hi:span;
                cfg.SetValue(80+q*4,clippedLo/16);cfg.SetValue(81+q*4,clippedHi/16);
                cfg.SetValue(82+q*4,0x55555555U<<((clippedLo%16)*2));
                cfg.SetValue(83+q*4,((1U<<((clippedHi%16)*2))-1U)&0x55555555U);
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
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,stop,
                0,
                wordCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);stop=cfg.GetValue(48);
        }
        if(ctx->lruSlots && first<kLruSlots && candidateCount) {
            Simt::VF_CALL<RestoreVictimTokens>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        V9_FETCH_PROFILE(scanTicks+=GetSystemCycle()-scanBegin;totalCandidates+=candidateCount;)
        V9_FETCH_PROFILE(uint64_t candidateBegin=GetSystemCycle(),postBegin=0;)
        uint32_t count=0,missing=0;
        if(candidateCount) {
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY;
            uint32_t available=PrefetchAllowance(slots,requests,state.GetValue(208),
                state.GetValue(209),state.GetValue(210),ctx->windowLimit);
            if(!slots && !ctx->tailBounds) {
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
                auto post64=postInput.ReinterpretCast<uint64_t>();
                postInput.SetValue(64,count);postInput.SetValue(65,SIM_RESIDENT+slots);
                postInput.SetValue(66,pending);post64.SetValue(34,addresses);
                post64.SetValue(35,output);
                if constexpr(kUbmemLocalIdGroups==2) {
                    if(UbmemWindowUseLocalIds(count)) {
                        auto retained=rowStorage[kUbmemLocalIdOffsetWords+((b-firstBatch)/kServiceLanes)*kUbmemLocalIdCapacity];
                        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                        StageUbmemWindowIds(reinterpret_cast<__ubuf__ uint32_t*>(retained.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),count);
                        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    }
                }
                if(!UbmemWindowUseLocalIds(count))
                {
                    GlobalTensor<uint32_t> jobs;
                    jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
                    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                    DataCopy(jobs,ids,((count+7)/8)*8);
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                }
                slots+=count;requests+=count;
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
    if(postedCount) {
        V9_FETCH_PROFILE(uint64_t begin=GetSystemCycle();)
        CopyUbmemWindow(ctx,firstBatch,scratch,work,cfg,rowStorage,event
        );
        V9_FETCH_PROFILE(postTicks+=GetSystemCycle()-begin;)
    }
    if constexpr(kBatchedRows) {
        if(firstBatch<SIM_BATCH
        ) {
            DataCopyExtParams copy;copy.blockCount=groupCount;
            copy.blockLen=sizeof(Row);copy.srcStride=0;copy.dstStride=(kServiceLanes-1)*sizeof(Row);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(allRows[firstBatch*256],rowStorage,copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    V9_FETCH_PROFILE(totalPosted+=postedCount;)
    } // groupBase
    V9_FETCH_PROFILE(event->scanTicks=scanTicks;event->candidateTicks=candidateTicks;event->postTicks=postTicks;)
    V9_FETCH_PROFILE(event->bitmapBytes=bitmapBytes;event->candidateCount=totalCandidates;event->postedCount=totalPosted;)
    V9_FETCH_PROFILE(event->end=GetSystemCycle();)event->complete=1;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),
        (V9_INTERMEDIATE_PROFILING || (V9_UBMEM && V9_WINDOW_COPY_ENGINE==1))?sizeof(Event):64);
}
}
