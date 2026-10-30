#pragma once
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"
namespace rotating {
// The demo's SIMD compaction keeps hot TopK entries out of the SIMT worklist.
// Runtime work scales with selected cold entries, not all four 2048-entry rows.
__simd_vf__ inline void ColdPositions(__ubuf__ uint32_t* selected,
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
        MaskReg invalid;CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(SIM_HISTORY),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(SIM_RESIDENT),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),cold);
        Add(total,total,flags,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,cold);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(positions,store);ReduceSum(total,total,all);StoreAlign(count,total,lane);
    ReduceSum(errors,errors,all);StoreAlign(count+8,errors,lane);
}
// All speculative CQ completions precede this kernel. Recycle ONLY temporary
// slots that no final query references. Never overwrite a protected hot slot.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Resolve(__ubuf__ int32_t* selected, __ubuf__ int32_t* locations,
    __ubuf__ uint32_t* scratch, __ubuf__ uint32_t* positions, uint32_t coldCount, uint32_t usedSlots,
    __gm__ uint32_t* map, __gm__ uint64_t* addresses,
    __gm__ int32_t* blocks, __gm__ int32_t* offsets, __gm__ int32_t* destinations,
    __gm__ uint32_t* total, __gm__ uint32_t* status, uint32_t row) {
    const uint32_t tid=threadIdx.x;
    constexpr uint32_t seenWords=(SIM_HISTORY+31)/32;
    auto seen=scratch;
    auto protectedSlots=seen+seenWords;
    auto ids=protectedSlots+kSlots;
    auto freeSlots=ids+kSlots;
    auto counters=freeSlots+kSlots; // missing count, free capacity/base, error
    const bool append=usedSlots<=kSlots && coldCount<=kSlots-usedSlots;
    for(uint32_t i=tid;i<seenWords;i+=Threads) seen[i]=0;
    if(!append) for(uint32_t i=tid;i<kSlots;i+=Threads) protectedSlots[i]=0;
    if(tid<4) counters[tid]=0;
    asc_syncthreads();
    // Rounded iteration count keeps ballot/shuffle participation warp-uniform.
    for(uint32_t item=tid;item<((coldCount+Threads-1)/Threads)*Threads;item+=Threads) {
        const uint32_t i=item<coldCount ? positions[item] : 0;
        const int32_t token=item<coldCount ? selected[i] : 0;
        bool fresh=false;
        if(token<0 || uint32_t(token)>=SIM_HISTORY) {
            asc_atomic_or(counters+2,1U);
        } else if(item<coldCount) {
            const uint32_t slot=asc_ldcg(map+token);
            locations[i]=int32_t(slot);
            if(uint32_t(token)>=SIM_RESIDENT) {
                if(slot!=UINT32_MAX) {
                    if(slot<SIM_RESIDENT || slot>=SIM_RESIDENT+kSlots) asc_atomic_or(counters+2,2U);
                    else if(!append) asc_atomic_or(protectedSlots+slot-SIM_RESIDENT,1U);
                } else {
                    const uint32_t bit=1U<<(uint32_t(token)&31U);
                    fresh=!(asc_atomic_or(seen+(uint32_t(token)>>5),bit)&bit);
                }
            }
        }
        // One reservation per warp, rather than a serialized counter for
        // every candidate. All lanes participate, including hot/invalid IDs.
        const uint32_t mask=asc_ballot(fresh),lane=tid%warpSize;
        uint32_t base=0;
        if(lane==0 && mask) base=asc_atomic_add(counters,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        if(fresh) ids[base+uint32_t(__popc(mask&((1U<<lane)-1U)))]=uint32_t(token);
    }
    asc_syncthreads();
    if(append) {
        // The common case has ample unallocated temporary slots. Avoid
        // scanning/recycling 8192 slots when monotonic allocation suffices.
        for(uint32_t i=tid;i<counters[0];i+=Threads) freeSlots[i]=SIM_RESIDENT+usedSlots+i;
        if(!tid)counters[1]=kSlots-usedSlots;
        asc_syncthreads();
    } else {
        uint32_t localFree=0;
        for(uint32_t slot=tid;slot<kSlots;slot+=Threads) localFree+=!protectedSlots[slot];
        freeSlots[tid]=localFree;
        asc_syncthreads();
        // Warp scans replace the former O(Threads^2) shared-memory reduction.
        const uint32_t lane=tid%warpSize,warp=tid/warpSize;
        uint32_t prefix=localFree;
    #pragma unroll
        for(uint32_t delta=1;delta<warpSize;delta*=2) {
            uint32_t value=asc_shfl_up(prefix,delta);
            if(lane>=delta) prefix+=value;
        }
        if(lane==warpSize-1) freeSlots[warp]=prefix;
        asc_syncthreads();
        if(warp==0) {
            uint32_t value=lane<Threads/warpSize ? freeSlots[lane] : 0;
    #pragma unroll
            for(uint32_t delta=1;delta<warpSize;delta*=2) {
                uint32_t prior=asc_shfl_up(value,delta);
                if(lane>=delta)value+=prior;
            }
            freeSlots[lane]=value;
        }
        asc_syncthreads();
        uint32_t warpPrefix=warp ? freeSlots[warp-1] : 0;
        uint32_t begin=warpPrefix+prefix-localFree;
        if(tid==Threads-1) counters[1]=begin+localFree;
        asc_syncthreads();
        for(uint32_t slot=tid;slot<kSlots;slot+=Threads)
            if(!protectedSlots[slot]) freeSlots[begin++]=SIM_RESIDENT+slot;
        asc_syncthreads();
    }
    if(counters[1]<counters[0] || counters[2]) {
        if(!tid) { counters[2]=counters[2] ? counters[2] : 4U; asc_stcg(status,counters[2]); }
        return;
    }
    asc_syncthreads();
    if(!tid) counters[1]=asc_atomic_add(total,counters[0]);
    asc_syncthreads();
    for(uint32_t i=tid;i<counters[0];i+=Threads) {
        const uint32_t token=ids[i],slot=freeSlots[i],dst=counters[1]+i;
        const uint64_t remote=asc_ldcg(addresses+token);
        protectedSlots[i]=uint32_t(remote);ids[i]=uint32_t(remote>>32);
        asc_stcg(map+token,slot);
        asc_stcg(blocks+dst,int32_t(row*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(row*(SIM_RESIDENT+kSlots)+slot));
    }
    asc_threadfence(); asc_syncthreads();
    for(uint32_t item=tid;item<coldCount;item+=Threads) {
        const uint32_t i=positions[item];
        if(locations[i]==-1) locations[i]=int32_t(asc_ldcg(map+selected[i]));
    }
    if(!tid) { asc_stcg(status,0U); asc_stcg(status+1,counters[0]); asc_stcg(status+2,counters[1]); }
    asc_threadfence();
}
}
