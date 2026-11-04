#pragma once
#include "prefetch.h"

// Imported from final_resolve.h

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
        MaskReg invalid,notPadding,valid;
        CompareScalar<uint32_t,CMPMODE::NE>(notPadding,ids,UINT32_MAX,all);
        CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(SIM_HISTORY),all);
        MaskAnd(invalid,invalid,notPadding,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(SIM_RESIDENT),all);
        CompareScalar<uint32_t,CMPMODE::LT>(valid,ids,uint32_t(SIM_HISTORY),all);
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

// Imported from final_dense.h

namespace rotating {
// Demo FinalResolve, applied to bounded 8K map chunks. Only compacted cold
// TopK positions are revisited per chunk; hot entries never enter the loop.
__simd_vf__ inline void DenseChunk(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* positions,
    __ubuf__ uint32_t* required,__ubuf__ uint32_t* mapping,__ubuf__ uint32_t* tokens,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* count,
    uint32_t base,uint32_t span,uint32_t coldCount,uint32_t slotBase) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> zero,one,position,token,index,flags,old,rank,slots,written,total,broadcast,zeroIndex,baseVector;
    Duplicate(baseVector,base);Duplicate(zero,uint32_t(0));Duplicate(one,uint32_t(1));Duplicate(zeroIndex,uint32_t(0));
    for(uint16_t i=0;i<span;i+=64)StoreAlign(required+i,zero,all);
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_STORE>();
    // Unique IDs within each TopK; barriers serialize identical cross-query
    // flag writes, as in the original demo.
    for(uint16_t q=0;q<4;++q) {
        uint32_t remaining=coldCount;
        for(uint16_t i=0;i<coldCount;i+=64) {
            auto valid=UpdateMask<uint32_t>(remaining);
            LoadAlign(position,positions+i);Gather(token,selected,position,valid);
            RegTensor<uint32_t> query;
            ShiftRights(query,position,int16_t(11),valid);
            Sub(index,token,baseVector,valid);
            MaskReg inside,owned;
            CompareScalar<uint32_t,CMPMODE::LT>(inside,index,span,valid);
            CompareScalar<uint32_t,CMPMODE::EQ>(owned,query,uint32_t(q),valid);
            MaskAnd(inside,inside,owned,valid);Scatter(required,one,index,inside);
        }
        LocalMemBar<MemType::VEC_STORE,MemType::VEC_STORE>();
    }
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(written,count);
    RegTensor<int32_t> tokenSequence;
    Arange(tokenSequence,int32_t(base));
    for(uint16_t i=0;i<span;i+=64) {
        LoadAlign(flags,required+i);LoadAlign(old,mapping+i);
        MaskReg take,missing;
        CompareScalar<uint32_t,CMPMODE::NE>(take,flags,uint32_t(0),all);
        CompareScalar<uint32_t,CMPMODE::EQ>(missing,old,uint32_t(-1),all);
        MaskAnd(take,take,missing,all);
        Unsqueeze(rank,take);Add(index,rank,written,all);Adds(slots,index,slotBase,all);
        Select(slots,slots,old,take);StoreAlign(mapping+i,slots,all);
        Scatter(tokens,reinterpret_cast<RegTensor<uint32_t>&>(tokenSequence),index,take);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),take);
        ReduceSum(total,flags,all);Gather(broadcast,total,zeroIndex);Add(written,written,broadcast,all);
        Adds(tokenSequence,tokenSequence,int32_t(64),all);
    }
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    uint32_t remaining=coldCount;
    for(uint16_t i=0;i<coldCount;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);
        LoadAlign(position,positions+i);Gather(token,selected,position,valid);
        Sub(index,token,baseVector,valid);
        MaskReg inside;CompareScalar<uint32_t,CMPMODE::LT>(inside,index,span,valid);
        Gather(slots,mapping,index,inside);Scatter(locations,slots,position,inside);
    }
    StoreAlign(count,written,lane);
}
// Collect arbitrary (not necessarily block-affine) remote addresses once.
// This also emits the descriptors needed by the small-SQ transport fallback.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void DenseRecords(__ubuf__ uint32_t* tokens,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* remoteLow,__ubuf__ uint32_t* remoteHigh,__ubuf__ uint32_t* control,
    __gm__ uint64_t* addresses,__gm__ int32_t* blocks,__gm__ int32_t* offsets,
    __gm__ int32_t* destinations,__gm__ uint32_t* total,__gm__ uint32_t* status,
    uint32_t count,uint32_t slotBase,uint32_t row) {
    const uint32_t tid=threadIdx.x;
    if(!tid)control[1]=asc_atomic_add(total,count);
    asc_syncthreads();
    for(uint32_t i=tid;i<count;i+=Threads) {
        const uint32_t token=tokens[i],slot=slotBase+i,dst=control[1]+i;
        const uint64_t remote=asc_ldcg(addresses+token);
        // remoteHigh aliases tokens; each lane reads its token before replacing it.
        remoteLow[i]=uint32_t(remote);remoteHigh[i]=uint32_t(remote>>32);slots[i]=slot;
        asc_stcg(blocks+dst,int32_t(row*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(row*(SIM_RESIDENT+kSlots)+slot));
    }
    if(!tid) { asc_stcg(status,0U);asc_stcg(status+1,count);asc_stcg(status+2,control[1]); }
    asc_threadfence();
}
__aicore__ inline void DenseFinal(__gm__ Context* ctx,__gm__ Row* row,
    __gm__ uint32_t* status,LocalTensor<uint32_t> local,uint32_t coldCount,uint32_t b) {
    constexpr uint32_t seenWords=(SIM_HISTORY+31)/32;
    auto selected=local;auto locations=local[kSlots];
    auto required=local[2*kSlots];auto mapping=local[3*kSlots];auto tokens=local[4*kSlots];
    auto positions=local[5*kSlots+seenWords+64];auto control=local[5*kSlots+seenWords];
    control.SetValue(0,0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> map,locationsGm;
    map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY);
    locationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    if(coldCount)for(uint32_t base=SIM_RESIDENT;base<SIM_HISTORY;base+=kSlots) {
        const uint32_t span=SIM_HISTORY-base<kSlots ? SIM_HISTORY-base : kSlots;
        DataCopy(mapping,map[base],span);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DenseChunk(reinterpret_cast<__ubuf__ uint32_t*>(selected.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(positions.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(required.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(mapping.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(locations.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()),base,span,coldCount,SIM_RESIDENT+row->slots);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(map[base],mapping,span);
        // Next MTE2 load reuses the same UB map; a V-only wait does not protect it.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(locationsGm,locations,kSlots);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t count=control.GetValue(0);
    if(count) {
        Simt::VF_CALL<DenseRecords<512>>(Simt::Dim3{512,1,1},
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(mapping.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(required.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(tokens.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()),
            reinterpret_cast<__gm__ uint64_t*>(row->addresses),
            reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),
            reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
            reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
            reinterpret_cast<__gm__ uint32_t*>(ctx->count),status,count,SIM_RESIDENT+row->slots,b);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;trace[2]=GetSystemCycle();
    SubmitFinal(row,mapping,selected,required,tokens,control[8],count);
    status[3]=count && !row->finalPending ? 1U : 0U;
}
}

// Imported from final_sparse.h

namespace rotating {
// Long-history path. A bounded UB table is used only for new misses;
// each selected cold entry reads GM map once, with no GM publish/reload roundtrip.
// Admission guarantees at most 4096 keys in 8192 buckets (load <= 1/2).
__simt_callee__ inline uint32_t SparseHash(uint32_t token) {
    return (token*0x9e3779b9U)>>19;
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ResolveSparse(__ubuf__ int32_t* selected,__ubuf__ int32_t* locations,
    __ubuf__ uint32_t* positions,__ubuf__ uint32_t* table,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* slots,__ubuf__ uint32_t* protectedBits,__ubuf__ uint32_t* counters,
    __gm__ uint32_t* map,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,__gm__ uint32_t* status,uint32_t coldCount,uint32_t usedSlots,uint32_t row) {
    const uint32_t tid=threadIdx.x,lane=tid%warpSize;
    const uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    const bool append=usedSlots<=kSlots && coldCount<=kSlots-usedSlots;
    if(!append && tid<kSlots/32)protectedBits[tid]=0;
    for(uint32_t i=tid;i<buckets;i+=Threads)table[i]=UINT32_MAX;
    if(tid<4)counters[tid]=0;
    asc_syncthreads();
    for(uint32_t item=tid;item<((coldCount+Threads-1)/Threads)*Threads;item+=Threads) {
        uint32_t token=0,h=0;bool fresh=false;
        if(item<coldCount) {
            const uint32_t position=positions[item];token=uint32_t(selected[position]);
            const bool alias=position>=2*kTop && uint32_t(selected[position-2*kTop])==token;
            const uint32_t slot=alias ? UINT32_MAX-1 : (usedSlots ? asc_ldcg(map+token) : UINT32_MAX);
            locations[position]=int32_t(slot);
            if(slot==UINT32_MAX) {
                h=SparseHash(token)&(buckets-1);
                const uint32_t reservation=(token<<13)|0x80000000U;
                for(uint32_t probe=0;probe<buckets;++probe) {
                    uint32_t old=asc_atomic_cas(table+h,UINT32_MAX,reservation);
                    if(old==UINT32_MAX) { fresh=true;break; }
                    if(((old&0x7fffffffU)>>13)==token)break;
                    h=(h+1)&(buckets-1);
                }
            } else if(!alias && (slot<SIM_RESIDENT || slot>=SIM_RESIDENT+kSlots))asc_atomic_or(counters+2,2U);
            else if(!alias && !append)asc_atomic_or(protectedBits+(slot-SIM_RESIDENT)/32,1U<<((slot-SIM_RESIDENT)%32));
        }
        uint32_t mask=asc_ballot(fresh),base=0;
        if(!lane && mask)base=asc_atomic_add(counters,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        if(fresh) {
            const uint32_t rank=base+uint32_t(__popc(mask&((1U<<lane)-1U)));
            ids[rank]=token;
            if(append)slots[h]=rank;
            else asc_atomic_exch(table+h,(token<<13)|rank);
        }
    }
    asc_syncthreads();
    if(counters[2]) { if(!tid)asc_stcg(status,counters[2]);return; }
    if(!append) {
        // Stop once enough unreferenced slots have been collected. All lanes
        // enter each ballot/barrier, including lanes beyond the last useful slot.
        for(uint32_t base=0;base<kSlots && counters[3]<counters[0];base+=Threads) {
            uint32_t slot=base+tid;
            bool free=!(protectedBits[slot/32]&(1U<<(slot%32)));
            uint32_t mask=asc_ballot(free),start=0;
            if(!lane && mask)start=asc_atomic_add(counters+3,uint32_t(__popc(mask)));
            start=asc_shfl(start,0);
            uint32_t rank=start+uint32_t(__popc(mask&((1U<<lane)-1U)));
            if(free && rank<counters[0])slots[rank]=SIM_RESIDENT+slot;
            asc_syncthreads();
        }
        if(counters[3]<counters[0]) { if(!tid)asc_stcg(status,4U);return; }
    }
    asc_syncthreads();
    if(!tid)counters[1]=asc_atomic_add(total,counters[0]);
    for(uint32_t item=tid;item<coldCount;item+=Threads) {
        const uint32_t position=positions[item];
        if(locations[position]!=-1)continue;
        const uint32_t token=uint32_t(selected[position]);
        uint32_t h=SparseHash(token)&(buckets-1);
        for(uint32_t probe=0;probe<buckets;++probe) {
            const uint32_t word=table[h];
            if(((word&0x7fffffffU)>>13)==token) {
                locations[position]=int32_t(append ? SIM_RESIDENT+usedSlots+slots[h] : slots[word&8191U]);break;
            }
            h=(h+1)&(buckets-1);
        }
    }
    asc_syncthreads();
    // A query occurrence can reuse its already resolved matching occurrence
    // in q0/q1. This is checked per token; arbitrary query layouts still hash.
    for(uint32_t item=tid;item<coldCount;item+=Threads) {
        uint32_t position=positions[item];
        if(locations[position]==-2)locations[position]=locations[position-2*kTop];
    }
    // All lookups finished. The table and IDs now hold packed remote addresses.
    for(uint32_t i=tid;i<counters[0];i+=Threads) {
        const uint32_t token=ids[i],slot=append?SIM_RESIDENT+usedSlots+i:slots[i],dst=counters[1]+i;
        slots[i]=slot;
        const uint64_t remote=asc_ldcg(addresses+token);
        table[i]=uint32_t(remote);ids[i]=uint32_t(remote>>32);
        asc_stcg(map+token,slot);
        asc_stcg(blocks+dst,int32_t(row*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(row*(SIM_RESIDENT+kSlots)+slot));
    }
    if(!tid) { asc_stcg(status,0U);asc_stcg(status+1,counters[0]);asc_stcg(status+2,counters[1]); }
    asc_threadfence();
}
}

// Imported from final_worker_submit.h

namespace rotating {
__aicore__ inline __ubuf__ uint32_t* FinalPtr(LocalTensor<uint32_t> x) {return reinterpret_cast<__ubuf__ uint32_t*>(x.GetPhyAddr());}

__aicore__ inline __gm__ FinalPost* FinalPostAt(__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+kFinalPostOffset)+core;
}
__aicore__ inline void PublishFinalWorker(__gm__ Context* ctx,__gm__ Row* row,
    uint32_t b,uint32_t w,uint32_t own,uint32_t batchCount,uint32_t descriptor,uint32_t error,
    LocalTensor<uint32_t> local,LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,
    LocalTensor<uint32_t> slots,LocalTensor<uint32_t> ctl) {
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+FinalQueue(b,w);
    auto post=FinalPostAt(ctx,b*kFinalWorkers+w);
    auto cfg=ctl[1024],meta=ctl[1280],queueLocal=ctl[1344];
    GlobalTensor<uint32_t> entityGm,queueGm;
    entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->channel_entity));
    queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args->queue_state));
    DataCopy(meta,entityGm,48);DataCopy(queueLocal,queueGm,16);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    auto meta64=meta.ReinterpretCast<uint64_t>(),queue64=queueLocal.ReinterpretCast<uint64_t>();
    uint64_t head=queue64.GetValue(1),doorbell=meta64.GetValue(8);uint32_t depth=meta.GetValue(19);
    if(depth<2 || (depth&(depth-1)) || !meta64.GetValue(7) || !doorbell ||
       head>UINT64_MAX-own || queue64.GetValue(4))error|=64;
#if !V9_SIMULATION
    if(queueLocal.GetValue(1) && queueLocal.GetValue(0)!=kvaf_device::kQueueMagic)error|=64;
    queueLocal.SetValue(0,kvaf_device::kQueueMagic);queueLocal.SetValue(1,1);
#endif
    // Recycling a speculative slot must not race its still-in-flight prefetch.
    bool deferred=own && (row->slots+batchCount>kSlots || own>=depth);
    bool submit=own && !deferred && !error;
    if(submit){
        cfg.SetValue(1,6U<<8);cfg.SetValue(2,(meta.GetValue(20)&0xffffffU)|0x1000000U);cfg.SetValue(3,meta.GetValue(6)&0xfffffU);
        for(uint32_t j=0;j<4;++j)cfg.SetValue(4+j,meta.GetValue(22+j));
        cfg.SetValue(8,meta.GetValue(7));cfg.SetValue(9,0);cfg.SetValue(12,kTokenBytes);cfg.SetValue(13,meta.GetValue(12));
        cfg.SetValue(14,uint32_t(row->output));cfg.SetValue(15,uint32_t(row->output>>32));cfg.SetValue(16,uint32_t(head));cfg.SetValue(17,depth);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(meta64.GetValue(7)));
        for(uint32_t begin=0;begin<own;begin+=1024){uint32_t n=own-begin<1024?own-begin:1024;
            FinalBuild(FinalPtr(remoteLow),FinalPtr(remoteHigh),FinalPtr(slots),FinalPtr(local),FinalPtr(cfg),begin,n,own);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            uint32_t start=uint32_t(head+begin)&(depth-1),first=n<depth-start?n:depth-start;
            DataCopy(sq[start*16],local,first*16);if(first<n)DataCopy(sq,local[first*16],(n-first)*16);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        queue64.SetValue(1,head+own);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(queueGm,queueLocal,16);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if V9_SIMULATION
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+own;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        st_dev(uint32_t(head+own),reinterpret_cast<__gm__ uint32_t*>(doorbell),0);
#endif
    }
    post->count=own;post->pending=submit;post->error=error;post->deferred=deferred;
    post->descriptor=descriptor;post->head=head;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}
}

// Imported from final_parallel.h


// Imported from final_scratch.h

#include <cstdint>
#ifndef V9_USER_UB_BYTES
#error "V9_USER_UB_BYTES must match the compiler user-reserved-ub-size option"
#endif
namespace rotating {
// Six uint32 banks, one uint16 first-occurrence-position bank, and control.
// Storing a TopK position (0..8191) instead of its token saves 16 KiB without
// narrowing token IDs. Recover tokens from selected[position] at address gather.
constexpr uint32_t kFinalSlotsOffset = 4 * 8192 + 4096;
constexpr uint32_t kFinalPositionsOffset = kFinalSlotsOffset + 8192;
constexpr uint32_t kFinalControlOffset = kFinalPositionsOffset + 8192;
constexpr uint32_t kFinalScratchWords = kFinalControlOffset + 2560;
constexpr uint32_t kFinalScratchBytes = kFinalScratchWords * sizeof(uint32_t);
static_assert(kFinalScratchBytes <= V9_USER_UB_BYTES,
              "final scratch exceeds the compiler user UB reservation");
static_assert(V9_USER_UB_BYTES + 6144 <= 229376,
              "user plus CANN 9.1 compiler UB exceeds the SIMT limit");
}

namespace rotating {
__simd_vf__ inline void OwnedColdPositions(__ubuf__ uint32_t* selected,
    __ubuf__ uint32_t* locations,__ubuf__ uint32_t* positions,__ubuf__ uint32_t* count,uint32_t worker) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();
    auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> ids,flags,total,packed,errors,ownedTotal,hash,ownerMask;
    RegTensor<int32_t> signedIndex;
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(signedIndex);
    Arange(signedIndex,int32_t(0));Duplicate(total,uint32_t(0));Duplicate(errors,uint32_t(0));
    Duplicate(ownedTotal,uint32_t(0));Duplicate(ownerMask,uint32_t(kFinalWorkers-1));
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<kSlots;i+=64) {
        LoadAlign(ids,selected+i);StoreAlign(locations+i,ids,all);
        MaskReg invalid;CompareScalar<uint32_t,CMPMODE::GE>(invalid,ids,uint32_t(SIM_HISTORY),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),invalid);Or(errors,errors,flags,all);
        MaskReg cold;CompareScalar<uint32_t,CMPMODE::GE>(cold,ids,uint32_t(SIM_RESIDENT),all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),cold);
        Add(total,total,flags,all);
        Muls(hash,ids,uint32_t(0x9e3779b9U),all);ShiftRights(hash,hash,int16_t(16),all);And(hash,hash,ownerMask,all);
        MaskReg owned;CompareScalar<uint32_t,CMPMODE::EQ>(owned,hash,worker,all);MaskAnd(owned,owned,cold,all);
        if(i>=2*kTop){RegTensor<uint32_t> first;LoadAlign(first,selected+i-2*kTop);
            MaskReg distinct;Compare<uint32_t,CMPMODE::NE>(distinct,ids,first,all);MaskAnd(owned,owned,distinct,all);}
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),owned);Add(ownedTotal,ownedTotal,flags,all);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,owned);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(positions,packed,store);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(positions,store);ReduceSum(total,total,all);StoreAlign(count+16,total,lane);
    ReduceSum(ownedTotal,ownedTotal,all);StoreAlign(count,ownedTotal,lane);
    ReduceSum(errors,errors,all);StoreAlign(count+8,errors,lane);
}

// Token ownership keeps all query duplicates on one worker. Only counts and
// protected-slot bitsets cross workers; the hot path has no global hash atomics.
__simt_callee__ inline uint32_t FinalOwner(uint32_t token) {
    return ((token*0x9e3779b9U)>>16)&(kFinalWorkers-1);
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ParallelCollect(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,
    __ubuf__ uint32_t* keys,__ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,
    __ubuf__ uint32_t* control,__gm__ uint32_t* map,uint32_t worker,uint32_t used,bool protect) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize;
    const uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads)keys[i]=UINT32_MAX;
    if(tid<256)control[32+tid]=0;
    if(tid<8)control[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<((coldCount+Threads-1)/Threads)*Threads;i+=Threads) {
        bool fresh=false;uint32_t token=0,h=0;
        if(i<coldCount) {
            uint32_t pos=positions[i];token=selected[pos];
            if(token>=SIM_HISTORY)asc_atomic_or(control+1,1U);
            else if(FinalOwner(token)==worker) {
                bool alias=pos>=2*kTop && selected[pos-2*kTop]==token;
                uint32_t slot=alias?UINT32_MAX-1:(used?asc_ldcg(map+token):UINT32_MAX);
                loc[pos]=slot;
                if(slot==UINT32_MAX) {
                    h=SparseHash(token)&(buckets-1);
                    for(uint32_t probe=0;probe<buckets;++probe) {
                        uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                        if(old==UINT32_MAX){fresh=true;break;}
                        if(old==token)break;
                        h=(h+1)&(buckets-1);
                    }
                } else if(alias){}
                else if(slot<SIM_RESIDENT || slot>=SIM_RESIDENT+kSlots)asc_atomic_or(control+1,2U);
                else if(protect)asc_atomic_or(control+32+(slot-SIM_RESIDENT)/32,1U<<((slot-SIM_RESIDENT)%32));
            }
        }
        uint32_t mask=asc_ballot(fresh),base=0;
        if(!lane && mask)base=asc_atomic_add(control,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        if(fresh){uint32_t rank=base+uint32_t(__popc(mask&((1U<<lane)-1U)));ranks[h]=rank;ids[rank]=uint16_t(positions[i]);}
    }
    asc_syncthreads();
}
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ParallelEmit(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* control,__ubuf__ uint32_t* protections,
    __gm__ uint32_t* map,__gm__ uint64_t* addresses,__gm__ uint32_t* locations,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    uint32_t worker,uint32_t used,uint32_t ownCount,uint32_t batchCount,
    uint32_t prefix,uint32_t globalBase,uint32_t batch) {
    uint32_t tid=threadIdx.x;
    const uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    if(used+batchCount<=kSlots) {
        for(uint32_t i=tid;i<ownCount;i+=Threads)slots[i]=SIM_RESIDENT+used+prefix+i;
    } else {
        if(tid<256) {
            uint32_t bits=0;
            for(uint32_t w=0;w<kFinalWorkers;++w)bits|=protections[w*256+tid];
            control[32+tid]=bits;control[288+tid]=uint32_t(__popc(~bits));
        }
        asc_syncthreads();
        const uint32_t lane=tid%warpSize,warp=tid/warpSize;
        uint32_t n=tid<256?control[288+tid]:0,scan=n;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
        if(lane==warpSize-1)control[544+warp]=scan;
        asc_syncthreads();
        if(!warp){uint32_t value=lane<Threads/warpSize?control[544+lane]:0;
            for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}
            control[544+lane]=value;}
        asc_syncthreads();
        if(tid<256)control[288+tid]=scan-n+(warp?control[544+warp-1]:0);
        if(!tid && control[544+Threads/warpSize-1]<batchCount)control[1]=4;
        asc_syncthreads();
        for(uint32_t s=tid;s<kSlots;s+=Threads) {
            uint32_t bits=control[32+s/32],bit=s%32;
            uint32_t rank=control[288+s/32]+uint32_t(__popc((~bits)&((1U<<bit)-1U)));
            if(!(bits&(1U<<bit)) && rank>=prefix && rank<prefix+ownCount)slots[rank-prefix]=SIM_RESIDENT+s;
        }
    }
    asc_syncthreads();
    if(control[1])return;
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i],token=selected[pos];
        if(FinalOwner(token)!=worker)continue;
        uint32_t slot=loc[pos];
        if(slot==UINT32_MAX-1)slot=loc[pos-2*kTop];
        if(slot==UINT32_MAX){uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);slot=slots[ranks[h]];}
        asc_stcg(locations+pos,slot);
        if(pos<2*kTop && selected[pos+2*kTop]==token)asc_stcg(locations+pos+2*kTop,slot);
    }
    asc_syncthreads(); // hash arrays become remote-address vectors only after all lookups
    for(uint32_t i=tid;i<ownCount;i+=Threads) {
        uint32_t token=selected[ids[i]],slot=slots[i],dst=globalBase+i;
        uint64_t remote=asc_ldcg(addresses+token);
        keys[i]=uint32_t(remote);ranks[i]=uint32_t(remote>>32);
        asc_stcg(map+token,slot);
        asc_stcg(blocks+dst,int32_t(batch*(SIM_HISTORY/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*kTokenBytes));
        asc_stcg(destinations+dst,int32_t(batch*(SIM_RESIDENT+kSlots)+slot));
    }
    asc_threadfence();
}
__aicore__ inline void ParallelFinal(__gm__ Context* ctx) {
    const uint32_t core=GetBlockIdx(),b=core/kFinalWorkers,w=core%kFinalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
    const uint64_t started=GetSystemCycle();
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;
    static_assert(kSlots==8192,"final scratch bank geometry changed");
    pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output,shared;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    shared.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->owners)+SIM_BATCH*32);
    DataCopy(selected,input,kSlots);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    OwnedColdPositions(FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]),w);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),invalid=ctl.GetValue(2056);
    Simt::VF_CALL<ParallelCollect<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(ctl),reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,w,row->slots,row->slots+ctl.GetValue(2064)>kSlots);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t error=ctl.GetValue(1)|invalid|row->error|(row->slots>kSlots?32U:0U);

    ctl.SetValue(1,error);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(shared[core*kFinalStateWords],ctl,kFinalStateWords);
    // Initialize hot locations exactly once; cold overwrites follow the barrier.
    DataCopy(output[w*(kSlots/kFinalWorkers)],selected[w*(kSlots/kFinalWorkers)],kSlots/kFinalWorkers);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SyncAll<true>();
    auto counts=ctl[1024];
    DataCopyExtParams cp{uint16_t(kFinalCores),32,uint32_t(kFinalStateWords*4-32),0,0};
    DataCopyPad(counts,shared,cp,DataCopyPadExtParams<uint32_t>{false,0,0,0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t globalBase=0,batchBase=0,batchCount=0,prefix=0;
    for(uint32_t c=0;c<kFinalCores;++c){uint32_t n=counts.GetValue(c*8);if(c<core)globalBase+=n;if(c<b*kFinalWorkers)batchBase+=n;if(c/kFinalWorkers==b){batchCount+=n;error|=counts.GetValue(c*8+1);if(c<core)prefix+=n;}}
    uint32_t own=counts.GetValue(core*8);
    if(!core){uint32_t total=0;for(uint32_t c=0;c<kFinalCores;++c)total+=counts.GetValue(c*8);*reinterpret_cast<__gm__ uint32_t*>(ctx->count)=total;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),32);}
    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    // Counts occupy [1024,1536); protection bits occupy [1536,2560).
    auto protect=ctl[1536];
    if(row->slots+batchCount>kSlots){
        DataCopyExtParams bp{uint16_t(kFinalWorkers),1024,128,0,0};
        DataCopyPad(protect,shared[b*kFinalWorkers*kFinalStateWords+32],bp,DataCopyPadExtParams<uint32_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    ctl.SetValue(1,error);SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if(!error)Simt::VF_CALL<ParallelEmit<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),FinalPtr(protect),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots,reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),reinterpret_cast<__gm__ int32_t*>(ctx->destinations),
        w,row->slots,own,batchCount,prefix,globalBase,b);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);error=ctl.GetValue(1);
    if(!w)trace[2]=GetSystemCycle();
    PublishFinalWorker(ctx,row,b,w,own,batchCount,globalBase,error,local,keys,ranks,slots,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalHead=FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}

// Imported from final_replicated.h


namespace rotating {
// Every worker independently builds the same rank/slot plan from immutable
// prefetch mapping. Rank is the first TopK occurrence, NOT atomic arrival order.
// Only disjoint location/descriptor/WQE outputs are partitioned across workers.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void ReplicatedResolve(__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* positions,uint32_t coldCount,__ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* ranks,__ubuf__ uint16_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,__gm__ uint32_t* mapping,__gm__ uint64_t* addresses,
    __gm__ int32_t* blocks,__gm__ int32_t* offsets,__gm__ int32_t* destinations,
    __gm__ uint32_t* total,uint32_t worker,uint32_t used,uint32_t batch,__gm__ Context* ctx) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    auto lru=ctx->lruSlots ? reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+batch*kLruSlots : nullptr;
    uint32_t outputStride=OutputStride(ctx),ownerBase=OwnerRow(ctx,batch)*outputStride;
    uint32_t buckets=coldCount<=1024?2048U:(coldCount<=2048?4096U:8192U);
    for(uint32_t i=tid;i<buckets;i+=Threads){keys[i]=UINT32_MAX;ranks[i]=UINT32_MAX;}
    if(tid<256)cfg[32+tid]=0;
    if(tid<8)cfg[tid]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<coldCount;i+=Threads){
        uint32_t pos=positions[i],token=selected[pos];
        if(token>=SIM_HISTORY){asc_atomic_or(cfg+1,1U);continue;}
        if(pos>=2*kTop && selected[pos-2*kTop]==token){loc[pos]=UINT32_MAX-2;continue;}
        uint32_t slot=(used || ctx->tailBounds)?asc_ldcg(mapping+token):UINT32_MAX;loc[pos]=slot;
        if(slot==UINT32_MAX){
            uint32_t h=SparseHash(token)&(buckets-1);
            for(uint32_t n=0;n<buckets;++n){uint32_t old=asc_atomic_cas(keys+h,UINT32_MAX,token);
                if(old==UINT32_MAX || old==token){asc_atomic_min(ranks+h,pos);break;}h=(h+1)&(buckets-1);}
        } else if(slot<SIM_RESIDENT || slot>=outputStride)asc_atomic_or(cfg+1,2U);
        else if(slot<SIM_RESIDENT+kSlots && used+coldCount>kSlots)asc_atomic_or(cfg+32+(slot-SIM_RESIDENT)/32,1U<<((slot-SIM_RESIDENT)%32));
    }
    asc_syncthreads();
    // Mark the first occurrence before replacing hash-table positions by ranks.
    uint32_t per=(coldCount+Threads-1)/Threads,localCount=0;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX)continue;
        uint32_t token=selected[pos],h=SparseHash(token)&(buckets-1);
        while(keys[h]!=token)h=(h+1)&(buckets-1);
        if(ranks[h]==pos){loc[pos]=UINT32_MAX-1;++localCount;}}
    uint32_t scan=localCount;
    for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(scan,d);if(lane>=d)scan+=prior;}
    if(lane==warpSize-1)cfg[544+warp]=scan;
    asc_syncthreads();
    if(!warp){uint32_t value=lane<Threads/warpSize?cfg[544+lane]:0;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}cfg[544+lane]=value;}
    asc_syncthreads();
    uint32_t begin=scan-localCount+(warp?cfg[544+warp-1]:0),count=cfg[544+Threads/warpSize-1];
    if(!tid)cfg[0]=count;
    for(uint32_t i=tid*per;i<(tid+1)*per && i<coldCount;++i){uint32_t pos=positions[i];
        if(loc[pos]!=UINT32_MAX-1)continue;
        uint32_t token=selected[pos],h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);
        ids[begin]=uint16_t(pos);ranks[h]=begin++;}
    asc_syncthreads();
    if(used+count<=kSlots){for(uint32_t i=tid;i<count;i+=Threads)slots[i]=SIM_RESIDENT+used+i;}
    else {
        uint32_t n=tid<256?uint32_t(__popc(~cfg[32+tid])):0,value=n;
        for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(value,d);if(lane>=d)value+=prior;}
        if(lane==warpSize-1)cfg[544+warp]=value;
        asc_syncthreads();
        if(!warp){uint32_t v=lane<Threads/warpSize?cfg[544+lane]:0;
            for(uint32_t d=1;d<warpSize;d*=2){uint32_t prior=asc_shfl_up(v,d);if(lane>=d)v+=prior;}cfg[544+lane]=v;}
        asc_syncthreads();
        if(tid<256)cfg[288+tid]=value-n+(warp?cfg[544+warp-1]:0);
        if(!tid && cfg[544+Threads/warpSize-1]<count)cfg[1]=4;
        asc_syncthreads();
        for(uint32_t slot=tid;slot<kSlots;slot+=Threads){uint32_t bits=cfg[32+slot/32],bit=slot%32;
            uint32_t rank=cfg[288+slot/32]+uint32_t(__popc((~bits)&((1U<<bit)-1U)));
            if(!(bits&(1U<<bit)) && rank<count)slots[rank]=SIM_RESIDENT+slot;}
    }
    asc_syncthreads();
    // Only an empty hot row outside the LRU pool retains its own destination.
    // Reserved hot rows are ordinary misses and keep the allocator's slots.
    for(uint32_t i=tid;i<count;i+=Threads) {
        uint32_t token=selected[ids[i]];
        if(token<SIM_RESIDENT && !IsTailToken(ctx,batch,token)) {
            int32_t ordinal=ctx->reservedHot ? asc_ldcg(reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+batch*SIM_RESIDENT+token) : -1;
            if(ordinal<0)slots[i]=token;
        }
    }
    asc_syncthreads();
    if(cfg[1])return;
    uint32_t ownBegin=count*worker/kFinalWorkers,ownEnd=count*(worker+1)/kFinalWorkers;
    if(!tid){cfg[2]=ownBegin;cfg[3]=ownEnd-ownBegin;if(!worker)asc_atomic_add(total,count);}
    for(uint32_t i=tid;i<coldCount;i+=Threads){uint32_t pos=positions[i];
        if(pos/(kSlots/kFinalWorkers)!=worker)continue;
        uint32_t slot=loc[pos],token=selected[pos];
        if(slot==UINT32_MAX-2)slot=loc[pos-2*kTop];
        if(slot>=UINT32_MAX-1){uint32_t h=SparseHash(token)&(buckets-1);while(keys[h]!=token)h=(h+1)&(buckets-1);slot=slots[ranks[h]];}
        loc[pos]=slot;
    }
    asc_syncthreads(); // hash and position arrays may now be reused
    // All duplicate-query reads of logical slots joined above. The output
    // adapter shares the original resolver invocation; no extra VF launch.
    for(uint32_t i=tid;i<coldCount;i+=Threads) {
        uint32_t pos=positions[i];
        if(pos/(kSlots/kFinalWorkers)==worker)
            loc[pos]=ResolvePhysicalSlot(lru,loc[pos]);
    }
    asc_syncthreads(); // Join position readers before descriptor scratch reuse.
    for(uint32_t i=tid;i<ownEnd-ownBegin;i+=Threads){uint32_t rank=ownBegin+i,token=selected[ids[rank]],slot=slots[rank],dst=batch*kSlots+rank;
        uint64_t remote=asc_ldcg(addresses+token);keys[i]=uint32_t(remote);ranks[i]=uint32_t(remote>>32);positions[i]=ResolvePhysicalSlot(lru,slot);
        RecordChange(ctx,batch,slot,token);
        // Index rows and their token addresses move together. Deriving a DDR
        // block from the *physical Index position* is wrong after a swap.
        StoreFallbackSource(ctx,batch,dst,token,remote,blocks,offsets);
        asc_stcg(destinations+dst,int32_t(ownerBase+positions[i]));
    }
    // Mapping remains the immutable prefetch snapshot. Final locations and
    // descriptors are authoritative; mutating mapping here races other readers.
    asc_threadfence();
}
__aicore__ inline void ReplicatedFinal(__gm__ Context* ctx) {
    const uint32_t core=GetBlockIdx(),b=core/kFinalWorkers,w=core%kFinalWorkers;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto status=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto trace=reinterpret_cast<__gm__ uint64_t*>(ctx->owners)+b*16;
    uint64_t started=GetSystemCycle();
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
    auto selected=local,loc=local[kSlots],keys=local[2*kSlots],ranks=local[3*kSlots];
    auto ids=local[4*kSlots].ReinterpretCast<uint16_t>();
    auto slots=local[kFinalSlotsOffset],positions=local[kFinalPositionsOffset],ctl=local[kFinalControlOffset];
    GlobalTensor<uint32_t> input,output;input.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*kSlots);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*kSlots);
    DataCopy(selected,input,kSlots);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr(SIM_RESIDENT<=2*kSlots) {
        static_assert(6*kSlots<=kFinalControlOffset,"metadata staging overlaps resolver control");
        // Hash/ID/slot scratch is not live yet. Reuse it for two dense metadata
        // rows; release it before ColdPositions and the original resolver.
        GlobalTensor<uint32_t> metadata;
        uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        if(ctx->hotValid) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+owner*SIM_RESIDENT);
            DataCopy(keys,metadata,SIM_RESIDENT);
        }
        if(ctx->reservedHot) {
            metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+b*SIM_RESIDENT);
            DataCopy(local[4*kSlots],metadata,SIM_RESIDENT);
        }
        uint32_t begin=0,end=0;
        if(ctx->tailBounds) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
            begin=bounds[0];end=bounds[1];
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Simt::VF_CALL<AdaptColdPredicateLocal>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),
            FinalPtr(keys),reinterpret_cast<__ubuf__ int32_t*>(local[4*kSlots].GetPhyAddr()),
            ctx->hotValid!=0,ctx->reservedHot!=0,begin,end);
        PipeBarrier<PIPE_V>();
        ColdPositions(FinalPtr(loc),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    } else {
        // Larger supported resident layouts cannot hold both rows in this UB.
        Simt::VF_CALL<AdaptColdPredicate>(Simt::Dim3{512,1,1},ctx,b,FinalPtr(selected),FinalPtr(keys));
        PipeBarrier<PIPE_V>();
        ColdPositions(FinalPtr(keys),FinalPtr(loc),FinalPtr(positions),FinalPtr(ctl[2048]));
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t cold=ctl.GetValue(2048),error=ctl.GetValue(2056)|row->error|(row->slots>kSlots?32U:0U);

    if(!w){trace[0]=started;trace[1]=GetSystemCycle();}
    if(!error)Simt::VF_CALL<ReplicatedResolve<512>>(Simt::Dim3{512,1,1},FinalPtr(selected),FinalPtr(loc),FinalPtr(positions),cold,
        FinalPtr(keys),FinalPtr(ranks),reinterpret_cast<__ubuf__ uint16_t*>(ids.GetPhyAddr()),FinalPtr(slots),FinalPtr(ctl),
        reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY,reinterpret_cast<__gm__ uint64_t*>(row->addresses),
        reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks),reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets),
        reinterpret_cast<__gm__ int32_t*>(ctx->destinations),reinterpret_cast<__gm__ uint32_t*>(ctx->count),w,row->slots,b,ctx);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t batchCount=error?0:ctl.GetValue(0),prefix=error?0:ctl.GetValue(2),own=error?0:ctl.GetValue(3),batchBase=b*kSlots;
    if(!error)error=ctl.GetValue(1);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(output[w*(kSlots/kFinalWorkers)],loc[w*(kSlots/kFinalWorkers)],kSlots/kFinalWorkers);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if(!w)trace[2]=GetSystemCycle();
    PublishFinalWorker(ctx,row,b,w,own,batchCount,batchBase+prefix,error,local,keys,ranks,positions,ctl);
    if(!w){
        status[0]=error;status[1]=batchCount;status[2]=batchBase;status[3]=0;status[4]=0;
        row->finalBegin=started;row->finalRequests=batchCount;row->finalHead=FinalPostAt(ctx,core)->head;
        row->finalPending=batchCount?1:0;row->finalEnd=GetSystemCycle();trace[3]=row->finalEnd;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(trace),64);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),32);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row)+960,64);
    }
    pipe.Destroy();
}
}
