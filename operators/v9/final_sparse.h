#pragma once
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
