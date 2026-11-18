// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "high_index_cold_chain.h"

// One-owner plan for cold sources used once across all five phases. Victim
// slots may repeat across chunks. Within each chunk the source and victim
// sets are disjoint and injective, so Consumer's ordered closure pairs each
// victim with its own cold source. Different hot slots form independent cycles.
constexpr uint32_t kColdHotLimit=16384, kColdHistoryLimit=210944;
constexpr uint32_t kColdCtl=0, kColdScan=32, kColdSeen=64;
constexpr uint32_t kColdHotSeen=kColdSeen+(kColdHistoryLimit+31)/32;
constexpr uint32_t kColdChunkSeen=kColdHotSeen+kColdHotLimit/32;
constexpr uint32_t kColdFirst=kColdChunkSeen+kColdHotLimit/32;
constexpr uint32_t kColdLast=kColdFirst+kColdHotLimit;
constexpr uint32_t kColdTouched=kColdLast+kColdHotLimit;
constexpr uint32_t kColdWords=kColdTouched+kCapacity/2;
static_assert(kColdWords*4<=223232, "cold chain planning must fit in private AIV UB");

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void InitColdPlan(uint32_t history, uint32_t resident, __ubuf__ uint32_t* m) {
    for(uint32_t i=threadIdx.x;i<(history+31)/32;i+=THREADS)m[kColdSeen+i]=0;
    for(uint32_t i=threadIdx.x;i<(resident+31)/32;i+=THREADS)m[kColdHotSeen+i]=0;
    if(threadIdx.x<8)m[kColdCtl+threadIdx.x]=0;
    asc_syncthreads();
}

__simt_callee__ inline bool MarkColdBit(__ubuf__ uint32_t* bits,uint32_t id) {
    uint32_t bit=uint32_t(1)<<(id%32);
    return (asc_atomic_or(bits+id/32,bit)&bit)==0;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void VisitColdBatch(__gm__ uint32_t* sources,__gm__ uint32_t* victims,
        uint32_t count,uint32_t resident,uint32_t history,__gm__ Edge* edges,__ubuf__ uint32_t* m) {
    uint32_t tid=threadIdx.x,newCount=m[kColdCtl+1],repeatCount=m[kColdCtl+2];
    for(uint32_t base=0;base<count;base+=THREADS){
        uint32_t n=count-base<THREADS ? count-base : THREADS;
        for(uint32_t i=tid;i<(resident+31)/32;i+=THREADS)m[kColdChunkSeen+i]=0;
        asc_syncthreads();
        uint32_t src=0,dst=0;
        if(tid<n){
            src=asc_ldcg(sources+base+tid); dst=asc_ldcg(victims+base+tid);
            if(src<resident || src>=history || dst>=resident ||
               !MarkColdBit(m+kColdSeen,src) || !MarkColdBit(m+kColdChunkSeen,dst))
                asc_atomic_or(m+kColdCtl,1U);
        }
        asc_syncthreads();
        if(m[kColdCtl])return; // Discard private scratch; general Plan reads the raw events again.
        bool fresh=tid<n && MarkColdBit(m+kColdHotSeen,dst);
        uint32_t rank=prefix(fresh,m+kColdScan), added=m[kColdScan+15];
        if(tid<n)VisitColdSlot(dst,src,fresh,m+kColdFirst,m+kColdLast,m+kColdTouched,
                              edges,newCount+rank,repeatCount+tid-rank);
        newCount+=added; repeatCount+=n-added;
        asc_syncthreads();
    }
    if(!tid){m[kColdCtl+1]=newCount; m[kColdCtl+2]=repeatCount;}
    asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void FinishColdPlan(__gm__ Edge* edges,__ubuf__ uint32_t* m) {
    uint32_t tid=threadIdx.x,n=m[kColdCtl+1],pairs=0,singles=m[kColdCtl+2];
    for(uint32_t base=0;base<n;base+=THREADS){
        uint32_t size=n-base<THREADS ? n-base : THREADS,hot=0,first=0,last=0;
        if(tid<size){
            hot=m[kColdTouched+base+tid]; first=m[kColdFirst+hot]; last=m[kColdLast+hot];
        }
        bool isPair=tid<size && first==last;
        uint32_t rank=prefix(isPair,m+kColdScan),added=m[kColdScan+15];
        if(tid<size)CloseColdSlot(hot,first,last,edges,pairs+rank,singles+2*(tid-rank));
        pairs+=added; singles+=2*(size-added);
        asc_syncthreads();
    }
    if(!tid){m[kColdCtl+1]=pairs; m[kColdCtl+2]=singles;}
    asc_threadfence();
}

__aicore__ __attribute__((noinline)) inline bool TryColdPlan(__gm__ Args* a,uint32_t b,__gm__ Edge* edges,
                                                            bool profile,PrivatePlan& plan){
    if(a->resident>kColdHotLimit || a->history>kColdHistoryLimit)return false;
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,kColdWords*4);
    auto local=buf.Get<uint32_t>(); auto mem=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    auto src=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+uint64_t(b)*a->maxEvents;
    auto dst=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*a->maxEvents;
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
    uint64_t begin=profile ? GetSystemCycle() : 0;
    Simt::VF_CALL<InitColdPlan>(Simt::Dim3{THREADS,1,1},a->history,a->resident,mem);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t finalCount=0,chunks=0;
    bool eligible=true;
    for(uint32_t phase=0;phase<5 && eligible;++phase){
        auto flag=flags+(phase ? kRawQueryFlag+phase-1 : 0);
        if(profile)plan.probeCycles+=GetSystemCycle()-begin;
        uint32_t error=AwaitPublication(flag);
        if(profile)begin=GetSystemCycle();
        uint32_t count=flag->count,offset=phase ? kPrefetch+finalCount : 0;
        if(error || count>kTop || offset+count>a->maxEvents){
            // Producer rejection/timeout is definitive; no Index write is authorized.
            plan.error=error ? error : 1U; break;
        }
        if(!count)continue;
        Simt::VF_CALL<VisitColdBatch>(Simt::Dim3{THREADS,1,1},src+offset,dst+offset,count,
                                     a->resident,a->history,edges,mem);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        eligible=local.GetValue(kColdCtl)==0;
        if(profile)plan.probedEvents+=count;
        chunks+=(count+THREADS-1)/THREADS;
        if(phase)finalCount+=count;
    }
    if(eligible && !plan.error){
        Simt::VF_CALL<FinishColdPlan>(Simt::Dim3{THREADS,1,1},edges,mem);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        plan.pairs=local.GetValue(kColdCtl+1); plan.singles=local.GetValue(kColdCtl+2);
        plan.chunks=chunks; // Freed sources are all outside resident by construction.
    }
    if(profile){plan.probeCycles+=GetSystemCycle()-begin;plan.activeCycles+=plan.probeCycles;}
    pipe.Destroy();
    return eligible;
}
