// SPDX-License-Identifier: Apache-2.0
#pragma once

// The first mode is a list of proven disjoint swaps. On the first interfering
// chunk convert that prefix once, in UB. Thereafter original IDs and physical
// destinations share a dictionary: fwd[slot] is its original payload and
// inv[payload] is its current slot. Both store dictionary indices.
constexpr uint32_t kTouchedMagic = 0x48543331U;
constexpr uint32_t kTouchedEvents = 3072, kTouchedBuckets = 8192;
constexpr uint32_t kTouchedHistory = 210944, kTouchedResident = 16384;
enum TouchedState : uint32_t { Err, Mode, Processed, Unique, Invalid, Chunks, DirectTested, Pairs, Singles, Conflict };
constexpr uint32_t tScan=32, tSeen=64, tPrefixSrc=tSeen+(kTouchedHistory+31)/32;
constexpr uint32_t tPrefixDst=tPrefixSrc+kTouchedEvents, tKeys=tPrefixDst+kTouchedEvents;
constexpr uint32_t tFwd=tKeys+kTouchedBuckets, tInv=tFwd+kTouchedBuckets;
constexpr uint32_t tList=tInv+kTouchedBuckets, tSrcBits=tList+kTouchedBuckets;
constexpr uint32_t tDstBits=tSrcBits+kTouchedBuckets/32, tOut=tDstBits+kTouchedBuckets/32;
constexpr uint32_t tFree=tOut+THREADS, tWords=tFree+THREADS;
static_assert(tWords*4<=223232, "Index plan exceeds private AIV UB");

__simt_callee__ inline bool FirstVisit(__ubuf__ uint32_t* bits, uint32_t id) {
    uint32_t mask=1U<<(id%32);
    return !(asc_atomic_or(bits+id/32,mask)&mask);
}

__simt_callee__ inline bool WasVisited(__ubuf__ uint32_t* bits, uint32_t id) {
    return (bits[id/32]>>(id%32))&1U;
}

__simt_callee__ inline uint32_t InternPosition(__ubuf__ uint32_t* keys, uint32_t position, bool& fresh) {
    uint32_t bucket=(position*0x9e3779b9U)>>19;
    // At most two positions per input event, <=75% of the dictionary.
    while(true){
        uint32_t prior=asc_atomic_cas(keys+bucket,EMPTY,position);
        if(prior==EMPTY || prior==position){fresh=prior==EMPTY; return bucket;}
        bucket=(bucket+1)&(kTouchedBuckets-1);
    }
}

__simt_callee__ inline void ImportDisjointPrefix(__ubuf__ uint32_t* m, uint32_t count) {
    uint32_t tid=threadIdx.x;
    for(uint32_t i=tid;i<kTouchedBuckets;i+=THREADS)m[tKeys+i]=EMPTY;
    asc_syncthreads();
    for(uint32_t i=tid;i<count;i+=THREADS){
        bool fresh;
        uint32_t a=InternPosition(m+tKeys,m[tPrefixSrc+i],fresh);
        uint32_t b=InternPosition(m+tKeys,m[tPrefixDst+i],fresh);
        m[tFwd+a]=b; m[tFwd+b]=a; m[tInv+a]=b; m[tInv+b]=a;
        m[tList+2*i]=a; m[tList+2*i+1]=b;
    }
    if(!tid){m[Unique]=2*count; m[Mode]=3;}
    asc_syncthreads();
}

__simt_callee__ inline void ComposeTouchedChunk(__ubuf__ uint32_t* m,uint32_t count,
        uint32_t raw,uint32_t dest,uint32_t resident,uint32_t history,__gm__ uint32_t* invalidations) {
    uint32_t tid=threadIdx.x;
    for(uint32_t i=tid;i<2*(kTouchedBuckets/32);i+=THREADS)m[tSrcBits+i]=0;
    uint32_t orig=0,dst=0;
    bool freshOrig=false,freshDst=false;
    if(tid<count){
        if(raw>=history || dest>=resident)asc_atomic_or(m+Err,1U);
        else{
            orig=InternPosition(m+tKeys,raw,freshOrig);
            dst=InternPosition(m+tKeys,dest,freshDst);
            if(freshOrig){m[tFwd+orig]=orig; m[tInv+orig]=orig;}
            if(freshDst){m[tFwd+dst]=dst; m[tInv+dst]=dst;}
        }
    }
    uint32_t rank=prefix(uint32_t(freshOrig)+uint32_t(freshDst),m+tScan);
    if(m[Err])return;
    uint32_t added=m[tScan+15], start=m[Unique]+rank;
    if(freshOrig)m[tList+start]=orig;
    if(freshDst)m[tList+start+uint32_t(freshOrig)]=dst;
    uint32_t src=0;
    if(tid<count){
        src=m[tInv+orig];
        if(!FirstVisit(m+tSrcBits,src) || !FirstVisit(m+tDstBits,dst))asc_atomic_or(m+Err,1U);
    }
    asc_syncthreads();
    if(m[Err])return;
    bool displaced=tid<count && !WasVisited(m+tSrcBits,dst);
    bool freed=tid<count && !WasVisited(m+tDstBits,src);
    bool invalidate=freed && m[tKeys+src]<resident;
    rank=prefix(uint32_t(displaced)|(uint32_t(freed)<<10)|(uint32_t(invalidate)<<20),m+tScan);
    uint32_t totals=m[tScan+15], extras=totals&1023U;
    if(displaced)m[tOut+(rank&1023U)]=dst;
    if(freed)m[tFree+((rank>>10)&1023U)]=src;
    if(invalidate)asc_stcg(invalidations+(m[Invalid]+(rank>>20))*(sizeof(Tuple)/4),m[tKeys+src]);
    asc_syncthreads();
    uint32_t displacedOrig=0, freeDst=0;
    if(tid<extras){displacedOrig=m[tFwd+m[tOut+tid]]; freeDst=m[tFree+tid];}
    asc_syncthreads();
    if(tid<count){m[tFwd+dst]=orig; m[tInv+orig]=dst;}
    if(tid<extras){m[tFwd+freeDst]=displacedOrig; m[tInv+displacedOrig]=freeDst;}
    if(!tid){m[Unique]+=added; m[Invalid]+=totals>>20;}
    asc_syncthreads();
}

__simt_callee__ inline void FinishTouched(__ubuf__ uint32_t* m,__gm__ Edge* edges) {
    uint32_t tid=threadIdx.x, pairs=0, singles=0;
    if(m[Mode]==1){
        pairs=m[Processed];
        for(uint32_t i=tid;i<pairs;i+=THREADS){
            asc_stcg(&edges[i].src,m[tPrefixSrc+i]);
            asc_stcg(&edges[i].dst,m[tPrefixDst+i]);
        }
    }else{
        for(uint32_t base=0;base<m[Unique];base+=THREADS){
            uint32_t at=base+tid, dst=0, src=0;
            bool pair=false, single=false;
            if(at<m[Unique]){
                dst=m[tList+at]; src=m[tFwd+dst];
                if(src!=dst){
                    single=m[tFwd+src]!=dst;
                    pair=!single && dst<src;
                }
            }
            uint32_t rank=prefix(uint32_t(pair)|(uint32_t(single)<<16),m+tScan);
            uint32_t totals=m[tScan+15];
            if(pair || single){
                uint32_t i=pair ? pairs+(rank&65535U) : kCapacity-1-singles-(rank>>16);
                asc_stcg(&edges[i].src,m[tKeys+src]); asc_stcg(&edges[i].dst,m[tKeys+dst]);
            }
            pairs+=totals&65535U; singles+=totals>>16;
            asc_syncthreads();
        }
    }
    if(!tid){m[Pairs]=pairs; m[Singles]=singles;}
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void ApplyPublishedPhases(__gm__ uint32_t* sources,__gm__ uint32_t* destinations,
        uint32_t first,uint32_t end,uint32_t finalOffset,uint32_t n0,uint32_t n1,uint32_t n2,
        uint32_t n3,uint32_t n4,uint32_t resident,uint32_t history,__gm__ Edge* edges,
        __gm__ uint32_t* invalidations,__ubuf__ uint32_t* m) {
    uint32_t tid=threadIdx.x;
    if(first==0){
        for(uint32_t i=tid;i<(history+31)/32;i+=THREADS)m[tSeen+i]=0;
        if(tid<=Conflict)m[tid]=tid==Mode ? 1U : 0U;
        asc_syncthreads();
    }
    for(uint32_t q=first;q<end;++q){
        uint32_t size=q==0?n0:q==1?n1:q==2?n2:q==3?n3:n4;
        uint32_t offset=q ? kPrefetch+finalOffset : 0;
        for(uint32_t base=0;base<size;base+=THREADS){
            uint32_t n=size-base<THREADS ? size-base : THREADS;
            uint32_t processed=m[Processed];
            if(processed+n>kTouchedEvents){
                if(!tid)m[Mode]=2;
                asc_syncthreads(); return;
            }
            uint32_t raw=0, dst=0;
            if(tid<n){raw=asc_ldcg(sources+offset+base+tid); dst=asc_ldcg(destinations+offset+base+tid);}
            if(m[Mode]==1){
                if(tid<n && (raw<resident || raw>=history || dst>=resident ||
                   !FirstVisit(m+tSeen,raw) || !FirstVisit(m+tSeen,dst)))asc_atomic_or(m+Conflict,1U);
                asc_syncthreads();
                if(!tid)m[DirectTested]+=n;
                if(!m[Conflict]){
                    if(tid<n){m[tPrefixSrc+processed+tid]=raw; m[tPrefixDst+processed+tid]=dst;}
                }else ImportDisjointPrefix(m,processed);
            }
            if(m[Mode]==3)ComposeTouchedChunk(m,n,raw,dst,resident,history,invalidations);
            if(m[Err])return;
            if(!tid){m[Processed]+=n; ++m[Chunks];}
            asc_syncthreads();
        }
        if(q)finalOffset+=size;
    }
    if(end==5)FinishTouched(m,edges);
    asc_threadfence();
}

__aicore__ __attribute__((noinline)) inline bool TryTouchedPlan(__gm__ Args* a,uint32_t b,
        __gm__ Edge* edges,__gm__ uint32_t* invalidations,bool profile,PrivatePlan& plan) {
    if(a->resident>kTouchedResident || a->history>kTouchedHistory)return false;
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,tWords*4);
    auto local=buf.Get<uint32_t>(); auto mem=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    auto src=reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+uint64_t(b)*a->maxEvents;
    auto dst=reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*a->maxEvents;
    auto flags=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
    uint32_t finalOffset=0, first=0, strategy=1;
    uint64_t begin=profile ? GetSystemCycle() : 0;
    while(first<5){
        auto flag=flags+(first ? kRawQueryFlag+first-1 : 0);
        if(profile)plan.probeCycles+=GetSystemCycle()-begin;
        uint32_t error=AwaitPublication(flag);
        if(profile)begin=GetSystemCycle();
        if(error){plan.error=error; break;}
        uint32_t counts[5]={}, end=first, next=finalOffset;
        do {
            auto f=flags+(end ? kRawQueryFlag+end-1 : 0);
            if(end!=first){
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(f),sizeof(BatchFlag));
                if(!f->ready)break;
            }
            uint32_t count=f->count, off=end ? kPrefetch+next : 0;
            if(f->error || count>kTop || off+count>a->maxEvents){error=f->error ? f->error : 1U; break;}
            counts[end]=count;
            if(end) next+=count;
            ++end;
        }while(end<5);
        if(error){plan.error=error; break;}
        Simt::VF_CALL<ApplyPublishedPhases>(Simt::Dim3{THREADS,1,1},src,dst,first,end,finalOffset,
                counts[0],counts[1],counts[2],counts[3],counts[4],a->resident,a->history,edges,invalidations,mem);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        strategy=local.GetValue(Mode);
        plan.error=local.GetValue(Err);
        if(strategy==2 || plan.error)break;
        first=end; finalOffset=next;
    }
    if(profile){plan.probeCycles+=GetSystemCycle()-begin; plan.activeCycles+=plan.probeCycles;}
    if(first || strategy==2 || !plan.error)plan.probedEvents=local.GetValue(DirectTested);
    if(strategy!=2 && !plan.error){
        plan.pairs=local.GetValue(Pairs); plan.singles=local.GetValue(Singles);
        plan.invalidations=local.GetValue(Invalid); plan.chunks=local.GetValue(Chunks);
    }
    plan.strategy=strategy; pipe.Destroy();
    return strategy!=2;
}
