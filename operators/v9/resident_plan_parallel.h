#pragma once
#include "resident_plan.h"
#include "contract_parallel.h"
namespace rotating {
// Same deterministic first-occurrence/victim/cursor order as BuildResidentPlan.
// Native V9's CAS hash, stable block scans and parallel descriptor preparation
// are retained; only the source of reusable slots follows the resident contract.
__simt_callee__ inline void BuildResidentPlanParallel(
    uint32_t budget,uint32_t length,int32_t originalCursor,
    __gm__ uint32_t* hot,__gm__ int32_t* forward,__gm__ int32_t* reservations,
    __gm__ int32_t* victims,__gm__ uint32_t* victimBits,
    __ubuf__ uint32_t* selected,__ubuf__ uint32_t* locations,
    __ubuf__ uint32_t* tokens,__ubuf__ uint32_t* slots,__ubuf__ uint32_t* hash,
    __ubuf__ uint32_t* protectedSlots,__ubuf__ uint32_t* result,
    __ubuf__ uint32_t* prefetchedSlots,__ubuf__ uint32_t* scan) {
    uint32_t tid=threadIdx.x;
    for(uint32_t i=tid;i<kPlanHash;i+=256)hash[i]=UINT32_MAX;
    for(uint32_t i=tid;i<kPlanBitmap;i+=256)protectedSlots[i]=0;
    for(uint32_t i=tid;i<kPlanEntries;i+=256)locations[i]=UINT32_MAX;
    if(tid<5)result[tid]=tid==kPlanCursor?uint32_t(originalCursor):0;
    asc_syncthreads();
    // Original V9 principle: hot hits bypass the cold hash/slot worklist.
    // Reuse the hash as a direct slot-owner table first. It also validates
    // duplicate IDs per query and aliases without allocating additional UB.
    // tokens is dead until cold compaction. Stage the small H0 bitmap once
    // rather than performing 8192 random GM membership reads per worker.
    uint32_t hotWords=(length+31)/32;
    bool stageHot=hotWords<=kPlanEntries;
    if(stageHot)for(uint32_t i=tid;i<hotWords;i+=256)tokens[i]=hot[i];
    asc_syncthreads();
    uint32_t hotUnique=0,coldEntries=0;
    for(uint32_t i=tid;i<8192;i+=256) {
        uint32_t t=selected[i]; if(t==UINT32_MAX)continue;
        if(int64_t(t)>=int64_t(length)-4+i/2048+1) {
            asc_atomic_or(result+kPlanError,uint32_t(kInvalid)); continue;
        }
        if(!(stageHot?PlanBit(tokens,t):PlanHot(hot,t))){++coldEntries;continue;}
        uint32_t slot=uint32_t(forward[t]),bit=1U<<(28+i/2048);
        if(slot>=budget){asc_atomic_or(result+kPlanError,uint32_t(kInvalid));continue;}
        for(;;) {
            uint32_t old=hash[slot];
            if(old!=UINT32_MAX && ((old&0x0fffffffU)!=t || (old&bit))) {
                asc_atomic_or(result+kPlanError,uint32_t(kInvalid));break;
            }
            uint32_t next=old==UINT32_MAX?t|bit:old|bit;
            if(asc_atomic_cas(hash+slot,old,next)!=old)continue;
            if(old==UINT32_MAX) {
                ++hotUnique;asc_atomic_or(protectedSlots+slot/32,1U<<(slot%32));
            }
            break;
        }
        locations[i]=0x40000000U|slot;
        selected[i]=UINT32_MAX;
    }
    asc_syncthreads();
    if(result[kPlanError])return;
    auto hotSummary=ContractPrefixUB(hotUnique+(coldEntries<<16),scan);
    hotUnique=hotSummary.total&0xffffU;
    if(!(hotSummary.total>>16)) {
        for(uint32_t i=tid;i<8192;i+=256)if(locations[i]!=UINT32_MAX)locations[i]&=0x3fffffffU;
        if(!tid)result[kPlanUnique]=hotUnique;
        asc_syncthreads();return;
    }
    // Only occupied direct slot entries were changed by the hot pass. The
    // remaining hash entries still carry the initial empty marker.
    for(uint32_t i=tid;i<budget;i+=256)
        if(PlanBit(protectedSlots,i))hash[i]=UINT32_MAX;
    asc_syncthreads();
    for(uint32_t i=tid;i<8192;i+=256) {
        uint32_t t=selected[i]; if(t==UINT32_MAX)continue;
        uint32_t h=PlanHash(t),bit=1U<<(16+i/2048);
        for(;;) {
            uint32_t old=asc_atomic_cas(hash+h,UINT32_MAX,i|bit);
            if(old==UINT32_MAX)break;
            if(selected[old&8191U]!=t) { h=(h+1)&(kPlanHash-1); continue; }
            uint32_t first=old&8191U,next=(old&~8191U)|bit|(first<i?first:i);
            uint32_t seen=asc_atomic_cas(hash+h,old,next);
            if(seen!=old)continue;
            if(old&bit)asc_atomic_or(result+kPlanError,uint32_t(kInvalid));
            break;
        }
    }
    asc_syncthreads();
    if(result[kPlanError])return;
    uint32_t begin=tid*32,end=begin+32,count=0;
    for(uint32_t i=begin;i<end;++i) {
        uint32_t t=selected[i]; if(t==UINT32_MAX)continue;
        uint32_t h=PlanHash(t);
        while(selected[hash[h]&8191U]!=t)h=(h+1)&(kPlanHash-1);
        locations[i]=h; count+=(hash[h]&8191U)==i;
    }
    auto prefix=ContractPrefixUB(count,scan);
    uint32_t unique=prefix.total,rank=prefix.begin;
    if(unique+hotUnique>budget) { if(!tid)result[kPlanError]=uint32_t(kCapacity); asc_syncthreads(); return; }
    // locations holds hash buckets. Read first positions before overwriting the
    // hash with compact union ranks (other lanes may refer to the same bucket).
    for(uint32_t i=begin;i<end;++i) {
        uint32_t h=locations[i];
        if(h!=UINT32_MAX && !(h&0x40000000U) && (hash[h]&8191U)==i)locations[i]=h|0x80000000U;
    }
    asc_syncthreads();
    for(uint32_t i=begin;i<end;++i)if(locations[i]!=UINT32_MAX && (locations[i]&0x80000000U)) {
        uint32_t t=selected[i],h=locations[i]&0x7fffffffU,r=rank++;
        hash[h]=r; tokens[r]=t;
        int32_t slot=reservations[t]; slots[r]=uint32_t(slot);
        if(slot>=0) {
            if(uint32_t(slot)>=budget)asc_atomic_or(result+kPlanError,uint32_t(kInvalid));
            else if(asc_atomic_or(protectedSlots+slot/32,1U<<(slot%32))&(1U<<(slot%32)))
                asc_atomic_or(result+kPlanError,uint32_t(kInvalid));
        }
    }
    asc_syncthreads();
    if(result[kPlanError])return;
    for(uint32_t i=tid;i<8192;i+=256)if(locations[i]!=UINT32_MAX && !(locations[i]&0x40000000U))locations[i]=hash[locations[i]&0x7fffffffU];
    asc_syncthreads(); // hash is now free for the ordered reusable-slot list.
    uint32_t chunk=(unique+255)/256,lo=tid*chunk,hi=lo+chunk; if(hi>unique)hi=unique;
    count=0; for(uint32_t i=lo;i<hi;++i)count+=slots[i]==UINT32_MAX;
    prefix=ContractPrefixUB(count,scan); uint32_t needed=prefix.total;
    uint32_t available=0,cursor=originalCursor<0?0:uint32_t(originalCursor)%budget;
    if(!tid)scan[8]=0;
    asc_syncthreads();
    for(uint32_t phase=0;phase<3 && available<needed;++phase) {
        uint32_t n=phase==1?budget:2048;
        // Stable 1024-slot tiles preserve victim/cursor order. Cache each
        // lane's four decisions; stop once the remaining supplement is covered.
        for(uint32_t base=0;base<n && available<needed;base+=1024) {
            begin=base+tid*4;end=begin+4;if(end>n)end=n;
            uint32_t mask=0;count=0;
            for(uint32_t i=begin;i<end;++i) {
                uint32_t s=phase==1?(cursor+i)%budget:uint32_t(victims[i]);
                bool take=!PlanBit(protectedSlots,s) && (phase==1?!PlanHot(victimBits,s):
                    (phase==0?!PlanBit(prefetchedSlots,s):PlanBit(prefetchedSlots,s)));
                if(take){mask|=1U<<(i-begin);++count;}
            }
            prefix=ContractPrefixUB(count,scan);rank=available+prefix.begin;
            for(uint32_t i=begin;i<end;++i)if(mask&(1U<<(i-begin))) {
                if(rank<needed) {
                    hash[rank]=phase==1?(cursor+i)%budget:uint32_t(victims[i]);
                    if(phase==1)asc_atomic_max(scan+8,i+1);
                }
                ++rank;
            }
            available+=prefix.total;
            asc_syncthreads();
        }
    }
    if(available<needed) { if(!tid)result[kPlanError]=uint32_t(kCapacity); asc_syncthreads(); return; }
    chunk=(unique+255)/256; lo=tid*chunk; hi=lo+chunk; if(hi>unique)hi=unique;
    if(needed) {
    count=0; for(uint32_t i=lo;i<hi;++i)count+=slots[i]==UINT32_MAX;
    prefix=ContractPrefixUB(count,scan); rank=prefix.begin;
    for(uint32_t i=lo;i<hi;++i)if(slots[i]==UINT32_MAX) {
        uint32_t s=hash[rank++]; slots[i]=s; asc_atomic_or(protectedSlots+s/32,1U<<(s%32));
    }
    asc_syncthreads();
    }
    for(uint32_t i=tid;i<8192;i+=256)if(locations[i]!=UINT32_MAX) {
        uint32_t at=locations[i];
        locations[i]=(at&0x40000000U)?at&0x3fffffffU:slots[at];
    }
    // tokens/slots already contain only cold unique entries in first-use order.
    if(!tid) {
        result[kPlanUnique]=unique+hotUnique; result[kPlanMisses]=unique; result[kPlanSupplement]=needed;
        if(scan[8])result[kPlanCursor]=(cursor+scan[8])%budget;
    }
    asc_syncthreads();
}
}
