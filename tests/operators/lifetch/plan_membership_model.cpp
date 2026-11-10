// SPDX-License-Identifier: Apache-2.0
// Production planners run with one host thread; multiword flag storage is
// intentional here. The device's 512-thread specialization needs one word.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
struct {uint32_t x=0;} threadIdx;
uint64_t membershipQueries=0;
uintptr_t recordBegin=0,recordEnd=0;
uint32_t recordHalf=0,liveCount=0,candidateHalf=0;
template<class T>T asc_ldcg(T* p){
    const uintptr_t address=reinterpret_cast<uintptr_t>(p);
    if(address>=recordBegin && address<recordEnd) {
        const uint32_t i=((address-recordBegin)/sizeof(uint32_t))%recordHalf;
        assert(i<liveCount || (i>=candidateHalf && i<candidateHalf+liveCount));
    }
    return *p;
}
template<class T>void asc_stcg(T* p,T value){*p=value;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t value){auto before=*p;if(before==old)*p=value;return before;}
void asc_atomic_or(uint32_t* p,uint32_t value){*p|=value;}
void asc_syncthreads(){}
void asc_threadfence(){}
uint32_t asc_shfl_up(uint32_t value,uint32_t){return value;}
#include "planner_under_test.h"

constexpr uint32_t history=131328,resident=16384;
void Check(bool extended,uint32_t count,const std::vector<uint32_t>& inputSlots,
           const std::vector<uint32_t>& changes) {
    recordBegin=recordEnd=0;
    const uint32_t capacity=extended?kExtendedCandidates:kPlanCandidates;
    const uint32_t records=capacity*2,limit=extended?count:capacity;
    std::unordered_set<uint32_t> dstSet,srcSet;
    std::vector<uint32_t> expectedDst(records,UINT32_MAX),expectedSrc(records,UINT32_MAX);
    bool invalid=limit>capacity;
    uint32_t valid=0;
    if(!invalid)for(uint32_t i=0;i<limit;++i) {
        const auto s=changes[i],d=inputSlots[i];
        if(s==UINT32_MAX)continue;
        ++valid;
        invalid|=s>=history || d>=resident;
        invalid|=!srcSet.insert(s).second || !dstSet.insert(d).second;
        expectedSrc[i]=s;expectedDst[i]=d;
    }
    if(!invalid) {
        // Independently close the injective paths: outgoing destinations in
        // original order pair with freed sources in original order.
        uint32_t outgoing=capacity,freed=capacity;
        for(uint32_t i=0;i<limit;++i)if(changes[i]!=UINT32_MAX) {
            if(!srcSet.count(inputSlots[i]))expectedSrc[outgoing++]=inputSlots[i];
            if(!dstSet.count(changes[i]))expectedDst[freed++]=changes[i];
        }
        assert(outgoing==freed);
    }
    std::vector<uint32_t> work(extended?2*kExtendedRecords:kPlanWords,0x12345678);
    std::vector<uint32_t> hashes(2*kExtendedHash,0x87654321),status(128,0);
    std::vector<uint32_t> plan(extended?kExtendedTransferWords:kTransferPlanWords,0x91919191);
    if(extended) {
        // Reuse a genuinely populated maximum-sized previous round. The next
        // round may shrink to zero, holes, a short prefix, or an invalid count.
        std::vector<uint32_t> priorSlots(capacity),priorSources(capacity);
        std::iota(priorSlots.begin(),priorSlots.end(),0);
        std::iota(priorSources.begin(),priorSources.end(),resident);
        BuildExtendedPlan<1>(priorSlots.data(),priorSources.data(),capacity,history,resident,
                            hashes.data(),status.data(),work.data());
        assert(status[0]==0);
        std::fill(status.begin(),status.end(),0);
    }
    const auto priorWork=work;
    // Native API is mutable-pointer even though these two inputs are read-only.
    auto slots=inputSlots,input=changes;
    recordBegin=extended?reinterpret_cast<uintptr_t>(work.data()):0;
    recordEnd=extended?reinterpret_cast<uintptr_t>(work.data()+work.size()):0;
    recordHalf=records;candidateHalf=capacity;liveCount=count;
    membershipQueries=0;
    if(extended)BuildExtendedPlan<1>(slots.data(),input.data(),count,history,resident,
        hashes.data(),status.data(),work.data());
    else BuildUpdatePlan<1>(slots.data(),input.data(),history,resident,work.data(),status.data());
    assert(bool(status[0])==invalid);
    assert(slots==inputSlots && input==changes);
    if(invalid){assert(membershipQueries==0);return;}
    assert(membershipQueries==uint64_t(valid)*2); // Previously four probes.
    const auto dst=work.data()+(extended?0:2*kPlanHash),src=dst+records;
    for(uint32_t i=0;i<records;++i) {
        const bool live=!extended || i<count || (i>=capacity && i<capacity+count);
        assert(dst[i]==(live?expectedDst[i]:priorWork[i]));
        assert(src[i]==(live?expectedSrc[i]:priorWork[records+i]));
    }
    if(extended)PublishExtendedPlan<1>(work.data(),status.data(),plan.data(),resident);
    else PublishUpdatePlan<1>(work.data(),status.data(),plan.data(),resident);
    uint32_t rank=0;
    for(uint32_t i=0;i<records;++i)if(expectedSrc[i]!=UINT32_MAX && expectedSrc[i]!=expectedDst[i]) {
        const uint32_t tagged=expectedDst[i]|(i>=capacity && expectedDst[i]<resident?kInvalidateHot:0);
        assert(plan[kTransferHeaderWords+2*rank]==tagged);
        assert(plan[kTransferHeaderWords+2*rank+1]==expectedSrc[i]);++rank;
    }
    assert(plan[0]==rank);
}

int main() {
    std::mt19937 rng(216);
    for(bool extended:{false,true}) {
        const uint32_t capacity=extended?kExtendedCandidates:kPlanCandidates;
        std::vector<uint32_t> slots(capacity,UINT32_MAX),changes(capacity,UINT32_MAX);
        Check(extended,0,slots,changes);
        // A chain, a cycle crossing a bitset word, a self-move, and holes.
        slots[0]=0;changes[0]=1;slots[1]=1;changes[1]=2;slots[2]=2;changes[2]=resident+17;
        slots[31]=3;changes[31]=4;slots[32]=4;changes[32]=3;slots[63]=5;changes[63]=5;
        Check(extended,64,slots,changes);
        std::vector<uint32_t> destinations(resident),sources(history);
        for(uint32_t count:{1U,31U,32U,33U,63U,64U,65U,511U,512U,513U,2048U,2049U,4096U,4097U,10240U}) {
            if(count>capacity)continue;
            for(uint32_t trial=0;trial<3;++trial) {
                std::iota(destinations.begin(),destinations.end(),0);
                std::iota(sources.begin(),sources.end(),0);
                std::shuffle(destinations.begin(),destinations.end(),rng);
                std::shuffle(sources.begin(),sources.end(),rng);
                std::fill(changes.begin(),changes.end(),UINT32_MAX);
                for(uint32_t i=0;i<count;++i)if(trial==0 || (i+trial)%5) {
                    slots[i]=destinations[i];changes[i]=sources[i];
                }
                Check(extended,count,slots,changes);
            }
        }
        for(uint32_t bad=0;bad<4;++bad) {
            std::fill(changes.begin(),changes.end(),UINT32_MAX);
            slots[0]=0;changes[0]=resident+1;slots[1]=1;changes[1]=resident+2;
            if(bad==0)slots[1]=0; // Repeated destination.
            if(bad==1)changes[1]=changes[0]; // Repeated source.
            if(bad==2)slots[1]=resident;
            if(bad==3)changes[1]=history;
            Check(extended,2,slots,changes);
        }
        if(extended)Check(true,capacity+1,slots,changes);
    }
}
