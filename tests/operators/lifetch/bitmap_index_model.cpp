// SPDX-License-Identifier: Apache-2.0
// Execute the production planners against the original hash specialization and
// an independent ordered-set oracle. No hardware timing is inferred here.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
struct {uint32_t x=0;} threadIdx;
std::atomic<uint64_t> casCalls{0},orCalls{0};
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t expected,uint32_t value) {
    ++casCalls;__atomic_compare_exchange_n(p,&expected,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return expected;
}
uint32_t asc_atomic_or(uint32_t* p,uint32_t value) {
    ++orCalls;return __atomic_fetch_or(p,value,__ATOMIC_SEQ_CST);
}
void asc_syncthreads(){}
void asc_threadfence(){}
uint32_t asc_shfl_up(uint32_t value,uint32_t){return value;}
#include "index_cache_update.h"

constexpr uint32_t poison=0xdeadbeef,guard=16;
struct Result {uint32_t error;std::vector<uint32_t> dst,src,plan;uint64_t cas,ors;};
template<bool Bitmap>
Result Run(bool extended,uint32_t count,uint32_t history,uint32_t resident,
    const std::vector<uint32_t>& slots,const std::vector<uint32_t>& changes) {
    const uint32_t capacity=extended?kExtendedCandidates:kPlanCandidates;
    const uint32_t recordWords=extended?kExtendedWorkWords:kPlanWords;
    const uint32_t hashWords=extended?2*kExtendedHash:2*kPlanHash;
    std::vector<uint32_t> backing(recordWords+2*guard,poison),hashBacking(hashWords+2*guard,poison),status(128);
    auto work=backing.data()+guard,hashes=hashBacking.data()+guard;
    auto inputSlots=slots,input=changes;
    casCalls=0;orCalls=0;
    if(extended)BuildExtendedPlan<1,Bitmap>(inputSlots.data(),input.data(),count,history,resident,hashes,status.data(),work);
    else BuildUpdatePlan<1,Bitmap>(inputSlots.data(),input.data(),history,resident,work,status.data());
    const auto cas=casCalls.load(),ors=orCalls.load();
    assert(slots==inputSlots && changes==input);
    for(const auto* storage:{&backing,&hashBacking})for(uint32_t i=0;i<guard;++i) {
        assert((*storage)[i]==poison && (*storage)[storage->size()-1-i]==poison);
    }
    const bool bitmap=Bitmap && uint64_t(PlanBitmapWords(history))+PlanBitmapWords(resident)<=hashWords;
    if(bitmap && (!extended || count<=capacity)) {
        const auto sets=extended?hashes:work;
        // A selected bitmap must not clear the unused hash capacity.
        for(uint32_t i=PlanBitmapWords(history)+PlanBitmapWords(resident);i<hashWords;++i)assert(sets[i]==poison);
        assert(cas==0);
    }
    auto dst=work+(extended?0:2*kPlanHash),src=dst+2*capacity;
    std::vector<uint32_t> plan(extended?kExtendedTransferWords:kTransferPlanWords,poison);
    if(!status[0]) {
        if(extended)PublishExtendedPlan<1>(work,status.data(),plan.data(),resident);
        else PublishUpdatePlan<1>(work,status.data(),plan.data(),resident);
        plan.resize(kTransferHeaderWords+2*plan[0]);
    } else plan.clear();
    return {status[0],{dst,dst+2*capacity},{src,src+2*capacity},std::move(plan),cas,ors};
}

void Check(bool extended,uint32_t count,uint32_t history,uint32_t resident,
    const std::vector<uint32_t>& slots,const std::vector<uint32_t>& changes) {
    const uint32_t capacity=extended?kExtendedCandidates:kPlanCandidates,limit=extended?count:capacity;
    std::unordered_set<uint32_t> srcSet,dstSet;
    bool bad=count>capacity;
    std::vector<uint32_t> expectedDst(2*capacity,UINT32_MAX),expectedSrc(2*capacity,UINT32_MAX);
    if(!bad)for(uint32_t i=0;i<limit;++i) {
        uint32_t s=changes[i],d=slots[i];if(s==UINT32_MAX)continue;
        bad|=s>=history || d>=resident;
        bad|=!srcSet.insert(s).second || !dstSet.insert(d).second;
        expectedDst[i]=d;expectedSrc[i]=s;
    }
    if(!bad) {
        uint32_t outgoing=capacity,freed=capacity;
        for(uint32_t i=0;i<limit;++i)if(changes[i]!=UINT32_MAX) {
            if(!srcSet.count(slots[i]))expectedSrc[outgoing++]=slots[i];
            if(!dstSet.count(changes[i]))expectedDst[freed++]=changes[i];
        }
        assert(outgoing==freed);
    }
    const auto baseline=Run<false>(extended,count,history,resident,slots,changes);
    const auto bitmap=Run<true>(extended,count,history,resident,slots,changes);
    assert(bool(baseline.error)==bad && bitmap.error==baseline.error);
    if(bad)return;
    assert(baseline.dst==bitmap.dst && baseline.src==bitmap.src && baseline.plan==bitmap.plan);
    uint32_t rank=0;
    for(uint32_t i=0;i<2*capacity;++i) {
        bool live=!extended || i<count || (i>=capacity && i<capacity+count);
        assert(bitmap.dst[i]==(live?expectedDst[i]:poison));
        assert(bitmap.src[i]==(live?expectedSrc[i]:poison));
        if(expectedSrc[i]==UINT32_MAX || expectedSrc[i]==expectedDst[i])continue;
        const uint32_t tagged=expectedDst[i]|(i>=capacity && expectedDst[i]<resident?kInvalidateHot:0);
        assert(bitmap.plan[kTransferHeaderWords+2*rank]==tagged);
        assert(bitmap.plan[kTransferHeaderWords+2*rank+1]==expectedSrc[i]);++rank;
    }
    assert(bitmap.plan[0]==rank);
}

void AtomicWordContention() {
    // Different IDs share a word; duplicate claims must succeed exactly once,
    // including bit31. This calls the production helper with host atomics.
    uint32_t word=0;std::atomic<uint32_t> inserted{0};std::vector<std::thread> workers;
    for(uint32_t i=0;i<64;++i)workers.emplace_back([&,i]{
        for(uint32_t repeat=0;repeat<32;++repeat)inserted+=PlanBitmapInsert(&word,i%32);
    });
    for(auto& worker:workers)worker.join();
    assert(word==UINT32_MAX && inserted==32);
}
int main() {
    std::mt19937 random(9201);
    assert(PlanBitmapWords(0)==0 && PlanBitmapWords(31)==1 && PlanBitmapWords(32)==1);
    assert(PlanBitmapWords(33)==2 && PlanBitmapWords(UINT32_MAX)==134217728);
    AtomicWordContention();
    for(bool extended:{false,true}) {
        const uint32_t capacity=extended?kExtendedCandidates:kPlanCandidates;
        std::vector<uint32_t> slots(capacity,UINT32_MAX),changes(capacity,UINT32_MAX);
        for(auto geometry:std::vector<std::pair<uint32_t,uint32_t>>{
            {0,0},{31,17},{33,65},{131328,16384},{524288-16384,16384},
            {524289-16384,16384},{1048576-16384,16384},{1048577-16384,16384},{UINT32_MAX,16384}}) {
            auto [history,resident]=geometry;
            std::fill(changes.begin(),changes.end(),UINT32_MAX);
            Check(extended,0,history,resident,slots,changes);
            if(!history || !resident)continue;
            slots[0]=resident-1;changes[0]=history-1;
            if(changes[0]==UINT32_MAX) --changes[0];
            Check(extended,1,history,resident,slots,changes);
            slots[1]=0;changes[1]=0;Check(extended,2,history,resident,slots,changes);
        }
        constexpr uint32_t history=131328,resident=16384;
        for(uint32_t count:{1U,31U,32U,33U,511U,512U,513U,2048U,2049U,4096U,8192U,10240U}) {
            if(count>capacity)continue;
            std::vector<uint32_t> ds(resident),ss(history);
            for(uint32_t trial=0;trial<4;++trial) {
                std::iota(ds.begin(),ds.end(),0);std::iota(ss.begin(),ss.end(),0);
                if(trial%2){std::shuffle(ds.begin(),ds.end(),random);std::shuffle(ss.begin(),ss.end(),random);}
                std::fill(changes.begin(),changes.end(),UINT32_MAX);
                for(uint32_t i=0;i<count;++i)if(trial<2 || i%5) {
                    slots[i]=ds[i];changes[i]=trial==0?resident+i:ss[i];
                }
                Check(extended,count,history,resident,slots,changes);
            }
        }
        // Cycles, paths, self moves and holes cross word boundaries.
        std::fill(changes.begin(),changes.end(),UINT32_MAX);
        const uint32_t destinations[]={0,1,2,31,32,63,64},sources[]={1,2,resident+1,32,31,63,history-1};
        for(uint32_t i=0;i<7;++i){slots[i*9]=destinations[i];changes[i*9]=sources[i];}
        Check(extended,64,history,resident,slots,changes);
        for(uint32_t bad=0;bad<5;++bad) {
            std::fill(changes.begin(),changes.end(),UINT32_MAX);
            slots[0]=31;changes[0]=history-1;slots[1]=32;changes[1]=history-2;
            if(bad==0)slots[1]=slots[0];
            if(bad==1)changes[1]=changes[0];
            if(bad==2)slots[1]=resident;
            if(bad==3)changes[1]=history;
            if(bad==4){slots[0]=slots[1]=0;changes[0]=changes[1]=0;}
            Check(extended,2,history,resident,slots,changes);
        }
        if(extended)Check(true,capacity+1,history,resident,slots,changes);
    }
}
