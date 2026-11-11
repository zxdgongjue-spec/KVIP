// SPDX-License-Identifier: Apache-2.0
#include "copy_model.h"
#include "context.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace model {
struct Barrier {
    std::mutex mutex;std::condition_variable changed;
    unsigned threads,arrived=0,generation=0;
    explicit Barrier(unsigned n):threads(n){}
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);unsigned current=generation;
        if(++arrived==threads){arrived=0;++generation;changed.notify_all();}
        else changed.wait(lock,[&]{return generation!=current;});
    }
};
inline Barrier* barrier=nullptr;
inline uint8_t* payloadSource=nullptr;
inline uint8_t* payloadOutput=nullptr;
inline size_t sourceBytes=0,outputBytes=0;
inline unsigned worker=0;
inline std::atomic<unsigned> loads{0},stores{0},fences{0};
inline std::mutex ownerMutex;
inline std::map<uintptr_t,unsigned> sectorOwners;
inline thread_local std::vector<std::pair<uint32_t*,uint32_t>> pending;
}
struct ThreadIndex {unsigned x;};
thread_local ThreadIndex threadIdx;
inline void asc_syncthreads(){model::barrier->Wait();}
template<class T> inline T asc_ldcg(T* p) {
    uintptr_t address=reinterpret_cast<uintptr_t>(p),base=reinterpret_cast<uintptr_t>(model::payloadSource);
    if(address>=base && address<base+model::sourceBytes) {
        assert(sizeof(T)==4 && !(address&3) && address+sizeof(T)<=base+model::sourceBytes);
        ++model::loads;
    }
    return *p;
}
inline void asc_stcg(uint32_t* p,uint32_t value) {
    uintptr_t address=reinterpret_cast<uintptr_t>(p),base=reinterpret_cast<uintptr_t>(model::payloadOutput);
    assert(!(address&3) && address>=base && address+4<=base+model::outputBytes);
    model::pending.emplace_back(p,value);
}
inline unsigned asc_atomic_add(unsigned* p,unsigned value){return __atomic_fetch_add(p,value,__ATOMIC_RELAXED);}
inline unsigned asc_atomic_or(unsigned* p,unsigned value){return __atomic_fetch_or(p,value,__ATOMIC_RELAXED);}
inline void asc_threadfence() {
    // Model stores as deferred until each writer releases them. Recording
    // readiness before the collective fence would expose unchanged output.
    for(auto [p,value]:model::pending) {
        {
            std::lock_guard<std::mutex> lock(model::ownerMutex);
            auto result=model::sectorOwners.emplace(reinterpret_cast<uintptr_t>(p)/32,model::worker);
            assert(result.second || result.first->second==model::worker);
        }
        *p=value;++model::stores;
    }
    model::pending.clear();++model::fences;
}
#include "../op_kernel/copy/final_simt_copy.h"
namespace ubmem_device {inline void CleanCacheLines(uint8_t*,size_t){}}
#include "../op_kernel/copy/ubmem_copy.h"
#include "../op_kernel/resolver/final_worker_submit.h"
using namespace rotating;
uint64_t Address(const void* p){return reinterpret_cast<uintptr_t>(p);}

template<unsigned Threads>
void Invoke(Context& ctx,unsigned count,unsigned worker,std::vector<unsigned>& low,std::vector<unsigned>& high,
    std::vector<unsigned>& slots,std::vector<unsigned>& cfg) {
    model::Barrier barrier(Threads);model::barrier=&barrier;model::worker=worker;model::fences=0;
    std::vector<std::thread> threads;
    for(unsigned tid=0;tid<Threads;++tid)threads.emplace_back([&,tid] {
        threadIdx.x=tid;
        CopyFinalPayloadSimt<Threads>(&ctx,0,worker,count,low.data(),high.data(),slots.data(),cfg.data());
        assert(model::pending.empty());
    });
    for(auto& thread:threads)thread.join();
    if(cfg[kSimtCopyReady])assert(model::fences==Threads && cfg[kSimtCopyCompleted]==cfg[kSimtCopyScheduled]);
    else assert(model::fences==0 && model::stores==0);
}

template<unsigned Threads>
void Check(unsigned count,unsigned fault=0) {
    const unsigned stride=SIM_RESIDENT+384,owner=2;
    const size_t outputBytes=size_t(owner+1)*stride*kTokenBytes,sourceBytes=(count+1)*kTokenBytes+8;
    std::unique_ptr<uint8_t,decltype(&std::free)> output(
        static_cast<uint8_t*>(std::aligned_alloc(128,(outputBytes+255)/128*128)),&std::free);
    std::vector<uint32_t> storage((sourceBytes+3)/4+32);
    auto* source=reinterpret_cast<uint8_t*>(storage.data());
    for(size_t i=0;i<sourceBytes;++i)source[i]=uint8_t((i*17+i/19)%251);
    std::memset(output.get(),0xa5,outputBytes+128);
    std::vector<uint8_t> expected(outputBytes,0xa5);
    UbmemRegion region{kUbmemMagic,kUbmemVersion,0x100000000ULL,Address(source),sourceBytes,1,
        Address(output.get()),outputBytes,0,kUbmemHostLocal};
    Row row{};row.output=Address(output.get())+uint64_t(owner)*stride*kTokenBytes;
    std::vector<uint64_t> ownerStorage((kFinalOwnerBytes+7)/8);
    Context ctx{};ctx.ubmemRegion=Address(&region);ctx.rows=Address(&row);ctx.owners=Address(ownerStorage.data());
    std::vector<unsigned> low(count),high(count),slots(count),cfg(32,0);
    for(unsigned rank=0;rank<count;++rank) {
        slots[rank]=(rank*109+17)%SIM_RESIDENT; // Permuted destinations, including odd/even partners.
        const uint64_t peer=region.peerBase+uint64_t(count-rank-1)*kTokenBytes;
        low[rank]=uint32_t(peer);high[rank]=uint32_t(peer>>32);
    }
    if(fault==1){region.magic=0;}
    if(fault==2 && count){low[0]=uint32_t(region.peerBase+region.bytes-4);high[0]=uint32_t((region.peerBase+region.bytes-4)>>32);}
    if(fault==3 && count){slots[0]=SIM_RESIDENT;}
    if(fault==4){row.output=region.outputBase+region.outputBytes;}
    if(fault==5 && count){++low[0];} // Valid unaligned source: delegate the entire owning AIV to MTE.
    if(fault==6){region.mappedBase=UINT64_MAX-8;}
    if(fault==7){ctx.ubmemRegion=0;}
    model::payloadSource=source;model::sourceBytes=sourceBytes;
    model::payloadOutput=output.get();model::outputBytes=outputBytes;model::sectorOwners.clear();
    std::vector<uint8_t> scratch(2*kSlots*sizeof(uint32_t));
    for(unsigned worker=0;worker<kFinalWorkers;++worker) {
        std::fill(cfg.begin(),cfg.end(),0);model::loads=model::stores=0;
        Invoke<Threads>(ctx,count,worker,low,high,slots,cfg);
        const bool ownedFault=count && (slots[0]>>1)%kFinalWorkers==worker;
        const bool error=fault==1 || fault==4 || fault==6 || fault==7 || ((fault==2 || fault==3) && ownedFault);
        if(error){assert(cfg[1] && !model::loads && !model::stores && !cfg[kSimtCopyReady]);}
        else if(fault==5 && ownedFault){assert(!cfg[1] && !cfg[kSimtCopyReady] && cfg[kSimtCopyFallback]);}
        else {assert(!cfg[1] && cfg[kSimtCopyReady]);}
        auto post=FinalPostAt(&ctx,worker);
        copy_model::worker=worker;copy_model::owners.clear();
        const auto loadCommands=copy_model::loadCommands,storeCommands=copy_model::storeCommands;
        PublishFinalWorker(&ctx,&row,0,worker,0,count,0,cfg[1],
            {reinterpret_cast<uint32_t*>(scratch.data()),scratch.size()},
            {low.data(),low.size()*4},{high.data(),high.size()*4},{slots.data(),slots.size()*4},{cfg.data(),cfg.size()*4});
        copy_model::Drained();
        if(error){assert(post->error && post->head==0);continue;}
        unsigned owned=0;
        for(unsigned rank=0;rank<count;++rank)if((slots[rank]>>1)%kFinalWorkers==worker) {
            const uint64_t peer=uint64_t(low[rank])|(uint64_t(high[rank])<<32);
            std::memcpy(expected.data()+(size_t(owner)*stride+slots[rank])*kTokenBytes,
                        source+(peer-region.peerBase),kTokenBytes);++owned;
        }
        assert(post->count==owned && post->head==owned && !post->pending && !post->error);
        if(cfg[kSimtCopyReady]) {
            assert(model::loads==owned*(kTokenBytes/4) && model::stores==model::loads);
            assert(copy_model::loadCommands==loadCommands && copy_model::storeCommands==storeCommands);
        }
    }
    assert(std::memcmp(output.get(),expected.data(),outputBytes)==0);
    for(size_t i=outputBytes;i<outputBytes+128;++i)assert(output.get()[i]==0xa5);
}

int main() {
    for(unsigned count:{0U,1U,2U,17U,129U,2049U,8192U})Check<32>(count);
    Check<64>(257);
    for(unsigned fault=1;fault<=7;++fault)Check<32>(17,fault);
    std::cout<<"SIMT final actual-helper byte/sector/fence/fallback/FinalPost model PASS workers="<<kFinalWorkers<<"\n";
}
