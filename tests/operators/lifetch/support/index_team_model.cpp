// SPDX-License-Identifier: Apache-2.0
// Run the device field-transfer body on CPU with real SIMT block barriers.
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(n)
#define __gm__
#define __ubuf__
thread_local struct { unsigned x; } threadIdx;
thread_local std::barrier<>* rendezvous;
void asc_syncthreads() { rendezvous->arrive_and_wait(); }
void asc_threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
uint32_t asc_atomic_cas(uint32_t* p,uint32_t expected,uint32_t desired) {
    std::atomic_ref<uint32_t>(*p).compare_exchange_strong(expected,desired);
    return expected;
}
template<class T> T asc_ldcg(T* address) { return *address; }
template<class T> void asc_stcg(T* address,T value) { *address=value; }
constexpr unsigned CHUNK=512,kPrefetch=2048,kMaxFinal=8192,THREADS=32,kMaxMoves=1024;
#include "index_team_plan.h"
#include "index_team_fields.h"

struct Row {
    std::array<uint64_t,16> key;
    uint64_t addr,id;
    uint32_t scale;
    bool operator==(const Row&) const=default;
};

int main() {
    constexpr unsigned history=1280,resident=384,chunks=23;
    std::mt19937 rng(771636);
    std::vector<Row> expected(history);
    std::vector<uint64_t> keys(history*16),addresses(history),ids(history);
    std::vector<uint32_t> scales(history),valid(resident,1),expectedValid=valid;
    for(unsigned i=0;i<history;++i) {
        for(unsigned j=0;j<16;++j) keys[16*i+j]=expected[i].key[j]=(uint64_t(rng())<<32)|rng();
        addresses[i]=expected[i].addr=uint64_t(rng())<<16;
        ids[i]=expected[i].id=(uint64_t(rng())<<32)|i;
        scales[i]=expected[i].scale=rng();
    }
    std::vector<uint32_t> plan(kIndexEdgesOffset+4*(kPrefetch+kMaxFinal));
    std::array<uint32_t,32> control{};
    std::vector<unsigned> positions(history),sources;
    std::iota(positions.begin(),positions.end(),0);
    for(unsigned c=0;c<chunks;++c) {
        unsigned n=c==0 ? 0 : c%3==0 ? kMaxMoves : c%3==1 ? 3 : 447;
        std::shuffle(positions.begin(),positions.end(),rng);
        sources.assign(positions.begin(),positions.begin()+n);
        std::shuffle(sources.begin(),sources.end(),rng);
        const auto prior=expected;
        const uint32_t primaries=(n+1)/2;
        std::array<uint32_t,kMaxMoves> physSrc{},physDst{};
        control[4]=c;
        for(unsigned i=0;i<n;++i){
            unsigned src=sources[i],dst=positions[i];
            bool freed=i>=primaries && dst<resident;
            unsigned j=i<primaries?i:CHUNK+i-primaries;
            physSrc[j]=src;physDst[j]=dst;
            expected[dst]=prior[src];
            if(freed)expectedValid[dst]=0;
        }
        std::barrier sync(THREADS);
        std::vector<std::thread> writers;
        for(unsigned t=0;t<THREADS;++t)writers.emplace_back([&,t]{
            threadIdx.x=t;rendezvous=&sync;
            RecordIndexChunk(plan.data(),physSrc.data(),physDst.data(),control.data(),primaries,n,resident);
        });
        for(auto& t:writers)t.join();
    }
    // Independently delayed owners complete in reverse lane order. No
    // metadata, key word or scale can depend on a different lane's gather.
    uint32_t claim=0;
    for(unsigned pass=0;pass<4;++pass) {
        unsigned lane=3-pass;
        std::vector<uint64_t> ub(kMaxMoves*6);
        std::barrier sync(256);
        std::vector<std::thread> workers;
        for(unsigned t=0;t<256;++t) workers.emplace_back([&,t]{
            threadIdx.x=t;rendezvous=&sync;
            if(lane<3)CopyFields<true>(plan.data(),chunks,lane,keys.data(),scales.data(),addresses.data(),
                                      ids.data(),valid.data(),ub.data(),&claim,lane);
            else CopyFields<false>(plan.data(),chunks,lane,keys.data(),scales.data(),addresses.data(),
                                   ids.data(),valid.data(),ub.data());
        });
        for(auto& t:workers)t.join();
        if(lane<3 && uint32_t(ub[0])!=unsigned(lane==2))return 3;
    }
    if(claim!=3)return 4;
    for(unsigned i=0;i<history;++i){
        Row actual{{},addresses[i],ids[i],scales[i]};
        std::copy_n(keys.begin()+16*i,16,actual.key.begin());
        if(!(actual==expected[i]))return 1;
    }
    if(valid!=expectedValid)return 2;
    std::cout<<"INDEX_TEAM_CPU_PASS chunks="<<chunks<<" ordered_moves="<<plan[chunks-1]
             <<" lanes=4 fields=keys,scales,addresses,logical,hot_valid\n";
}
