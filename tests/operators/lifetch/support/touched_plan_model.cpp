// SPDX-License-Identifier: Apache-2.0
// Execute the production VF body on host lanes. The independent oracle composes
// dense physical permutations a chunk at a time, including freed-hot invalidations.
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

constexpr uint32_t THREADS=16, EMPTY=UINT32_MAX, kCapacity=20480, kPrefetch=2048;
struct Edge { uint32_t src,dst; };
struct Tuple { uint32_t fields[38]; };
struct Lane { uint32_t x; };
thread_local Lane threadIdx;
#define __ubuf__
#define __gm__
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(x)
struct BlockBarrier {
    std::mutex mutex;
    std::condition_variable cv;
    uint32_t count=0, epoch=0;
    void join() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto start=epoch;
        if(++count==THREADS){count=0;++epoch;cv.notify_all();}
        else cv.wait(lock,[&]{return epoch!=start;});
    }
} block;
void asc_syncthreads(){block.join();}
void asc_threadfence(){__atomic_thread_fence(__ATOMIC_SEQ_CST);}
uint32_t asc_atomic_or(uint32_t* address, uint32_t bits){
    return __atomic_fetch_or(address,bits,__ATOMIC_SEQ_CST);
}
uint32_t asc_atomic_cas(uint32_t* address,uint32_t old,uint32_t value){
    __atomic_compare_exchange_n(address,&old,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return old;
}
template<class T>T asc_ldcg(T* address){return *address;}
template<class T>void asc_stcg(T* address,T value){*address=value;}
uint32_t prefix(uint32_t x,uint32_t* scan){
    static uint32_t values[THREADS];
    values[threadIdx.x]=x;
    asc_syncthreads();
    uint32_t rank=0;
    for(uint32_t i=0;i<threadIdx.x;++i)rank+=values[i];
    if(threadIdx.x==THREADS-1)scan[15]=rank+x;
    asc_syncthreads();return rank;
}
#include "touched_device.h"

struct Event {uint32_t src,dst;};
using Phases=std::array<std::vector<Event>,5>;
constexpr uint32_t resident=128, history=131328;

void run(const Phases& events, const std::array<uint32_t,5>& dispatch, bool valid=true, bool overCapacity=false){
    std::array<uint32_t,5> counts;
    std::vector<uint32_t> sources(10240), destinations(10240);
    std::vector<uint32_t> position(history), inverse(history), invalid;
    std::vector<uint8_t> seen(history,0);
    bool direct=true;
    uint32_t finalCount=0, tested=0, chunks=0;
    std::iota(position.begin(),position.end(),0); inverse=position;
    for(uint32_t q=0;q<5;++q){
        const auto& e=events[q];counts[q]=e.size();
        auto off=q?kPrefetch+finalCount:0;
        if(q)finalCount+=counts[q];
        for(uint32_t i=0;i<e.size();++i){sources[off+i]=e[i].src;destinations[off+i]=e[i].dst;}
        if(!valid || overCapacity)continue;
        for(uint32_t base=0;base<e.size();base+=THREADS){
            uint32_t end=std::min<uint32_t>(base+THREADS,e.size());
            if(direct)tested+=end-base;
            ++chunks;
            std::vector<uint32_t> s,d,displaced,free;
            for(uint32_t i=base;i<end;++i){
                uint32_t r=e[i].src, v=e[i].dst;
                if(r<resident||seen[r]||seen[v])direct=false;
                seen[r]=seen[v]=1;
                s.push_back(inverse[r]);d.push_back(v);
            }
            for(uint32_t x:d)if(std::find(s.begin(),s.end(),x)==s.end())displaced.push_back(x);
            for(uint32_t x:s)if(std::find(d.begin(),d.end(),x)==d.end()){
                free.push_back(x);if(x<resident)invalid.push_back(x);
            }
            s.insert(s.end(),displaced.begin(),displaced.end());
            d.insert(d.end(),free.begin(),free.end());
            std::vector<uint32_t> old;
            for(uint32_t x:s)old.push_back(position[x]);
            for(uint32_t i=0;i<s.size();++i){position[d[i]]=old[i];inverse[old[i]]=d[i];}
        }
    }
    std::vector<uint32_t> m(tWords,0xa5a5a5a5), marks(10240*sizeof(Tuple)/4);
    std::vector<Edge> edges(kCapacity);
    std::vector<std::thread> pool;
    for(uint32_t tid=0;tid<THREADS;++tid)pool.emplace_back([&,tid]{
        threadIdx.x=tid;
        uint32_t first=0, offset=0;
        for(uint32_t end:dispatch){
            if(first==end)continue;
            ApplyPublishedPhases(sources.data(),destinations.data(),first,end,offset,
                counts[0],counts[1],counts[2],counts[3],counts[4],resident,history,
                edges.data(),marks.data(),m.data());
            asc_syncthreads();
            if(m[Err] || m[Mode]==2) break;
            for(uint32_t j=first;j<end;++j)if(j)offset+=counts[j];
            first=end;
        }
    });
    for(auto& t:pool)t.join();
    if(overCapacity){assert(!m[Err] && m[Mode]==2 && m[Processed]<=kTouchedEvents);return;}
    if(!valid){assert(m[Err]);return;}
    assert(!m[Err] && m[Mode]==(direct?1U:3U) && m[DirectTested]==tested && m[Chunks]==chunks);
    std::vector<uint32_t> actual(history);std::iota(actual.begin(),actual.end(),0);
    for(uint32_t i=0;i<m[Pairs];++i){
        auto e=edges[i];std::swap(actual[e.src],actual[e.dst]);
    }
    for(uint32_t i=0;i<m[Singles];++i){
        auto e=edges[kCapacity-1-i];actual[e.dst]=e.src;
    }
    assert(actual==position);
    std::vector<uint32_t> invLog;
    for(uint32_t i=0;i<m[Invalid];++i)invLog.push_back(marks[i*(sizeof(Tuple)/4)]);
    assert(invalid==invLog);
}

int main(){
    Phases e;
    run(e,{5,5,5,5,5});
    for(uint32_t i=0;i<63;++i)e[i%5].push_back({resident+i*97,i});
    run(e,{5,5,5,5,5});run(e,{1,2,3,4,5});
    e[4].push_back({e[0][0].src,127}); // late reselect must log the freed hot source
    run(e,{1,3,3,5,5});
    std::mt19937 rng(84177);
    for(uint32_t trial=0;trial<90;++trial){
        e={};
        for(auto& phase:e){
            for(uint32_t k=0;k<(1+rng()%3);++k){
                std::vector<uint32_t> s(400),d(resident);
                std::iota(s.begin(),s.end(),0);std::iota(d.begin(),d.end(),0);
                std::shuffle(s.begin(),s.end(),rng);std::shuffle(d.begin(),d.end(),rng);
                for(uint32_t i=0;i<THREADS;++i)phase.push_back({s[i],d[i]});
            }
        }
        run(e,trial%2?std::array<uint32_t,5>{1,2,3,4,5}:std::array<uint32_t,5>{5,5,5,5,5});
    }
    e={};e[0]={{10000,42},{10000,5}};run(e,{5,5,5,5,5},false);
    e[0]={{10000,42},{10001,42}};run(e,{5,5,5,5,5},false);
    e[0]={{history,0}};run(e,{5,5,5,5,5},false);
    e[0]={{resident,resident}};run(e,{5,5,5,5,5},false);
    e={};
    for(uint32_t q=0;q<2;++q)for(uint32_t j=0;j<1600;++j)
        e[q].push_back({resident+q*1600+j,j%resident});
    run(e,{5,5,5,5,5},true,true);
}
