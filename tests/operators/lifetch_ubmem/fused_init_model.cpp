// SPDX-License-Identifier: Apache-2.0
// Execute the production init-to-LI entry with a blocking host collective.
// Actual reset bytes and validation publication use the separate init model.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>
#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (isAiv)
using GM_ADDR=uint8_t*;
thread_local bool isAiv;
thread_local uint32_t core;
inline uint32_t GetBlockIdx() {return core;}
constexpr uint32_t AIVS=MODEL_AICS*2;
std::array<std::atomic<uint32_t>,SIM_BATCH> initialized;
std::atomic<uint32_t> arrivals,vectorStarts,cubeStarts,cubeProceeds;
std::mutex mutex;
std::condition_variable wake;
bool releaseInit=false;
namespace AscendC {
struct TPipe {void Destroy() {}};
template<bool OnlyVectors> void SyncAll() {
    static_assert(OnlyVectors);
    assert(isAiv);
    std::unique_lock<std::mutex> lock(mutex);
    if(++arrivals==AIVS)wake.notify_all();
    else wake.wait(lock,[]{return arrivals==AIVS;});
}
}
namespace rotating {
constexpr uint32_t kLiCores=MODEL_AICS;
struct Context {uint64_t queryEnds;};
namespace init {
using namespace AscendC;
void InitializeFetchCore(Context*,uint32_t b) {
    if(b>=SIM_BATCH)return;
    // Delay the final owner; inactive rows still participate and initialize.
    if(b==SIM_BATCH-1) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[]{return releaseInit;});
    }
    assert(initialized[b]++==0);
}
#include "before_li_actual.h"
}
}
namespace QLICommon {
struct fp8_e4m3fn_t {};struct bfloat16_t {};
enum class LI_LAYOUT {BSND,PA_BSND};
template<class...> struct Unused {};
template<class Q,class K,class O,bool PA,LI_LAYOUT QL,LI_LAYOUT KL,class... Others> struct QLIType {};
}
struct QLITilingData {uint32_t b,t,h,q,s,top,cores,block,maxBlocks,mode,keyStride,scaleStride,resident,split,aics;};
namespace QLIKernel {
struct VectorService {
    void SetPrefixStorage(GM_ADDR) {}
    void SetRotationContext(GM_ADDR) {}
};
template<class Type> struct QuantLightningIndexerKernel {
    uint64_t queryEnds_;
    VectorService vectorService;
    template<class... Args> void Init(Args...) {
        if(isAiv) {
            assert(arrivals==AIVS);
            for(auto& value:initialized)assert(value==1);
            ++vectorStarts;
        } else ++cubeStarts;
    }
    void Process() {
        if(isAiv) {std::lock_guard<std::mutex> lock(mutex);wake.notify_all();return;}
        // Model the existing ComputeMm1 two-vector credits. No cube read or
        // output can pass before the AIV entry has completed the collective.
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[]{return vectorStarts==AIVS;});
        ++cubeProceeds;
    }
};
}
#include "run_indexer_actual.h"
int main() {
    rotating::Context ctx{};std::array<uint64_t,15> bindings{};bindings[14]=uint64_t(&ctx);
    for(uint32_t replay=0;replay<8;++replay) {
        for(auto& value:initialized)value=0;
        arrivals=vectorStarts=cubeStarts=cubeProceeds=0;releaseInit=false;
        std::vector<std::thread> workers;
        for(uint32_t i=0;i<MODEL_AICS;++i)workers.emplace_back([&,i]{isAiv=false;core=i;RunIndexer<true>(reinterpret_cast<GM_ADDR>(bindings.data()));});
        for(uint32_t i=0;i<AIVS;++i)workers.emplace_back([&,i]{isAiv=true;core=(i+replay)%AIVS;RunIndexer<true>(reinterpret_cast<GM_ADDR>(bindings.data()));});
        while(cubeStarts!=MODEL_AICS || arrivals!=AIVS-1)std::this_thread::yield();
        assert(vectorStarts==0 && cubeProceeds==0 && initialized[SIM_BATCH-1]==0);
        {std::lock_guard<std::mutex> lock(mutex);releaseInit=true;wake.notify_all();}
        for(auto& worker:workers)worker.join();
        assert(arrivals==AIVS && vectorStarts==AIVS && cubeProceeds==MODEL_AICS);
        for(auto& value:initialized)assert(value==1);
    }
}
