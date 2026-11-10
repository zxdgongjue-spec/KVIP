// SPDX-License-Identifier: Apache-2.0
// Execute production init bodies with delayed DMA, not simulated device timing.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __simt_callee__
#define __gm__
#define LAUNCH_BOUND(n)
struct {uint32_t x;} threadIdx;
inline uint32_t currentPart=0,negativeFills=0,vfCalls=0;
inline uint32_t* reservedBegin=nullptr;
inline uint32_t* reservedEnd=nullptr;
inline uint32_t asc_ldcg(const uint32_t* p) {return *p;}
inline void asc_stcg(uint32_t* p,uint32_t value) {
    if(uintptr_t(p)>=uintptr_t(reservedBegin) && uintptr_t(p)<uintptr_t(reservedEnd))assert(currentPart==0);
    *p=value;
}
inline void asc_threadfence() {}
namespace AscendC {
constexpr uint32_t EVENT_ID0=0;
enum class HardEvent {V_MTE3,MTE3_V,V_S,MTE3_S,MTE2_MTE3};
inline bool flags[5]{};
inline std::vector<std::function<void()>> pendingMte2,pendingMte3;
inline void Drain(std::vector<std::function<void()>>& pending) {for(auto& op:pending)op();pending.clear();}
template<HardEvent event> void SetFlag(uint32_t) {assert(!flags[unsigned(event)]);flags[unsigned(event)]=true;}
template<HardEvent event> void WaitFlag(uint32_t) {
    assert(flags[unsigned(event)]);flags[unsigned(event)]=false;
    if constexpr(event==HardEvent::MTE3_V || event==HardEvent::MTE3_S)Drain(pendingMte3);
    if constexpr(event==HardEvent::MTE2_MTE3)Drain(pendingMte2);
}
template<class T> struct LocalTensor {T* data;};
template<class T> struct GlobalTensor {
    T* data=nullptr;
    void SetGlobalBuffer(T* value) {data=value;}
    GlobalTensor operator[](uint32_t offset)const {return {data+offset};}
};
template<class T> void Duplicate(LocalTensor<T> output,T value,uint32_t count) {
    assert(pendingMte2.empty() && pendingMte3.empty());
    negativeFills+=(value==UINT32_MAX);
    std::fill_n(output.data,count,value);
}
template<class T> void DataCopy(GlobalTensor<T> output,LocalTensor<T> input,uint32_t count) {
    assert(count%8==0 && pendingMte2.empty());
    pendingMte3.push_back([=] {std::copy_n(input.data,count,output.data);});
}
template<class T> void DataCopy(LocalTensor<T> output,GlobalTensor<T> input,uint32_t count) {
    // Reusing the sentinel UB before its GM copies complete would corrupt
    // both mapping and reservedHot. Model reads the UB at completion time.
    assert(count%8==0 && pendingMte3.empty());
    pendingMte2.push_back([=] {std::copy_n(input.data,count,output.data);});
}
enum class TPosition {VECCALC};
template<TPosition> struct TBuf {
    std::array<uint32_t,4096> words;
    template<class T> LocalTensor<T> Get() {return {reinterpret_cast<T*>(words.data())};}
};
struct TPipe {
    template<class Buffer> void InitBuffer(Buffer&,uint32_t size) {assert(size==4096*4);}
    void Destroy() {assert(pendingMte2.empty() && pendingMte3.empty());for(bool flag:flags)assert(!flag);}
};
namespace Simt {
struct Dim3 {uint32_t x,y,z;};
template<auto function,class... Args> void VF_CALL(Dim3 grid,Args... args) {
    assert(pendingMte2.empty() && pendingMte3.empty());++vfCalls;
    for(threadIdx.x=0;threadIdx.x<grid.x;++threadIdx.x)function(args...);
}
}
}
namespace kvaf_device {inline void CleanCacheLines(uint8_t*,uint32_t) {}}
namespace ubmem_device {inline void CleanCacheLines(uint8_t*,uint32_t) {}}
namespace rotating {
constexpr uint32_t kSlots=8192,kLruSlots=2048,kChangeCapacity=kSlots+kLruSlots;
constexpr uint32_t kRounds=8,kServiceLanes=4,kFinalWorkers=4,kFinalStateWords=288;
constexpr uint32_t kFinalPostOffset=SIM_BATCH*128+SIM_BATCH*kFinalWorkers*kFinalStateWords*4;
constexpr uint32_t kRowStatusWords=32,kUbmemStatusWords=16,kInitWorkersPerBatch=4;
struct Row {uint32_t words[256];};
struct Progress {uint32_t words[32];};
struct Event {uint32_t words[32];};
struct FinalPost {uint32_t words[32];};
struct KvafBatchGetCoreStatus {uint32_t words[32];};
struct Context {
    uint64_t rows,progress,owners,rowStatus,events,count,finalStatus,ubmemStatus;
    uint64_t mapping,selected,changes,changeSlots,changeCount,lruSlots,reservedHot,queryEnds,tailBounds,outputStride;
};
}
#include "reserved_hot.h"
#include "init_partition.h"
using namespace AscendC;
using namespace rotating;
// Extracted verbatim from the selected production lifecycle/header.
#include "reserved_hot_lifecycle.h"

int main() {
    constexpr uint32_t batches=SIM_BATCH,resident=SIM_RESIDENT,history=SIM_HISTORY;
    constexpr uint32_t parts=MODEL_PARALLEL?kInitWorkersPerBatch:1;
    std::vector<uint32_t> candidates(batches*kLruSlots),ends(batches*4),expected;
    std::vector<uint32_t> output(batches*resident+32,91),map(batches*history+32,91);
    std::vector<uint32_t> selected(batches*kSlots+32,91),changes(batches*kChangeCapacity+32,91);
    std::vector<uint32_t> slots(batches*kChangeCapacity+32,91),counts(batches),bounds(batches*2);
    std::vector<uint32_t> control(32768,0);
    Context ctx{};
    ctx.rows=ctx.progress=ctx.owners=ctx.rowStatus=ctx.events=ctx.count=ctx.finalStatus=uint64_t(control.data());
    ctx.mapping=uint64_t(map.data());ctx.selected=uint64_t(selected.data());ctx.changes=uint64_t(changes.data());
    ctx.changeSlots=uint64_t(slots.data());ctx.changeCount=uint64_t(counts.data());
    ctx.lruSlots=uint64_t(candidates.data());ctx.reservedHot=uint64_t(output.data());ctx.queryEnds=uint64_t(ends.data());
    ctx.outputStride=resident+384;
    reservedBegin=output.data();reservedEnd=output.data()+batches*resident;
    // Full -> shrink/reorder -> empty -> owner/row reuse -> optional bounds.
    for(uint32_t round=0;round<6;++round) {
        const uint32_t active=round==0?4:round==1?1:round==2?0:round==3?3:4;
        std::fill(ends.begin(),ends.end(),0);expected=output;
        ctx.queryEnds=round==4?0:uint64_t(ends.data());
        ctx.tailBounds=round==1?0:uint64_t(bounds.data());
        for(uint32_t b=0;b<batches;++b) {
            bounds[b*2]=history/2-128;bounds[b*2+1]=bounds[b*2]+(round==2?0:384);
            for(uint32_t i=0;i<kLruSlots;++i)
                candidates[b*kLruSlots+i]=(resident-1024+i+round*173+b*991)%resident;
            if(b<active) {
                ends[b*4+round%4]=130000;
#if V9_COMPACT_RESERVED
                std::fill_n(expected.begin()+b*resident,resident,UINT32_MAX);
                for(uint32_t i=0;i<kLruSlots;++i)expected[b*resident+candidates[b*kLruSlots+i]]=i;
#endif
            }
            for(uint32_t part=0;part<parts;++part) {
                // Reverse part order on alternate rounds: no other shard may
                // touch the complete reserved row owned by part zero.
                currentPart=round%2?parts-1-part:part;
                auto fills=negativeFills,calls=vfCalls;
#if MODEL_URMA
                auto reserved=InitializeRound(batches,history,resident,kRounds,kServiceLanes,kFinalWorkers,kFinalPostOffset,&ctx,b);
                Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},history,resident,&ctx,b,reserved);
#elif MODEL_PARALLEL
                InitializeRoundParallel(&ctx,b,currentPart);
#else
                InitializeRound(&ctx,b);
#endif
                assert(negativeFills-fills==1); // No second Duplicate(-1).
                assert(vfCalls-calls==1); // Tail and protection map share a VF.
                assert(pendingMte2.empty() && pendingMte3.empty());
                for(bool flag:flags)assert(!flag);
            }
            assert(counts[b]==kLruSlots);
            assert(std::equal(candidates.begin()+b*kLruSlots,candidates.begin()+(b+1)*kLruSlots,
                slots.begin()+b*kChangeCapacity));
            for(uint32_t token=0;token<history;++token) {
                uint32_t want=UINT32_MAX;
                if(ctx.tailBounds && token>=bounds[b*2] && token<bounds[b*2+1])
                    want=resident+kSlots+token%384;
                assert(map[b*history+token]==want);
            }
        }
        assert(output==expected);
        assert(std::all_of(changes.begin(),changes.begin()+batches*kChangeCapacity,[](uint32_t x){return x==UINT32_MAX;}));
#if !MODEL_URMA
        assert(std::all_of(selected.begin(),selected.begin()+batches*kSlots,[](uint32_t x){return x==UINT32_MAX;}));
#endif
        for(const auto* values:{&output,&map,&selected,&changes,&slots})
            assert(std::all_of(values->end()-32,values->end(),[](uint32_t x){return x==91;}));
    }
#if V9_COMPACT_RESERVED
    std::array<uint32_t,4096> local{};
    assert(ClearReservedHot(0,ctx.reservedHot,0,0,resident,LocalTensor<uint32_t>{local.data()})==0);
    assert(ClearReservedHot(ctx.lruSlots,0,0,0,resident,LocalTensor<uint32_t>{local.data()})==0);
    assert(pendingMte3.empty());
#endif
}
