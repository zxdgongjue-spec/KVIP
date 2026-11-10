// SPDX-License-Identifier: Apache-2.0
// Execute production init bodies with delayed DMA, not simulated device timing.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __simt_callee__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
struct {uint32_t x;} threadIdx;
inline uint32_t currentPart=0,negativeFills=0,vfCalls=0;
inline uint32_t* reservedBegin=nullptr;
inline uint32_t* reservedEnd=nullptr;
inline uint32_t* mappingBegin=nullptr;
inline uint32_t* mappingEnd=nullptr;
inline uint64_t mappingWords=0;
inline uint64_t reservedStores=0,reservedCopies=0;
// Match CANN's intrinsic signature: it loads through a mutable GM pointer,
// even when the caller's source data is logically read-only.
template<class T> inline T asc_ldcg(T* p) {return *p;}
inline void asc_stcg(uint32_t* p,uint32_t value) {
    if(uintptr_t(p)>=uintptr_t(reservedBegin) && uintptr_t(p)<uintptr_t(reservedEnd)) {
        assert(currentPart==0);++reservedStores;
    }
    *p=value;
}
inline void asc_threadfence() {}
namespace AscendC {
constexpr uint32_t EVENT_ID0=0;
enum class HardEvent {V_MTE3,MTE3_V,V_S,MTE3_S,MTE2_MTE3,MTE2_V};
inline bool flags[6]{};
inline std::vector<std::function<void()>> pendingMte2,pendingMte3;
inline void Drain(std::vector<std::function<void()>>& pending) {for(auto& op:pending)op();pending.clear();}
template<HardEvent event> void SetFlag(uint32_t) {assert(!flags[unsigned(event)]);flags[unsigned(event)]=true;}
template<HardEvent event> void WaitFlag(uint32_t) {
    assert(flags[unsigned(event)]);flags[unsigned(event)]=false;
    if constexpr(event==HardEvent::MTE3_V || event==HardEvent::MTE3_S)Drain(pendingMte3);
    if constexpr(event==HardEvent::MTE2_MTE3 || event==HardEvent::MTE2_V)Drain(pendingMte2);
}
template<class T> struct LocalTensor {
    T* data;
    LocalTensor operator[](uint32_t offset)const {return {data+offset};}
    T* GetPhyAddr()const {return data;}
};
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
    if(uintptr_t(output.data)>=uintptr_t(mappingBegin) && uintptr_t(output.data)<uintptr_t(mappingEnd))mappingWords+=count;
    if(uintptr_t(output.data)>=uintptr_t(reservedBegin) && uintptr_t(output.data)<uintptr_t(reservedEnd))++reservedCopies;
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
    std::array<uint32_t,20480> words;
    template<class T> LocalTensor<T> Get() {return {reinterpret_cast<T*>(words.data())};}
};
struct TPipe {
    template<class Buffer> void InitBuffer(Buffer&,uint32_t size) {assert(size==4096*4 || size==20480*4);}
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
constexpr uint32_t kPostThreads=256,kPostBank=240,kUbmemLocalIdGroups=0,kUbmemLocalIdCapacity=448;
struct Row {uint64_t mappingState;uint32_t words[254];};
struct Progress {uint32_t words[32];};
struct Event {uint32_t words[32];};
struct FinalPost {uint32_t words[32];};
struct KvafBatchGetCoreStatus {uint32_t words[32];};
struct Context {
    uint64_t rows,progress,owners,rowStatus,events,count,finalStatus,ubmemStatus;
    uint64_t mapping,selected,changes,changeSlots,changeCount,lruSlots,reservedHot,queryEnds,tailBounds,outputStride;
    uint64_t locations;
};
}
#include "reserved_hot.h"
#include "init_partition.h"
#include "mapping_epoch.h"
#include "mapping.h"
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
    std::vector<Row> rows(batches);
    std::vector<uint32_t> jobs(batches*kSlots);
    std::vector<uint64_t> addresses(history,0);
    Context ctx{};
    ctx.rows=ctx.progress=ctx.owners=ctx.rowStatus=ctx.events=ctx.count=ctx.finalStatus=uint64_t(control.data());
#if V9_UBMEM_TAGGED_MAP
    ctx.rows=uint64_t(rows.data());
#endif
    ctx.locations=uint64_t(jobs.data());
    ctx.mapping=uint64_t(map.data());ctx.selected=uint64_t(selected.data());ctx.changes=uint64_t(changes.data());
    ctx.changeSlots=uint64_t(slots.data());ctx.changeCount=uint64_t(counts.data());
    ctx.lruSlots=uint64_t(candidates.data());ctx.reservedHot=uint64_t(output.data());ctx.queryEnds=uint64_t(ends.data());
    ctx.outputStride=resident+384;
    reservedBegin=output.data();reservedEnd=output.data()+batches*resident;
    mappingBegin=map.data();mappingEnd=map.data()+batches*history;
    uint64_t resultHash=1469598103934665603ULL,activeRows=0;
    // Full -> shrink/reorder -> empty -> owner/row reuse -> optional bounds.
    for(uint32_t round=0;round<(V9_UBMEM_TAGGED_MAP?12:6);++round) {
        const uint32_t active=round==0?batches:round==1?1:round==2?0:round==3?batches-1:batches;
        std::fill(ends.begin(),ends.end(),0);expected=output;
        ctx.queryEnds=round==4?0:uint64_t(ends.data());
        ctx.tailBounds=round==1?0:uint64_t(bounds.data());
        for(uint32_t b=0;b<batches;++b) {
#if V9_UBMEM_TAGGED_MAP
            // Replays preserve round-local rows/maps; zeroing the state is an
            // explicit reset. Force both sides of wrap and invalid states.
            if(round==6)rows[b].mappingState=(uint64_t(kMappingEpochMagic)<<32)|(kMappingEpochMax-1);
            if(round==8)rows[b].mappingState=0;
            if(round==9)rows[b].mappingState=(uint64_t(kMappingEpochMagic-1)<<32)|37;
            if(round==10)rows[b].mappingState=(uint64_t(kMappingEpochMagic)<<32)|(kMappingEpochMax+1);
            if(round==11)rows[b].mappingState=uint64_t(kMappingEpochMagic)<<32;
            const auto generation=NextMappingGeneration(rows[b].mappingState);
            const uint64_t oldMappingWords=mappingWords;
            if(generation.clear)std::fill_n(map.begin()+b*history,history,7U);
#endif
            bounds[b*2]=history/2-128;bounds[b*2+1]=bounds[b*2]+(round==2?0:384);
            for(uint32_t i=0;i<kLruSlots;++i)
                candidates[b*kLruSlots+i]=(resident-1024+i+round*173+b*991)%resident;
            if(b<active) {
                ends[b*4+round%4]=130000;
                ++activeRows;
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
#if !V9_RESERVED_UB
                Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},history,resident,&ctx,b,reserved);
#endif
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
#if V9_UBMEM_TAGGED_MAP
            assert(rows[b].mappingState==generation.state);
            assert(mappingWords-oldMappingWords==(generation.clear?history:0));
            assert(std::all_of(std::begin(rows[b].words),std::end(rows[b].words),[](uint32_t x){return x==0;}));
#endif
            assert(std::equal(candidates.begin()+b*kLruSlots,candidates.begin()+(b+1)*kLruSlots,
                slots.begin()+b*kChangeCapacity));
            for(uint32_t token=0;token<history;++token) {
                uint32_t want=UINT32_MAX;
                if(ctx.tailBounds && token>=bounds[b*2] && token<bounds[b*2+1])
                    want=resident+kSlots+token%384;
                const uint32_t decoded=DecodeMappingSlot(map[b*history+token],
#if V9_UBMEM_TAGGED_MAP
                    generation.tag,kMappingSlotMask
#else
                    0,UINT32_MAX
#endif
                );
                assert(decoded==want);
            }
        }
        assert(output==expected);
        for(uint32_t b=0;b<batches;++b) {
            auto row=changes.begin()+b*kChangeCapacity;
            assert(std::all_of(row,row+kLruSlots,[](uint32_t x){return x==UINT32_MAX;}));
            assert(std::all_of(row+kLruSlots,row+kChangeCapacity,[](uint32_t x){return x==91;}));
        }
#if !MODEL_URMA
        assert(std::all_of(selected.begin(),selected.begin()+batches*kSlots,[](uint32_t x){return x==UINT32_MAX;}));
#endif
        for(const auto* values:{&output,&map,&selected,&changes,&slots})
            assert(std::all_of(values->end()-32,values->end(),[](uint32_t x){return x==91;}));
        for(const auto* values:{&output,&map,&selected,&changes,&slots,&counts})
            for(uint32_t value:*values){resultHash^=value;resultHash*=1099511628211ULL;}
#if V9_UBMEM_TAGGED_MAP
        // Execute the actual grouped window publisher and its gather reader.
        // The next init must invalidate these values even for inactive rows.
        for(uint32_t b=0;b<batches;++b) {
            alignas(8) uint32_t input[160]{};uint64_t peers[2*kPostBank]{};uint32_t plans[64]{};
            uint32_t ids[3]{100+b,resident+31,history-1},got[3]{};
            for(uint32_t i=0;i<3;++i){jobs[b*kSlots+i]=ids[i];addresses[ids[i]]=100000+ids[i];}
            input[64]=3;input[65]=resident;
            reinterpret_cast<uint64_t*>(input)[34]=uint64_t(addresses.data());
            for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
                PrepareUbmemWindow(&ctx,b,0,input,nullptr,peers,plans,0,UINT64_MAX);
            const auto state=rows[b].mappingState;
            const auto tag=uint32_t(state)<<kMappingSlotBits;
            for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
                GatherCandidateMap(ids,got,map.data()+b*history,3,tag);
            for(uint32_t i=0;i<3;++i){assert(got[i]==resident+i);assert(peers[i]==addresses[ids[i]]);}
            assert(rows[b].mappingState==state);
        }
#endif
    }
#if V9_UBMEM_TAGGED_MAP
    uint64_t state=0;
    for(uint32_t round=0;round<=kMappingEpochMax;++round) {
        const auto generation=NextMappingGeneration(state);
        assert(generation.clear==(round==0 || round==kMappingEpochMax));
        assert(uint32_t(generation.state)==round%kMappingEpochMax+1);
        state=generation.state;
    }
    const uint32_t lastTag=kMappingEpochMax<<kMappingSlotBits;
    assert(EncodeMappingSlot(kMappingSlotMask,lastTag,kMappingSlotMask)==UINT32_MAX);
    assert(EncodeMappingSlot(kMappingSlotMask+1,1U<<kMappingSlotBits,kMappingSlotMask)==UINT32_MAX);
    assert(DecodeMappingSlot(EncodeMappingSlot(kMappingSlotMask-1,lastTag,kMappingSlotMask),lastTag,kMappingSlotMask)==kMappingSlotMask-1);
    assert(DecodeMappingSlot(UINT32_MAX,lastTag,kMappingSlotMask)==UINT32_MAX);
    assert(DecodeMappingSlot(EncodeMappingSlot(resident,1U<<kMappingSlotBits,kMappingSlotMask),lastTag,kMappingSlotMask)==UINT32_MAX);
    assert(DecodeMappingSlot(UINT32_MAX,0,UINT32_MAX)==UINT32_MAX);
    assert(DecodeMappingSlot(resident,0,UINT32_MAX)==resident);
#endif
#if V9_COMPACT_RESERVED
    std::array<uint32_t,4096> local{};
    assert(ClearReservedHot(0,ctx.reservedHot,0,0,resident,LocalTensor<uint32_t>{local.data()})==0);
    assert(ClearReservedHot(ctx.lruSlots,0,0,0,resident,LocalTensor<uint32_t>{local.data()})==0);
    assert(pendingMte3.empty());
#endif
    // Report actual modeled operations, never a simulated latency. The new
    // path publishes the same inverse with one DMA and no random GM stores.
#if V9_RESERVED_UB
    assert(reservedStores==0 && reservedCopies==activeRows);
#endif
    std::cout<<resultHash<<' '<<reservedStores<<' '<<reservedCopies<<' '<<vfCalls<<' '<<negativeFills<<'\n';
}
