// SPDX-License-Identifier: Apache-2.0
// Execute the production service control flow; vector math/transport are probes.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
#define V9_FETCH_PROFILE(...)
#define V9_INTERMEDIATE_PROFILING 0
#define V9_PARTIAL_SCAN_SIMD 1
#define V9_BATCH_POST 1
#define V9_SIMULATION 1
#define V9_UBMEM 1
#define V9_WINDOW_COPY_ENGINE 0
using GM_ADDR=uint8_t*;
#include "context.h"
#include "mapping_epoch.h"
#include "prefetch_budget.h"
unsigned bitmapCopies=0,scanCalls=0,postCalls=0,previousChecks=0;
unsigned hotBitmapCopies=0,coldBitmapCopies=0,bitmapHistory=0,bitmapResident=0;
uint8_t* bitmapBegin=nullptr;
uint8_t* bitmapEnd=nullptr;
rotating::Event* previousEvent=nullptr;
namespace AscendC {
constexpr unsigned EVENT_ID0=0;
enum class HardEvent {S_MTE2,MTE2_S,MTE2_V,S_V,V_S,MTE2_MTE3,S_MTE3,MTE3_S,V_MTE3,MTE3_V};
bool flags[10]{};
template<HardEvent event> void SetFlag(unsigned){assert(!flags[unsigned(event)]);flags[unsigned(event)]=true;}
template<HardEvent event> void WaitFlag(unsigned){assert(flags[unsigned(event)]);flags[unsigned(event)]=false;}
template<class T> struct LocalTensor {
    T* data;
    LocalTensor operator[](unsigned i)const{return {data+i};}
    T GetValue(unsigned i)const{return data[i];}
    void SetValue(unsigned i,T value){data[i]=value;}
    template<class U>LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(data)};}
    uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(data);}
};
template<class T> struct GlobalTensor {
    T* data=nullptr;
    void SetGlobalBuffer(T* value){data=value;}
    GlobalTensor operator[](unsigned i)const{return {data+i};}
};
struct DataCopyExtParams {unsigned blockCount{},blockLen{},srcStride{},dstStride{};};
template<class T>struct DataCopyPadExtParams {bool enabled;unsigned left,right;T value;};
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,unsigned n){std::copy_n(src.data,n,dst.data);}
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,unsigned n){std::copy_n(src.data,n,dst.data);}
template<class T>void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
    if(uintptr_t(src.data)>=uintptr_t(bitmapBegin) && uintptr_t(src.data)<uintptr_t(bitmapEnd)) {
        ++bitmapCopies;
        unsigned token=unsigned(reinterpret_cast<uint8_t*>(src.data)-bitmapBegin)%bitmapHistory*4;
        if(token<bitmapResident)++hotBitmapCopies;else ++coldBitmapCopies;
    }
    for(unsigned b=0;b<p.blockCount;++b)
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data)+b*(p.blockLen+p.dstStride),
                    reinterpret_cast<uint8_t*>(src.data)+b*(p.blockLen+p.srcStride),p.blockLen);
}
template<class T>void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams p){
    for(unsigned b=0;b<p.blockCount;++b)
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data)+b*(p.blockLen+p.dstStride),
                    reinterpret_cast<uint8_t*>(src.data)+b*(p.blockLen+p.srcStride),p.blockLen);
}
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto function,class... Args>void VF_CALL(Dim3,Args... args){function(args...);}
}
}
void Clean(uint8_t* p,unsigned){
    if(p==reinterpret_cast<uint8_t*>(previousEvent)){
        ++previousChecks;previousEvent->complete=1; // Model predecessor becoming mature on its acquire.
    }
}
namespace kvaf_device {constexpr unsigned kInvalidArgs=1;void CleanCacheLines(uint8_t* p,unsigned n){Clean(p,n);}}
namespace ubmem_device {void CleanCacheLines(uint8_t* p,unsigned n){Clean(p,n);}}
using namespace AscendC;
namespace rotating {
constexpr unsigned SIM_BATCH=4,SIM_HISTORY=8192,SIM_RESIDENT=4096,kServiceLanes=2;
constexpr unsigned kLruSlots=2048,kScanTokens=23808;
constexpr bool kBatchedRows=true;
constexpr unsigned kUbmemLocalIdGroups=0,kUbmemLocalIdOffsetWords=0,kUbmemLocalIdCapacity=448;
unsigned Trigger(unsigned epoch){return Trigger(SIM_HISTORY,SIM_RESIDENT,4,epoch);}
template<bool Partial>void UnionWords(uint32_t*,uint32_t*,uint32_t*,uint32_t* cfg,unsigned,unsigned){
    ++scanCalls;cfg[72]=0; // No bitmap hits: positive budget must still inspect the bitmap.
}
void ExpandWords(uint32_t*,uint32_t*,uint32_t*,uint32_t*,unsigned,unsigned,unsigned,unsigned){assert(false);}
void ScanPartial(uint32_t*,uint32_t*,uint32_t*,unsigned,unsigned,unsigned){assert(false);}
void RestoreVictimTokens(uint32_t*,uint32_t*,unsigned){assert(false);}
#if MODEL_UBMEM
void GatherCandidateMap(uint32_t*,uint32_t*,uint32_t*,unsigned,unsigned){assert(false);}
#else
void GatherCandidateMap(uint32_t*,uint32_t*,uint32_t*,unsigned){assert(false);}
#endif
void CompactMisses(uint32_t*,uint32_t*,uint32_t*,unsigned,unsigned){assert(false);}
void PrepareCandidateWqes(unsigned,uint32_t*,uint32_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,unsigned,unsigned,Context*,unsigned){assert(false);}
template<class... Args>bool SubmitWqesAt(Args...){++postCalls;return true;}
template<bool B>void BatchBuild(unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,Context*,unsigned,unsigned,unsigned,uint64_t*,uint32_t*,uint32_t*,uint32_t*){assert(false);}
template<class... Args>void FinalBuild(Args...){assert(false);}
bool UbmemWindowUseLocalIds(unsigned){return false;}
void StageUbmemWindowIds(uint32_t*,uint32_t*,unsigned){assert(false);}
void CopyUbmemWindow(Context*,unsigned,LocalTensor<uint32_t>,LocalTensor<uint32_t>,LocalTensor<uint32_t>,LocalTensor<uint32_t>,Event*){assert(false);}
#include "service_function.h"
}

int main(){
    using namespace rotating;
    std::vector<Row> rows(SIM_BATCH);
    std::vector<Progress> progress(SIM_BATCH*4);
    std::vector<Event> events(4*kServiceLanes);
    std::vector<uint8_t> bitmap(SIM_BATCH*SIM_HISTORY,0xa5);
    std::vector<uint32_t> ends(SIM_BATCH*4,SIM_HISTORY),mapping(SIM_BATCH*SIM_HISTORY,191),jobs(SIM_BATCH*8192,313);
    std::vector<uint64_t> scratch(5000),work(5000),storage(2000);
    bitmapBegin=bitmap.data();bitmapEnd=bitmap.data()+bitmap.size();
    bitmapHistory=SIM_HISTORY;bitmapResident=SIM_RESIDENT;
    Context ctx{};ctx.rows=uint64_t(rows.data());ctx.progress=uint64_t(progress.data());ctx.events=uint64_t(events.data());
    ctx.hits=uint64_t(bitmap.data());ctx.queryEnds=uint64_t(ends.data());ctx.mapping=uint64_t(mapping.data());
    ctx.locations=uint64_t(jobs.data());ctx.lruSlots=1;ctx.windowLimit=64;
    for(bool batched:{false,true})for(unsigned reason=0;reason<5;++reason){
#if MODEL_UBMEM
        if(!batched)continue;
#endif
        rows.assign(SIM_BATCH,{});events.assign(4*kServiceLanes,{});
        for(auto& p:progress)p.end[0]=SIM_HISTORY;
        for(auto& row:rows){
            row.slots=7;row.requests=11;row.pending=3;row.suppressed=19;
            row.limit=8192;row.prefetchLimit=512;row.queueLimit=8191;
            for(unsigned q=0;q<4;++q)row.consumed[q]=q*16;
            if(reason==0)row.prefetchLimit=row.requests;
            if(reason==1)row.queueLimit=row.requests;
            if(reason==2)row.limit=row.slots;
            if(reason==3)row.prefetchLimit=0;
        }
        const auto initial=rows;
        bitmapCopies=scanCalls=postCalls=previousChecks=0;
        hotBitmapCopies=coldBitmapCopies=0;
        for(unsigned epoch=1;epoch<4;++epoch)for(unsigned lane=0;lane<kServiceLanes;++lane){
            std::fill(scratch.begin(),scratch.end(),0x7777777777777777ULL);
            std::fill(work.begin(),work.end(),0x7777777777777777ULL);
            std::fill(storage.begin(),storage.end(),0x7777777777777777ULL);
            previousEvent=&events[(epoch-1)*kServiceLanes+lane];previousEvent->complete=0;
#if MODEL_UBMEM
            FastService(reinterpret_cast<GM_ADDR>(&ctx),epoch,lane,lane,
                {reinterpret_cast<uint32_t*>(scratch.data())},{reinterpret_cast<uint16_t*>(work.data())},
                {reinterpret_cast<uint32_t*>(storage.data())});
#else
            FastService(SIM_BATCH,SIM_HISTORY,SIM_RESIDENT,4,kServiceLanes,2624,256,240,batched,23808,
                reinterpret_cast<GM_ADDR>(&ctx),epoch,lane,lane,
                {reinterpret_cast<uint32_t*>(scratch.data())},{reinterpret_cast<uint16_t*>(work.data())},
                {reinterpret_cast<uint32_t*>(storage.data())});
#endif
            assert(events[epoch*kServiceLanes+lane].complete==1);
            assert(std::all_of(std::begin(flags),std::end(flags),[](bool x){return !x;}));
        }
        assert(previousChecks==6 && postCalls==0);
        assert(reason==4 ? bitmapCopies>0 && scanCalls>0 : bitmapCopies==0 && scanCalls==0);
        if(reason==4) {
            assert(coldBitmapCopies>0);
            assert(V9_COLD_ONLY_PREFETCH ? hotBitmapCopies==0 : hotBitmapCopies>0);
        }
        for(unsigned b=0;b<SIM_BATCH;++b){
            const auto& row=rows[b];assert(row.sequence==3);
            assert(row.slots==initial[b].slots && row.requests==initial[b].requests && row.pending==3);
            assert(row.suppressed==19 && row.error==0);
            for(unsigned epoch=1;epoch<4;++epoch){
                assert(row.epochOrder[epoch]==epoch && row.epochCount[epoch]==0 && row.epochSuppressed[epoch]==0);
                if(reason!=4)for(unsigned q=0;q<4;++q)assert(row.snapshot[epoch][q]==initial[b].consumed[q]);
            }
            if(reason!=4)for(unsigned q=0;q<4;++q)assert(row.consumed[q]==initial[b].consumed[q]);
        }
        assert(std::all_of(mapping.begin(),mapping.end(),[](unsigned x){return x==191;}));
        assert(std::all_of(jobs.begin(),jobs.end(),[](unsigned x){return x==313;}));
    }
}
