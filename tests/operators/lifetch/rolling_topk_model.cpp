// SPDX-License-Identifier: Apache-2.0
// Run the unchanged LITopk<uint16_t> implementation and the actual new rolling
// driver. SIMD selection instructions are deterministic CPU probes; this does
// not model NPU latency or replace the real-snapshot ordered-TopK oracle.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
constexpr uint32_t SIM_Q=4,EVENT_ID0=0,PIPE_V=0;
template<class T>T Min(T a,T b){return std::min(a,b);}
namespace QLICommon {
template<class T>T Align(T a,T b){return (a+b-1)/b*b;}
struct RunInfo {uint32_t s2Idx{},gS1Idx{},bIdx{},actS2SizeOrig{},actS1Size{4},actS2Size{},indiceOutOffset{};bool isLastS2InnerLoop{};};
}
namespace AscendC {
enum class HardEvent {MTE3_MTE2,V_MTE2,MTE3_V,MTE2_V,MTE2_MTE3};
inline bool events[5][8]{};
inline std::vector<std::function<void()>> loads,stores;
void DrainLoads(){for(auto& f:loads)f();loads.clear();}
void DrainStores(){for(auto& f:stores)f();stores.clear();}
template<HardEvent E>void SetFlag(unsigned i){assert(!events[unsigned(E)][i]);events[unsigned(E)][i]=true;}
template<HardEvent E>void WaitFlag(unsigned i){assert(events[unsigned(E)][i]);events[unsigned(E)][i]=false;
    if constexpr(E==HardEvent::MTE2_V || E==HardEvent::MTE2_MTE3)DrainLoads();
    if constexpr(E==HardEvent::MTE3_MTE2 || E==HardEvent::MTE3_V)DrainStores();}
template<unsigned>void PipeBarrier(){}
template<class T>struct LocalTensor {
    T* data{};size_t count{};
    LocalTensor operator[](size_t i)const{assert(i<=count);return {data+i,count-i};}
    uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(data);}
    template<class U>LocalTensor<U>ReinterpretCast()const{return {reinterpret_cast<U*>(data),count*sizeof(T)/sizeof(U)};}
};
template<class T>struct GlobalTensor {
    T* data{};void SetGlobalBuffer(T* p){data=p;}
    GlobalTensor operator[](size_t i)const{return {data+i};}
};
template<class T>void Duplicate(LocalTensor<T> dst,T value,unsigned n){assert(n<=dst.count);std::fill_n(dst.data,n,value);}
template<class T>void DataCopy(LocalTensor<T> dst,LocalTensor<T> src,unsigned n){assert(n<=dst.count && n<=src.count);std::memmove(dst.data,src.data,n*sizeof(T));}
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,unsigned n){assert(n<=dst.count);loads.push_back([=]{std::copy_n(src.data,n,dst.data);});}
struct DataCopyExtParams {uint32_t blockCount,blockLen,srcStride,dstStride,reserved;};
template<class T>struct DataCopyPadExtParams {bool enabled;uint32_t left,right;T value;};
template<class T>void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
    assert(p.blockCount==1 && !p.srcStride && !p.dstStride && p.blockLen<=dst.count*sizeof(T));
    loads.push_back([=]{std::memcpy(dst.data,src.data,p.blockLen);});}
template<class T>void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams p){
    assert(p.blockCount==1 && p.blockLen<=src.count*sizeof(T));
    stores.push_back([=]{std::memcpy(dst.data,src.data,p.blockLen);});}
enum class RoundMode {CAST_NONE};
void Cast(LocalTensor<uint32_t> dst,LocalTensor<uint16_t> src,RoundMode,unsigned n){for(unsigned i=0;i<n;++i)dst.data[i]=src.data[i];}
}
using namespace AscendC;
namespace topkb16gather {
// Choose the highest score keys, with deterministic ties, then emit in input
// order. Both production call paths must preserve this exact ordered list.
template<bool Scores>void LiTopKVF(LocalTensor<uint16_t> indices,LocalTensor<uint16_t> out,
    LocalTensor<uint16_t> input,LocalTensor<uint32_t>,LocalTensor<uint32_t>,LocalTensor<uint32_t>,LocalTensor<uint32_t>,uint32_t k,uint32_t n){
    std::vector<uint32_t> order(n);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return input.data[a]>input.data[b];});
    order.resize(k);std::sort(order.begin(),order.end());
    for(unsigned i=0;i<k;++i){indices.data[i]=order[i];if(Scores)out.data[i]=input.data[order[i]];}
}
void LiTopKGatherVF(LocalTensor<uint32_t> dst,LocalTensor<uint16_t> out,LocalTensor<uint16_t> input,
    LocalTensor<uint16_t> selected,LocalTensor<uint32_t> previous,uint32_t k,uint32_t offset,uint32_t){
    std::vector<uint32_t> mapped(k);for(unsigned i=0;i<k;++i){auto p=selected.data[i];mapped[i]=p<k?previous.data[p]:p+offset;out.data[i]=input.data[p];}
    std::copy(mapped.begin(),mapped.end(),dst.data);
}
}
#include "topk_under_test.h"
namespace rotating {struct Context {uintptr_t queryEnds{};};}
struct Qlit{};
namespace QLIKernel {
template<class>class QLIVector {
public:
    using SCORE_T=uint16_t;
    static constexpr unsigned TOPK_V_MTE2_EVENT=4,TOPK_MTE3_V_EVENT=7,V_MTE2_EVENT3=5,TOPK_MTE2_V_EVENT=5,MTE3_MTE2_EVENT=0;
    bool rollingReady_=true;uint32_t rollingEnd_=16384,s2BaseSize_=128,trunkLen_=16384,s1BaseSize_=2,topkCount_=2048,blockId_{};
    struct {bool attenMaskFlag{};int32_t kSeqSize{};}constInfo_;
    void* rotationContext_{};uint8_t* prefixOutput_{};
    GlobalTensor<uint16_t> scoreGm;GlobalTensor<int32_t> indiceOutGm;
    std::vector<uint16_t> merge=std::vector<uint16_t>(18432),scores=std::vector<uint16_t>(2048);
    std::vector<uint32_t> indices=std::vector<uint32_t>(2112),work=std::vector<uint32_t>(20000);
    LocalTensor<uint16_t> mrgValueLocal_{merge.data(),merge.size()},scoreOutLocal_{scores.data(),scores.size()};
    LocalTensor<uint32_t> indicesOutLocal_{indices.data(),indices.size()},topkSharedTmpLocal_{work.data(),work.size()};
    topk::LITopk<uint16_t> topkOp_;
    uint16_t threshold{};unsigned updates{},progressRebuilds{};
    QLIVector(){topkOp_.Init(2048,16384);topkOp_.InitBuffers(topkSharedTmpLocal_,indicesOutLocal_);}
    void ProcessRollingTopK(const QLICommon::RunInfo&);
    void PublishRollingTopK(const QLICommon::RunInfo&);
    void TraceThreshold(uint32_t,uint32_t,uint32_t,uint32_t) {}
    void SaveTopKState(uint32_t query){
        assert(loads.empty());auto* row=prefixOutput_+(query/4)*50176;
        auto* vals=reinterpret_cast<uint16_t*>(row)+query%4*2048;
        auto* ids=reinterpret_cast<uint32_t*>(row+16384)+query%4*2048;
        std::copy(scores.begin(),scores.end(),vals);std::copy_n(indices.begin(),2048,ids);
        uint16_t next=*std::min_element(scores.begin(),scores.end());assert(next>=threshold);threshold=next;++updates;
    }
    void PrepareRotationProgress(uint32_t){assert(loads.empty());std::fill(scores.begin(),scores.end(),0xdead);++progressRebuilds;}
    void ServicePoison(){assert(loads.empty());std::fill(merge.begin(),merge.end(),0xbad);std::fill(work.begin(),work.end(),0xbad);std::fill(indices.begin(),indices.end(),0xbad);std::fill(scores.begin(),scores.end(),0xbad);}
};
}
#include "rolling_topk.h"

std::vector<uint32_t> Original(const std::vector<uint16_t>& source,uint32_t valid,bool prefix=false){
    if(valid<2048){std::vector<uint32_t> out(2048,UINT32_MAX);std::iota(out.begin(),out.begin()+valid,0);return out;}
    QLIKernel::QLIVector<Qlit> op;const uint32_t loops=(valid+16383)/16384;
    for(unsigned loop=0;loop<loops;++loop){
        unsigned n=std::min(16384U,valid-loop*16384),pad=QLICommon::Align(n+(loop?2048U:0U),256U);
        std::fill(op.merge.begin(),op.merge.end(),0);
        if(loop)std::copy(op.scores.begin(),op.scores.end(),op.merge.begin());
        std::copy_n(source.begin()+loop*16384,n,op.merge.begin()+(loop?2048:0));
        op.topkOp_(op.mrgValueLocal_,op.indicesOutLocal_,op.scoreOutLocal_,pad,loop,loops,prefix);
    }
    return {op.indices.begin(),op.indices.begin()+2048};
}

int main(){
    constexpr uint32_t history=131328;
    std::mt19937 random(9281);
    for(unsigned pattern=0;pattern<3;++pattern)for(bool causal:{false,true})
    for(uint32_t query=0;query<12;++query){
        uint32_t bounds[]={0,1,2047,2048,16383,16384,16385,32768,32769,130049,131072,131328};
        uint32_t end=bounds[query];if(causal)end=std::min(end,history-4+query%4+1);
        std::vector<uint16_t> src(history);
        for(unsigned i=0;i<history;++i)src[i]=pattern==0?uint16_t(0x8001):pattern==1?uint16_t(0x8100+(random()%17)):uint16_t(1+random()%65535);
        auto expected=Original(src,end);
        std::vector<uint8_t> prefix(3*50176,0xa7);
        auto* row=prefix.data()+(query/4)*50176;
        auto* prefixValues=reinterpret_cast<uint16_t*>(row)+query%4*2048;
        auto* prefixIndices=reinterpret_cast<uint32_t*>(row+16384)+query%4*2048;
        auto initial=Original(src,std::min(end,16384U),true);
        std::copy(initial.begin(),initial.end(),prefixIndices);
        for(unsigned i=0;i<2048;++i)prefixValues[i]=initial[i]==UINT32_MAX?0:src[initial[i]];
        QLIKernel::QLIVector<Qlit> op;
        op.threshold=*std::min_element(prefixValues,prefixValues+2048);
        std::vector<uint16_t> gm(2*history,0x1234);
        std::copy_n(src.begin(),16384,gm.begin()+(query%2)*history);
        std::vector<int32_t> output(12*2048,-91);std::vector<uint32_t> ends(12);ends[query]=bounds[query];
        rotating::Context context{reinterpret_cast<uintptr_t>(ends.data())};
        op.rotationContext_=&context;op.prefixOutput_=prefix.data();op.scoreGm={gm.data()};op.indiceOutGm={output.data()};
        op.blockId_=query%2;op.constInfo_={causal,history};
        std::memset(events,0,sizeof(events));SetFlag<HardEvent::V_MTE2>(4);SetFlag<HardEvent::MTE3_V>(7);
        QLICommon::RunInfo info;info.bIdx=query/4;info.gS1Idx=(query%4)/2;info.actS2Size=history;info.actS2SizeOrig=history;info.indiceOutOffset=(query/4)*4*2048;
        for(uint32_t available=16512;available<=history;available+=128){
            info.s2Idx=available/128-1;info.isLastS2InnerLoop=available==history;
            // Delay each producer's score-store until the rolling MTE3->MTE2
            // join, and poison the service's borrowed scratch between updates.
            stores.push_back([&,available]{std::copy_n(src.begin()+available-128,128,
                gm.begin()+(query%2)*history+available-128);});
            op.ProcessRollingTopK(info);op.ServicePoison();
        }
        assert(op.updates==(end>16384?(end-16384+16383)/16384:0));
        assert(op.progressRebuilds==op.updates);
        op.PublishRollingTopK(info);WaitFlag<HardEvent::MTE3_V>(7);WaitFlag<HardEvent::V_MTE2>(4);
        assert(loads.empty() && stores.empty());
        for(auto& group:events)for(bool event:group)assert(!event);
        for(unsigned i=0;i<output.size();++i)assert(output[i]==(i/2048==query?int32_t(expected[i%2048]):-91));
        for(unsigned q=0;q<12;++q)if(q!=query){
            const auto* other=prefix.data()+(q/4)*50176;
            for(unsigned i=0;i<4096;++i)assert(other[(q%4)*4096+i]==0xa7);
            for(unsigned i=0;i<8192;++i)assert(other[16384+(q%4)*8192+i]==0xa7);
        }
    }
}
