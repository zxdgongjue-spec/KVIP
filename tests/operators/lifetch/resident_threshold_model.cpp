// SPDX-License-Identifier: Apache-2.0
// Actual threshold/publication functions, bounded CPU instruction probes.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__
constexpr unsigned SIM_Q=4,SIM_RESIDENT=16384,SIM_HISTORY=32768,EVENT_ID0=0;
constexpr unsigned PIPE_V=0;
unsigned histogramPhase=0,victimLoads=0,gathers=0,classifications=0,bitmapWrites=0,cleans=0;
namespace AscendC {
template<unsigned>void PipeBarrier(){}
enum class HardEvent {MTE2_V,V_MTE3,MTE3_S,V_S};
bool events[4]{};
template<HardEvent e>void SetFlag(unsigned){assert(!events[unsigned(e)]);events[unsigned(e)]=true;}
template<HardEvent e>void WaitFlag(unsigned){assert(events[unsigned(e)]);events[unsigned(e)]=false;}
template<class T>struct LocalTensor {
    T* data;
    LocalTensor operator[](unsigned n)const{return {data+n};}
    uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(data);}
    template<class U>LocalTensor<U>ReinterpretCast()const{return {reinterpret_cast<U*>(data)};}
};
template<class T>struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* p){data=p;}
    uintptr_t GetPhyAddr()const{return reinterpret_cast<uintptr_t>(data);}
};
struct DataCopyExtParams {unsigned blockCount{},blockLen{},srcStride{},dstStride{};};
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,unsigned n){
    ++victimLoads;assert(n==2048);std::copy_n(src.data,n,dst.data);
}
void DataCopyPad(GlobalTensor<uint8_t> dst,LocalTensor<uint8_t> src,DataCopyExtParams p){
    ++bitmapWrites;assert(p.blockCount==16 && p.blockLen==32 && p.srcStride==7);
    for(unsigned i=0;i<p.blockCount;++i)
        std::memcpy(dst.data+i*(p.blockLen+p.dstStride),src.data+i*(p.blockLen+p.srcStride*32),p.blockLen);
}
namespace Simt {
struct Dim3{unsigned x,y,z;};
template<auto function,class... Args>void VF_CALL(Dim3,Args... args){function(args...);}
}
namespace MicroAPI {
enum class MaskPattern{ALL};enum class MemType{VEC_STORE,VEC_LOAD};enum class LoadDist{DIST_BRC_B16};
template<class T>struct RegTensor{T data[256/sizeof(T)];};
template<class T,MaskPattern>bool CreateMask(){return true;}
template<MemType,MemType>void LocalMemBar(){}
template<class T,LoadDist>void LoadAlign(RegTensor<T>& dst,T* src){std::fill(std::begin(dst.data),std::end(dst.data),src[0]);}
template<class T>void StoreAlign(T* dst,const RegTensor<T>& src,bool){std::copy(std::begin(src.data),std::end(src.data),dst);}
}
}
using namespace AscendC;
namespace topkb16gather {
// Histograms in the production API are cumulative. These probes model the
// unchanged four SIMD primitives; nth_element below is an independent oracle.
template<class T>void HistogramsHighVFImpl(uint32_t* hist,uint16_t* data,uint16_t loops,bool init){
    assert(histogramPhase++==0 && init);std::fill_n(hist,256,0U);
    for(unsigned i=0;i<unsigned(loops)*256;++i)++hist[data[i]>>8];
    std::partial_sum(hist,hist+256,hist);
}
void FindHighTargetBinVFImpl(uint32_t* high,uint32_t* rank,uint32_t* hist,uint32_t bottom){
    assert(histogramPhase++==1 && bottom>0);high[0]=unsigned(std::lower_bound(hist,hist+256,bottom)-hist);
    assert(high[0]<256);rank[0]=bottom-(high[0]?hist[high[0]-1]:0);
}
template<class T>void HistogramsLowVFImpl(uint32_t* hist,uint16_t* data,uint32_t* high,uint16_t loops,bool init){
    assert(histogramPhase++==2 && init);std::fill_n(hist,256,0U);
    for(unsigned i=0;i<unsigned(loops)*256;++i)if((data[i]>>8)==high[0])++hist[data[i]&255];
    std::partial_sum(hist,hist+256,hist);
}
void FindKthVFImpl(uint32_t* kth,uint32_t* hist,uint32_t* high,uint32_t* low){
    assert(histogramPhase++==3);low[0]=unsigned(std::lower_bound(hist,hist+256,kth[0])-hist);
    assert(low[0]<256);reinterpret_cast<uint16_t*>(kth)[0]=uint16_t((high[0]<<8)|low[0]);
}
}
#include "threshold_under_test.h"
namespace QLICommon {template<class T>T Align(T n,T a){return (n+a-1)/a*a;}}
namespace rotating {
struct Context{uintptr_t lruSlots{},hits{},queryEnds{},progress{};};
void GatherVictimScores(uint32_t* victims,uint32_t* scores,const uint32_t* source){
    ++gathers;auto out=reinterpret_cast<uint16_t*>(scores);auto values=reinterpret_cast<const uint16_t*>(source);
    for(unsigned i=0;i<2048;++i)out[i]=values[victims[i]];
}
}
namespace vector1 {
void ClassifyResidentScores(uint16_t* scores,uint16_t* victims,uint16_t* threshold,uint16_t* hits,unsigned valid){
    ++classifications;auto ids=reinterpret_cast<uint32_t*>(victims);auto bits=reinterpret_cast<uint8_t*>(hits);
    std::fill_n(bits,4096,0);
    for(unsigned i=0;i<2048;++i)if(ids[i]<valid && scores[i]>=threshold[0])
        bits[(i/128)*256+(i%128)/4]|=uint8_t(1U<<((i%4)*2));
}
}
namespace ubmem_device {void CleanCacheLines(uint8_t*,unsigned n){assert(n==128);++cleans;}}
namespace kvaf_device {void CleanCacheLines(uint8_t* p,unsigned n){ubmem_device::CleanCacheLines(p,n);}}
struct Publisher {
    void* rotationContext_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;
    LocalTensor<uint16_t> prefixThreshold_;
    GlobalTensor<uint16_t> scoreGm;
    struct{uint64_t kSeqSize=SIM_RESIDENT;}constInfo_;
    struct{unsigned history=SIM_HISTORY,resident=SIM_RESIDENT;}runtimeShape_;
    uint64_t s2BaseSize_=128;
#include "publication_under_test.h"
};
void CheckThreshold(const std::vector<uint16_t>& valid){
    unsigned n=unsigned(valid.size()),padded=(n+255)/256*256;
    std::vector<uint16_t> input(padded,0);std::copy(valid.begin(),valid.end(),input.begin());
    auto sorted=input;std::nth_element(sorted.begin(),sorted.end()-2048,sorted.end());
    uint16_t expected=sorted[sorted.size()-2048];
    constexpr uint32_t guard=0xa5a5a5a5;
    std::array<uint32_t,834> scratch;scratch.fill(guard);
    std::array<uint16_t,258> output;output.fill(0x55aa);
    histogramPhase=0;
    kvip_prefix::ResidentThresholdOnly({input.data()},{scratch.data()+1},{output.data()+1},padded,2048);
    assert(histogramPhase==4 && scratch.front()==guard && scratch.back()==guard);
    for(unsigned i=1;i<=128;++i)assert(output[i]==expected);
    for(unsigned i=129;i<output.size();++i)assert(output[i]==0x55aa);
    assert(output[0]==0x55aa);
    // Exact cutoff means the entire uint16 classifier domain agrees, including
    // encoded signed zero, infinities, NaNs, sentinel, and any BF16 tie plateau.
    for(unsigned score=0;score<65536;++score)assert((score>=output[1])==(score>=expected));
    assert(std::equal(input.begin(),input.begin()+n,valid.begin()));
}
int main(){
    for(int n:{-1,0,1,2047,2048,2049,16384,16385}){
        assert(kvip_prefix::UseThresholdOnly(true,n,2048,16384)==bool(V9_THRESHOLD_ONLY && n>=2048 && n<=16384));
        assert(!kvip_prefix::UseThresholdOnly(false,n,2048,16384));
    }
    assert(!kvip_prefix::UseThresholdOnly(true,4096,2048,2048));
    assert(!kvip_prefix::UseThresholdOnly(true,4096,1024,16384));
    std::mt19937 random(7919);
    for(unsigned n:{2048U,2049U,2175U,4095U,8192U,16383U,16384U}){
        std::vector<uint16_t> scores(n);
        for(unsigned pattern=0;pattern<6;++pattern){
            const uint16_t special[]={0,1,0x7f7f,0x7f80,0x7fc0,0x7fff,0x8000,0xff7f,0xff80,0xffc0,0xffff};
            for(unsigned i=0;i<n;++i)scores[i]=pattern==0?0:pattern==1?65535:pattern==2?0x8000:
                pattern==3?special[i%11]:pattern==4?uint16_t(random()):uint16_t(i%7+65529);
            CheckThreshold(scores);
        }
    }
    // Run the actual publication for all physical Q4 rows, with dirty hot bits
    // and cold bytes. Only the optional compact hot bitmap may change.
    std::vector<uint32_t> scratch(4096),victims(4096),ends(8,SIM_RESIDENT),progress(8*32,0);
    std::vector<uint16_t> scores(SIM_RESIDENT*2,0x9000),threshold(256,0x8000);
    std::vector<uint8_t> bitmap(8*SIM_HISTORY/4,0xa5);
    std::iota(victims.begin(),victims.begin()+2048,0);std::copy_n(victims.begin(),2048,victims.begin()+2048);
    rotating::Context ctx{uintptr_t(victims.data()),uintptr_t(bitmap.data()),uintptr_t(ends.data()),uintptr_t(progress.data())};
    Publisher publisher{&ctx,{scratch.data()},{threshold.data()},{scores.data()}};
    for(unsigned query=0;query<8;++query)publisher.PublishResident(query,query%2);
    assert(cleans==8 && (V9_COLD_ONLY_PREFETCH?victimLoads==0:victimLoads==8));
    assert(gathers==victimLoads && classifications==victimLoads && bitmapWrites==victimLoads);
    for(unsigned query=0;query<8;++query){
        assert(progress[query*32]==SIM_RESIDENT);
        for(unsigned i=1;i<32;++i)assert(progress[query*32+i]==0);
        for(unsigned i=0;i<SIM_HISTORY/4;++i)
            assert(bitmap[query*(SIM_HISTORY/4)+i]==(!V9_COLD_ONLY_PREFETCH && i<512?0x55:0xa5));
    }
    assert(std::all_of(std::begin(events),std::end(events),[](bool x){return !x;}));
    assert(std::all_of(threshold.begin(),threshold.end(),[](uint16_t x){return x==0x8000;}));
}
