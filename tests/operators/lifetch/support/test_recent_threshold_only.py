"""Execute the native threshold-only loader and its unchanged radix stages.

The four existing device histogram VFs are modeled at their public boundaries;
the newly added native loader/selector and their scratch/output addresses run
unchanged. Device instruction behavior still requires CANN/NPU validation.
"""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function


def threshold_model_source():
    vector = (INDEXER / "indexer_vector.h").read_text()
    topk = (INDEXER / "topk.h").read_text()
    loader = function(vector, "    __aicore__ inline bool TryUnifiedResidentTopK(")
    select = function(topk, "    __aicore__ inline void Threshold(")
    return (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(n)
struct {unsigned x;} threadIdx;
#include "recent_index.h"
constexpr unsigned SIM_Q=4,EVENT_ID0=0,V_MTE2_EVENT3=5,TOPK_MTE2_V_EVENT=5;
enum class HardEvent {V_MTE2,MTE2_V,V_S};bool credits[3][8]{};
template<HardEvent D> void SetFlag(unsigned id){assert(!credits[unsigned(D)][id]);credits[unsigned(D)][id]=true;}
template<HardEvent D> void WaitFlag(unsigned id){assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;}
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto F,class... Args> void VF_CALL(Dim3 dim,Args... args){
 for(threadIdx.x=0;threadIdx.x<dim.x;++threadIdx.x)F(args...);
}
}
namespace QLICommon {unsigned Align(unsigned a,unsigned n){return (a+n-1)/n*n;}}
template<class T> T Min(T a,T b){return std::min(a,b);}
template<class T> struct Tensor {
 T* p;unsigned n;
 Tensor operator[](uint64_t off)const{assert(off<=n);return {p+off,n-unsigned(off)};}
 T* GetPhyAddr()const{return p;}
};
template<class T> using LocalTensor=Tensor<T>;
unsigned dma=0;
template<class T> void Duplicate(Tensor<T> dst,T v,unsigned n){assert(n<=dst.n);std::fill_n(dst.p,n,v);}
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool pad;unsigned a,b;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 assert(p.blockCount==1 && !p.srcStride && !p.dstStride);unsigned n=p.blockLen/sizeof(T);
 assert(n<=dst.n && n<=src.n);std::copy_n(src.p,n,dst.p);++dma;
}
namespace topkb16gather {
unsigned stage=0,passes=0;
// Exact contracts of the unchanged cumulative byte-histogram VFs.
template<class T> void HistogramsHighVFImpl(uint32_t* hist,uint16_t* input,uint16_t loops,bool init){
 assert(stage==0 && init);std::fill_n(hist,256,0);
 for(unsigned i=0;i<loops*256;++i)++hist[input[i]>>8];
 std::partial_sum(hist,hist+256,hist);stage=1;
}
void FindHighTargetBinVFImpl(uint32_t* high,uint32_t* next,uint32_t* hist,uint32_t rank){
 assert(stage==1);unsigned bin=0;while(bin<256 && hist[bin]<rank)++bin;assert(bin<256);
 *high=bin;std::fill_n(next,64,rank-(bin?hist[bin-1]:0));stage=2;
}
template<class T> void HistogramsLowVFImpl(uint32_t* hist,uint16_t* input,uint32_t* high,uint16_t loops,bool init){
 assert(stage==2 && init);std::fill_n(hist,256,0);
 for(unsigned i=0;i<loops*256;++i)if((input[i]>>8)==*high)++hist[input[i]&255];
 std::partial_sum(hist,hist+256,hist);stage=3;
}
void FindKthVFImpl(uint32_t* next,uint32_t* hist,uint32_t* high,uint32_t* low){
 assert(stage==3);unsigned bin=0;while(bin<256 && hist[bin]<*next)++bin;assert(bin<256);
 *low=bin;std::fill_n(reinterpret_cast<uint16_t*>(next),64,uint16_t((*high<<8)|bin));
 stage=0;++passes;
}
}
struct TopK {
 unsigned topK=2048,calls=0;
 std::array<uint32_t,3*256> scratch{};
 LocalTensor<uint32_t> histogramsLocal{scratch.data(),256},idxHighLocal{scratch.data()+256,256},idxLowLocal{scratch.data()+512,256};
 void operator()(Tensor<uint16_t>,Tensor<uint32_t>,Tensor<uint16_t>,unsigned,unsigned,unsigned,bool){++calls;assert(false);}
"""
        + select
        + r"""
};
namespace rotating {
struct Context {uint64_t thresholdBounds=0,queryEnds=0;};
struct RuntimeShape {unsigned resident=16384,history=131328;};
}
struct Model {
 using SCORE_T=uint16_t;rotating::Context ctx;rotating::RuntimeShape runtimeShape_;void* rotationContext_=&ctx;
 unsigned topkCount_=2048,topkCountAlign256_=2048,trunkLen_=16384;
 std::vector<uint16_t> scores=std::vector<uint16_t>(4*131328),merge=std::vector<uint16_t>(18432,0xdddd),out=std::vector<uint16_t>(2048,0xcccc);
 std::vector<uint32_t> ids=std::vector<uint32_t>(2048,0xbbbbbbbb);
 // Guard both banks: the selector temporarily writes exactly 256 bytes.
 std::array<uint16_t,384> threshold{};
 Tensor<uint16_t> scoreGm{scores.data(),unsigned(scores.size())},mrgValueLocal_{merge.data(),18432},scoreOutLocal_{out.data(),2048},prefixThreshold_{threshold.data()+64,256};
 Tensor<uint32_t> indicesOutLocal_{ids.data(),2048};TopK topkOp_;
 Model(){threshold.fill(0xaaaa);}
"""
        + loader
        + r"""
};
int main(){
 using namespace kvip_recent_index;
 for(unsigned id:{0U,1U,2U,6U,7U})credits[unsigned(HardEvent::V_MTE2)][id]=true;
 std::mt19937 rng(41);
 for(unsigned r:{2048U,8192U,16384U})for(unsigned length:{0U,1U,127U,128U,129U,2048U,2049U,20000U})for(unsigned mode=0;mode<8;++mode){
  Model m;m.runtimeShape_.resident=r;const unsigned h=m.runtimeShape_.history;
  for(auto& v:m.scores)v=mode==0?0:mode==1?65535:mode==2?uint16_t(32760+rng()%17):uint16_t(rng());
  const auto original=m.scores;int32_t bounds[2]={20001,int32_t(20001+length)};
  if(mode==3){bounds[0]=r-7;bounds[1]=r+length;}
  if(mode==4)bounds[0]=-1;
  if(mode==5)bounds[1]=h+1;
  uint32_t ends[4]={h,20256,20003,1024};m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
  m.ctx.thresholdBounds=mode==6?0:reinterpret_cast<uint64_t>(bounds);
  for(unsigned q=0;q<4;++q){
   unsigned n=std::min(r,ends[q]);if(mode==7 && q==2)n=2049;
   unsigned available=mode==7?20200:h;
   auto range=m.ctx.thresholdBounds?Clip(bounds[0],bounds[1],r,h,std::min(available,ends[q])):Range{0,0};
   unsigned count=n+range.end-range.begin,aligned=QLICommon::Align(count,256);
   bool expected=n>=2048 && n%16==0 && aligned<=18432;
   auto before=m.threshold;unsigned pass=topkb16gather::passes;dma=0;
   bool selected=m.TryUnifiedResidentTopK(q,q,n,available,true);assert(selected==expected);
   assert(topkb16gather::passes==pass+unsigned(expected));assert(m.topkOp_.calls==0);
   if(expected){
    std::vector<uint16_t> ref(m.scores.begin()+q*h,m.scores.begin()+q*h+n);
    ref.insert(ref.end(),m.scores.begin()+q*h+range.begin,m.scores.begin()+q*h+range.end);
    std::nth_element(ref.begin(),ref.begin()+2047,ref.end(),std::greater<uint16_t>());
    assert(m.prefixThreshold_.p[(q%2)*128]==ref[2047]);
    assert(dma==1U+unsigned(range.end>range.begin));
   }else assert(dma==0 && m.threshold==before);
   for(unsigned i=0;i<m.threshold.size();++i){
    if(i<64+(q%2)*128 || i>=64+(q%2+1)*128)assert(m.threshold[i]==before[i]);
   }
   assert(m.scores==original);
   assert(std::all_of(m.ids.begin(),m.ids.end(),[](auto v){return v==0xbbbbbbbb;}));
   assert(std::all_of(m.out.begin(),m.out.end(),[](auto v){return v==0xcccc;}));
   for(unsigned id=0;id<8;++id){
    assert(credits[0][id]==(id==0 || id==1 || id==2 || id==6 || id==7));
    assert(!credits[1][id] && !credits[2][id]);
   }
  }
 }
}
"""
    )


def test_exact_kth_loader_preserves_scores_and_avoids_candidate_outputs(tmp_path):
    cpp = tmp_path / "threshold.cpp"
    cpp.write_text(threshold_model_source())
    exe = tmp_path / "threshold"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-DKVIP_RECENT_THRESHOLD_ONLY=1",
            "-DKVIP_RECENT_UNIFIED_TOPK=1",
            "-I",
            str(COMMON),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_threshold_only_keeps_topk_event_ownership_and_publication():
    source = (INDEXER / "indexer_vector.h").read_text()
    process = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessTopK(")
    start = process.index("#if KVIP_RECENT_THRESHOLD_ONLY")
    branch = process[start : process.index("AscendC::DataCopyPadExtParams<SCORE_T> padParams", start)]
    assert process.rfind("WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);", 0, start) >= 0
    assert process.rfind("WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);", 0, start) >= 0
    assert branch.index("SetFlag<HardEvent::V_MTE2>") < branch.index("SaveResident(")
    assert branch.index("SaveResident(") < branch.index("SetFlag<HardEvent::MTE3_V>")
    save = function(source, "    __aicore__ inline void SaveResident(")
    list_block = function(save, "        if (!thresholdReady)")
    assert "MergeRecentScores(" in list_block and "ReduceMin(" in list_block
    assert save.index("DataCopy(thresholds[") < save.index("PipeBarrier<PIPE_ALL>();")
    assert save.index("PipeBarrier<PIPE_ALL>();") < save.index("GatherVictimScores>")
    assert save.index("GatherVictimScores>") < save.index("ready[0]=runtimeShape_.resident;")


@pytest.mark.parametrize("prefix", [False, True])
def test_topk_profile_is_separate_and_keeps_query_cacheline(tmp_path, prefix):
    source = (INDEXER / "indexer_vector.h").read_text()
    method = function(source, "__aicore__ inline void QLIVector<QLIT>::MarkTopK(").replace("QLIVector<QLIT>::", "")
    cpp = tmp_path / "markers.cpp"
    cpp.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdint>
#define __aicore__
#define __gm__
constexpr unsigned SIM_Q=4,PIPE_ALL=0;
unsigned joins=0,flushes=0;uint64_t cycle=100;
template<unsigned> void PipeBarrier(){++joins;}
uint64_t GetSystemCycle(){return ++cycle;}
namespace kvaf_device {void CleanCacheLines(uint8_t*,unsigned n){assert(n==128);++flushes;}}
namespace pipeline_ab {struct Args {uint32_t prefetchCount;uint64_t times;};}
struct Model {
 pipeline_ab::Args args{};void* streamingWorkspace_=&args;bool prefixActive_=MODEL_PREFIX;
 struct {unsigned batch=16;} runtimeShape_;
 std::array<uint64_t,16*64+64*16> ticks{};
 Model(){args.times=reinterpret_cast<uint64_t>(ticks.data());}
"""
        + method
        + r"""
};
int main(){
 Model m;
 for(unsigned q=0;q<64;++q){m.MarkTopK(q,true);m.MarkTopK(q,false);}
 assert(joins==0 && flushes==0);
 m.args.prefetchCount=0x80000000U;
 for(unsigned q=0;q<64;++q){m.MarkTopK(q,true);m.MarkTopK(q,false);}
 assert(joins==128 && flushes==128);
 for(unsigned i=0;i<16*64;++i)assert(m.ticks[i]==0);
 for(unsigned q=0;q<64;++q)for(unsigned k=0;k<16;++k){
  auto value=m.ticks[16*64+q*16+k];unsigned slot=MODEL_PREFIX?12:14;
  assert(value==((k==slot || k==slot+1)?101+2*q+k-slot:0));
 }
 m.MarkTopK(64,true);assert(joins==128 && flushes==128);
}
"""
    )
    exe = tmp_path / "markers"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fsanitize=address,undefined",
            f"-DMODEL_PREFIX={int(prefix)}",
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
