"""Execute the production two-range TopK loader/remap with CPU tensor mocks."""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function


ROOT = Path(__file__).resolve().parents[4]
INDEXER = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer"
COMMON = ROOT / "csrc/operators/common/lifetch"


@pytest.mark.parametrize("enabled", [0, 1])
def test_unified_topk_exact_threshold_and_physical_ids(tmp_path, enabled):
    source = (INDEXER / "indexer_vector.h").read_text()
    unified = function(source, "    __aicore__ inline bool TryUnifiedResidentTopK(")
    merge = function(source, "    __aicore__ inline void MergeRecentScores(")
    cpp = tmp_path / "unified.cpp"
    cpp.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
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
enum class HardEvent {V_MTE2,MTE2_V,V_S};
bool credits[3][8]{};
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
 T* GetPhyAddr(){return p;}
};
template<class T> void DataCopy(Tensor<T> dst,Tensor<T> src,unsigned n){
 assert(n<=dst.n && n<=src.n);std::copy_n(src.p,n,dst.p);
}
template<class T> void Duplicate(Tensor<T> dst,T v,unsigned n){assert(n<=dst.n);std::fill_n(dst.p,n,v);}
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool pad;unsigned a,b;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 assert(p.blockCount==1 && !p.srcStride && !p.dstStride);DataCopy(dst,src,p.blockLen/sizeof(T));
}
namespace rotating {
struct Context {uint64_t thresholdBounds=0,queryEnds=0;};
struct RuntimeShape {unsigned resident=16384,history=131328;};
}
// Same selection contract as LiTopKVF<uint16_t>: first all IDs greater than
// the K-th sortable BF16 key in physical order, then boundary ties in order.
// Device histogram intrinsics themselves cannot run on a CPU.
struct TopK {
 unsigned calls=0;
 void operator()(Tensor<uint16_t> input,Tensor<uint32_t> ids,Tensor<uint16_t> out,
                 unsigned count,unsigned loop,unsigned loops,bool values){
  assert(!loop && loops==1 && values && count<=input.n && count%256==0);
  ++calls;std::vector<uint16_t> sorted(input.p,input.p+count);
  std::nth_element(sorted.begin(),sorted.begin()+2047,sorted.end(),std::greater<uint16_t>());
  const auto kth=sorted[2047];unsigned dst=0;
  for(unsigned i=0;i<count;++i)if(input.p[i]>kth){assert(dst<2048);ids.p[dst++]=i;}
  for(unsigned i=0;i<count && dst<2048;++i)if(input.p[i]==kth)ids.p[dst++]=i;
  assert(dst==2048);for(unsigned i=0;i<2048;++i)out.p[i]=input.p[ids.p[i]];
 }
};
struct Model {
 using SCORE_T=uint16_t;
 rotating::Context ctx;rotating::RuntimeShape runtimeShape_;
 void* rotationContext_=&ctx;
 unsigned topkCount_=2048,topkCountAlign256_=2048,trunkLen_;
 std::vector<uint32_t> tmp=std::vector<uint32_t>(2048),ids=std::vector<uint32_t>(2048);
 std::vector<uint16_t> merge,out=std::vector<uint16_t>(2048),scores;
 Tensor<uint32_t> topkSharedTmpLocal_{tmp.data(),2048},indicesOutLocal_{ids.data(),2048};
 Tensor<uint16_t> mrgValueLocal_,scoreOutLocal_{out.data(),2048},scoreGm;
 TopK topkOp_;
 Model(unsigned trunk):trunkLen_(trunk),merge(trunk+2048,0xdddd),scores(4*runtimeShape_.history),
  mrgValueLocal_{merge.data(),unsigned(merge.size())},scoreGm{scores.data(),unsigned(scores.size())}{}
"""
        + unified
        + merge
        + r"""
 void Resident(unsigned q,unsigned count){
  // Independently compute the resident-only result, equivalent to the old
  // multi-chunk reduction; seed its physical IDs before production merge.
  std::vector<uint16_t> input(scores.begin()+q*runtimeShape_.history,
                              scores.begin()+q*runtimeShape_.history+count);
  input.resize(QLICommon::Align(count,256),0);
  if(count>=2048)topkOp_({input.data(),unsigned(input.size())},indicesOutLocal_,scoreOutLocal_,input.size(),0,1,true);
  else {std::fill(ids.begin(),ids.end(),UINT32_MAX);std::fill(out.begin(),out.end(),0);
        std::iota(ids.begin(),ids.begin()+count,0);}
 }
};
int main(){
 using namespace kvip_recent_index;
 // Outstanding banks at ProcessTopK after its input/output waits. The new
 // method must neither consume other credits nor leave a private credit set.
 for(unsigned id:{0U,1U,2U,6U,7U})credits[unsigned(HardEvent::V_MTE2)][id]=true;
 std::mt19937 rng(29);
 for(unsigned resident:{2048U,8192U,16384U})for(unsigned trunk:{8192U,16384U})
 for(unsigned length:{0U,1U,127U,128U,129U,2048U,2049U,20000U})for(unsigned mode=0;mode<9;++mode){
  Model old(trunk),now(trunk);old.runtimeShape_.resident=now.runtimeShape_.resident=resident;
  const unsigned h=now.runtimeShape_.history;
  for(unsigned i=0;i<now.scores.size();++i)now.scores[i]=
   mode==0?0:mode==1?32768:mode==2?uint16_t(32760+rng()%17):uint16_t(1+rng()%65534);
  old.scores=now.scores; // Capacity stays unchanged; tensor pointers remain valid.
  const auto original=now.scores;
  int32_t bounds[2]={20001,int32_t(20001+length)};
  if(mode==3){bounds[0]=int32_t(resident-7);bounds[1]=int32_t(resident+length);}
  if(mode==4)bounds[0]=-1;
  if(mode==5)bounds[1]=h+1;
  uint32_t ends[4]={131000,20256,20003,1024};
  old.ctx.thresholdBounds=now.ctx.thresholdBounds=mode==6?0:reinterpret_cast<uint64_t>(bounds);
  old.ctx.queryEnds=now.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
  for(unsigned q=0;q<4;++q){
   unsigned n=std::min(resident,ends[q]);unsigned available=mode==7?20200U:mode==8?1024U:h;
   old.topkOp_.calls=now.topkOp_.calls=0;
   old.Resident(q,n);old.MergeRecentScores(q,q,available);
   bool unified=now.TryUnifiedResidentTopK(q,q,n,available);
   if(!unified){now.Resident(q,n);now.MergeRecentScores(q,q,available);}
   auto range=now.ctx.thresholdBounds?Clip(bounds[0],bounds[1],resident,h,std::min(ends[q],available)):Range{0,0};
   unsigned total=n+range.end-range.begin;
   bool expected=KVIP_RECENT_UNIFIED_TOPK && n>=2048 && range.end>range.begin && n%16==0 &&
    QLICommon::Align(total,256)<=trunk+2048;
   assert(unified==expected);
   if(unified){assert(now.topkOp_.calls==1 && old.topkOp_.calls>=2);}
   else assert(now.topkOp_.calls==old.topkOp_.calls);
   auto a=old.out,b=now.out;std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
   assert(a==b); // All 2048 selected score keys and the exact early K-th key.
   if(n>=2048){
    auto unique=now.ids;std::sort(unique.begin(),unique.end());
    assert(std::adjacent_find(unique.begin(),unique.end())==unique.end());
    for(unsigned i=0;i<2048;++i){unsigned p=now.ids[i];
     assert(p<n || (p>=range.begin && p<range.end));
     assert(now.out[i]==now.scores[q*h+p]);
    }
   }
   assert(now.scores==original && old.scores==original);
   for(unsigned id=0;id<8;++id){
    assert(credits[unsigned(HardEvent::V_MTE2)][id]==(id==0 || id==1 || id==2 || id==6 || id==7));
    assert(!credits[unsigned(HardEvent::MTE2_V)][id] && !credits[unsigned(HardEvent::V_S)][id]);
   }
  }
 }
 // Each concatenated index maps once; a padding index is never an Index ID.
 uint32_t ids[2048];std::iota(ids,ids+2048,16300U);
 Simt::VF_CALL<RemapUnified>({512,1,1},ids,16384U,130001U,129U);
 assert(ids[83]==16383 && ids[84]==130001 && ids[212]==130129 && ids[213]==UINT32_MAX);
}
"""
    )
    exe = tmp_path / "unified"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
            f"-DKVIP_RECENT_UNIFIED_TOPK={enabled}", "-I", str(COMMON), str(cpp), "-o", str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_unified_selection_is_prefix_only_and_preserves_fallback():
    source = (INDEXER / "indexer_vector.h").read_text()
    final = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessTopK(")
    assert "prefixActive_ && TryUnifiedResidentTopK(" in final
    assert "if (!recentMerged && validS2Len >= topkCount_)" in final
    assert "else if (!recentMerged)" in final
    save = function(source, "    __aicore__ inline void SaveResident(")
    assert "if (!recentMerged) MergeRecentScores(query, scoreRow, availableEnd);" in save
    assert "prefixOutput_" not in final
    assert "scoreGm[vecOffset *" in final
