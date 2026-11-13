"""CPU execution of the native recent-range and threshold-merge code."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
INDEXER = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer"
COMMON = ROOT / "csrc/operators/common/lifetch"


def function(source, signature):
    begin = source.index(signature)
    brace = source.index("{", begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_native_recent_merge_and_bounds(tmp_path):
    merge = function((INDEXER / "indexer_vector.h").read_text(), "    __aicore__ inline void MergeRecentScores(")
    source = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(n)
struct {unsigned x;} threadIdx;
#include "recent_index.h"
constexpr unsigned SIM_Q=4, EVENT_ID0=0, V_MTE2_EVENT3=5;
enum class HardEvent {V_MTE2,MTE2_V,V_S};
bool credits[3][8]{};
template<HardEvent D> void SetFlag(unsigned id){assert(!credits[unsigned(D)][id]);credits[unsigned(D)][id]=true;}
template<HardEvent D> void WaitFlag(unsigned id){assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;}
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto F, class... Args> void VF_CALL(Dim3 dim,Args... args) {
 for(threadIdx.x=0;threadIdx.x<dim.x;++threadIdx.x)F(args...);
}
}
namespace QLICommon {unsigned Align(unsigned a,unsigned n){return (a+n-1)/n*n;}}
template<class T> T Min(T a,T b){return std::min(a,b);}
template<class T> struct Tensor {
 T* p;unsigned n;
 Tensor operator[](unsigned offset)const{assert(offset<=n);return {p+offset,n-offset};}
 T* GetPhyAddr(){return p;}
};
template<class T> void DataCopy(Tensor<T> dst,Tensor<T> src,unsigned n){
 assert(n<=dst.n && n<=src.n);std::copy_n(src.p,n,dst.p);
}
template<class T> void Duplicate(Tensor<T> dst,T value,unsigned n){assert(n<=dst.n);std::fill_n(dst.p,n,value);}
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool pad;unsigned a,b;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 assert(p.blockCount==1 && !p.srcStride && !p.dstStride);
 DataCopy(dst,src,p.blockLen/sizeof(T));
}
namespace rotating {
struct Context {uint64_t thresholdBounds=0,queryEnds=0;};
struct RuntimeShape {unsigned resident=16384,history=131328;};
}
// Numerical stand-in for the established device TopK. The new production
// merge, IDs remap, clipping and buffer accesses themselves run unchanged.
struct TopK {
 void operator()(Tensor<uint16_t> input,Tensor<uint32_t> ids,Tensor<uint16_t> out,
                 unsigned count,unsigned loop,unsigned loops,bool scores){
  assert(!loop && loops==1 && scores && count<=input.n);
  std::vector<unsigned> order(count);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return input.p[a]>input.p[b];});
  for(unsigned i=0;i<2048;++i){ids.p[i]=order[i];out.p[i]=input.p[order[i]];}
 }
};
struct Model {
 using SCORE_T=uint16_t;
 rotating::Context ctx;
 rotating::RuntimeShape runtimeShape_;
 void* rotationContext_=&ctx;
 unsigned topkCount_=2048,trunkLen_=16384;
 std::vector<uint32_t> tmp=std::vector<uint32_t>(22000),ids=std::vector<uint32_t>(2048);
 std::vector<uint16_t> merge=std::vector<uint16_t>(18432),out=std::vector<uint16_t>(2048),scores;
 Tensor<uint32_t> topkSharedTmpLocal_{tmp.data(),unsigned(tmp.size())},indicesOutLocal_{ids.data(),2048};
 Tensor<uint16_t> mrgValueLocal_{merge.data(),unsigned(merge.size())},scoreOutLocal_{out.data(),2048},scoreGm;
 TopK topkOp_;
 Model():scores(4*runtimeShape_.history),scoreGm{scores.data(),unsigned(scores.size())}{}
"""
        + merge
        + r"""
};
int main(){
 using namespace kvip_recent_index;
 // Real AllocEventID/ProcessTopK leave these banks outstanding at SaveResident.
 for(unsigned id:{0U,1U,2U,4U,6U,7U})credits[unsigned(HardEvent::V_MTE2)][id]=true;
 for(unsigned length: {0U,1U,127U,128U,129U,4095U,20000U}){
  auto range=Clip(20001,20001+length,16384,131328,131328);
  auto all=Tiles(range),a=Tiles(range,0,2),b=Tiles(range,1,2);
  assert(a.begin==all.begin && a.end==b.begin && b.end==all.end);
  if(length)assert(all.begin*128<=range.begin && all.end*128>=range.end);
 }
 assert(Clip(-1,10,2048,4096,4096).end==0);
 assert(Clip(4000,5000,2048,4096,4096).end==0);
 assert(Clip(10,9,2048,4096,4096).end==0);
 std::mt19937 rng(17);
 for(unsigned resident:{2048U,8192U,16384U})for(unsigned trunk:{8192U,16384U})
 for(int mode=0;mode<12;++mode){
  Model m;m.runtimeShape_.resident=resident;m.trunkLen_=trunk;
  const unsigned h=m.runtimeShape_.history;
  for(auto& value:m.scores)value=mode==8?uint16_t(0):uint16_t(1+rng()%65534);
  auto original=m.scores;
  int32_t bounds[2]={int32_t(mode==0?resident-100:20001),int32_t(mode==1?20001:mode==2?60003:20513)};
  if(mode==3){bounds[0]=0;bounds[1]=int32_t(resident+259);}
  if(mode==4){bounds[0]=20513;bounds[1]=20001;}
  if(mode==5){bounds[0]=-1;bounds[1]=20001;}
  if(mode==6){bounds[0]=20001;bounds[1]=int32_t(h+1);}
  uint32_t ends[4]={131000,20256,20003,0};
  if(mode==7)ends[0]=1024;
  m.ctx.thresholdBounds=mode==9?0:reinterpret_cast<uint64_t>(bounds);
  m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
  for(unsigned q=0;q<4;++q){
   unsigned n=std::min(resident,ends[q]);
   std::vector<uint32_t> residentIds(n);std::iota(residentIds.begin(),residentIds.end(),0);
   std::stable_sort(residentIds.begin(),residentIds.end(),[&](unsigned a,unsigned b){return m.scores[q*h+a]>m.scores[q*h+b];});
   std::fill(m.ids.begin(),m.ids.end(),UINT32_MAX);std::fill(m.out.begin(),m.out.end(),0);
   for(unsigned i=0;i<std::min(n,2048U);++i){m.ids[i]=residentIds[i];if(n>=2048)m.out[i]=m.scores[q*h+m.ids[i]];}
   auto beforeIds=m.ids;auto beforeScores=m.out;
   unsigned availableEnd=mode==10?20200U:mode==11?1024U:h;
   m.MergeRecentScores(q,q,availableEnd);
   auto range=m.ctx.thresholdBounds?Clip(bounds[0],bounds[1],resident,h,std::min(ends[q],availableEnd)):Range{0,0};
   if(n<2048 || availableEnd<2048 || range.begin==range.end){assert(m.ids==beforeIds && m.out==beforeScores);continue;}
   std::vector<uint16_t> expected;
   for(unsigned i=0;i<n;++i)expected.push_back(m.scores[q*h+i]);
   for(unsigned i=range.begin;i<range.end;++i)expected.push_back(m.scores[q*h+i]);
   std::sort(expected.begin(),expected.end(),std::greater<uint16_t>());
   assert(*std::min_element(m.out.begin(),m.out.end())==expected[2047]);
   // With finite nonzero keys, padding cannot enter the selected set.
   if(mode!=8){
    auto unique=m.ids;std::sort(unique.begin(),unique.end());
    assert(std::adjacent_find(unique.begin(),unique.end())==unique.end());
    for(unsigned i=0;i<2048;++i){unsigned id=m.ids[i];assert(id<n || (id>=range.begin && id<range.end));assert(m.out[i]==m.scores[q*h+id]);}
   }
   assert(m.scores==original); // Never overwrites the full-LI score input.
   for(unsigned id=0;id<8;++id)assert(credits[unsigned(HardEvent::V_MTE2)][id]==(id!=3 && id!=5));
   for(unsigned id=0;id<8;++id)assert(!credits[unsigned(HardEvent::MTE2_V)][id] && !credits[unsigned(HardEvent::V_S)][id]);
  }
 }
}
"""
    )
    cpp = tmp_path / "recent.cpp"
    cpp.write_text(source)
    exe = tmp_path / "recent"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined", "-I", str(COMMON), str(cpp), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_recent_does_not_change_final_topk_input():
    source = (INDEXER / "indexer_vector.h").read_text()
    final = source[source.index("__aicore__ inline void QLIVector<QLIT>::ProcessTopK(") :]
    # The final driver always rebuilds from scoreGm, never the early prefix.
    assert "prefixOutput_" not in final
    assert "scoreGm[vecOffset *" in final
    assert "if (prefixActive_)" in final
    score = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessVec1(")
    assert score.count("!scoreOnly && curS2Idx >= runtimeShape_.resident") == 2
    assert "if (!scoreOnly) PublishScoreProgress(info, progressShard, publishShardEnd);" in score


def test_native_score_pass_preserves_regular_scan_state(tmp_path):
    source = (INDEXER / "indexer.h").read_text()
    method = function(source, "__aicore__ inline bool QuantLightningIndexerKernel<QLIT>::ScoreRecent(")
    method = method.replace("QuantLightningIndexerKernel<QLIT>::", "")
    cpp = tmp_path / "schedule.cpp"
    cpp.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#define ASCEND_IS_AIC (false)
#include "recent_index.h"
constexpr unsigned SIM_Q=4,S2_BASE_SIZE=128;
namespace rotating {struct Context {uint64_t thresholdBounds=0,queryEnds=0;};}
namespace QLICommon {
#ifndef MODEL_SCALE_TILES
#define MODEL_SCALE_TILES 16
#endif
constexpr unsigned KScaleTiles=MODEL_SCALE_TILES;
struct RunInfo {unsigned loop,s2Idx,s2Start,validS2Len,qScaleLoop,kScaleLoop;
 bool isFirstS2InnerLoop,isLastS2InnerLoop; bool retainQuery=false;};
}
struct Vector {
 std::vector<QLICommon::RunInfo> calls;
 void ProcessVec1(QLICommon::RunInfo info,unsigned shard,bool publish,bool scoresOnly){
  assert(shard==0 && !publish && scoresOnly);calls.push_back(info);
 }
};
struct Cube {void ComputeMm1(QLICommon::RunInfo){assert(false);}};
struct Model {
 unsigned resident_=16384;
 struct {unsigned history=131328;} runtimeShape_;
 rotating::Context ctx;
 void* rotationContext_=&ctx;
 struct {unsigned s2Start=0;} splitCoreInfo;
 struct {unsigned bIdx=0,actS2Size=131328,s2LoopEnd=1025,validS2Len=131328;} tempLoopInfo;
 Vector vectorService;Cube matmulService;
 void CalcRunInfo(unsigned loop,unsigned tile,QLICommon::RunInfo& i,unsigned q,unsigned k){
  i={loop,tile,splitCoreInfo.s2Start,tempLoopInfo.validS2Len,q,k,
     tile==splitCoreInfo.s2Start,tile==tempLoopInfo.s2LoopEnd};
 }
"""
        + method
        + r"""
};
int main(){
 for(unsigned length: {0U,1U,127U,128U,129U,2050U,8193U})for(unsigned shards:{1U,2U}){
  std::vector<unsigned> visits(131328/128,0);
  for(unsigned shard=0;shard<shards;++shard){
   Model m;int32_t bounds[]={20001,int32_t(20001+length)};
   uint32_t ends[]={131000,130999,130998,130997};
   m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
   unsigned loop=6;bool had=m.ScoreRecent(loop,3,shard,shards);
   assert(had==!m.vectorService.calls.empty());
   assert(m.splitCoreInfo.s2Start==0 && m.tempLoopInfo.s2LoopEnd==1025 && m.tempLoopInfo.validS2Len==131328);
   auto calls=m.vectorService.calls;assert(loop==6+calls.size());
   for(unsigned i=0;i<calls.size();++i){auto c=calls[i];
    ++visits[c.s2Idx];assert(c.loop==6+i && c.qScaleLoop==3);
    assert(c.isFirstS2InnerLoop==(i==0));assert(c.isLastS2InnerLoop==(i+1==calls.size()));
    assert(c.kScaleLoop==1+i/QLICommon::KScaleTiles);assert(c.validS2Len==(calls.back().s2Idx+1)*128);
   }
  }
  auto range=kvip_recent_index::Clip(20001,20001+length,16384,131328,131000);
  auto tiles=kvip_recent_index::Tiles(range);
  for(unsigned t=0;t<visits.size();++t)assert(visits[t]==unsigned(t>=tiles.begin && t<tiles.end));
 }
 Model m;unsigned loop=7;assert(!m.ScoreRecent(loop,0,0,1) && loop==7);
 int32_t bounds[]={130500,131000};uint32_t ends[]={0,0,0,0};
 m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
 assert(!m.ScoreRecent(loop,0,0,1) && loop==7);
}
"""
    )
    for tiles in (8, 16):
        exe = tmp_path / f"schedule-{tiles}"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                f"-DMODEL_SCALE_TILES={tiles}",
                "-I",
                str(COMMON),
                str(cpp),
                "-o",
                str(exe),
            ],
            check=True,
        )
        subprocess.run([str(exe)], check=True)
