"""Run the native recent-hit producer against CPU vector/DMA models."""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function


@pytest.mark.parametrize("enabled", [0, 1])
def test_recent_hits_physical_halos_causal_masks_and_cached_scores(tmp_path, enabled):
    vector = (INDEXER / "indexer_vector.h").read_text()
    score = (INDEXER / "indexer_score.h").read_text()
    classify = function(score, "__simd_vf__ inline void ClassifyRecentScoreTiles(")
    publish = function(vector, "    __aicore__ inline void PublishRecentHits(")
    cpp = tmp_path / "recent_hits.cpp"
    cpp.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>
#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__
#include "recent_index.h"
constexpr unsigned SIM_Q=4,EVENT_ID0=0,V_MTE2_EVENT3=5;
enum class HardEvent {V_MTE2,MTE2_V,V_MTE3,MTE3_V};
bool credits[4][8]{};
template<HardEvent D> void SetFlag(unsigned id){assert(!credits[unsigned(D)][id]);credits[unsigned(D)][id]=true;}
template<HardEvent D> void WaitFlag(unsigned id){assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;}
template<class T> T Min(T a,T b){return std::min(a,b);}
template<class T> T Max(T a,T b){return std::max(a,b);}
namespace CMPMODE {constexpr int GT=0,LT=1;}
namespace AscendC::MicroAPI {
enum class MaskPattern{ALL};enum class LoadDist{DIST_BRC_B16};
struct MaskReg {std::array<bool,128> v;};
template<class T> struct RegTensor {std::array<T,128> v;};
template<class T,MaskPattern> MaskReg CreateMask(){MaskReg m;m.v.fill(true);return m;}
template<class T> void Arange(RegTensor<T>& out,T initial){for(unsigned i=0;i<128;++i)out.v[i]=initial+i;}
template<class T> void Duplicate(RegTensor<T>& out,T value){out.v.fill(value);}
template<class T,LoadDist> void LoadAlign(RegTensor<T>& out,T* src){out.v.fill(*src);}
template<class T> void Adds(RegTensor<T>& out,RegTensor<T>& a,T b,MaskReg mask){
 for(unsigned i=0;i<128;++i)out.v[i]=mask.v[i]?T(a.v[i]+b):0;
}
template<class T> void Gather(RegTensor<T>& out,T* src,RegTensor<T>& offsets,MaskReg mask){
 for(unsigned i=0;i<128;++i)out.v[i]=mask.v[i]?src[offsets.v[i]]:0;
}
template<class T,int Mode> void Compare(MaskReg& out,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<128;++i)out.v[i]=mask.v[i] && (Mode==CMPMODE::GT?a.v[i]>b.v[i]:a.v[i]<b.v[i]);
}
template<class T,int Mode> void CompareScalar(MaskReg& out,RegTensor<T>& a,T b,MaskReg mask){
 for(unsigned i=0;i<128;++i)out.v[i]=mask.v[i] && (Mode==CMPMODE::GT?a.v[i]>b:a.v[i]<b);
}
inline void MaskAnd(MaskReg& out,MaskReg a,MaskReg b,MaskReg mask){
 for(unsigned i=0;i<128;++i)out.v[i]=mask.v[i] && a.v[i] && b.v[i];
}
// Native BF16 predicates use even bits: 128 token flags in 32 bytes.
inline void StoreAlign(uint16_t* dest,MaskReg mask){
 auto bytes=reinterpret_cast<uint8_t*>(dest);std::fill_n(bytes,32,0);
 for(unsigned i=0;i<128;++i)if(mask.v[i])bytes[i/4]|=uint8_t(1U<<((i%4)*2));
}
}
namespace vector1 {
"""
        + classify
        + r"""
}
template<class T> struct Tensor {
 T* p=nullptr;uint64_t n=0;
 Tensor operator[](uint64_t offset)const{assert(offset<=n);return {p+offset,n-offset};}
 template<class U> Tensor<U> ReinterpretCast(){assert(n*sizeof(T)%sizeof(U)==0);return {reinterpret_cast<U*>(p),n*sizeof(T)/sizeof(U)};}
 T* GetPhyAddr(){return p;}
 void SetGlobalBuffer(T* base){p=base;n=131328/4;}
};
template<class T> using GlobalTensor=Tensor<T>;
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool pad;unsigned a,b;T value;};
unsigned scoreReads=0,bitmapWrites=0;
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 assert(p.blockCount==1 && !p.srcStride && !p.dstStride && p.blockLen%32==0);
 unsigned n=p.blockLen/sizeof(T);assert(n<=dst.n && n<=src.n);
 std::copy_n(src.p,n,dst.p);scoreReads+=n;
}
void DataCopyPad(Tensor<uint8_t> dst,Tensor<uint8_t> src,DataCopyExtParams p){
 assert(p.blockLen==32 && p.srcStride==7 && p.dstStride==0);
 for(unsigned i=0;i<p.blockCount;++i){
  assert(i*32+32<=dst.n && i*256+32<=src.n);
  std::copy_n(src.p+i*256,32,dst.p+i*32);
 }bitmapWrites+=p.blockCount;
}
namespace rotating {
struct Context {uint64_t thresholdBounds=0,queryEnds=0,hits=0;};
struct RuntimeShape {unsigned resident=16384,history=131328;};
}
struct Model {
 using SCORE_T=uint16_t;
 rotating::Context ctx;rotating::RuntimeShape runtimeShape_;void* rotationContext_=&ctx;
 std::vector<uint32_t> scratch=std::vector<uint32_t>(2048,0xcccccccc);
 std::vector<uint16_t> cached=std::vector<uint16_t>(18432,0xdddd),threshold=std::vector<uint16_t>(256,32768),scores;
 std::vector<uint8_t> bitmap;
 Tensor<uint32_t> topkSharedTmpLocal_{scratch.data(),scratch.size()};
 Tensor<uint16_t> mrgValueLocal_{cached.data(),cached.size()},prefixThreshold_{threshold.data(),threshold.size()},scoreGm;
 Model():scores(4*runtimeShape_.history),bitmap(4*runtimeShape_.history/4,0xff),scoreGm{scores.data(),scores.size()}{
  ctx.hits=reinterpret_cast<uint64_t>(bitmap.data());
 }
"""
        + publish
        + r"""
};
int main(){
 using namespace kvip_recent_index;
 // At SaveResident the final input token remains pending, alongside the
 // independent query/key banks. Only EVENT5 is borrowed for recent score DMA.
 for(unsigned id:{0U,1U,2U,4U,6U,7U})credits[unsigned(HardEvent::V_MTE2)][id]=true;
 std::mt19937 rng(13);
 for(unsigned mode=0;mode<13;++mode)for(bool cache:{false,true})for(unsigned length:{1U,127U,128U,129U,512U,2048U,4097U}){
  Model m;const unsigned h=m.runtimeShape_.history,r=m.runtimeShape_.resident;
  for(auto& x:m.scores)x=uint16_t(rng()%65536);
  const auto original=m.scores;
  int32_t bounds[2]={20001,int32_t(20001+length)};
  if(mode==1)bounds[0]=20096; // Tile-aligned, usually wholly reusable from UB.
  if(mode==2){bounds[0]=int32_t(r-19);bounds[1]=int32_t(r+length);}
  if(mode==3)bounds[0]=-1;
  if(mode==4)bounds[1]=h+1;
  if(mode==5)bounds[1]=bounds[0];
  uint32_t ends[4]={131000,20205,20001,0};
  if(mode==6)std::fill_n(ends,4,0);
  if(mode==7)std::fill_n(ends,4,20002);
  if(mode==8)std::fill_n(ends,4,h);
  unsigned available=mode==9?h-128:h;
  m.ctx.thresholdBounds=mode==10?0:reinterpret_cast<uint64_t>(bounds);
  m.ctx.queryEnds=mode==11?0:reinterpret_cast<uint64_t>(ends);
  if(mode==12)m.ctx.hits=0;
  unsigned qmax=m.ctx.queryEnds?*std::max_element(ends,ends+4):h;
  for(unsigned q=0;q<4;++q){
   unsigned qend=m.ctx.queryEnds?ends[q]:h;
   auto exact=Clip(bounds[0],bounds[1],r,h,std::min(qend,available));
   bool merged=cache && exact.end>exact.begin && exact.end-exact.begin<=2048;
   if(merged)std::copy(m.scores.begin()+q*h+exact.begin,m.scores.begin()+q*h+exact.end,m.cached.begin()+r);
   auto before=m.bitmap;
   scoreReads=bitmapWrites=0;
   m.PublishRecentHits(q,q,available,merged);
   auto tiles=(m.ctx.thresholdBounds && m.ctx.hits)?ReusableTiles(bounds[0],bounds[1],r,h,available,qmax):Range{0,0};
   unsigned expectedReads=0;
   for(unsigned physical=0;physical<h;++physical){
    uint8_t bits=uint8_t(3U<<((physical%4)*2));
    uint8_t actual=m.bitmap[q*h/4+physical/4]&bits;
    if(physical>=tiles.begin*128 && physical<tiles.end*128){
     bool selected=physical<std::min(qend,available) && m.scores[q*h+physical]>32768;
     assert(actual==uint8_t(unsigned(selected)<<((physical%4)*2)));
    }else assert(actual==(before[q*h/4+physical/4]&bits));
   }
   if(tiles.end>tiles.begin){
    assert(bitmapWrites==tiles.end-tiles.begin);
    if(merged && exact.begin%128==0 && exact.end%128==0 && qend>=exact.end)assert(scoreReads==0);
   }else assert(scoreReads==0 && bitmapWrites==0);
   // No query writes a neighboring bitmap or resident victim coordinates.
   assert(std::equal(m.bitmap.begin(),m.bitmap.begin()+q*h/4,before.begin()));
   assert(std::equal(m.bitmap.begin()+(q+1)*h/4,m.bitmap.end(),before.begin()+(q+1)*h/4));
   assert(m.scores==original);
   for(unsigned id=0;id<8;++id){
    assert(credits[unsigned(HardEvent::V_MTE2)][id]==(id!=3 && id!=5));
    for(unsigned d=1;d<4;++d)assert(!credits[d][id]);
   }
  }
 }
 // Exercise an offset that is not DMA/vector-load aligned, exact threshold
 // ties, and old all-one masks. This is the production VF, not a reimplementation.
 std::vector<uint16_t> input(1024,32768),limit(128,32768),hits(256,0xffff);
 input[17]=32769;input[18]=32767;input[145]=65535;
 vector1::ClassifyRecentScoreTiles(input.data(),limit.data(),hits.data(),17,130000,256,130129);
 auto bytes=reinterpret_cast<uint8_t*>(hits.data());
 assert(bytes[0]==1 && bytes[256]==1);
 for(unsigned i=1;i<32;++i)assert(bytes[i]==0 && bytes[256+i]==0);
}
"""
    )
    exe = tmp_path / "recent_hits"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
            f"-DKVIP_RECENT_SKIP={enabled}", "-I", str(COMMON), str(cpp), "-o", str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_skip_progress_is_independent_of_gemm_events():
    source = (INDEXER / "indexer_vector.h").read_text()
    publish = function(source, "__aicore__ inline void QLIVector<QLIT>::PublishScoreProgress(")
    assert "CrossCore" not in publish
    assert "VEC1_MTE3_V_EVENT" not in publish
    assert "publishShardEnd ||" in publish
    assert "rotating::IsTrigger(" in publish
    process = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessVec1(")
    assert "if (!scoreOnly) PublishScoreProgress(info, progressShard, publishShardEnd);" in process
    save = function(source, "    __aicore__ inline void SaveResident(")
    assert save.index("PublishRecentHits(") < save.index("if (context->lruSlots)")
    assert "ready[0]=runtimeShape_.resident;" in save


def test_progress_dma_keeps_query_shard_offsets_and_completion_boundaries(tmp_path):
    source = (INDEXER / "indexer_vector.h").read_text()
    method = function(source, "__aicore__ inline void QLIVector<QLIT>::PublishScoreProgress(")
    method = method.replace("QLIVector<QLIT>::", "")
    cpp = tmp_path / "progress.cpp"
    cpp.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#define __aicore__
#define __ubuf__
constexpr unsigned SIM_Q=4,EVENT_ID0=0,ROTATION_PUBLISH_TILES=4;
enum class HardEvent {MTE3_V,V_MTE3};bool credits[2][8]{};
template<HardEvent D> void SetFlag(unsigned id){assert(!credits[unsigned(D)][id]);credits[unsigned(D)][id]=true;}
template<HardEvent D> void WaitFlag(unsigned id){assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;}
template<class A,class B> auto CeilDiv(A a,B b){return (a+b-1)/b;}
template<class T> struct Tensor {
 T* p;unsigned n;Tensor operator[](unsigned off){assert(off<=n);return {p+off,n-off};}
 T* GetPhyAddr(){return p;}
};
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride;};
unsigned writes=0;
void DataCopyPad(Tensor<uint32_t> dst,Tensor<uint32_t> src,DataCopyExtParams p){
 assert(p.blockLen==32 && !p.srcStride && p.dstStride==96);
 for(unsigned i=0;i<p.blockCount;++i){assert(i*32+8<=dst.n && i*8+8<=src.n);
  std::copy_n(src.p+i*8,8,dst.p+i*32);
 }++writes;
}
namespace QLICommon {struct RunInfo {unsigned gS1Idx,s2Idx,actS1Size,bIdx;bool isLastS2InnerLoop;};}
namespace rotating {
bool IsTrigger(unsigned,unsigned,unsigned,unsigned end){return end==257 || end==601;}
void PrepareProgress(uint32_t* table,unsigned base){
 for(unsigned tile=0;tile<64;++tile)std::fill_n(table+tile*16,16,(base+tile+1)*128);
}
}
struct Model {
 unsigned s1BaseSize_=4,s2BaseSize_=128,blockId_=0,rotationProgressBase_=128;
 struct {unsigned history=131328,resident=16384,rounds=8;}runtimeShape_;
 std::array<uint32_t,1024> bank{},output{};
 Tensor<uint32_t> rotationProgress_{bank.data(),unsigned(bank.size())},rotationProgressGm_{output.data(),unsigned(output.size())};
 Model(){rotating::PrepareProgress(bank.data(),rotationProgressBase_);output.fill(0xcccccccc);}
"""
        + method
        + r"""
};
int main(){
 for(unsigned size:{2U,4U})for(unsigned active=0;active<=size;++active)
 for(unsigned core:{0U,1U})for(unsigned shard:{0U,1U}){
  Model m;m.s1BaseSize_=size;m.blockId_=core;
  for(unsigned tile:{127U,128U,131U,191U,192U,256U,300U,600U,1025U})for(unsigned cause=0;cause<3;++cause){
   QLICommon::RunInfo info{0,tile,active,1,cause==1};bool end=cause==2;
   auto before=m.output;writes=0;
   m.PublishScoreProgress(info,shard,end);
   unsigned count=core?active/2:(active+1)/2,start=core*((active+1)/2);
   bool expected=count && tile*128>=16384 && (end || cause==1 || (tile+1)%4==0 || tile+1==257 || tile+1==601);
   assert(writes==unsigned(expected));
   for(unsigned pos=0;pos<m.output.size();++pos){
    bool written=false;
    for(unsigned q=0;q<count;++q){unsigned first=(4+start+q)*32+shard*16;
     written|=expected && pos>=first && pos<first+8;
    }
    assert(m.output[pos]==(written?(tile+1)*128:before[pos]));
   }
   for(auto& bank:credits)for(bool pending:bank)assert(!pending);
  }
 }
}
"""
    )
    exe = tmp_path / "progress"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined", str(cpp), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
