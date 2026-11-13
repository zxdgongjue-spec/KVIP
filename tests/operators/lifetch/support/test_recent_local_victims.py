"""Execute SaveResident and both victim classifiers with delayed DMA models."""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function


@pytest.mark.parametrize("ubmem", [0, 1])
def test_local_victims_preserve_bitmap_and_publication(tmp_path, ubmem):
    vector = (INDEXER / "indexer_vector.h").read_text()
    score = (INDEXER / "indexer_score.h").read_text()
    save = function(vector, "    __aicore__ inline void SaveResident(")
    old = function(score, "__simd_vf__ inline void ClassifyResidentScores(")
    local = function(score, "__simd_vf__ inline void ClassifyLocalResidentScores(")
    gather = function((COMMON / "victim_ops.h").read_text(), "inline void GatherVictimScores(")
    cpp = tmp_path / "local_victims.cpp"
    cpp.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__
#define __local_mem__
#define __VEC_SCOPE__
#include "recent_index.h"
constexpr unsigned SIM_Q=4,EVENT_ID0=0,TOPK_V_MTE3_EVENT=6,V_MTE2_EVENT3=5;
constexpr unsigned PIPE_ALL=0,PIPE_V=1;
std::vector<std::function<void()>> mte2,mte3;
void finish(std::vector<std::function<void()>>& pending){for(auto& f:pending)f();pending.clear();}
template<unsigned P> void PipeBarrier(){if(P==PIPE_ALL){finish(mte2);finish(mte3);}}
enum class HardEvent {V_MTE3,MTE2_V,MTE3_S,V_S,V_MTE2};
bool credits[5][8]{};
template<HardEvent E> void SetFlag(unsigned id){assert(!credits[unsigned(E)][id]);credits[unsigned(E)][id]=true;}
template<HardEvent E> void WaitFlag(unsigned id){
 assert(credits[unsigned(E)][id]);credits[unsigned(E)][id]=false;
 if(E==HardEvent::MTE2_V)finish(mte2);
 if(E==HardEvent::MTE3_S)finish(mte3);
}
namespace CMPMODE {constexpr int GT=0,LT=1,EQ=2;}
namespace AscendC::MicroAPI {
enum class MaskPattern {ALL};enum class LoadDist {DIST_BRC_B16};
enum class MemType {VEC_STORE,VEC_LOAD};
struct MaskReg {std::array<bool,128> v;};
template<class T> struct RegTensor {std::array<T,128> v;};
template<class T,MaskPattern> MaskReg CreateMask(){MaskReg m;m.v.fill(true);return m;}
template<class T> void Arange(RegTensor<T>& r,T initial){for(unsigned i=0;i<128;++i)r.v[i]=initial+i;}
template<class T> void Duplicate(RegTensor<T>& r,T value){r.v.fill(value);}
template<class T> void Muls(RegTensor<T>& out,RegTensor<T>& in,T v,MaskReg m){
 for(unsigned i=0;i<128;++i)out.v[i]=m.v[i]?T(in.v[i]*v):0;
}
template<class T> void LoadAlign(RegTensor<T>& r,T* src){assert(mte2.empty());std::copy_n(src,128,r.v.begin());}
template<class T,LoadDist> void LoadAlign(RegTensor<T>& r,T* src){assert(mte2.empty());r.v.fill(*src);}
template<class T> void Min(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<128;++i)if(m.v[i])r.v[i]=std::min(a.v[i],b.v[i]);
}
template<class T> void ReduceMin(RegTensor<T>& r,RegTensor<T>& a,MaskReg){r.v.fill(*std::min_element(a.v.begin(),a.v.end()));}
template<class T> void StoreAlign(T* dst,RegTensor<T>& r,MaskReg m){for(unsigned i=0;i<128;++i)if(m.v[i])dst[i]=r.v[i];}
template<class T> void Gather(RegTensor<T>& r,T* base,RegTensor<T>& offsets,MaskReg m){
 assert(mte2.empty());for(unsigned i=0;i<128;++i)r.v[i]=m.v[i]?base[offsets.v[i]]:0;
}
template<class T,int C> void CompareScalar(MaskReg& out,RegTensor<T>& a,T b,MaskReg m){
 for(unsigned i=0;i<128;++i)out.v[i]=m.v[i]&&(C==CMPMODE::GT?a.v[i]>b:C==CMPMODE::LT?a.v[i]<b:a.v[i]==b);
}
template<class T,int C> void Compare(MaskReg& out,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<128;++i)out.v[i]=m.v[i]&&(C==CMPMODE::GT?a.v[i]>b.v[i]:a.v[i]<b.v[i]);
}
void MaskAnd(MaskReg& out,MaskReg a,MaskReg b,MaskReg m){for(unsigned i=0;i<128;++i)out.v[i]=a.v[i]&&b.v[i]&&m.v[i];}
template<class T> void Select(RegTensor<T>& out,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<128;++i)out.v[i]=m.v[i]?a.v[i]:b.v[i];
}
void StoreAlign(uint16_t* dst,MaskReg mask){
 auto bytes=reinterpret_cast<uint8_t*>(dst);std::fill_n(bytes,32,0);
 for(unsigned i=0;i<128;++i)if(mask.v[i])bytes[i/4]|=uint8_t(1U<<((i%4)*2));
}
void LoadAlign(MaskReg& mask,uint16_t* src){
 auto bytes=reinterpret_cast<uint8_t*>(src);
 for(unsigned i=0;i<128;++i)mask.v[i]=bool(bytes[i/4]&(1U<<((i%4)*2)));
}
template<MemType,MemType> void LocalMemBar(){}
}
namespace MicroAPI=AscendC::MicroAPI;
namespace vector1 {
void ClassifyScore(MicroAPI::RegTensor<uint16_t>& scores,uint16_t* threshold,uint16_t* dst,MicroAPI::MaskReg all){
 MicroAPI::MaskReg selected;
 MicroAPI::CompareScalar<uint16_t,CMPMODE::GT>(selected,scores,*threshold,all);
 MicroAPI::StoreAlign(dst,selected);
}
"""
        + old
        + "\n#if !V9_UBMEM\n"
        + local
        + r"""
#endif
}
unsigned gmReads=0,simtCalls=0,bitmapDma=0,cleanCalls=0;
uint32_t asc_ldcg(uint32_t* p){++gmReads;return *p;}
struct {unsigned x;} threadIdx;
namespace Simt {
struct Dim3 {unsigned x,y,z;};
template<auto F,class... A> void VF_CALL(Dim3 d,A... args){
 ++simtCalls;assert(mte2.empty());for(threadIdx.x=0;threadIdx.x<d.x;++threadIdx.x)F(args...);
}
}
namespace rotating {
constexpr unsigned kLruSlots=2048;
struct Context {uint64_t lruSlots=0,hits=0,queryEnds=0,progress=0;};
struct Shape {unsigned resident=16384,history=131328;};
"""
        + gather
        + r"""
}
namespace kvaf_device {void CleanCacheLines(uint8_t*,unsigned bytes){assert(mte2.empty()&&mte3.empty()&&bytes==128);++cleanCalls;}}
namespace ubmem_device=kvaf_device;
namespace QLICommon {uint64_t Align(uint64_t n,uint64_t a){return (n+a-1)/a*a;}}
template<class T> struct Tensor {
 T* p=nullptr;uint64_t n=0;bool gm=false;
 Tensor operator[](uint64_t off)const{assert(off<=n);return {p+off,n-off,gm};}
 T* GetPhyAddr(){return p;}
 template<class U> Tensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p),n*sizeof(T)/sizeof(U),gm};}
 void SetGlobalBuffer(T* base){p=base;n=100000000;gm=true;}
};
template<class T> using GlobalTensor=Tensor<T>;
template<class T> void DataCopy(Tensor<T> dst,Tensor<T> src,unsigned count){
 assert(count<=dst.n&&count<=src.n&&dst.gm!=src.gm);
 auto copy=[=](){std::copy_n(src.p,count,dst.p);};(dst.gm?mte3:mte2).push_back(copy);
}
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
void DataCopyPad(Tensor<uint8_t> dst,Tensor<uint8_t> src,DataCopyExtParams p){
 assert(dst.gm&&!src.gm&&p.blockCount==16&&p.blockLen==32&&p.srcStride==7&&!p.dstStride);
 ++bitmapDma;mte3.push_back([=](){for(unsigned i=0;i<16;++i)std::copy_n(src.p+i*256,32,dst.p+i*32);});
}
struct Model {
 rotating::Context ctx;rotating::Shape runtimeShape_;void* rotationContext_=&ctx;
 struct {uint64_t kSeqSize=131328;} constInfo_;unsigned s2BaseSize_=128;
 std::vector<uint16_t> merge=std::vector<uint16_t>(18432),scores=std::vector<uint16_t>(4*131328);
 std::vector<uint32_t> scratch=std::vector<uint32_t>(4096,0xcccccccc),victims=std::vector<uint32_t>(3*2048);
 std::array<uint16_t,256> thresholds{};std::array<uint16_t,2048> out{};std::array<int32_t,2048> indices{};
 std::vector<uint8_t> prefix=std::vector<uint8_t>(3*(4*2048*6+4*256),0xdd),bitmap=std::vector<uint8_t>(12*131328/4,0xee);
 std::array<uint32_t,12> ends{};std::array<uint32_t,12*32> progress{};
 uint8_t* prefixOutput_=prefix.data();
 Tensor<uint16_t> mrgValueLocal_{merge.data(),merge.size()},scoreGm{scores.data(),scores.size(),true},prefixThreshold_{thresholds.data(),256},scoreOutLocal_{out.data(),2048};
 Tensor<uint32_t> topkSharedTmpLocal_{scratch.data(),scratch.size()};
 Tensor<int32_t> indicesOutLocal_{indices.data(),2048};
 Model(){ctx.lruSlots=uint64_t(victims.data());ctx.hits=uint64_t(bitmap.data());ctx.queryEnds=uint64_t(ends.data());ctx.progress=uint64_t(progress.data());}
 void MergeRecentScores(unsigned,unsigned,unsigned){}
 // Null adaptive controls preserve the original victim/protection path.
 void PrepareAdmissionThreshold(unsigned){}
"""
        + save
        + r"""
};
int main(){
 using kvip_recent_index::UseLocalVictims;
 assert(UseLocalVictims(true,16384,16384,131328));
 assert(!UseLocalVictims(false,16384,16384,131328));
 assert(!UseLocalVictims(true,16384,16383,131328));
 assert(!UseLocalVictims(true,16384,16384,16383));
 assert(!UseLocalVictims(true,1024,1024,131328));
 assert(!UseLocalVictims(true,32768,32768,131328));
 std::mt19937 rng(49);
 for(unsigned r:{2048U,2176U,8192U,16384U,32768U})for(unsigned mode=0;mode<7;++mode)for(unsigned q:{0U,1U,6U,11U}){
  Model m;unsigned row=q%4,h=m.runtimeShape_.history;m.runtimeShape_.resident=r;
  bool ready=mode!=1;unsigned end=mode==2?r-1:mode==3?0:r;
  unsigned available=mode==4?r-1:h;
  bool fast=!V9_UBMEM&&UseLocalVictims(ready,r,end,available);
  if(mode==5&&!fast)continue; // Malformed IDs are outside the old GM contract.
  for(auto& x:m.scores)x=uint16_t(rng());
  if(r<=m.merge.size())std::copy_n(m.scores.data()+row*h,r,m.merge.data());
  const auto beforeScores=m.scores;const auto beforeMerge=m.merge;
  uint16_t threshold=mode==6?65535:32768;m.thresholds.fill(threshold);m.out.fill(threshold);
  m.ends.fill(h);m.ends[q]=end;
  for(auto& x:m.victims)x=rng()%r;
  auto ids=m.victims.data()+(q/4)*2048;
  ids[0]=0;ids[1]=r-1;ids[2]=1;ids[3]=1; // Duplicates preserve ordinal bits.
  m.scores[row*h+1]=threshold;if(r<=m.merge.size())m.merge[1]=threshold;
  if(mode==5){ids[4]=r;ids[5]=UINT32_MAX;ids[6]=0x10000;ids[7]=0x10001;}
  auto scoreExpected=m.scores;auto mergeExpected=m.merge;auto prefixBefore=m.prefix;
  gmReads=simtCalls=bitmapDma=cleanCalls=0;
  m.SaveResident(q,row,q%2==0,available,ready);
  assert(mte2.empty()&&mte3.empty()&&bitmapDma==1);
  assert(simtCalls==unsigned(!fast));assert(gmReads==(fast?0U:2048U));
  assert(cleanCalls==unsigned(q%2==0));assert(m.progress[q*32]==(q%2==0?r:0));
  auto actual=m.bitmap.data()+q*h/4;
  for(unsigned i=0;i<2048;++i){
   bool valid=ids[i]<end;
   bool selected=valid&&m.scores[row*h+ids[i]]>threshold;
   unsigned bits=(actual[i/4]>>((i%4)*2))&3U;assert(bits==unsigned(selected));
  }
  for(unsigned i=512;i<h/4;++i)assert(actual[i]==0xee);
  for(unsigned other=0;other<12;++other)if(other!=q)assert(m.bitmap[other*h/4]==0xee);
  assert(m.scores==scoreExpected&&m.merge==mergeExpected);
  if(ready){ // No prefix candidate list is published by threshold-only mode.
   unsigned base=(q/4)*(4*2048*6+4*256);
   for(unsigned i=0;i<4*2048*6;++i)assert(m.prefix[base+i]==prefixBefore[base+i]);
  }
  auto prefixThreshold=reinterpret_cast<uint16_t*>(m.prefix.data()+(q/4)*(4*2048*6+4*256)+4*2048*6+(q%4)*256);
  for(unsigned i=0;i<128;++i)assert(prefixThreshold[i]==threshold);
  for(auto& bank:credits)for(bool credit:bank)assert(!credit);
 }
}
"""
    )
    exe = tmp_path / "local_victims"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DV9_UBMEM={ubmem}",
            "-I",
            str(COMMON),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
