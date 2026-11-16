"""Execute the device compactor, gather, ID remapper and production driver.

The existing radix TopK is modeled by its GT/EQ contract; the new device
functions run unchanged. MTE2 writes are delayed until the matching join.
CANN and actual device scheduling still require separate validation.
"""

import subprocess

from test_recent_index import INDEXER, function


def source():
    select = function((INDEXER / "topk.h").read_text(), "    __aicore__ inline uint32_t SelectPruned(")
    driver = function(
        (INDEXER / "indexer_vector.h").read_text(),
        "__aicore__ inline bool QLIVector<QLIT>::TryPrunedFinalTopK(",
    ).replace("QLIVector<QLIT>::", "")
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
#define __aicore__
#define __simd_vf__
#define __simt_vf__
#define __ubuf__
#define LAUNCH_BOUND(n)
#define KVIP_FINAL_TOPK_PRUNE 1
#define V9_UBMEM 0
struct {unsigned x;} threadIdx;
std::vector<std::function<void()>> pending;
std::vector<std::function<void()>> vectorStores;
unsigned dma=0,selectCalls=0,selectElements=0;
namespace AscendC {
enum class SpecialPurposeReg {AR};unsigned ar=0,last=0;
template<SpecialPurposeReg> uint64_t GetSpr(){return ar;}
namespace MicroAPI {
enum class GatherMaskMode {STORE_REG};enum class PostLiteral {POST_MODE_UPDATE};
enum class CMPMODE {GE};
enum class MaskPattern {ALL};
struct MaskReg {std::array<bool,128> v{};};
template<class T> struct RegTensor {std::array<T,128> v{};};
struct UnalignRegForStore {};
template<SpecialPurposeReg> void ClearSpr(){ar=last=0;}
template<class T,MaskPattern> MaskReg CreateMask(){MaskReg m;m.v.fill(true);return m;}
template<class T> MaskReg UpdateMask(uint32_t& count){MaskReg m;auto n=std::min(128U,count);
 std::fill_n(m.v.begin(),n,true);count-=n;return m;}
template<class T> void LoadAlign(RegTensor<T>& dst,T* input){assert(pending.empty());std::copy_n(input,128,dst.v.begin());}
template<class T> void Arange(RegTensor<T>& dst,T offset){for(unsigned i=0;i<128;++i)dst.v[i]=T(offset+i);}
template<class T> void Duplicate(RegTensor<T>& dst,T value){dst.v.fill(value);}
template<class T> void Select(RegTensor<T>& dst,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<128;++i)dst.v[i]=mask.v[i]?a.v[i]:b.v[i];}
template<class T> void StoreAlign(T* dst,RegTensor<T>& value,MaskReg mask){
 auto data=value.v;vectorStores.push_back([=]{for(unsigned i=0;i<128;++i)if(mask.v[i])dst[i]=data[i];});}
template<class T,CMPMODE> void CompareScalar(MaskReg& dst,RegTensor<T>& input,T value,MaskReg valid){
 for(unsigned i=0;i<128;++i)dst.v[i]=valid.v[i]&&input.v[i]>=value;}
template<class T,GatherMaskMode> void Squeeze(RegTensor<T>& dst,RegTensor<T>& input,MaskReg mask){
 unsigned n=0;for(unsigned i=0;i<128;++i)if(mask.v[i])dst.v[n++]=input.v[i];
 last=n*sizeof(T);ar+=last;}
template<class T,PostLiteral> void StoreUnAlign(T* dst,RegTensor<T>& value,UnalignRegForStore&){
 auto data=value.v;unsigned n=last/sizeof(T),start=(ar-last)/sizeof(T);
 vectorStores.push_back([=]{std::copy_n(data.begin(),n,dst+start);});}
template<class T> void StoreUnAlign(T*& dst,RegTensor<T>& value,UnalignRegForStore&,uint32_t stride){
 auto data=value.v;T* target=dst;vectorStores.push_back([=]{std::copy_n(data.begin(),128,target);});dst+=stride;}
template<class T> void StoreUnAlignPost(T*,UnalignRegForStore&){}
template<class T> void StoreUnAlignPost(T*&,UnalignRegForStore&,int){}
template<class T> void Gather(RegTensor<T>& dst,T* input,RegTensor<uint16_t>& offsets,MaskReg mask){
 dst.v.fill(0);for(unsigned i=0;i<128;++i)if(mask.v[i]){assert(offsets.v[i]<16384);dst.v[i]=input[offsets.v[i]];}}
}
}
using namespace AscendC;
#include "final_topk_prune.h"
namespace Simt {
struct Dim3{unsigned x,y,z;};
template<auto F,class... Args> void VF_CALL(Dim3 dim,Args... args){assert(vectorStores.empty());for(threadIdx.x=0;threadIdx.x<dim.x;++threadIdx.x)F(args...);}
}
enum class HardEvent {V_S,V_MTE2,MTE2_V};bool credits[3][8]{};
constexpr unsigned EVENT_ID0=0,V_MTE2_EVENT3=5,TOPK_MTE2_V_EVENT=5;
enum {PIPE_ALL,PIPE_V};
void Flush(){for(auto& f:pending)f();pending.clear();}
void FlushVector(){for(auto& f:vectorStores)f();vectorStores.clear();}
template<int P> void PipeBarrier(){FlushVector();if constexpr(P==PIPE_ALL)Flush();}
template<HardEvent H> void SetFlag(unsigned id){assert(!credits[unsigned(H)][id]);credits[unsigned(H)][id]=true;}
template<HardEvent H> void WaitFlag(unsigned id){assert(credits[unsigned(H)][id]);credits[unsigned(H)][id]=false;
 if constexpr(H==HardEvent::MTE2_V)Flush();if constexpr(H==HardEvent::V_S)FlushVector();}
namespace QLICommon {unsigned Align(unsigned a,unsigned b){return (a+b-1)/b*b;}
uint64_t Align(uint64_t a,uint64_t b){return (a+b-1)/b*b;}}
template<class T> T Min(T a,T b){return std::min(a,b);}
template<class T> T CeilDiv(T a,T b){return (a+b-1)/b;}
template<class T> struct Tensor {
 T* p;unsigned n;
 Tensor operator[](uint64_t off)const{assert(off<=n);return {p+off,n-unsigned(off)};}
 T* GetPhyAddr()const{return p;}
 T GetValue(unsigned i)const{assert(i<n);return p[i];}
 template<class U> Tensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p),unsigned(n*sizeof(T)/sizeof(U))};}
};
template<class T> using LocalTensor=Tensor<T>;
template<class T> void Duplicate(Tensor<T> dst,T value,unsigned n){assert(n<=dst.n);std::fill_n(dst.p,n,value);}
template<class T> void DataCopy(Tensor<T> dst,Tensor<T> src,unsigned n){assert(n<=dst.n&&n<=src.n&&n*sizeof(T)%32==0);std::copy_n(src.p,n,dst.p);}
struct DataCopyExtParams {unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
template<class T> struct DataCopyPadExtParams {bool pad;unsigned a,b;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams params,DataCopyPadExtParams<T>){
 assert(params.blockCount==1&&!params.srcStride&&!params.dstStride);unsigned n=params.blockLen/sizeof(T);
 assert(n<=src.n&&n<=dst.n);pending.push_back([=]{std::copy_n(src.p,n,dst.p);});++dma;}
std::vector<unsigned> Select(const uint16_t* values,unsigned n,unsigned k){
 std::vector<uint16_t> copy(values,values+n);std::nth_element(copy.begin(),copy.begin()+n-k,copy.end());
 uint16_t threshold=copy[n-k];std::vector<unsigned> order;
 for(unsigned i=0;i<n;++i)if(values[i]>threshold)order.push_back(i);
 for(unsigned i=0;i<n;++i)if(values[i]==threshold)order.push_back(i);
 assert(order.size()>=k);return order;
}
namespace topkb16gather {
template<bool> void LiTopKVF(Tensor<uint16_t> tmp,Tensor<uint16_t> out,Tensor<uint16_t> input,
 Tensor<uint32_t>,Tensor<uint32_t>,Tensor<uint32_t>,Tensor<uint32_t>,unsigned k,unsigned n){
 assert(pending.empty()&&vectorStores.empty()&&n%256==0&&n<=input.n&&n<=18432);
 auto ids=Select(input.p,n,k);assert(ids.size()<=tmp.n);for(unsigned i=0;i<ids.size();++i)tmp.p[i]=uint16_t(ids[i]);
 for(unsigned i=0;i<k;++i)out.p[i]=input.p[ids[i]];++selectCalls;selectElements+=n;
}
}
struct TopK {
 unsigned topK=2048;std::vector<uint32_t> history=std::vector<uint32_t>(2048,0xaaaaaaaa);
 std::vector<uint16_t> temp=std::vector<uint16_t>(18432,0xbbbb);
 Tensor<uint16_t> tmpIndexLocal{temp.data(),18432};Tensor<uint32_t> hisIndexLocal[2]={{nullptr,0},{history.data(),2048}};
 Tensor<uint32_t> histogramsLocal{},idxHighLocal{},idxLowLocal{},nkValueLocal{};
"""
        + select
        + r"""
};
template<class T> struct Guard {
 std::vector<T> v;unsigned n;Guard(unsigned count):v(count+64,T(0x5a5a)),n(count){}
 Tensor<T> tensor(){return {v.data()+32,n};}
 void check(){for(unsigned i=0;i<32;++i)assert(v[i]==T(0x5a5a)&&v[n+32+i]==T(0x5a5a));}
};
struct Model {
 using SCORE_T=uint16_t;using QK_T=float;
 bool prefixActive_=false,streamingTopkOnly_=false,noPrefetchRound_=false;
 struct {bool isLDOpen=false;unsigned mBaseSize=64,kSeqSize=131328;} constInfo_;
 struct {unsigned splitQ=1,batch=16,aicCount=32,resident=16384;} runtimeShape_;
 unsigned s1BaseSize_=2,s2BaseSize_=128,topkCount_=2048,trunkLen_=16384;
 Guard<uint16_t> merge{18432},out{2048},offsets{16384},thresholds{256};Guard<uint32_t> ids{2048};
 std::vector<uint16_t> gm=std::vector<uint16_t>(4*131328);
 Tensor<uint16_t> scoreGm{gm.data(),unsigned(gm.size())},mrgValueLocal_=merge.tensor(),scoreOutLocal_=out.tensor(),prefixThreshold_=thresholds.tensor();
 Tensor<float> resMm1UB_{reinterpret_cast<float*>(offsets.tensor().p),8192};
 Tensor<uint32_t> indicesOutLocal_=ids.tensor();TopK topkOp_;
 void check(){merge.check();out.check();offsets.check();thresholds.check();ids.check();assert(pending.empty()&&vectorStores.empty());for(auto& row:credits)for(bool c:row)assert(!c);}
"""
        + driver
        + r"""
};
// Independent reference: reproduce the unpruned 16K merge contract, including
// its intermediate GT/EQ order rather than comparing just selected sets.
std::vector<uint32_t> Reference(const uint16_t* values,unsigned n){
 std::vector<uint16_t> scores;std::vector<uint32_t> ids;
 for(unsigned begin=0;begin<n;begin+=16384){unsigned count=std::min(16384U,n-begin);
  auto merged=scores;auto positions=ids;
  merged.insert(merged.end(),values+begin,values+begin+count);
  for(unsigned i=0;i<count;++i)positions.push_back(begin+i);
  unsigned valid=merged.size();merged.resize(QLICommon::Align(valid,256U),0);positions.resize(merged.size(),0xffffffff);
  auto chosen=Select(merged.data(),merged.size(),2048);scores.clear();ids.clear();
  for(unsigned i=0;i<2048;++i){scores.push_back(merged[chosen[i]]);ids.push_back(positions[chosen[i]]);}
 }
 return ids;
}
void Check(unsigned n,unsigned pattern,unsigned seed,unsigned q){
 Model m;std::mt19937 rng(seed);auto* values=m.gm.data()+(q%2)*m.constInfo_.kSeqSize;
 for(unsigned i=0;i<m.constInfo_.kSeqSize;++i){
  uint16_t v=uint16_t(1+rng()%65535);
  if(pattern==1)v=7;
  if(pattern==2)v=uint16_t(1+rng()%5);
  if(pattern==3)v=i>=n-2048?65535:uint16_t(1+rng()%100);
  if(pattern==4)v=i<16384?uint16_t(40000+rng()%20000):uint16_t(1+rng()%45000);
  if(pattern==5)v=i>=n-512?uint16_t(60000+rng()%5000):uint16_t(1+rng()%60000);
  if(pattern==6)v=i>=n-2048?(i==n-1?65535:50000):1;
  values[i]=v;
 }
 std::vector<uint16_t> seedValues(values,values+16384);
 if(pattern==3||pattern==5||pattern==6)seedValues.insert(seedValues.end(),values+std::max(16384U,n-2048),values+n);
 std::nth_element(seedValues.begin(),seedValues.end()-2048,seedValues.end());
 uint16_t threshold=seedValues[seedValues.size()-2048];m.prefixThreshold_.p[(q%2)*128]=threshold;
 auto original=m.gm;auto expected=Reference(values,n);unsigned calls=selectCalls;
 assert(m.TryPrunedFinalTopK(q,q%2,n));
 assert(std::equal(expected.begin(),expected.end(),m.indicesOutLocal_.p));
 assert(original==m.gm);assert(selectCalls>calls);m.check();
}
int main(){
 for(unsigned n:{16384U,16385U,32767U,130048U,131072U,131328U})
  for(unsigned pattern=0;pattern<7;++pattern)for(unsigned seed:{17U,29U})Check(n,pattern,seed,seed%4);
 // Original thresholds with zero/tiny domains and unsupported shapes must
 // bypass without touching dead Cube scratch or starting a DMA.
 for(unsigned mode=0;mode<9;++mode){Model m;m.prefixThreshold_.p[0]=99;unsigned before=dma,calls=selectCalls;
  unsigned end=131072;if(mode==0)m.prefixActive_=true;if(mode==1)m.streamingTopkOnly_=true;
  if(mode==2)m.constInfo_.isLDOpen=true;if(mode==3)m.runtimeShape_.aicCount=31;
  if(mode==4)m.s1BaseSize_=4;if(mode==5)m.constInfo_.mBaseSize=32;
  if(mode==6)end=16000;if(mode==7)m.prefixThreshold_.p[0]=0;
  if(mode==8)m.noPrefetchRound_=true;
  assert(!m.TryPrunedFinalTopK(0,0,end)&&dma==before&&selectCalls==calls);m.check();
 }
 // A corrupt/high threshold cannot publish an incomplete final list.
 {Model m;std::fill(m.gm.begin(),m.gm.end(),1);m.prefixThreshold_.p[0]=2;
  auto original=m.gm;assert(!m.TryPrunedFinalTopK(0,0,131328));assert(original==m.gm);m.check();}
}
"""
    )


def test_actual_pruned_topk_driver_order_and_boundaries(tmp_path):
    cpp = tmp_path / "final_prune.cpp"
    cpp.write_text(source())
    exe = tmp_path / "final_prune"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
            "-I", str(INDEXER), str(cpp), "-o", str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
