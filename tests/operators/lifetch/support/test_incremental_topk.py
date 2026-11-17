"""Run the actual producer compaction/remapping under ASan/UBSan on CPU."""

import subprocess

import pytest

from test_recent_index import INDEXER, function


@pytest.mark.parametrize("window,prefilter", [(0, 0), (1, 0), (1, 1)])
def test_actual_compact_pairs_padding_and_physical_remap(tmp_path, window, prefilter):
    text = (INDEXER / "incremental_topk_vector.h").read_text()
    methods = "\n".join(function(text, name) for name in (
        "__simd_callee__ inline void StoreWindow(",
        "template<bool positions>\n__simd_vf__ inline void PackWindow(",
        "__simd_callee__ inline void Compact(",
        "__simd_vf__ inline void ZeroTail(",
        "__simd_vf__ inline void BroadcastThreshold(",
        "inline void Remap(",
        "inline void TagSeed(",
        "__simd_vf__ inline void CompactIds(",
        "__simd_vf__ inline void UntagIds(",
    ))
    model = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>
#define __simd_callee__
#define __simd_vf__
#define __gm__
#define __ubuf__
#define KVIP_REVERSE_COLD 1
constexpr uint32_t kTop=2048,kNeedsFetch=0x80000000U,kIdMask=0x7fffffffU;
struct {unsigned x;} threadIdx;
// Model both SDK enums: importing both namespaces must not resolve a bare name.
namespace AscendC {
enum class SpecialPurposeReg{AR};
enum class GatherMaskMode{VERSION_V1,VERSION_V2};
namespace Reg {enum class GatherMaskMode{NO_STORE_REG,STORE_REG};}
namespace MicroAPI = Reg;
}
using namespace AscendC;
using namespace AscendC::MicroAPI;
enum class CMPMODE{GT,LT,GE,NE};
enum class MaskPattern{ALL};enum class LoadDist{DIST_BRC_B16};
enum class HighLowPart{LOWEST};
enum class PostLiteral{POST_MODE_UPDATE};
struct MaskReg{std::array<bool,128> v{};};
template<class T>struct RegTensor{std::array<T,256/sizeof(T)> v{};};
struct UnalignRegForStore{};
unsigned ar=0,packed=0;
template<AscendC::SpecialPurposeReg>void ClearSpr(){ar=0;}
template<class T,MaskPattern>MaskReg CreateMask(){MaskReg m;std::fill_n(m.v.begin(),256/sizeof(T),true);return m;}
template<class T>void LoadAlign(RegTensor<T>& r,T* p){std::copy_n(p,r.v.size(),r.v.begin());}
template<class T,LoadDist>void LoadAlign(RegTensor<T>& r,T* p){r.v.fill(*p);}
template<class T>void StoreAlign(T* p,RegTensor<T>& r,MaskReg m){for(unsigned i=0;i<r.v.size();++i)if(m.v[i])p[i]=r.v[i];}
template<class T>void Duplicate(RegTensor<T>& r,T v){r.v.fill(v);}
template<class T,HighLowPart>void Duplicate(RegTensor<T>& r,RegTensor<T>& a,MaskReg m){T v=a.v[0];for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=v;}
template<class T>void Arange(RegTensor<T>& r,T v){for(unsigned i=0;i<r.v.size();++i)r.v[i]=T(v+i);}
template<class T>void Adds(RegTensor<T>& r,RegTensor<T>& a,T v,MaskReg m){for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=T(a.v[i]+v);}
template<class T,CMPMODE C>void Compare(MaskReg& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<a.v.size();++i)r.v[i]=m.v[i]&&(C==CMPMODE::GT?a.v[i]>b.v[i]:C==CMPMODE::LT?a.v[i]<b.v[i]:C==CMPMODE::GE?a.v[i]>=b.v[i]:a.v[i]!=b.v[i]);}
template<class T,CMPMODE C>void CompareScalar(MaskReg& r,RegTensor<T>& a,T v,MaskReg m){RegTensor<T>b;b.v.fill(v);Compare<T,C>(r,a,b,m);}
void MaskOr(MaskReg& r,MaskReg a,MaskReg b,MaskReg m){for(unsigned i=0;i<128;++i)r.v[i]=m.v[i]&&(a.v[i]||b.v[i]);}
void MaskAnd(MaskReg& r,MaskReg a,MaskReg b,MaskReg m){for(unsigned i=0;i<128;++i)r.v[i]=m.v[i]&&a.v[i]&&b.v[i];}
template<class T>void Select(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<r.v.size();++i)r.v[i]=m.v[i]?a.v[i]:b.v[i];}
template<class T>void Min(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=std::min(a.v[i],b.v[i]);}
template<class T>void ReduceMin(RegTensor<T>& r,RegTensor<T>& a,MaskReg m){T v=65535;for(unsigned i=0;i<a.v.size();++i)if(m.v[i])v=std::min(v,a.v[i]);r.v.fill(0xdead);r.v[0]=v;}
template<class T,AscendC::MicroAPI::GatherMaskMode>void Squeeze(RegTensor<T>& r,RegTensor<T>& a,MaskReg m){packed=0;for(unsigned i=0;i<a.v.size();++i)if(m.v[i])r.v[packed++]=a.v[i];}
template<class T,PostLiteral>void StoreUnAlign(T*& p,RegTensor<T>& r,UnalignRegForStore&){std::copy_n(r.v.begin(),packed,p);p+=packed;ar+=packed*sizeof(T);}
template<class T>void StoreUnAlignPost(T*&,UnalignRegForStore&){}
template<class T>void And(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=a.v[i]&b.v[i];}
template<class T>T asc_ldcg(T* p){return *p;}
namespace rotating {
struct Context{uint64_t reservedHot;uint32_t begin,end;uint32_t* valid;};
bool HotValid(uint32_t resident,Context* ctx,uint32_t b,uint32_t t){return ctx->valid[b*resident+t]!=0;}
bool IsTailToken(Context* ctx,uint32_t,uint32_t t){return t>=ctx->begin&&t<ctx->end;}
}
METHODS
int main(){
 std::mt19937 gen(42);
 std::array<uint32_t,4096> seed;std::array<uint32_t,16384> valid;std::array<int32_t,16384> reserved;
 for(unsigned i=0;i<16384;++i){valid[i]=(i%7)!=0;reserved[i]=(i%11)?-1:1;}
 rotating::Context ctx{uint64_t(reserved.data()),16384,17000,valid.data()};
 for(unsigned i=0;i<2048;++i)seed[i]=16000+i;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)TagSeed(&ctx,0,16384,seed.data());
 for(unsigned i=0;i<2048;++i){unsigned t=16000+i;bool need=(t<16384||t>=17000)&&(t>=16384||!valid[t]||reserved[t]>=0);assert(seed[i]==(t|(need?kNeedsFetch:0)));}

 for(unsigned trial=0;trial<1000;++trial){
  RegTensor<uint16_t> input;for(auto& v:input.v)v=uint16_t(gen()%16);
  uint16_t threshold=uint16_t(gen()%16);unsigned valid=gen()%129,lo=gen()%129,hi=gen()%129;
  if(lo>hi)std::swap(lo,hi);
  std::array<uint16_t,256> score,offset;score.fill(0xbeef);offset.fill(0xbeef);
  unsigned start=gen()%64,base=gen()%16000;
  Compact(input,&threshold,score.data()+start,offset.data()+start,uint16_t(base),valid|(lo<<8)|(hi<<16));
  unsigned n=0;for(unsigned i=0;i<128;++i)if(i<valid&&(i<lo||i>=hi)&&input.v[i]>threshold){
   assert(score[start+n]==input.v[i]&&offset[start+n]==base+i);++n;
  }
  assert(ar==2*n&&score[start+n]==0xbeef&&offset[start+n]==0xbeef);
  // Execute actual fixed-address store with poisoned neighbors and AR. It
  // must overwrite all 128 lanes, leave AR unchanged, and preserve guards.
  score.fill(0xbeef); ar=0x123456;
  StoreWindow(input,score.data()+64,valid|(lo<<8)|(hi<<16),&threshold);
  for(unsigned i=0;i<256;++i){
   unsigned j=i-64;
   uint16_t expected=i>=64&&i<192 ?
     (j<valid&&(j<lo||j>=hi)&&(!KVIP_INCREMENTAL_PREFILTER||input.v[j]>threshold) ? input.v[j] : 0) : 0xbeef;
   assert(score[i]==expected);
  }
  assert(ar==0x123456);
 }
 // Pack full/dense/empty windows without per-tile scalar counts. The 2048
 // boundary is a storage capacity, not a limit on accepted candidates.
 for(unsigned count:{0U,1U,127U,128U,2047U,2048U,2049U,16383U,16384U}){
  std::vector<uint16_t> raw(16384,0),packedScores(16384+128,0xbeef),packedOffsets(16384+128,0xbeef);
  std::vector<unsigned> order(16384);for(unsigned i=0;i<16384;++i)order[i]=i;
  std::shuffle(order.begin(),order.end(),gen);
  for(unsigned i=0;i<count;++i)raw[order[i]]=uint16_t(1+gen()%65535);
  PackWindow<false>(raw.data(),packedScores.data(),16384);assert(ar==2*count);
  PackWindow<true>(raw.data(),packedOffsets.data(),16384);assert(ar==2*count);
  unsigned n=0;for(unsigned i=0;i<16384;++i)if(raw[i]){assert(packedScores[n]==raw[i]&&packedOffsets[n]==i);++n;}
  for(unsigned i=n;i<packedScores.size();++i)assert(packedScores[i]==0xbeef&&packedOffsets[i]==0xbeef);
 }
 std::array<uint16_t,4096> values;values.fill(0xffff);
 for(unsigned n=2048;n<4096;++n){
  values.fill(0xffff);unsigned end=(n+255)/256*256;
  ZeroTail(values.data(),n,end);
  for(unsigned i=0;i<4096;++i)assert(values[i]==(i>=n&&i<end?0:0xffff));
 }
 std::array<uint16_t,128> out;values.fill(0xffff);values[0]=17;
 BroadcastThreshold(values.data(),out.data());for(auto v:out)assert(v==17);
 std::array<uint32_t,2048> ids,old,flags,selected;std::array<uint16_t,2048> offsets;
 for(unsigned i=0;i<2048;++i){ids[i]=i%2?2048+i:i;old[i]=i+3;offsets[i]=uint16_t(2*i);}
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)Remap(ids.data(),old.data(),offsets.data(),20000,16384,129024,0,0,0,0);
 for(unsigned i=0;i<2048;++i){unsigned token=20000+((KVIP_INCREMENTAL_WINDOW&&!KVIP_INCREMENTAL_PREFILTER)?i:2*i);unsigned physical=16384+129024-2048-(token&~2047U)+(token&2047U);assert(ids[i]==(i%2?(physical|kNeedsFetch):old[i]));flags[i]=i%3==0;}
 for(unsigned i=0;i<2048;++i)ids[i]=(ids[i]&kIdMask)|(flags[i]?kNeedsFetch:0);
 CompactIds(ids.data(),selected.data());unsigned n=0;
 for(unsigned i=0;i<2048;++i)if(flags[i])assert(selected[n++]==(ids[i]&kIdMask));assert(ar==4*n);
 UntagIds(ids.data(),selected.data());for(unsigned i=0;i<2048;++i)assert(selected[i]==(ids[i]&kIdMask));
#if KVIP_INCREMENTAL_PREFILTER
 for(unsigned i=0;i<2048;++i)ids[i]=2048+i;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)Remap(ids.data(),old.data(),nullptr,20000,16384,129024,0,0,0,0);
 for(unsigned i=0;i<2048;++i){unsigned token=20000+i;assert(ids[i]==((16384+129024-2048-(token&~2047U)+(token&2047U))|kNeedsFetch));}
#endif
 for(unsigned i=0;i<2048;++i)ids[i]=i+16000;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)Remap(ids.data(),old.data(),offsets.data(),0,0,0,16384,128000,0,0);
 for(unsigned i=0;i<2048;++i)assert(ids[i]==(i+16000<16384?i+16000:128000+i+16000-16384));
}
'''.replace("METHODS", methods)
    source = tmp_path / "incremental.cpp"
    source.write_text(model)
    binary = source.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined", f"-DKVIP_INCREMENTAL_WINDOW={window}", f"-DKVIP_INCREMENTAL_PREFILTER={prefilter}",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
