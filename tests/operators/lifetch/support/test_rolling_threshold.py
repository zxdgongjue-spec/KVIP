"""Execute the native rolling CDF and candidate filter with host lane models.

These cover numerical/visibility contracts, not CANN instruction latency.
"""

from pathlib import Path
import subprocess

import pytest

from test_recent_register_kth import register_model_source
from test_recent_index import function, INDEXER

NATIVE = INDEXER.parents[1]


def execute(tmp_path, source, name):
    cpp = tmp_path / (name + ".cpp")
    cpp.write_text(source)
    binary = cpp.with_suffix("")
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        "-I", str(INDEXER), "-I", str(NATIVE / "include"),
        "-I", str(NATIVE.parent / "common/lifetch"),
        str(cpp), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)


@pytest.mark.parametrize("batched,windowed,compact,first", [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 1, 1)])
@pytest.mark.parametrize("reverse", [0, 1])
def test_actual_cdf_saturation_ties_causal_tail_and_seed_dedup(tmp_path, batched, windowed, compact, first, reverse):
    model = register_model_source('#include "recent_index.h"')
    model = model[:model.index('#include "recent_threshold.h"')]
    model = model.replace("{DIST_DINTLV_B8}", "{DIST_DINTLV_B8,DIST_BRC_B16}")
    model = model.replace("GE=0,LT=1,EQ=2", "GE=0,LT=1,EQ=2,GT=3")
    model = model.replace(":a.v[i]==b.v[i]", ":C==CMPMODE::GT?a.v[i]>b.v[i]:a.v[i]==b.v[i]")
    score_source = (INDEXER / "indexer_score.h").read_text()
    classify = function(score_source, "__simd_callee__ inline void ClassifyScore(")
    classify += "\n" + function(score_source, "__simd_callee__ inline void ClassifyRollingScore(")
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>
#define __aicore__
#define __ubuf__
#define KVIP_ROLLING_THRESHOLD 1
''' + model + r'''
namespace AscendC::MicroAPI {
enum class MemType {VEC_STORE,VEC_LOAD};
template<MemType,MemType> void LocalMemBar(){}
template<class T> void LoadAlign(RegTensor<T>& r,T* p){std::copy_n(p,r.v.size(),r.v.begin());}
template<class T,LoadDist> void LoadAlign(RegTensor<T>& r,T* p){r.v.fill(*p);}
inline void StoreAlign(uint16_t* p,MaskReg mask){for(unsigned i=0;i<128;++i)p[i]=mask.v[i];}
template<class T> MaskReg UpdateMask(uint32_t& n){MaskReg m;unsigned take=std::min(n,unsigned(256/sizeof(T)));
 std::fill_n(m.v.begin(),take,true);n-=take;return m;}
template<class T,int C> void CompareScalar(MaskReg& r,RegTensor<T>& a,T value,MaskReg mask){
 RegTensor<T> b;Duplicate(b,value);Compare<T,C>(r,a,b,mask);}
// Match CANN's required fourth execution-mask argument; no permissive overload.
inline void MaskAnd(MaskReg& r,MaskReg& a,MaskReg& b,MaskReg& mask){
 for(unsigned i=0;i<256;++i)r.v[i]=mask.v[i]&&(a.v[i]&&b.v[i]);}
inline void MaskOr(MaskReg& r,MaskReg& a,MaskReg& b,MaskReg& mask){
 for(unsigned i=0;i<256;++i)r.v[i]=mask.v[i]&&(a.v[i]||b.v[i]);}
template<class T> void DeInterleave(RegTensor<T>& a,RegTensor<T>& b,RegTensor<T> x,RegTensor<T> y){
 unsigned n=x.v.size();for(unsigned i=0;i<n/2;++i){a.v[i]=x.v[2*i];b.v[i]=x.v[2*i+1];
 a.v[i+n/2]=y.v[2*i];b.v[i+n/2]=y.v[2*i+1];}}
}
#include "rolling_threshold_vector.h"
''' + classify + r'''
using namespace AscendC::MicroAPI;
uint16_t kth(std::vector<uint16_t> a){std::nth_element(a.begin(),a.end()-2048,a.end());return a[a.size()-2048];}
int main(){
 {
  RegTensor<uint16_t> score;for(unsigned i=0;i<128;++i)score.v[i]=i;
  auto all=CreateMask<uint16_t,MaskPattern::ALL>();
  std::array<uint16_t,128> threshold{},hits{};threshold.fill(90);
  std::array<uint16_t,kvip_rolling::kStateWords> state{};
  std::fill_n(state.data()+kvip_rolling::kBins,128,80);
  RegTensor<uint16_t> limit,anchor;LoadAlign(limit,threshold.data());Duplicate(anchor,uint16_t(80));
  kvip_rolling::SaveAdmission(state.data(),limit,anchor);
  ClassifyRollingScore(score,threshold.data(),hits.data(),all,state.data());
  for(unsigned i=0;i<128;++i)assert(hits[i]==unsigned(i>=90));
  threshold.fill(80);Duplicate(limit,uint16_t(80));
  kvip_rolling::SaveAdmission(state.data(),limit,anchor);
  ClassifyRollingScore(score,threshold.data(),hits.data(),all,state.data());
  for(unsigned i=0;i<128;++i)assert(hits[i]==unsigned(i>80));
 }

#if KVIP_ROLLING_COMPACT
 // Exhaust the full sortable-u16 domain at zero, byte boundaries and max.
 // The compact strict test must never lose an exact-Kth tie.
 for(unsigned seed:{0U,1U,255U,256U,15000U,65534U,65535U}){
  for(unsigned step:{0U,1U,2U,127U,255U}){
   unsigned kthKey=std::min(seed+step,65535U);
   std::array<uint16_t,kvip_rolling::kStateWords> state{};
   std::array<uint16_t,128> threshold,hits;
   threshold.fill(kthKey);
   RegTensor<uint16_t> key,anchor,score;
   Duplicate(key,uint16_t(kthKey));Duplicate(anchor,uint16_t(seed));
   kvip_rolling::SaveAdmission(state.data(),key,anchor);
   auto all=CreateMask<uint16_t,MaskPattern::ALL>();
   for(unsigned offset=0;offset<65536;offset+=128){
    for(unsigned lane=0;lane<128;++lane)score.v[lane]=offset+lane;
    ClassifyRollingScore(score,threshold.data(),hits.data(),all,state.data());
    for(unsigned lane=0;lane<128;++lane)
     assert(hits[lane]==unsigned(offset+lane>seed && offset+lane>=kthKey));
   }
  }
 }
#endif
 std::mt19937 random(716);
 for(unsigned mode=0;mode<5;++mode){
  const unsigned n=16384,recentBegin=131072-2048,recentEnd=131072-67;
  std::vector<uint16_t> values(262144);
  for(unsigned i=0;i<values.size();++i){
   values[i]=mode==0?uint16_t(15000+random()%300):mode==1?uint16_t(random()):
             mode==2?uint16_t(65000+random()%536):mode==3?uint16_t(15000):uint16_t(i%2?0:65535);
  }
  std::vector<uint16_t> truth(values.begin(),values.begin()+n);
  truth.insert(truth.end(),values.begin()+recentBegin,values.begin()+recentEnd);
  uint16_t initial=kth(truth);
  std::array<uint16_t,128> threshold;threshold.fill(initial);
  std::array<uint16_t,kvip_rolling::kStateWords+32> storage;storage.fill(0xa55a);
  auto seed=truth;seed.resize((seed.size()+255)/256*256,0);
  kvip_rolling::Seed(seed.data(),truth.size(),threshold.data(),storage.data()+16);
  const unsigned end=values.size()-71;
  bool frozen=false;
  unsigned previous=initial,pending=0,expectedHistograms=0,batchCount=0,windowCount=0;
  std::array<uint16_t,kvip_rolling::kWindowScores> window{};
  std::array<uint16_t,128> windowHits{};
  histograms=0;
  std::vector<unsigned> tiles;
  for(unsigned block=n;block<values.size();block+=2048)tiles.push_back(block);
  if(REVERSE_TEST)std::reverse(tiles.begin(),tiles.end()-1);
  unsigned ordinal=0;
  for(unsigned block:tiles)for(unsigned tile=block;tile<block+2048;tile+=128){
   RegTensor<uint16_t> score;LoadAlign(score,values.data()+tile);
   unsigned mask=kvip_rolling::TileMask(tile,end,recentBegin,recentEnd);
   for(unsigned i=tile;i<std::min(tile+128,end);++i)
    if(i<recentBegin||i>=recentEnd)truth.push_back(values[i]);
   // An odd 137-tile window exercises partial and odd batches, including
   // reversed block boundaries and the final partial causal tile.
   bool advance=(++ordinal)%137==0 || ordinal*128==values.size()-n;
   ++batchCount;
   if(KVIP_ROLLING_WINDOWED && !frozen){
    assert(windowCount+128<=window.size());
    std::copy_n(values.data()+tile,128,window.data()+windowCount);windowCount+=128;
    if(advance){expectedHistograms+=2*((batchCount+1)/2);batchCount=0;}
   }else if(!KVIP_ROLLING_WINDOWED && KVIP_ROLLING_BATCHED){
    mask=kvip_rolling::BatchMask(mask,pending,advance);
    if(batchCount==8||advance){expectedHistograms+=2*((batchCount+1)/2);batchCount=0;}
   }else if(!KVIP_ROLLING_WINDOWED){expectedHistograms+=2;}
#if KVIP_ROLLING_WINDOWED
   if(advance && !frozen){
    kvip_rolling::ObserveWindow(window.data(),windowCount,storage.data()+16,threshold.data(),windowHits.data(),
      n+ordinal*128-windowCount,end,recentBegin,recentEnd,n,end/2048*2048);
    windowCount=0;
    for(unsigned i=0;i<128;++i)assert(windowHits[i]==unsigned(score.v[i]>=threshold[0]&&score.v[i]>initial));
   }
#else
   kvip_rolling::Observe(score,storage.data()+16,threshold.data(),mask,advance);
#endif
   if(advance && !frozen){unsigned exact=kth(truth),expected=std::min(exact,unsigned(initial)+255);
    assert(threshold[0]==expected&&threshold[0]>=previous&&threshold[0]<=exact);
    previous=threshold[0];for(auto key:threshold)assert(key==expected);
    if(KVIP_ROLLING_FIRST_WINDOW)frozen=true;}
   if(frozen)assert(threshold[0]<=kth(truth));
   assert(threshold[0]==previous);
   for(unsigned bin=0;bin<256;++bin)assert(storage[16+bin]<=2048);
  }
  assert(histograms==expectedHistograms);
  if(KVIP_ROLLING_BATCHED||KVIP_ROLLING_WINDOWED)assert(pending==0&&windowCount==0&&histograms<ordinal*1.05);
  for(unsigned i=0;i<16;++i)assert(storage[i]==0xa55a&&storage[16+kvip_rolling::kStateWords+i]==0xa55a);
 }
}
'''
    source = (f"#define KVIP_ROLLING_BATCHED {batched}\n#define KVIP_ROLLING_WINDOWED {windowed}\n"
              f"#define KVIP_ROLLING_COMPACT {compact}\n#define KVIP_ROLLING_FIRST_WINDOW {first}\n"
              f"#define KVIP_REVERSE_COLD {reverse}\n#define REVERSE_TEST {reverse}\n") + source
    execute(tmp_path, source, "rolling_cdf")


def test_actual_candidate_filter_preserves_visibility_and_boundary_ties(tmp_path):
    prefetch = (NATIVE / "op_kernel/prefetch.h").read_text()
    decoder = function(prefetch, "__simt_callee__ inline uint32_t DecodeScanToken(")
    gather = function(prefetch, "inline void GatherCandidateMap(")
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __simt_callee__
#define __ubuf__
#define __gm__
#define KVIP_REVERSE_COLD 1
#define KVIP_WINDOW_REFILTER 1
struct {unsigned x;} threadIdx;
template<class T> T asc_ldcg(T* p){return *p;}
''' + decoder + '\n' + gather + r'''
int main(){
 constexpr unsigned h=8192;
 std::vector<uint32_t> map(h,UINT32_MAX),scores(4*h/2,0),cfg(128,0);
 cfg[60]=2048;cfg[61]=8192;
 auto put=[&](unsigned q,unsigned token,unsigned score){
  auto& word=scores[(q*h+token)/2];word|=score<<((token%2)*16);};
 std::array<uint32_t,5> ids{6144,6145,2048,6146,6147},slots{};
 for(unsigned q=0;q<4;++q){cfg[96+q]=100;cfg[100+q]=4096;cfg[104+q]=8192;}
 put(0,6144,100);put(1,6145,99);put(2,2048,65535); // future scan position, must not admit
 put(3,6146,102);map[6147]=17; // already resident, not a miss
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
  GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
 assert(slots[0]==UINT32_MAX&&slots[1]==0&&slots[2]==0&&slots[3]==UINT32_MAX&&slots[4]==17);
 assert(map[6144]==UINT32_MAX&&map[6145]==UINT32_MAX); // no cache mutation
 // No rolling state is the unchanged original map gather.
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size());
 for(unsigned i=0;i<ids.size();++i)assert(slots[i]==map[ids[i]]);
}
'''
    execute(tmp_path, source, "rolling_refilter")


def test_actual_threshold_publication_uses_writable_tensor_and_only_its_stripe(tmp_path):
    publish = function((INDEXER / "indexer_vector.h").read_text(),
                       "    __aicore__ inline void PublishRolling(")
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#define __aicore__
#define __gm__
#include "rolling_threshold.h"
template<class T> struct LocalTensor {
 T* p;unsigned n;
 LocalTensor operator[](unsigned i)const{assert(i<=n);return {p+i,n-i};}
 template<class U> LocalTensor<U> ReinterpretCast()const {
  assert(uintptr_t(p)%alignof(U)==0);return {reinterpret_cast<U*>(p),unsigned(n*sizeof(T)/sizeof(U))};}
};
template<class T> struct GlobalTensor {
 T* p;unsigned n;
 // Match the const GM address exposed by CANN, not a permissive T* stub.
 const T* GetPhyAddr()const{return p;}
 GlobalTensor operator[](unsigned i)const{assert(i<=n);return {p+i,n-i};}
};
struct DataCopyExtParams{unsigned blockCount,blockLen,srcStride,dstStride,rsv;};
template<class T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams copy){
 assert(copy.blockCount==1&&copy.blockLen==32&&!copy.srcStride&&!copy.dstStride);
 assert(copy.blockLen<=dst.n*sizeof(T)&&copy.blockLen<=src.n*sizeof(T));
 std::memcpy(dst.p,src.p,copy.blockLen);
}
struct Kernel {
 GlobalTensor<uint32_t> rotationProgressGm_;
 LocalTensor<uint16_t> prefixThreshold_;
''' + publish + r'''
};
int main(){
 std::array<uint32_t,64*32+32> gm;
 alignas(32) std::array<uint32_t,128> ub;
 std::fill_n(ub.data(),64,0x12341234U);std::fill_n(ub.data()+64,64,0x56785678U);
 Kernel kernel{{gm.data()+16,64*32},{reinterpret_cast<uint16_t*>(ub.data()),256}};
 for(unsigned query=0;query<64;++query){
  gm.fill(0xa55aa55aU);kernel.PublishRolling(query);
  unsigned begin=16+query*32+kvip_rolling::kProgressThreshold;
  for(unsigned i=0;i<gm.size();++i)
   assert(gm[i]==(i>=begin&&i<begin+8?ub[(query%2)*64]:0xa55aa55aU));
 }
}
'''
    execute(tmp_path, source, "rolling_publication")
