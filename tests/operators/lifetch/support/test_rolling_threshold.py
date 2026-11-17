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


@pytest.mark.parametrize("batched,windowed,compact,first,reuse,fast,sparse,boundary", [
    (0, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0),
    (0, 1, 1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 0, 0, 0, 0), (0, 1, 1, 0, 1, 0, 0, 0),
    (0, 1, 1, 1, 1, 0, 0, 0), (0, 1, 1, 0, 1, 1, 0, 0), (0, 1, 1, 1, 1, 1, 0, 0),
    (0, 1, 1, 0, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 0, 1, 1, 0, 1)])
@pytest.mark.parametrize("reverse", [0, 1])
def test_actual_cdf_saturation_ties_causal_tail_and_seed_dedup(tmp_path, batched, windowed, compact, first, reuse, fast, sparse, boundary, reverse):
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
// CANN exposes both the scalar GatherMask mode and the register GatherMask
// mode. Production headers also import AscendC; keep that lookup environment
// so an unqualified enum cannot pass host tests and fail the NPU compiler.
namespace AscendC {
enum class SpecialPurposeReg {AR};
enum class GatherMaskMode : uint8_t {VERSION_V1=0,VERSION_V2=1};
}
namespace AscendC::MicroAPI {
enum class GatherMaskMode {STORE_REG};enum class PostLiteral {POST_MODE_UPDATE};
struct UnalignRegForStore {};
unsigned arBytes=0,squeezeCount=0;
template<AscendC::SpecialPurposeReg> void ClearSpr(){arBytes=0;}
template<class T,GatherMaskMode> void Squeeze(RegTensor<T>& r,RegTensor<T>& a,MaskReg mask){
 squeezeCount=0;for(unsigned i=0;i<a.v.size();++i)if(mask.v[i])r.v[squeezeCount++]=a.v[i];}
template<class T,PostLiteral> void StoreUnAlign(T*& p,RegTensor<T>& r,UnalignRegForStore&){
 std::copy_n(r.v.begin(),squeezeCount,p);p+=squeezeCount;arBytes+=squeezeCount*sizeof(T);}
template<class T> void StoreUnAlignPost(T*&,UnalignRegForStore&){}
template<class T> void ShiftRights(RegTensor<T>& r,RegTensor<T>& a,int16_t n,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=a.v[i]>>n;}
template<class T> void Xor(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=a.v[i]^b.v[i];}
// Seed export must consume its live CDF registers, not temporarily overwrite
// and gather the rolling-state buffer. No UB-source Gather overload is given.
template<class T> void Gather(RegTensor<T>& r,RegTensor<T> source,RegTensor<uint16_t>& index){
 for(unsigned i=0;i<r.v.size();++i){assert(index.v[i]<source.v.size());r.v[i]=source.v[index.v[i]];}}
template<class T> void And(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=a.v[i]&b.v[i];}
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
using namespace AscendC;
#include "rolling_threshold_vector.h"
''' + classify + r'''
using namespace AscendC::MicroAPI;
#include "cold_scan.h"
void CheckClassify(RegTensor<uint16_t>& score,uint16_t* threshold,uint16_t* hits,
                   MaskReg& all,uint16_t* state){
#if KVIP_ROLLING_BOUNDARY
 std::array<uint16_t,128> cutoff;
 std::copy_n(state+kvip_rolling::kAdmission,128,cutoff.data());
 ClassifyRollingScore(score,cutoff.data(),hits,all,nullptr);
#else
 ClassifyRollingScore(score,threshold,hits,all,state);
#endif
}
uint16_t kth(std::vector<uint16_t> a){std::nth_element(a.begin(),a.end()-2048,a.end());return a[a.size()-2048];}
int main(){
#if KVIP_ROLLING_FAST_PATH
 // Exhaust every input key, including both byte-range saturation edges.
 for(unsigned byte:{0U,1U,127U,255U}){
  RegTensor<uint16_t> seedByte,ones,zero;
  Duplicate(seedByte,uint16_t(byte*257));Duplicate(ones,uint16_t(65535));Duplicate(zero,uint16_t(0));
  std::array<uint16_t,256> input;
  for(unsigned start=0;start<65536;start+=256){
   for(unsigned i=0;i<256;++i)input[i]=start+i;
   RegTensor<uint8_t> encoded;
   kvip_rolling::EncodeFullPair(encoded,input.data(),seedByte,ones,zero);
   for(unsigned i=0;i<256;++i){
    int delta=std::clamp(int(input[i])-int(byte*256),0,255);
    assert(encoded.v[i]==255-delta);
   }
  }
 }
#endif
 {
  RegTensor<uint16_t> score;for(unsigned i=0;i<128;++i)score.v[i]=i;
  auto all=CreateMask<uint16_t,MaskPattern::ALL>();
  std::array<uint16_t,128> threshold{},hits{};threshold.fill(90);
  std::array<uint16_t,kvip_rolling::kStateWords> state{};
  std::fill_n(state.data()+kvip_rolling::kBins,128,80);
  RegTensor<uint16_t> limit,anchor;LoadAlign(limit,threshold.data());Duplicate(anchor,uint16_t(80));
  kvip_rolling::SaveAdmission(state.data(),limit,anchor);
  CheckClassify(score,threshold.data(),hits.data(),all,state.data());
  for(unsigned i=0;i<128;++i)assert(hits[i]==unsigned(i>=90));
  threshold.fill(80);Duplicate(limit,uint16_t(80));
  kvip_rolling::SaveAdmission(state.data(),limit,anchor);
  CheckClassify(score,threshold.data(),hits.data(),all,state.data());
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
    CheckClassify(score,threshold.data(),hits.data(),all,state.data());
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
#if KVIP_ROLLING_REUSE_SEED
  unsigned seedHistograms=histograms,seedStores=stores;
  kvip_recent_threshold::RegisterThreshold(seed.data(),threshold.data(),seed.size()/256,
      seed.size()-2048+1,storage.data()+16);
  assert(threshold[0]==initial);
  assert(histograms-seedHistograms==4*seed.size()/256); // No third score pass.
  assert(stores-seedStores==5); // Threshold + final CDF/anchor/admission only.
  for(unsigned j=0;j<256;++j){
   unsigned key=(unsigned(initial)&0xff00)+255-j;
   unsigned count=std::count_if(seed.begin(),seed.end(),[&](auto value){return value>=key;});
   assert(storage[16+j]==std::min(count,2048U));
  }
#else
  kvip_rolling::Seed(seed.data(),truth.size(),threshold.data(),storage.data()+16);
#endif
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
    if(advance){if(!KVIP_ROLLING_SPARSE)expectedHistograms+=2*((batchCount+1)/2);batchCount=0;}
   }else if(!KVIP_ROLLING_WINDOWED && KVIP_ROLLING_BATCHED){
    mask=kvip_rolling::BatchMask(mask,pending,advance);
    if(batchCount==8||advance){expectedHistograms+=2*((batchCount+1)/2);batchCount=0;}
   }else if(!KVIP_ROLLING_WINDOWED){expectedHistograms+=2;}
#if KVIP_ROLLING_WINDOWED
   if(advance && !frozen){
#if KVIP_ROLLING_SPARSE
    const auto originalWindow=window;
    std::array<uint16_t,kvip_rolling::kWindowScores+32> compacted;compacted.fill(0xa55a);
    kvip_rolling::CompactWindow(window.data(),windowCount,compacted.data()+16,storage.data()+16,
      n+ordinal*128-windowCount,end,recentBegin,recentEnd,n,end/2048*2048);
    unsigned sparseCount=arBytes/2;
    std::vector<uint16_t> expectedSparse;
    for(unsigned offset=0;offset<windowCount;offset+=128){
     unsigned physical=n+ordinal*128-windowCount+offset;
     if(REVERSE_TEST && physical>=n && physical<end/2048*2048)
      physical=n+end/2048*2048-2048-(physical&~2047U)+(physical&2047U);
     for(unsigned lane=0;lane<128;++lane)
      if(physical+lane<end && (physical+lane<recentBegin||physical+lane>=recentEnd) && window[offset+lane]>initial)
       expectedSparse.push_back(window[offset+lane]);
    }
    assert(sparseCount==expectedSparse.size());
    assert(std::equal(expectedSparse.begin(),expectedSparse.end(),compacted.begin()+16));
    assert(window==originalWindow);
    for(unsigned i=0;i<16;++i)assert(compacted[i]==0xa55a&&compacted[16+kvip_rolling::kWindowScores+i]==0xa55a);
    expectedHistograms+=2*((sparseCount+255)/256);
#endif
#if KVIP_ROLLING_BOUNDARY
    expectedHistograms-=2*((windowCount/128+1)/2);
    unsigned scanBegin=n+ordinal*128-windowCount;
    auto reverse=REVERSE_TEST?kvip_cold_scan::Range{n,end/2048*2048}:kvip_cold_scan::Range{0,0};
    auto runs=kvip_cold_scan::WindowRuns(scanBegin,scanBegin+windowCount,reverse);
    for(unsigned r=0;r<runs.count;++r){
     auto span=runs.spans[r];unsigned count=span.end-span.begin;
     expectedHistograms+=2*((count/128+1)/2);
     kvip_rolling::ObserveWindow(values.data()+span.begin,count,storage.data()+16,threshold.data(),windowHits.data(),
       span.begin,end,recentBegin,recentEnd,0,0);
    }
    auto classifierThreshold=threshold;
    kvip_rolling::FinishBoundary(values.data()+tile,storage.data()+16,classifierThreshold.data(),windowHits.data());
    for(unsigned lane=0;lane<128;++lane){
     assert(storage[16+kvip_rolling::kAdmission+lane]==threshold[lane]);
     assert(classifierThreshold[lane]==std::max(unsigned(initial),std::max(unsigned(threshold[lane]),1U)-1));
    }
#else
    kvip_rolling::ObserveWindow(window.data(),windowCount,storage.data()+16,threshold.data(),windowHits.data(),
      n+ordinal*128-windowCount,end,recentBegin,recentEnd,n,end/2048*2048
#if KVIP_ROLLING_SPARSE
      ,compacted.data()+16,sparseCount
#endif
      );
#endif
    windowCount=0;
    for(unsigned i=0;i<128;++i)assert(windowHits[i]==unsigned(score.v[i]>=threshold[0]&&score.v[i]>initial));
   }
#else
   kvip_rolling::Observe(score,storage.data()+16,threshold.data(),mask,advance);
#endif
   if(advance && !frozen){unsigned exact=kth(truth);
    unsigned cap=KVIP_ROLLING_REUSE_SEED?(unsigned(initial)|255U):unsigned(initial)+255;
    unsigned expected=std::min(exact,cap);
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
              f"#define KVIP_ROLLING_FAST_PATH {fast}\n#define KVIP_ROLLING_SPARSE {sparse}\n#define KVIP_ROLLING_BOUNDARY {boundary}\n"
              f"#define KVIP_ROLLING_REUSE_SEED {reuse}\n#define KVIP_ROLLING_COMPACT {compact}\n#define KVIP_ROLLING_FIRST_WINDOW {first}\n"
              f"#define KVIP_REVERSE_COLD {reverse}\n#define REVERSE_TEST {reverse}\n") + source
    execute(tmp_path, source, "rolling_cdf")


@pytest.mark.parametrize("safe,fast", [(0, 0), (1, 0), (1, 1)])
def test_actual_candidate_filter_preserves_visibility_and_boundary_ties(tmp_path, safe, fast):
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
uintptr_t scoreBegin=0,scoreEnd=0;unsigned scoreLoads=0;
template<class T> T asc_ldcg(T* p){
 auto address=reinterpret_cast<uintptr_t>(p);
 if(address>=scoreBegin&&address<scoreEnd)++scoreLoads;
 return *p;}
''' + decoder + '\n' + gather + r'''
int main(){
 constexpr unsigned h=8192;
 std::vector<uint32_t> map(h,UINT32_MAX),scores(4*h/2,0),cfg(128,0);
 scoreBegin=reinterpret_cast<uintptr_t>(scores.data());scoreEnd=scoreBegin+scores.size()*4;
 cfg[60]=2048;cfg[61]=8192;
 auto put=[&](unsigned q,unsigned token,unsigned score){
  auto& word=scores[(q*h+token)/2];word|=score<<((token%2)*16);};
 std::array<uint32_t,5> ids{6144,6145,2048,6146,6147},slots{};
 for(unsigned q=0;q<4;++q){cfg[96+q]=100;cfg[100+q]=4096;cfg[104+q]=8192;}
 cfg[108]=4096;
 put(0,6144,100);put(1,6145,99);put(2,2048,65535); // future scan position, must not admit
 put(3,6146,102);map[6147]=17; // already resident, not a miss
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
  GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
 assert(slots[0]==UINT32_MAX&&slots[1]==0&&slots[2]==(KVIP_REFILTER_SAFE?UINT32_MAX:0)&&slots[3]==UINT32_MAX&&slots[4]==17);
 // A candidate from Q0 can be useful to Q3 even if Q0's threshold rose.
 // Keep it when Q3 is not ready; do not wait or read its unpublished score.
 cfg[103]=2048;put(3,6145,150);
 cfg[108]=2048;
 scoreLoads=0;
 threadIdx.x=1;GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
 assert(scoreLoads==(KVIP_ROLLING_FAST_PATH?0:3));
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
  GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
 assert(slots[1]==(KVIP_REFILTER_SAFE?UINT32_MAX:0));
 // A causal-invalid query is different from an unknown valid query.
 cfg[107]=6145;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
  GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
 assert(slots[1]==0);

 assert(map[6144]==UINT32_MAX&&map[6145]==UINT32_MAX); // no cache mutation
#if KVIP_REFILTER_SAFE
 // Every final-selected token is admissible under any Q4 publication mask:
 // known score >= final Kth >= rolling Kth, or unknown and kept conservatively.
 std::fill(scores.begin(),scores.end(),0);
 for(unsigned winner=0;winner<4;++winner)for(unsigned mask=0;mask<16;++mask){
  std::fill(scores.begin(),scores.end(),0);put(winner,6145,110);
  for(unsigned q=0;q<4;++q){cfg[96+q]=100;cfg[100+q]=(mask&(1U<<q))?4096:2048;cfg[104+q]=8192;}
  cfg[108]=*std::min_element(cfg.begin()+100,cfg.begin()+104);
  threadIdx.x=1;GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),scores.data(),cfg.data(),h);
  assert(slots[1]==UINT32_MAX);
 }
#endif
 // No rolling state is the unchanged original map gather.
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)GatherCandidateMap(ids.data(),slots.data(),map.data(),ids.size(),nullptr,cfg.data(),h);
 for(unsigned i=0;i<ids.size();++i)assert(slots[i]==map[ids[i]]);
}
'''
    execute(tmp_path, f"#define KVIP_REFILTER_SAFE {safe}\n#define KVIP_ROLLING_FAST_PATH {fast}\n" + source, "rolling_refilter")


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


@pytest.mark.parametrize("refill,out_of_place", [(0,0), (0,1), (1,0), (1,1)])
def test_actual_refill_loop_fills_only_frozen_window_without_budget_growth(tmp_path, refill, out_of_place):
    """Run the native admission loop and VF; mock only bitmap DMA/enumeration."""
    text = (NATIVE / "op_kernel/prefetch.h").read_text()
    decoder = function(text, "inline uint32_t DecodeScanToken(")
    gather = function(text, "inline void GatherCandidateMap(")
    compact = function(text, "inline void CompactMisses(")
    first = text.index("        auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;", text.index("inline void FastService"))
    last = text.index("            if(count) {", first)
    body = text[first:last].replace('#include "../../common/lifetch/sequence_candidates.h"', r'''
        // Exact candidate-capacity boundary; ready limits never advance.
        uint32_t first=state.GetValue(12),last=cfg.GetValue(128),stop=last;
        uint32_t candidateCount=std::min(2048U,last-first);
        uint32_t consumed[2][4]{},end[2][4]{};
        for(unsigned q=0;q<4;++q){consumed[0][q]=state.GetValue(12+q);end[0][q]=last;}
        for(unsigned i=0;i<candidateCount;++i)ids.SetValue(i,first+i);
        stop=first+candidateCount; ++chunks;
        assert(chunks<=4);
''')
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __ubuf__
#define __gm__
#define __simt_callee__
#define KVIP_WINDOW_REFILTER 1
#define KVIP_REFILTER_SAFE 1
#define KVIP_REVERSE_COLD 0
#define V9_FETCH_PROFILE(...)
struct {unsigned x;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
namespace MicroAPI {
enum class MaskPattern{ALL,VL1};enum class CMPMODE{EQ,LT};enum class MaskMergeMode{ZEROING};
enum class GatherMaskMode{STORE_REG};enum class PostLiteral{POST_MODE_UPDATE};enum class SpecialPurposeReg{AR};
struct MaskReg{std::array<bool,64> v{};};
template<class T>struct RegTensor{std::array<T,64> v{};unsigned n=64;};
struct UnalignRegForStore{};
template<class T,MaskPattern P>MaskReg CreateMask(){MaskReg m;std::fill_n(m.v.begin(),P==MaskPattern::ALL?64:1,true);return m;}
template<class T>MaskReg UpdateMask(unsigned& n){MaskReg m;unsigned take=std::min(n,64U);std::fill_n(m.v.begin(),take,true);n-=take;return m;}
template<class T>void Duplicate(RegTensor<T>& r,T n){r.v.fill(n);}
template<class T,MaskMergeMode>void Duplicate(RegTensor<T>& r,T n,MaskReg m){for(unsigned i=0;i<64;++i)r.v[i]=m.v[i]?n:0;}
template<class T>void LoadAlign(RegTensor<T>& r,T* p){std::copy_n(p,64,r.v.begin());}
template<class T,CMPMODE C>void CompareScalar(MaskReg& r,RegTensor<T> a,T v,MaskReg m){for(unsigned i=0;i<64;++i)r.v[i]=m.v[i]&&(C==CMPMODE::EQ?a.v[i]==v:a.v[i]<v);}
void Unsqueeze(RegTensor<uint32_t>& r,MaskReg m){unsigned n=0;for(unsigned i=0;i<64;++i){r.v[i]=n;n+=m.v[i];}}
void Add(RegTensor<uint32_t>& r,RegTensor<uint32_t> a,RegTensor<uint32_t> b,MaskReg m){for(unsigned i=0;i<64;++i)if(m.v[i])r.v[i]=a.v[i]+b.v[i];}
void ReduceSum(RegTensor<uint32_t>& r,RegTensor<uint32_t> a,MaskReg m){unsigned n=0;for(unsigned i=0;i<64;++i)if(m.v[i])n+=a.v[i];r.v[0]=n;}
void Gather(RegTensor<uint32_t>& r,RegTensor<uint32_t> a,RegTensor<uint32_t> idx){for(unsigned i=0;i<64;++i)r.v[i]=a.v[idx.v[i]];}
void MaskAnd(MaskReg& r,MaskReg a,MaskReg b,MaskReg m){for(unsigned i=0;i<64;++i)r.v[i]=m.v[i]&&a.v[i]&&b.v[i];}
template<class T,GatherMaskMode>void Squeeze(RegTensor<T>& r,RegTensor<T> a,MaskReg m){r.n=0;for(unsigned i=0;i<64;++i)if(m.v[i])r.v[r.n++]=a.v[i];}
template<SpecialPurposeReg>void ClearSpr(){}
template<class T,PostLiteral>void StoreUnAlign(T*& p,RegTensor<T> a,UnalignRegForStore&){std::copy_n(a.v.begin(),a.n,p);p+=a.n;}
template<class T>void StoreUnAlignPost(T*,UnalignRegForStore&){}
template<class T>void StoreAlign(T* p,RegTensor<T> a,MaskReg m){for(unsigned i=0;i<64;++i)if(m.v[i])p[i]=a.v[i];}
}
using namespace MicroAPI;
enum class HardEvent{V_S,S_V};constexpr unsigned EVENT_ID0=0;
template<HardEvent>void SetFlag(unsigned){} template<HardEvent>void WaitFlag(unsigned){}
namespace Simt {
struct Dim3{unsigned x,y,z;};
template<auto F,class... A>void VF_CALL(Dim3 d,A... a){for(threadIdx.x=0;threadIdx.x<d.x;++threadIdx.x)F(a...);}
}
template<class T>struct Tensor {
 T* p;unsigned size;
 T* GetPhyAddr(){return p;}
 T GetValue(unsigned i){assert(i<size);return p[i];}
 void SetValue(unsigned i,T v){assert(i<size);p[i]=v;}
 Tensor operator[](unsigned i){assert(i<size);return {p+i,size-i};}
};
uint32_t PrefetchAllowance(unsigned slots,unsigned requests,unsigned row,unsigned total,unsigned transport,unsigned window){return std::min({row-slots,total-requests,transport-requests,window});}
''' + decoder + '\n' + gather + '\n' + compact + r'''
void run(unsigned limit,bool hits){
 const unsigned b=0,rt_history=8192,epoch=3;unsigned chunks=0;
 const unsigned slots=0,requests=0,sequenceHalfWindow=0;
 std::vector<uint32_t> storage(9216,0xa5a5a5a5),idData(2048,0),stateData(256,0),mapData(rt_history,UINT32_MAX),scores(4*rt_history/2,0);
 Tensor<uint32_t> work{storage.data(),unsigned(storage.size())},cfg=work[4096];
 Tensor<uint32_t> ids{idData.data(),2048},state{stateData.data(),256};
 struct Context{uint64_t mapping,tailBounds;unsigned windowLimit;} context{uint64_t(mapData.data()),1,limit};
 auto ctx=&context;auto rollingScores=reinterpret_cast<uint8_t*>(scores.data());
 for(unsigned q=0;q<4;++q){cfg.SetValue(96+q,100);cfg.SetValue(100+q,5000);cfg.SetValue(104+q,5000);cfg.SetValue(128+q,5000);}
 cfg.SetValue(60,0);cfg.SetValue(61,0);
 if(hits)for(unsigned i=4090;i<4210;++i)scores[i/2]|=100U<<((i%2)*16);
 for(unsigned i=208;i<=210;++i)state.SetValue(i,512);
''' + body + r'''
 assert(count==(KVIP_REFILTER_REFILL&&hits?std::min(120U,limit):0));
 assert(missing==(KVIP_REFILTER_REFILL&&limit&&hits?(limit<=6?6U:120U):0));
 for(unsigned i=0;i<count;++i)assert(selectedIds.GetValue(i)==4090+i);
 assert(chunks==(KVIP_REFILTER_REFILL&&limit?(hits&&limit<=6?2:3):1));
 for(unsigned q=0;q<4;++q){assert(cfg.GetValue(128+q)==5000);assert(state.GetValue(12+q)<=5000);}
 // The appended list stays within its 480-word band, even for partial VF stores.
 for(unsigned i=4096+1536+count;i<storage.size();++i)assert(storage[i]==0xa5a5a5a5);
 for(auto value:mapData)assert(value==UINT32_MAX);
}
void compactEquivalence(){
 for(unsigned count:{0U,1U,63U,64U,65U,127U,128U,2047U,2048U})
 for(unsigned limit:{0U,1U,63U,64U,480U})for(unsigned pattern=0;pattern<4;++pattern){
  std::array<uint32_t,2048> ids,slots,reference;
  std::vector<uint32_t> expected;
  for(unsigned i=0;i<2048;++i){
   ids[i]=2047-i;slots[i]=(pattern==0||(pattern==2&&i%2)||(pattern==3&&(i*17%43)<7))?UINT32_MAX:3;
   if(i<count&&slots[i]==UINT32_MAX)expected.push_back(ids[i]);
  }
  reference=ids;
  std::array<uint32_t,128> cfg{};std::array<uint32_t,512> dst;dst.fill(0xabcddcbaU);
  CompactMisses(ids.data(),slots.data(),cfg.data(),count,limit,dst.data()+16);
  assert(cfg[56]==expected.size()&&ids==reference);
  unsigned n=std::min(unsigned(expected.size()),limit);
  for(unsigned i=0;i<n;++i)assert(dst[16+i]==expected[i]);
  for(unsigned i=0;i<16;++i)assert(dst[i]==0xabcddcbaU);
  for(unsigned i=16+n;i<512;++i)assert(dst[i]==0xabcddcbaU);
  CompactMisses(ids.data(),slots.data(),cfg.data(),count,limit,ids.data());
  assert(cfg[56]==expected.size());for(unsigned i=0;i<n;++i)assert(ids[i]==dst[16+i]);
 }
}
int main(){compactEquivalence();for(bool hits:{false,true})for(unsigned limit:{0U,1U,7U,64U,119U,120U,480U})run(limit,hits);}
'''
    execute(tmp_path, f"#define KVIP_REFILTER_REFILL {refill}\n#define KVIP_COMPACT_OUT_OF_PLACE {out_of_place}\n" + source, "refill_loop")
