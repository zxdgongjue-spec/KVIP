"""Execute early-score UB placement, exact masking and native Kth consumption."""

import subprocess

from test_recent_index import COMMON, INDEXER, function
from test_recent_threshold_only import threshold_model_source


def test_seed_ub_preserves_gm_and_exact_threshold_with_delayed_dma(tmp_path):
    source = (INDEXER / "indexer_vector.h").read_text()
    begin = function(source, "    __aicore__ inline void BeginResidentSeed(")
    process = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessVec1(")
    placement = function(process, "        if (residentSeedActive_)")
    code = threshold_model_source().split("int main(){", 1)[0]
    code = code.replace("#include <functional>", "#include <functional>\n#include <cstring>")
    code = code.replace("#define __simt_vf__", "#define __simt_vf__\n#define __simd_vf__")
    code = code.replace(
        '#include "recent_index.h"',
        r"""
std::vector<std::function<void()>> pendingCopies;
bool mte3JoinedMte2=false;
namespace CMPMODE {constexpr int GE=0,LT=1;}
namespace AscendC::MicroAPI {
enum class MaskPattern {ALL};struct MaskReg {std::array<bool,128> v;};
template<class T> struct RegTensor {std::array<T,128> v;};
template<class T,MaskPattern> MaskReg CreateMask(){MaskReg m;m.v.fill(true);return m;}
template<class T> void Duplicate(RegTensor<T>& r,T v){r.v.fill(v);}
template<class T> void Arange(RegTensor<T>& r,T start){for(unsigned i=0;i<128;++i)r.v[i]=start+i;}
template<class T,int Mode> void CompareScalar(MaskReg& dst,RegTensor<T>& r,T value,MaskReg mask){
 for(unsigned i=0;i<128;++i)dst.v[i]=mask.v[i] && (Mode==CMPMODE::GE?r.v[i]>=value:r.v[i]<value);
}
void MaskAnd(MaskReg& dst,MaskReg a,MaskReg b,MaskReg mask){for(unsigned i=0;i<128;++i)dst.v[i]=a.v[i]&&b.v[i]&&mask.v[i];}
template<class T> void LoadAlign(RegTensor<T>& r,T* p){std::copy_n(p,128,r.v.begin());}
template<class T> void Select(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<128;++i)r.v[i]=mask.v[i]?a.v[i]:b.v[i];
}
template<class T> void StoreAlign(T* p,RegTensor<T>& r,MaskReg mask){
 assert(pendingCopies.empty()); // A cached-score mask must follow all MTE3 reads.
 for(unsigned i=0;i<128;++i)if(mask.v[i])p[i]=r.v[i];
}
}
#include "recent_index.h"
""",
    )
    code = code.replace(
        "template<HardEvent D> void WaitFlag(unsigned id){assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;}",
        """template<HardEvent D> void WaitFlag(unsigned id){
 assert(credits[unsigned(D)][id]);credits[unsigned(D)][id]=false;
 if(D==HardEvent::MTE2_V && id==5 && !pendingCopies.empty()){
  assert(mte3JoinedMte2);for(auto& copy:pendingCopies)copy();pendingCopies.clear();
 }
}""",
    )
    code = code.replace(
        "template<class T> T Min(T a,T b){return std::min(a,b);}",
        "template<class T> T Min(T a,T b){return std::min(a,b);}\n"
        "template<class T> T Max(T a,T b){return std::max(a,b);}\n"
        "enum class LI_LAYOUT {BSND,TND};\n"
        "namespace QLICommon {struct RunInfo {unsigned bIdx,gS1Idx,actS1Size,actS2Size,s2Idx;};}",
    )
    code = code.replace(
        "struct Model {\n using SCORE_T=uint16_t;",
        """struct Model {
 bool residentSeedActive_=false;uint32_t residentSeedQuery_=UINT32_MAX;
 kvip_recent_index::SeedLayout residentSeed_{};
 unsigned s1BaseSize_=2,blockId_=0;
 static constexpr LI_LAYOUT Q_LAYOUT_T=LI_LAYOUT::BSND;
 struct {bool attenMaskFlag=false,isLDOpen=false;} constInfo_;
 using SCORE_T=uint16_t;""",
    )
    pos = code.rfind("};")
    code = (
        code[:pos]
        + begin
        + r"""
 void ScoreTile(unsigned curS2Idx,bool scoreOnly,unsigned scoreRow,const std::vector<uint16_t>& reference){
  assert(residentSeedActive_);LocalTensor<SCORE_T> outBase{};
"""
        + placement
        + r"""
  // Stand-in only for the unchanged QK/vector arithmetic. The production
  // outBase selection runs above; the original score DMA remains pending.
  auto expected=reference.data()+scoreRow*runtimeShape_.history+curS2Idx;
  std::copy_n(expected,128,outBase.p);
  auto dst=scoreGm.p+(blockId_%2)*runtimeShape_.history+curS2Idx;
  std::fill_n(dst,128,uint16_t(0));
  pendingCopies.push_back([outBase,dst](){std::copy_n(outBase.p,128,dst);});
 }
"""
        + code[pos:]
    )
    code += r"""
int main(){
 const unsigned h=131328,r=16384;
 std::mt19937 rng(731);
 for(unsigned length:{0U,1U,127U,128U,129U,512U,1921U,2048U,2049U,4096U})
 for(unsigned shift:{0U,1U,127U})for(unsigned q=0;q<4;++q){
  Model m;for(auto& v:m.scores)v=uint16_t(rng()%65536);
  auto original=m.scores;
  int32_t bounds[6]={0,0,0,0,int32_t(20096+shift),int32_t(20096+shift+length)};
  uint32_t ends[12];std::fill_n(ends,12,h);ends[9]=20096+shift+3;ends[10]=20096+shift;ends[11]=0;
  m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
  m.blockId_=q%2;m.scoreGm={m.scores.data()+(q/2)*2*h,2*h};
  QLICommon::RunInfo info{2,q/2,4,h,0};m.BeginResidentSeed(info);
  auto expected=kvip_recent_index::MakeSeedLayout(bounds[4],bounds[5],r,h,h,ends[8+q],h,18432);
  assert(m.residentSeedActive_==expected.enabled);
  if(!expected.enabled){assert(!m.TryUnifiedResidentTopK(8+q,q%2,std::min(r,ends[8+q]),h,true) || ends[8+q]>=r);continue;}
  for(unsigned tile=expected.tiles.begin;tile<expected.tiles.end;++tile)m.ScoreTile(tile*128,true,q,original);
  for(unsigned tile=0;tile<r/128;++tile)m.ScoreTile(tile*128,false,q,original);
  assert(!pendingCopies.empty());mte3JoinedMte2=true;dma=0;
  bool selected=m.TryUnifiedResidentTopK(8+q,q%2,r,h,true);
  assert(selected && !m.residentSeedActive_ && pendingCopies.empty() && dma==0);
  assert(m.scores==original); // Halo masking must never leak into scoreGM.
  auto exact=kvip_recent_index::Clip(bounds[4],bounds[5],r,h,ends[8+q]);
  std::vector<uint16_t> ref(original.begin()+q*h,original.begin()+q*h+r);
  ref.insert(ref.end(),original.begin()+q*h+exact.begin,original.begin()+q*h+exact.end);
  std::nth_element(ref.begin(),ref.begin()+2047,ref.end(),std::greater<uint16_t>());
  assert(m.prefixThreshold_.p[(q%2)*128]==ref[2047]);
  // Fetch may immediately overwrite the consumed cache. A second selection
  // must use scoreGM, never a stale bank.
  std::fill(m.merge.begin(),m.merge.end(),0xffff);dma=0;
  assert(m.TryUnifiedResidentTopK(8+q,q%2,r,h,true));
  assert(dma>=1 && m.prefixThreshold_.p[(q%2)*128]==ref[2047]);
 }
 // An empty recent range still needs alignment padding when R is only
 // 128-aligned. Clear exactly that suffix, preserving the adjacent guard.
 auto padded=kvip_recent_index::MakeSeedLayout(0,0,2176,h,h,h,h,18432);
 assert(padded.enabled && padded.words==2304 && padded.keep.end==0);
 std::array<uint16_t,256> padding;padding.fill(0xdddd);
 kvip_recent_index::MaskSeedPadding(padding.data(),padded.words-2176,0,0);
 for(unsigned i=0;i<256;++i)assert(padding[i]==(i<128?0:0xdddd));
 // Eligibility fallbacks cannot accidentally keep a previous query's cache.
 Model m;uint32_t ends[4]={h,h,h,h};int32_t bounds[2]={20001,20205};
 m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);
 for(unsigned kind=0;kind<7;++kind){
  m.residentSeedActive_=true;m.s1BaseSize_=kind==0?4:2;
  m.constInfo_.attenMaskFlag=kind==1;m.constInfo_.isLDOpen=kind==2;
  QLICommon::RunInfo info{0,0,kind==3?3U:4U,kind==4?h-128:h,kind==5?1U:0U};
  ends[0]=kind==6?r-1:h;m.BeginResidentSeed(info);assert(!m.residentSeedActive_);
 }
}
"""
    cpp = tmp_path / "seed.cpp"
    cpp.write_text(code)
    exe = tmp_path / "seed"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-DKVIP_RECENT_THRESHOLD_ONLY=1",
            "-DKVIP_RECENT_UNIFIED_TOPK=1",
            "-DKVIP_RECENT_SEED_UB=1",
            "-I",
            str(COMMON),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_release_preprocessing_removes_all_marker_calls(tmp_path):
    source = (INDEXER / "indexer_vector.h").read_text()
    marker = function(source, "__aicore__ inline void QLIVector<QLIT>::MarkTopK(")
    process = function(source, "__aicore__ inline void QLIVector<QLIT>::ProcessTopK(")
    cpp = tmp_path / "markers.cpp"
    cpp.write_text("#if KVIP_RECENT_PROFILE\n" + marker + "\n#endif\n" + process)
    for enabled in (0, 1):
        output = subprocess.check_output(
            [
                "c++",
                "-E",
                "-P",
                "-DKVIP_STREAMING_LI=1",
                "-DKVIP_RECENT_THRESHOLD_ONLY=1",
                f"-DKVIP_RECENT_PROFILE={enabled}",
                str(cpp),
            ],
            text=True,
        )
        assert ("MarkTopK(" in output) is bool(enabled)
        if enabled:
            assert output.count("MarkTopK(") == 5  # Definition + four actual call sites.
