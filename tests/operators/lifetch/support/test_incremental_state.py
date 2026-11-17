"""Execute actual direct state materialization and boundary-tie dispatch."""

import subprocess

import pytest

from test_incremental_prefilter import prefilter_model_source
from test_recent_index import INDEXER, function


@pytest.mark.parametrize("ties", [0, 1])
def test_actual_direct_and_tie_boundary_match_full_selection(tmp_path, ties):
    source = prefilter_model_source()
    vector = (INDEXER / "incremental_state_vector.h").read_text()
    additions = r'''
enum class StoreDist {DIST_PACK_B32};
template<class T,StoreDist>void StoreAlign(T* p,RegTensor<T>& a,MaskReg mask){
 static_assert(sizeof(T)==2);
 for(unsigned i=0;i<64;++i)if(mask.v[i])p[i]=a.v[2*i];
}
template<class T>void ShiftRights(RegTensor<T>& r,RegTensor<T>& a,T s,MaskReg m){
 for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=a.v[i]>>s;
}
template<class T>void Add(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=a.v[i]+b.v[i];
}
template<class T>void Sub(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=a.v[i]-b.v[i];
}
template<class T>void Or(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg m){
 for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=a.v[i]|b.v[i];
}
template<class T>void Gather(RegTensor<T>& r,T* src,RegTensor<uint32_t>& index,MaskReg m){
 for(unsigned i=0;i<r.v.size();++i)if(m.v[i])r.v[i]=src[index.v[i]];
}
namespace kvip_incremental {
using ::kTop;
FUNCTIONS
}
namespace topkb16gather {
void FindIdxGTOutputVFImpl(uint16_t* out,uint16_t* values,uint16_t begin,uint32_t* key,uint16_t loops){
 ar=0;unsigned n=0;
 for(unsigned i=0;i<loops*128;++i)if(values[i]>uint16_t(*key))out[n++]=begin+i;
 ar=n*2;
}
void FindIdxEQOutputVFImpl(uint16_t* out,uint16_t* values,uint16_t begin,uint32_t* key,uint16_t loops){
 unsigned n=ar/2;
 for(unsigned i=0;i<loops*128;++i)if(values[i]==uint16_t(*key))out[n++]=begin+i;
}
}
'''.replace("FUNCTIONS", "\n".join(function(vector, name) for name in (
        "__simd_callee__ inline void GatherU16(",
        "template<bool tieUpdate>\n__simd_vf__ inline void Materialize(")))
    source = source.replace("struct Selector {", additions + "\nstruct Selector {")
    source = source.replace(" uint32_t* scratch;unsigned calls=0;uint16_t threshold=0;", r'''
 uint32_t* scratch;unsigned calls=0;uint16_t threshold=0;
 std::array<uint32_t,64> key{};std::array<uint16_t,18432> ordinal{};
 Tensor<uint16_t> ThresholdResult(){return {reinterpret_cast<uint16_t*>(key.data()),128};}
 Tensor<uint16_t> IncrementalOrdinals(){return {ordinal.data(),18432};}
 void Threshold(Tensor<uint16_t> input,Tensor<uint16_t> cutoff,unsigned size){
  auto sorted=std::vector<uint16_t>(input.p,input.p+size);
  std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());
  threshold=sorted[2047];std::fill_n(cutoff.p,128,threshold);++calls;
 }
''')
    # Alternate sparse/empty windows with dense ones so a full Kth update
    # is followed by the tie fast path, exercising the retained strict count.
    source = source.replace("unsigned admitted=std::min(count,extent);",
                            "unsigned admitted=std::min(step%3==1?2049U:count,extent);")
    source = source.replace(" uint16_t kth=30000;", " uint16_t kth=30000;\n unsigned incrementalStrict_=0,rotationProgressBase_=0,incrementalPending_=0; bool incrementalPacked_=false;")
    source = source.replace("   unsigned calls=s.topkOp_.calls;", """
   unsigned calls=s.topkOp_.calls;
   bool fast=KVIP_INCREMENTAL_TIES&&admitted>0&&admitted<=2048&&admitted<2048-s.incrementalStrict_;
""")
    source = source.replace("calls+(admitted?1:0)", "calls+(admitted&&!fast?1:0)")
    source = source.replace("   assert(s.incrementalCount_==0&&s.kth==kth);", """
   assert(s.incrementalCount_==0&&s.kth==kth);
   if(admitted){
    unsigned n=0;for(auto id:s.ids)if(id&kNeedsFetch)assert(s.scratch[32+n++]==(id&kIdMask));
    assert(s.incrementalPacked_&&s.incrementalPending_==n);
   }else assert(!s.incrementalPacked_);
""")
    # Execute the real publication method too: contiguous header+payload,
    # both prepacked and regenerated (empty-window) paths, then ready stripe.
    publish = function((INDEXER / "incremental_topk_methods.h").read_text(),
                       "    __aicore__ inline void PublishIncremental(")
    compact = function((INDEXER / "incremental_topk_vector.h").read_text(),
                       "__simd_vf__ inline void CompactIds(")
    source = source.replace("namespace kvip_incremental {\nusing ::kTop;",
        "namespace kvip_incremental {\nusing ::kTop; constexpr unsigned kHeader=32,kReadyWord=8,kListWords=2080;\n" + compact)
    source = source.replace("uint32_t* valid;};", "uint32_t* valid;uint64_t incrementalLists=0;};")
    source = source.replace("enum class HardEvent{MTE3_V};", "enum class HardEvent{MTE3_V,V_S,S_MTE3,V_MTE3,MTE3_S};")
    source = source.replace(" T* GetPhyAddr(){return p;}", " T* GetPhyAddr(){return p;}\n T GetValue(unsigned i){assert(i<n);return p[i];}\n void SetValue(unsigned i,T v){assert(i<n);p[i]=v;}")
    source = source.replace("namespace Simt {", r"""
template<class T>struct GlobalTensor {
 T* p=nullptr;void SetGlobalBuffer(T* x){p=x;}
 GlobalTensor operator[](unsigned i){return {p+i};}
};
template<class T>void DataCopy(GlobalTensor<T> out,Tensor<T> in,unsigned n){assert(n<=in.n);std::copy_n(in.p,n,out.p);}
template<class T>void Duplicate(Tensor<T> x,T v,unsigned n){assert(n<=x.n);std::fill_n(x.p,n,v);}
namespace Simt {
""")
    source = source.replace("struct State{", r"""struct State{
 std::array<uint32_t,4*9*2080> published{};
 std::array<uint32_t,4*32> progress{};
 std::array<uint16_t,256> cutoff{};
 rotating::Context ctx{0,0,0,nullptr,uint64_t(published.data())};
 uint64_t rotationContext_=uint64_t(&ctx);
 struct{unsigned rounds=9;}runtimeShape_;
 GlobalTensor<uint32_t> rotationProgressGm_{progress.data()};
 Tensor<uint16_t> prefixThreshold_{cutoff.data(),256};
""")
    source = source.replace(" void SaveIncremental(unsigned){", publish + "\n void SaveIncremental(unsigned){")
    source = source.replace("   for(unsigned i=0;i<2048;++i){\n    unsigned pos=selected[i],expected;", """
   s.cutoff.fill(kth);
   s.PublishIncremental(0,step+1);
   unsigned pendingCount=0;
   for(auto id:s.ids)if(id&kNeedsFetch)assert(s.published[(step+1)*2080+32+pendingCount++]==(id&kIdMask));
   auto header=s.published.data()+(step+1)*2080;
   assert(header[0]==pendingCount&&header[1]==kth&&header[2]==step+2);
   assert(s.progress[8]==step+2&&!s.incrementalPacked_);
   for(unsigned i=0;i<2048;++i){
    unsigned pos=selected[i],expected;
""")
    cpp = tmp_path / "state.cpp"
    cpp.write_text(source)
    exe = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-DKVIP_INCREMENTAL_WINDOW=1", "-DKVIP_INCREMENTAL_PREFILTER=1",
                    "-DKVIP_INCREMENTAL_DIRECT=1", f"-DKVIP_INCREMENTAL_TIES={ties}",
                    str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
