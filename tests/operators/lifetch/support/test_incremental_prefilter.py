"""Execute the actual boundary method with poisoned shared scratch on CPU."""

import subprocess
from pathlib import Path

from test_recent_index import INDEXER, function


def prefilter_model_source():
    vector = (INDEXER / "incremental_topk_vector.h").read_text()
    methods = "\n".join(function(vector, name) for name in (
        "__simd_callee__ inline void StoreWindow(",
        "template<bool positions>\n__simd_vf__ inline void PackWindow(",
        "__simd_vf__ inline void ZeroTail(",
        "inline void Remap(",
    ))
    # Reuse the SIMD lane/instruction model, not a reimplementation of Merge.
    fixture = Path(__file__).with_name("test_incremental_topk.py").read_text()
    source = fixture.split("    model = r'''", 1)[1].split("METHODS", 1)[0]
    source += "\nnamespace kvip_incremental { constexpr unsigned kChunk=2048;\n" + methods + "\n}\n"
    source += r'''
#define __aicore__
constexpr unsigned PIPE_V=0,EVENT_ID0=0;
namespace QLICommon {unsigned Align(unsigned n,unsigned a){return (n+a-1)/a*a;}}
enum class HardEvent{MTE3_V};
template<HardEvent>void SetFlag(unsigned){}
template<HardEvent>void WaitFlag(unsigned){}
template<unsigned>void PipeBarrier(){}
template<SpecialPurposeReg>unsigned GetSpr(){return ar;}
template<class T>struct Tensor{
 T* p;unsigned n;
 T* GetPhyAddr(){return p;}
 Tensor operator[](unsigned i){assert(i<=n);return {p+i,n-i};}
 template<class U>Tensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p),unsigned(n*sizeof(T)/sizeof(U))};}
};
template<class T>void DataCopy(Tensor<T> out,Tensor<T> in,unsigned n){
 assert(n<=out.n&&n<=in.n);std::copy_n(in.p,n,out.p);
}
namespace Simt {
struct Dim3{unsigned x,y,z;};
template<auto F,class... A>void VF_CALL(Dim3 d,A&&... a){for(threadIdx.x=0;threadIdx.x<d.x;++threadIdx.x)F(a...);}
}
struct Selector {
 uint32_t* scratch;unsigned calls=0;uint16_t threshold=0;
 void operator()(Tensor<uint16_t> input,Tensor<uint32_t> indices,Tensor<uint16_t> output,
                 unsigned size,unsigned,unsigned,bool){
  assert(size>=2048&&size<=18432&&size%256==0);++calls;
  auto sorted=std::vector<uint16_t>(input.p,input.p+size);
  std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());threshold=sorted[2047];
  unsigned n=0;
  for(unsigned i=0;i<size;++i)if(input.p[i]>threshold){indices.p[n]=i;output.p[n++]=input.p[i];}
  for(unsigned i=0;i<size&&n<2048;++i)if(input.p[i]==threshold){indices.p[n]=i;output.p[n++]=input.p[i];}
  assert(n==2048);
  // Selector owns this buffer again. Any compact-offset alias here is invalid.
  std::fill_n(scratch,12096,0xdeadbeefU);
 }
};
struct State{
 std::array<uint16_t,18432> mrg{};
 std::array<uint16_t,4096> scores{};
 std::array<uint16_t,2048> offsets{};
 std::array<uint32_t,2048> ids{};
 std::array<uint32_t,12096> scratch{};
 std::array<uint32_t,2112> indexOut{};
 std::array<uint16_t,2048> scoreOut{};
 Tensor<uint16_t> mrgValueLocal_{mrg.data(),18432},incrementalScores_{scores.data(),4096},
   incrementalOffsets_{offsets.data(),2048},scoreOutLocal_{scoreOut.data(),2048};
 Tensor<uint32_t> incrementalIds_{ids.data(),2048},topkSharedTmpLocal_{scratch.data(),12096},
   indicesOutLocal_{indexOut.data(),2112};
 Selector topkOp_{scratch.data()};
 unsigned incrementalCount_=16384,incrementalBegin_=16384,incrementalReverseBegin_=16384,
   incrementalReverseEnd_=129024,incrementalTailBegin_=128000,incrementalTailEnd_=129536;
 uint16_t kth=30000;
 void SaveIncremental(unsigned){
  std::copy_n(scoreOut.data(),2048,scores.data());
  std::copy_n(indexOut.data(),2048,ids.data());kth=topkOp_.threshold;
 }
 MERGE
};
int main(){
 std::mt19937 gen(177);
 for(unsigned count:{0U,1U,127U,128U,2047U,2048U,2049U,16384U}){
  State s;
  for(unsigned i=0;i<2048;++i){s.scores[i]=30000;s.ids[i]=i|((i%3)?0:kNeedsFetch);}
  for(unsigned step=0;step<8;++step){
   unsigned extent=step%2?16384:16256;
   unsigned admitted=std::min(count,extent);
   auto oldScores=s.scores;auto oldIds=s.ids;
   std::vector<uint16_t> raw(extent);
   for(auto& value:raw)value=uint16_t(gen()%(s.kth+1));
   std::vector<unsigned> order(extent);for(unsigned i=0;i<extent;++i)order[i]=i;
   std::shuffle(order.begin(),order.end(),gen);
   for(unsigned i=0;i<admitted;++i)raw[order[i]]=s.kth+1;
   // Simulate Fetch's destruction of shared buffers between windows.
   s.mrg.fill(0xbeef);s.scratch.fill(0xdeadbeef);s.offsets.fill(0xbeef);
   s.incrementalBegin_=16384+step*12800;s.incrementalCount_=extent;
   for(unsigned base=0;base<extent;base+=128){
    RegTensor<uint16_t> tile;std::copy_n(raw.data()+base,128,tile.v.begin());
    kvip_incremental::StoreWindow(tile,s.mrg.data()+2048+base,128,&s.kth);
   }
   std::vector<uint16_t> all(oldScores.begin(),oldScores.begin()+2048);
   all.insert(all.end(),raw.begin(),raw.end());auto sorted=all;
   std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());uint16_t kth=sorted[2047];
   std::vector<unsigned> selected;
   for(unsigned i=0;i<all.size();++i)if(all[i]>kth)selected.push_back(i);
   for(unsigned i=0;i<all.size()&&selected.size()<2048;++i)if(all[i]==kth)selected.push_back(i);
   unsigned calls=s.topkOp_.calls;
   s.MergeIncremental(0);
   assert(s.incrementalCount_==0&&s.kth==kth);
   assert(s.topkOp_.calls==calls+(admitted?1:0));
   for(unsigned i=0;i<2048;++i){
    unsigned pos=selected[i],expected;
    if(pos<2048)expected=oldIds[pos];
    else{
     unsigned t=s.incrementalBegin_+pos-2048;
     if(t>=s.incrementalReverseBegin_&&t<s.incrementalReverseEnd_)
      t=s.incrementalReverseBegin_+s.incrementalReverseEnd_-2048-(t&~2047U)+(t&2047U);
     expected=t|((t<s.incrementalTailBegin_||t>=s.incrementalTailEnd_)?kNeedsFetch:0);
    }
    assert(s.ids[i]==expected&&s.scores[i]==all[pos]);
   }
  }
 }
}
'''.replace("MERGE", function((INDEXER / "incremental_topk_methods.h").read_text(),
                                  "    __aicore__ inline void MergeIncremental("))
    return source


def test_actual_boundary_sparse_dense_empty_and_scratch_reuse(tmp_path):
    source = prefilter_model_source()
    cpp = tmp_path / "prefilter.cpp"
    cpp.write_text(source)
    exe = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-DKVIP_INCREMENTAL_WINDOW=1", "-DKVIP_INCREMENTAL_PREFILTER=1",
                    str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
