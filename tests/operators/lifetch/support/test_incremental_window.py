"""Exercise native incremental window selection and shared-scratch lifetime."""

import subprocess

from test_recent_index import COMMON, INDEXER, function
from test_recent_register_kth import register_model_source
from test_recent_threshold_only import threshold_model_source


def test_register_dispatch_preserves_exact_kth_and_stable_ties(tmp_path):
    # Execute the actual new branch and the actual register Kth implementation.
    # The unchanged output VF contracts are modeled explicitly; CANN/NPU still
    # verifies hardware scheduling and instruction behavior.
    source = register_model_source(threshold_model_source().split("struct TopK {", 1)[0])
    source = source.replace(" T* GetPhyAddr()const{return p;}", """
 T* GetPhyAddr()const{return p;}
 template<class U> Tensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(p),unsigned(n*sizeof(T)/sizeof(U))};}
""")
    text = (INDEXER / "topk.h").read_text()
    start = text.index("        if (s2LoopNum == 1) {", text.index("bool returnScores = false"))
    end = text.index("\n        if (loopIdx == 0)", start)
    branch = text[start:end]
    source += r'''
#define KVIP_INCREMENTAL_REGISTER 1
#define KVIP_INCREMENTAL_TIES 1
enum class SpecialPurposeReg{AR};
unsigned scalarAr=0;
template<SpecialPurposeReg>unsigned GetSpr(){return scalarAr;}
constexpr unsigned PIPE_V=0;
template<unsigned>void PipeBarrier(){}
enum class RoundMode{CAST_NONE};
void Cast(Tensor<uint32_t> out,Tensor<uint16_t> in,RoundMode,unsigned n){
 assert(n<=out.n&&n<=in.n);for(unsigned i=0;i<n;++i)out.p[i]=in.p[i];
}
namespace topkb16gather {
unsigned written=0;
void FindIdxGTOutputVFImpl(uint16_t* out,uint16_t* in,uint16_t begin,uint32_t* key,uint16_t loops){
 written=0;auto kth=*reinterpret_cast<uint16_t*>(key);
 for(unsigned i=0;i<loops*128;++i)if(in[i]>kth)out[written++]=begin+i;
 scalarAr=2*written;
}
void FindIdxEQOutputVFImpl(uint16_t* out,uint16_t* in,uint16_t begin,uint32_t* key,uint16_t loops){
 auto kth=*reinterpret_cast<uint16_t*>(key);
 for(unsigned i=0;i<loops*128;++i)if(in[i]==kth)out[written++]=begin+i;
}
void FindValueOutputVFImpl(uint16_t* out,uint16_t* in,uint16_t* idx,uint16_t loops){
 for(unsigned i=0;i<loops*128;++i)out[i]=in[idx[i]];
}
template<bool B,class... Args>void LiTopKVF(Args...){assert(false);}
}
struct TopK {
 unsigned topK=2048,incrementalStrictCount_=0;
STRICT_GETTER
 std::array<uint32_t,832> scratch{};
 std::array<uint16_t,18432> index{};
 Tensor<uint32_t> histogramsLocal{scratch.data(),256},idxHighLocal{scratch.data()+256,256},
   idxLowLocal{scratch.data()+512,256},nkValueLocal{scratch.data()+768,64};
 Tensor<uint16_t> tmpIndexLocal{index.data(),18432};
THRESHOLD
 void Run(Tensor<uint16_t>& mrgValueLocal,Tensor<uint32_t>& indicesOutLocal,
     Tensor<uint16_t>& hisValueLocal,uint32_t s2SeqLen){
   uint32_t s2LoopNum=1;bool returnScores=true;
BRANCH
 }
};
int main(){
 std::mt19937 gen(177);
 for(unsigned n:{2048U,2304U,4096U,16640U,18176U,18432U})
  for(unsigned pattern=0;pattern<8;++pattern){
   std::vector<uint16_t> input(n);
   for(unsigned i=0;i<n;++i)input[i]=pattern==0?0:pattern==1?65535:
     pattern==2?uint16_t(gen()%4):pattern==3?uint16_t(i):
     pattern==4?uint16_t(n-i):pattern==5?uint16_t(0xff00+gen()%256):uint16_t(gen());
   if(pattern==7)std::fill(input.begin()+n-128,input.end(),uint16_t(0));
   const auto before=input;auto sorted=input;
   std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());
   uint16_t kth=sorted[2047];std::vector<unsigned> expected;
   for(unsigned i=0;i<n;++i)if(input[i]>kth)expected.push_back(i);
   for(unsigned i=0;i<n;++i)if(input[i]==kth)expected.push_back(i);
   std::array<uint32_t,2048> ids{};std::array<uint16_t,2048> scores{};
   Tensor<uint16_t> in{input.data(),n},out{scores.data(),2048};Tensor<uint32_t> idx{ids.data(),2048};
   TopK op;topkb16gather::passes=0;op.Run(in,idx,out,n);
   assert(input==before&&topkb16gather::passes==0);
   assert(op.IncrementalStrictCount()==std::count_if(input.begin(),input.end(),[&](auto v){return v>kth;}));
   assert(*reinterpret_cast<uint16_t*>(op.nkValueLocal.p)==kth);
   for(unsigned i=0;i<2048;++i){assert(ids[i]==expected[i]);assert(scores[i]==input[ids[i]]);}
  }
}
'''.replace("THRESHOLD", function(text, "    __aicore__ inline void Threshold("))
    source = source.replace("BRANCH", branch)
    source = source.replace("STRICT_GETTER", function(text, "    __aicore__ inline uint32_t IncrementalStrictCount("))
    cpp = tmp_path / "window_selector.cpp"
    cpp.write_text(source)
    exe = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(COMMON), "-I", str(INDEXER), str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_window_scratch_schedule_geometry(tmp_path):
    context = (INDEXER.parents[1] / "include/context.h").read_text()
    names = ("Trigger", "ServiceRank", "ServiceCore", "ServiceLane", "EpochAt")
    functions = "\n".join(function(context, "V9_DEVICE inline " +
        ("int32_t" if name == "EpochAt" else "uint32_t") + " " + name + "(") for name in names)
    source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define V9_DEVICE
FUNCTIONS
int main(){
 for(uint32_t history:{17408U,131200U,131328U,131584U,147456U}){
  // Scratch can be poisoned by Fetch after every boundary. A fresh window
  // writes all of its slots before use, including zeros and 256 padding.
  std::vector<uint16_t> scratch(18432,0xdead);
  for(uint32_t epoch=1;epoch<=8;++epoch){
   uint32_t begin=Trigger(history,16384,9,epoch-1)*128;
   uint32_t end=Trigger(history,16384,9,epoch)*128;
   uint32_t size=(2048+end-begin+255)/256*256;
   assert(size<=scratch.size());
   std::fill(scratch.begin(),scratch.end(),uint16_t(0xdead));
   for(uint32_t i=0;i<2048;++i)scratch[i]=17;
   for(uint32_t i=begin;i<end;i+=128)
    std::fill_n(scratch.begin()+2048+i-begin,128,uint16_t(0));
   std::fill(scratch.begin()+2048+end-begin,scratch.begin()+size,uint16_t(0));
   assert(std::find(scratch.begin(),scratch.begin()+size,uint16_t(0xdead))==scratch.begin()+size);
  }
  for(uint32_t core=0;core<32;++core)
   for(uint32_t tile=128;tile<=history/128;++tile){
    int epoch=EpochAt(16,history,16384,1,9,2,tile,core);
    if(epoch>=0)assert(tile==Trigger(history,16384,9,uint32_t(epoch)));
   }
 }
}
'''.replace("FUNCTIONS", functions)
    cpp = tmp_path / "schedule.cpp"
    cpp.write_text(source)
    exe = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
