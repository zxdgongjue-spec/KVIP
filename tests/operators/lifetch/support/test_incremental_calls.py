"""Compile actual incremental SIMT call sites, then check candidate reservation.

Calling a function directly hides the default-argument error that VF_CALL's
function-pointer invocation exposes. Extract both definitions and call sites.
"""

import re
import subprocess

from test_recent_index import INDEXER, function


def calls(source, name):
    return re.findall(r"Simt::VF_CALL<" + re.escape(name) + r">\(.*?;", source, re.S)


def test_native_simt_calls_and_q4_reservation(tmp_path):
    prefetch = (INDEXER.parent / "prefetch.h").read_text()
    vector = (INDEXER / "incremental_topk_vector.h").read_text()
    methods = (INDEXER / "incremental_topk_methods.h").read_text()
    gathers = calls(prefetch, "GatherCandidateMap")
    reserves = calls(prefetch, "ReserveIncremental")
    remaps = calls(methods, "kvip_incremental::Remap")
    tags = calls(methods, "kvip_incremental::TagSeed")
    assert (len(gathers), len(reserves), len(remaps), len(tags)) == (2, 1, 2, 1)
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __ubuf__
#define __gm__
#define KVIP_WINDOW_REFILTER 0
#define KVIP_REVERSE_COLD 1
constexpr uint32_t kTop=2048,kNeedsFetch=0x80000000U;
struct {uint32_t x;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_threadfence(){}
namespace Simt {
struct Dim3{uint32_t x,y,z;};
template<auto F,class... A>void VF_CALL(Dim3 d,A&&... a){
 for(threadIdx.x=0;threadIdx.x<d.x;++threadIdx.x)F(a...);
}
}
template<class T>struct Tensor{
 T* p;T* GetPhyAddr(){return p;}
 Tensor operator[](uint32_t i){return {p+i};}
};
namespace rotating {
struct Context{uint64_t reservedHot;uint32_t begin,end;uint32_t* valid;};
bool HotValid(uint32_t resident,Context* ctx,uint32_t b,uint32_t t){return ctx->valid[b*resident+t]!=0;}
bool IsTailToken(Context* ctx,uint32_t,uint32_t t){return t>=ctx->begin&&t<ctx->end;}
}
namespace kvip_incremental {
PRODUCER_DEFINITIONS
}
CONSUMER_DEFINITIONS
// Compile the actual expressions too: producer dataflow is tested separately.
void producerCalls(Tensor<uint32_t> indicesOutLocal_,Tensor<uint32_t> incrementalIds_,
 Tensor<uint16_t> incrementalOffsets_,rotating::Context* ctx){
 uint32_t residentCount=16384,begin=129024,query=0;
 uint32_t incrementalBegin_=16384,incrementalReverseBegin_=16384,incrementalReverseEnd_=129024;
 uint32_t incrementalTailBegin_=129024,incrementalTailEnd_=129536;
 struct{uint32_t resident;}runtimeShape_{16384};
 PRODUCER_CALLS
}
void gather(Tensor<uint32_t> ids,Tensor<uint32_t> work,uint32_t* map,uint32_t candidates,bool baseline){
 std::array<uint32_t,128> control{};Tensor<uint32_t>cfg{control.data()};
 uint32_t candidateCount=candidates,rt_history=4096,b=0;
 uint8_t* rollingScores=nullptr;
 if(baseline){BASELINE_GATHER}else{INCREMENTAL_GATHER}
}
void reserve(Tensor<uint32_t> selectedIds,uint32_t* map,uint32_t admitted){
 uint32_t count=5,rt_resident=16384,slots=77;
 RESERVE
}
int main(){
 for(uint32_t n:{0U,1U,63U,64U,65U,255U,256U,257U,2047U,2048U}){
  std::vector<uint32_t> ids(2048),map(4096,UINT32_MAX),slots(2048,0xa5a5a5a5),selected(32);
  for(uint32_t i=0;i<n;++i){ids[i]=(17*i)%4096;if(i%7==0)map[ids[i]]=123;}
  for(unsigned q=0;q<4;++q){
   gather({ids.data()},{slots.data()},map.data(),n,false);
   for(uint32_t i=0;i<n;++i)assert(slots[i]==map[ids[i]]);
   for(uint32_t i=n;i<2048;++i)assert(slots[i]==0xa5a5a5a5);
   auto before=slots;
   gather({ids.data()},{slots.data()},map.data(),n,true);
   assert(before==slots);
   uint32_t admitted=0;
   for(uint32_t i=0;i<n&&admitted<16;++i)
    if(slots[i]==UINT32_MAX)selected[5+admitted++]=ids[i];
   auto expected=map;
   for(uint32_t i=0;i<admitted;++i)expected[selected[5+i]]=16384+77+5+i;
   reserve({selected.data()},map.data(),admitted);
   assert(map==expected); // Only admitted IDs reserved; later Qs see them.
  }
 }
}
'''
    source = source.replace("PRODUCER_DEFINITIONS", "\n".join(
        function(vector, "inline void " + name + "(") for name in ("Remap", "TagSeed")))
    source = source.replace("CONSUMER_DEFINITIONS", "\n".join(
        function(prefetch, "inline void " + name + "(")
        for name in ("GatherCandidateMap", "ReserveIncremental")))
    source = source.replace("PRODUCER_CALLS", "\n".join([remaps[0], tags[0], remaps[1]]))
    source = source.replace("BASELINE_GATHER", gathers[1]).replace("INCREMENTAL_GATHER", gathers[0])
    source = source.replace("RESERVE", reserves[0])
    cpp = tmp_path / "incremental_calls.cpp"
    cpp.write_text(source)
    binary = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
