"""Run the actual final-TopK classifier for supported hot sizes under ASan."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def test_classifier_partial_metadata_chunks_and_full_capacity(tmp_path):
    header = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_topk_classify.h"
    code = (
        r"""
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#define __simt_vf__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx;
void asc_syncthreads() {}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
template<class T> T asc_ldcg(T* p){return *p;}
namespace rotating {
constexpr uint32_t kTop=2048,kLruSlots=2048;
struct Context { uint64_t mapping,lruSlots; };
uint32_t ResolvePhysicalSlot(uint32_t resident,uint32_t* lru,uint32_t slot){
 return uint32_t(slot-resident)<kLruSlots?lru[slot-resident]:slot;
}
}
"""
        + f'#include "{header}"\n'
        + r"""
int main(){using namespace rotating;
 for(uint32_t resident:{2048U,8192U,16384U,33024U,131072U}){
  uint32_t history=resident+32768,words=resident/32;
  std::vector<uint32_t> valid(resident,1),map(history,UINT32_MAX),lru(2048),indices(2048),loc(2048,0),bits(4104),expected(words);
  std::vector<int32_t> reserved(resident,-1);
  for(uint32_t i=0;i<2048;++i)lru[i]=i;
  for(uint32_t i=0;i<resident;i++){valid[i]=(i%7!=0);if(i%11==0)reserved[i]=i%2048;}
  for(uint32_t i=0;i<history;++i)if(i%3==0)map[i]=i%2?i%resident:resident+i%2048;
  for(uint32_t i=0;i<2048;++i){
   indices[i]=(i*997)%history;
   if(i%97==0)indices[i]=UINT32_MAX;
   if(i==2046)indices[i]=resident-1; // final partial metadata chunk
  }
  Context ctx{uint64_t(map.data()),uint64_t(lru.data())};
  uint32_t tailBegin=resident-128,tailEnd=resident+128;
  for(uint32_t base=0;base<resident;base+=8192){
   uint32_t size=std::min(8192U,resident-base);
   for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
    ClassifyTopkHalf(history,resident,0,512,base,size,tailBegin,tailEnd,true,true,true,&ctx,
      indices.data(),loc.data(),valid.data()+base,reserved.data()+base,bits.data(),bits.data()+4096);
  }
  for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)MergeClassifyBitmap(bits.data(),resident);
  assert(bits[4096]==0);
  for(uint32_t i=0;i<2048;++i){
   uint32_t t=indices[i],want=UINT32_MAX;
   if(t!=UINT32_MAX){
    if(t<resident && !(t>=tailBegin&&t<tailEnd) && valid[t] && reserved[t]<0)want=t;
    else want=ResolvePhysicalSlot(resident,lru.data(),map[t]);
   }
   assert(loc[i]==want);
   if(want<resident)expected[want/32]|=1U<<(want%32);
  }
  for(uint32_t i=0;i<words;++i)assert(bits[i]==expected[i]);
 }
}
"""
    )
    source = tmp_path / "classify.cpp"
    source.write_text(code)
    binary = tmp_path / "classify"
    subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
