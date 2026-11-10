"""Compare staged service-output publication with the existing implementation."""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def original_function():
    text = (REPO / 'csrc/operators/lightning_indexer_with_fetch_safe/op_kernel/index_cache_update.cpp').read_text()
    pos = text.index('inline void PublishOutputs(')
    start = text.rfind('\n', 0, pos) + 1
    brace = text.index('{', pos)
    end, depth = brace + 1, 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def test_staged_output_matches_existing_permuted_and_padded(tmp_path):
    source, executable = tmp_path / 'test.cpp', tmp_path / 'test'
    prefix = r'''
#include <cstdint>
#include <vector>
#include <cassert>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
void asc_syncthreads(){} void asc_threadfence(){}
namespace rotating { constexpr uint32_t kTop=2048; }
#include "staged_outputs.h"
'''
    main = r'''
int main(){
 constexpr uint32_t b=2,h=128,k=2048;
 std::vector<int32_t> topk(b*4*k),slots(b*4*k),oldId(b*4*k,-9),oldSlot(b*4*k,-9),newId(b*4*k,-8),newSlot(b*4*k,-8);
 std::vector<int64_t> ids(3*h),unpack={5,0,6,6,1,4,7,0},valid={1,1,1,1,0,1,1,0};uint32_t owners[]={2,1};
 for(uint32_t i=0;i<ids.size();++i)ids[i]=1000+i;
 for(uint32_t i=0;i<topk.size();++i){topk[i]=i%h;slots[i]=i%64;}topk[7]=-1;
 uint64_t a[28]={};a[3]=(uint64_t)ids.data();a[9]=(uint64_t)owners;a[18]=(uint64_t)topk.data();a[19]=(uint64_t)slots.data();a[20]=(uint64_t)unpack.data();a[21]=(uint64_t)valid.data();a[22]=(uint64_t)oldId.data();a[23]=(uint64_t)oldSlot.data();a[24]=h;
 for(uint32_t row=0;row<b;++row)for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishOutputs(a,row);
 a[22]=(uint64_t)newId.data();a[23]=(uint64_t)newSlot.data();
 uint32_t masks[4],values[k];
 for(uint32_t row=0;row<b;++row)for(uint32_t q=0;q<4;++q){
  for(auto &mask:masks)mask=0;
  for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
   rotating::PrepareStagedQueryOutputs(a,b,row,q,(uint32_t*)(topk.data()+row*4*k),masks,values);
  // Model the production MTE3 copies after all SIMT lanes have completed.
  for(uint32_t dst=0;dst<b*4;++dst){
   bool active=(masks[dst/32]>>(dst%32))&1U,pad=(masks[2+dst/32]>>(dst%32))&1U;
   if(active)for(uint32_t j=0;j<k;++j){newId[dst*k+j]=int32_t(values[j]);newSlot[dst*k+j]=slots[(row*4+q)*k+j];}
   if(pad)for(uint32_t j=0;j<k;++j){newId[dst*k+j]=-1;newSlot[dst*k+j]=-1;}
  }
 }
 assert(oldId==newId && oldSlot==newSlot);
}
'''
    source.write_text(prefix + original_function() + main)
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(HERE.parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming/op_kernel'), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_routing_masks_span_two_words_and_ignore_unrelated_rows(tmp_path):
    source, executable = tmp_path / 'mask.cpp', tmp_path / 'mask'
    source.write_text(r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
void asc_syncthreads(){}
namespace rotating { constexpr uint32_t kTop=2048; }
#include "staged_outputs.h"
int main(){
 constexpr uint32_t b=16,k=2048;std::vector<int64_t> ids(64),unpack(64,63),valid(64,1);
 std::vector<uint32_t> owners(b),selected(4*k,1),values(k),masks(4);uint64_t a[28]={};
 for(uint32_t i=0;i<64;++i)ids[i]=100+i;
 unpack[0]=0;unpack[31]=0;unpack[32]=0;unpack[63]=0;valid[1]=0;
 a[3]=(uint64_t)ids.data();a[9]=(uint64_t)owners.data();a[20]=(uint64_t)unpack.data();a[21]=(uint64_t)valid.data();a[24]=64;
 for(uint32_t i=0;i<512;++i){threadIdx.x=i;rotating::PrepareStagedQueryOutputs(a,b,0,0,selected.data(),masks.data(),values.data());}
 assert(masks[0]==((1U<<0)|(1U<<31)));assert(masks[1]==((1U<<0)|(1U<<31)));
 assert(masks[2]==(1U<<1));assert(masks[3]==0);
 for(auto v:values)assert(v==101);
}
''')
    include = HERE.parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming/op_kernel'
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(include), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)
