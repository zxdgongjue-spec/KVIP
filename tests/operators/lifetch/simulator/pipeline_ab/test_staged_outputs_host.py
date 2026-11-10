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
 uint32_t kinds[b*4];
 for(uint32_t row=0;row<b;++row)for(uint32_t q=0;q<4;++q){
  // The host shim has no SIMT barrier: model the lane-parallel first phase
  // before executing the individual lanes of the production function.
  for(uint32_t i=0;i<b*4;++i)kinds[i]=valid[i]?(unpack[i]==row*4+q?1:0):(q==0&&i/4==row?2:0);
  for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
   rotating::PublishStagedQueryOutputs(a,b,row,q,(uint32_t*)(topk.data()+row*4*k),(uint32_t*)(slots.data()+row*4*k),kinds);
 }
 assert(oldId==newId && oldSlot==newSlot);
}
'''
    source.write_text(prefix + original_function() + main)
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(HERE.parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming/op_kernel'), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)
