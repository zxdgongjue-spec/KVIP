# SPDX-License-Identifier: Apache-2.0
"""Execute the candidate's real scalar DMA orchestration and SIMT gather.

The CPU model copies DMA synchronously and executes independent gather lanes
sequentially. It validates byte semantics/bounds, not NPU cache or performance.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def test_mte_output_rows_match_element_oracle(tmp_path):
    header = ROOT / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/low_q4_outputs_mte.h'
    program = tmp_path / 'outputs.cpp'
    program.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(x)
constexpr unsigned EVENT_ID0=0;
thread_local struct {uint32_t x=0;} threadIdx;
void asc_syncthreads() {}
template<class T>T asc_ldcg(T* p) {return *p;}
void asc_atomic_or(uint32_t* p,uint32_t v) {*p|=v;}
namespace AscendC {
enum class TPosition {VECCALC};
enum class HardEvent {MTE2_V,V_S,V_MTE3,MTE3_S};
template<class T>struct LocalTensor {
 T* data;
 LocalTensor operator[](uint32_t i)const{return {data+i};}
 T* GetPhyAddr()const{return data;}
 T GetValue(uint32_t i)const{return data[i];}
};
template<class T>struct GlobalTensor {
 T* data;
 void SetGlobalBuffer(T* p){data=p;}
};
template<TPosition>struct TBuf {
 std::vector<uint32_t> values;
 template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(values.data())};}
};
uint64_t readWords=0,writeWords=0,gathers=0;
struct TPipe {
 template<TPosition P>void InitBuffer(TBuf<P>& b,uint32_t bytes){b.values.resize(bytes/4);}
 void Destroy(){}
};
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t n){
 std::copy_n(src.data,n,dst.data);readWords+=n;
}
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,uint32_t n){
 std::copy_n(src.data,n,dst.data);writeWords+=n;
}
template<class T>void Duplicate(LocalTensor<T> dst,T value,uint32_t n){std::fill_n(dst.data,n,value);}
template<HardEvent>void SetFlag(unsigned){}
template<HardEvent>void WaitFlag(unsigned){}
}
namespace Simt {
struct Dim3{uint32_t x,y,z;};
template<auto F,class...Args>void VF_CALL(Dim3 dims,Args...args){
 ++AscendC::gathers;
 for(uint32_t t=0;t<dims.x;++t){threadIdx.x=t;F(args...);}
}
}
namespace rotating {
constexpr uint32_t kTop=2048;
struct Shape{uint32_t batch,history,resident;};
struct Context{Shape shape;uint64_t outputStride,selected,locations;};
}
namespace pipeline_ab {struct Args{uint64_t updateBindings;};}
''' + '#include "' + str(header) + '"\n' + r'''
int main(){
 constexpr uint32_t history=131328,resident=16384,capacity=resident+512,ownersCount=20;
 std::mt19937 rng(414);
 for(uint32_t batch:{1U,3U,16U}){
  const uint32_t rows=batch*4,words=rows*2048;
  std::vector<uint32_t> selected(words),slots(words),logicalOut(words),slotOut(words),owners(batch);
  std::vector<int64_t> logical(uint64_t(ownersCount)*history),unpack(rows),valid(rows,1);
  std::iota(owners.begin(),owners.end(),2);
  std::shuffle(owners.begin(),owners.end(),rng);
  for(uint32_t i=0;i<logical.size();++i)logical[i]=uint64_t(i)*37+13;
  for(uint32_t i=0;i<words;++i){selected[i]=rng()%history;slots[i]=rng()%capacity;}
  for(uint32_t i=0;i<words;i+=37){selected[i]=UINT32_MAX;slots[i]=0;}
  for(uint32_t row=0;row<rows;++row){
   unpack[row]=(rows-row-1)/2; // reorder and duplicate source query rows
   if(row%7==2){valid[row]=0;unpack[row]=INT64_MIN;}
  }
  uint64_t update[28]={};
  update[3]=uint64_t(logical.data());update[9]=uint64_t(owners.data());
  update[20]=uint64_t(unpack.data());update[21]=uint64_t(valid.data());
  update[22]=uint64_t(logicalOut.data());update[23]=uint64_t(slotOut.data());update[24]=history;
  rotating::Context context{{batch,history,resident},capacity,uint64_t(selected.data()),uint64_t(slots.data())};
  pipeline_ab::Args args{uint64_t(update)};
  for(uint32_t cores:{8U,32U,64U})for(uint32_t repeat=0;repeat<2;++repeat){
   std::fill(logicalOut.begin(),logicalOut.end(),123);std::fill(slotOut.begin(),slotOut.end(),123);
   auto originalSelected=selected,originalSlots=slots;auto originalLogical=logical;
   AscendC::readWords=AscendC::writeWords=AscendC::gathers=0;
   uint32_t active=0;
   for(uint32_t block=0;block<cores;++block)for(uint32_t row=block;row<rows;row+=cores){
    assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,row)==0);
    active+=valid[row]!=0;
   }
   assert(AscendC::readWords==uint64_t(active)*4096);
   assert(AscendC::writeWords==uint64_t(rows)*4096);
   assert(AscendC::gathers==active);
   for(uint32_t row=0;row<rows;++row)for(uint32_t column=0;column<2048;++column){
    uint32_t expectedId=UINT32_MAX,expectedSlot=UINT32_MAX;
    if(valid[row]){
     uint32_t source=unpack[row],token=selected[source*2048+column];
     if(token!=UINT32_MAX){
      expectedId=uint32_t(logical[uint64_t(owners[source/4])*history+token]);
      expectedSlot=slots[source*2048+column];
     }
    }
    assert(logicalOut[row*2048+column]==expectedId);
    assert(slotOut[row*2048+column]==expectedSlot);
   }
   assert(originalSelected==selected&&originalSlots==slots&&originalLogical==logical);
   for(auto& id:logical)id+=17;
  }
  // Active input errors must be reported before a transport can be posted.
  const auto oldUnpack=unpack[0];unpack[0]=-1;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)!=0);
  unpack[0]=rows;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)!=0);
  unpack[0]=oldUnpack;
  const uint32_t source=unpack[0]*2048;
  const auto oldSelected=selected[source];selected[source]=history;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)!=0);
  assert(logicalOut[0]==UINT32_MAX&&slotOut[0]==UINT32_MAX);
  selected[source]=7;const auto oldSlot=slots[source];slots[source]=capacity;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)!=0);
  assert(logicalOut[0]==UINT32_MAX&&slotOut[0]==UINT32_MAX);
  selected[source]=oldSelected;slots[source]=oldSlot;
  update[9]=0;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)!=0);
  update[9]=uint64_t(owners.data());
  // Padding with poisoned identity pointers must neither gather nor read IDs.
  valid[0]=0;unpack[0]=INT64_MIN;update[9]=0;update[3]=0;
  assert(pipeline_ab::low_q4_outputs_mte::Publish(&context,&args,0)==0);
  assert(std::all_of(logicalOut.begin(),logicalOut.begin()+2048,[](auto x){return x==UINT32_MAX;}));
  assert(std::all_of(slotOut.begin(),slotOut.begin()+2048,[](auto x){return x==UINT32_MAX;}));
 }
}
''')
    binary = tmp_path / 'outputs'
    subprocess.run(['c++','-std=c++17','-O1','-g','-fsanitize=address,undefined',str(program),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=60)
