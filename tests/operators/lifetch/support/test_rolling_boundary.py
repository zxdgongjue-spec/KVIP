"""Execute the native boundary orchestration with bounded DMA/event stubs."""
from pathlib import Path
import subprocess

from test_recent_index import function, INDEXER


def test_boundary_dma_offsets_scratch_capacity_and_event_ownership(tmp_path):
    method = function((INDEXER / "indexer_vector.h").read_text(),
                      "__aicore__ inline void UpdateBoundary(")
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __ubuf__
#define KVIP_REVERSE_COLD 1
#include "cold_scan.h"
enum class HardEvent {MTE3_MTE2,V_MTE2,MTE2_V,V_MTE3};
constexpr unsigned EVENT_ID0=0,V_MTE2_EVENT3=5,TOPK_MTE2_V_EVENT=5,VEC1_V_MTE3_EVENT=0;
bool flags[4][8]{},written=false,scratchReady=false,readReady=false;
unsigned copies=0,readWords=0;
template<HardEvent E> void SetFlag(unsigned id){assert(!flags[unsigned(E)][id]);flags[unsigned(E)][id]=true;}
template<HardEvent E> void WaitFlag(unsigned id){
 assert(flags[unsigned(E)][id]);flags[unsigned(E)][id]=false;
 if(E==HardEvent::MTE3_MTE2)written=true;
 if(E==HardEvent::V_MTE2)scratchReady=true;
 if(E==HardEvent::MTE2_V)readReady=true;
}
template<class T> struct LocalTensor{
 T* p;unsigned size;
 LocalTensor operator[](unsigned i)const{assert(i<size);return {p+i,size-i};}
 T* GetPhyAddr()const{return p;}
};
struct DataCopyExtParams{unsigned blockCount,blockLen,srcStride,dstStride,reserved;};
template<class T> struct DataCopyPadExtParams{bool pad;unsigned left,right;T value;};
template<class T> void DataCopyPad(LocalTensor<T> dst,LocalTensor<T> src,
 DataCopyExtParams copy,DataCopyPadExtParams<T>){
 assert(written&&scratchReady&&copy.blockCount==1);
 unsigned n=copy.blockLen/sizeof(T);assert(n<=dst.size&&n<=src.size);
 std::copy_n(src.p,n,dst.p);++copies;readWords+=n;
}
namespace QLICommon{struct RunInfo{unsigned s2Idx,loop;};}
namespace rotating{
unsigned Trigger(unsigned h,unsigned r,unsigned rounds,unsigned epoch){
 unsigned tiles=(h-r)/128,n=rounds-1;
 return r/128+(tiles/n)*epoch+std::min(epoch,tiles%n);
}}
std::vector<unsigned> visits;
namespace kvip_rolling{
constexpr unsigned kStateWords=512,kAdmission=384;
void ObserveWindow(uint16_t* input,unsigned count,uint16_t*,uint16_t*,uint16_t*,
 unsigned begin,unsigned,unsigned,unsigned,unsigned,unsigned){
 assert(readReady);for(unsigned i=0;i<count;++i){assert(input[i]==uint16_t(begin+i));visits.push_back(begin+i);}
}
void FinishBoundary(uint16_t* last,uint16_t*,uint16_t*,uint16_t*){assert(readReady&&last);}
}
struct Model{
 using SCORE_T=uint16_t;
 struct{unsigned resident=16384,history=131328,rounds=9;}runtimeShape_;
 unsigned rollingWindowBegin_=16384,rollingReverseEnd_=129024;
 unsigned rollingWindowEpoch_=1,rollingWindowTrigger_=0;
 unsigned rollingEnds_[2]{130153,130154},rollingBegin_[2]{128000,128000},rollingEnd_[2]{130048,130048};
 std::array<uint16_t,18432> merge{};
 std::array<uint16_t,1024> state{};
 std::array<uint16_t,256> threshold{},indices{};
 std::vector<uint16_t> gm=std::vector<uint16_t>(2*131328);
 LocalTensor<uint16_t> mrgValueLocal_{merge.data(),unsigned(merge.size())};
 LocalTensor<uint16_t> rollingState_{state.data(),unsigned(state.size())};
 LocalTensor<uint16_t> prefixThreshold_{threshold.data(),unsigned(threshold.size())};
 LocalTensor<uint16_t> indicesOutLocal_{indices.data(),unsigned(indices.size())};
 LocalTensor<uint16_t> scoreGm{gm.data(),unsigned(gm.size())};
METHOD
};
int main(){
 for(unsigned q=0;q<2;++q){
  Model m;unsigned h=m.runtimeShape_.history;
  std::fill(m.gm.begin(),m.gm.end(),0xdead);
  for(unsigned i=0;i<h;++i)m.gm[q*h+i]=uint16_t(i);
  auto range=kvip_cold_scan::Range{16384,m.rollingReverseEnd_};
  for(unsigned epoch=1;epoch<=8;++epoch){
   unsigned start=m.rollingWindowBegin_,end=rotating::Trigger(h,16384,9,epoch)*128;
   QLICommon::RunInfo info{end/128-1,epoch};
   std::array<uint16_t,128> last{},hits{};
   written=scratchReady=readReady=false;copies=readWords=0;visits.clear();
   m.UpdateBoundary(info,q,q*h,{last.data(),128},{hits.data(),128});
   std::vector<unsigned> expected;
   for(unsigned i=start;i<end;++i)expected.push_back(kvip_cold_scan::Remap(i,range));
   std::sort(visits.begin(),visits.end());std::sort(expected.begin(),expected.end());
   assert(visits==expected&&readWords==end-start&&copies<=5);
   assert(m.rollingWindowBegin_==end&&m.rollingWindowEpoch_==epoch+1);
   for(auto& pipe:flags)for(bool pending:pipe)assert(!pending);
  }
 }
}
'''.replace("METHOD", method)
    cpp = tmp_path / "boundary.cpp"
    cpp.write_text(code)
    binary = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(INDEXER.parents[1] / "include"), str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
