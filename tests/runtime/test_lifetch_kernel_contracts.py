"""Run the actual final-TopK classifier for supported hot sizes under ASan."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


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
 for(uint32_t resident:{2048U,6400U,8192U,16384U,33024U,131072U}){
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


def test_shared_li_dispatch_and_shard_boundaries(tmp_path):
    common = ROOT / "csrc/operators/common/lifetch"
    source = r"""
#include <cstdint>
#include <cassert>
#include <vector>
namespace rotating {
struct RuntimeShape {uint32_t splitQ=1,indexHeads=32,rounds=8,resident=16384,history=131072,liCores=8;};
#include "sequence_schedule.h"
}
std::vector<int> calls;
int failure=-1;
uint32_t run(int phase,uint32_t cores,void*,void*) {calls.push_back(phase);return phase==failure?99:0;}
uint32_t launch(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0(c,s,b) run(0,c,s,b)
#define LI_LAUNCH_1(c,s,b) run(1,c,s,b)
#define LI_LAUNCH_2(c,s,b) run(2,c,s,b)
#define LI_LAUNCH_3(c,s,b) run(3,c,s,b)
#include "li_launch.h"
}
int main() {
 rotating::RuntimeShape shape;auto ptr=reinterpret_cast<void*>(1);
 for(auto rounds:{2U,4U,8U}) {
  shape.rounds=rounds;assert(rotating::UseSequenceSplit(shape));
  calls.clear();assert(!launch(ptr,ptr,&shape));assert((calls==std::vector<int>{0,1,2,3}));
  auto half=rotating::SequenceHalfWindow(shape);
  for(uint32_t token=0;token<shape.history;++token) {
   auto part=rotating::SequenceSliceAt(token,shape.resident,half);
   assert(part.end>token && part.end<=shape.history+2*half);
   if(token<shape.resident)assert(part.shard==0 && part.end==shape.resident);
   else assert(part.shard==((token-shape.resident)/half)%2);
  }
 }
 // BS2 uses the same split path, with an aligned truncated final shard.
 shape.rounds=4;assert(rotating::SequenceHalfWindow(shape)==19200);
 std::vector<unsigned> visits(shape.history,0);
 for(unsigned epoch=0;epoch<3;++epoch)for(unsigned shard=0;shard<2;++shard){
  unsigned begin=shape.resident+(2*epoch+shard)*19200;
  unsigned end=begin+19200;if(end>shape.history)end=shape.history;
  assert(begin%256==0 && end%256==0);
  for(unsigned t=begin;t<end;++t){++visits[t];assert(rotating::SequenceSliceAt(t,shape.resident,19200).shard==shard);}
 }
 for(unsigned t=shape.resident;t<shape.history;++t)assert(visits[t]==1);
 shape.rounds=8;
 for(int phase=0;phase<4;++phase) {failure=phase;calls.clear();assert(launch(ptr,ptr,&shape)==99);assert(calls.size()==phase+1);}
 assert(launch(nullptr,ptr,&shape)==100000);
}
"""
    path = tmp_path / "shared.cpp"
    path.write_text(source)
    binary = tmp_path / "shared"
    subprocess.run(["c++", "-std=c++17", "-I", str(common), str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_scale8_releases_partial_sequence_shard(tmp_path):
    """Exercise the actual bank-release condition with BS2's 150-tile shard."""
    import re

    header = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h"
    source = header.read_text()
    condition = re.search(r"if \((\(info.s2Idx - info.s2Start \+ 1\).*?)\) \{", source, re.S)[1]
    cpp = tmp_path / "scale_bank.cpp"
    cpp.write_text(
        """
#include <cassert>
namespace QLICommon {constexpr unsigned KScaleTiles=8;}
struct Info {unsigned s2Idx,s2Start,validS2Len;bool isLastS2InnerLoop;};
bool release(Info info) {unsigned s2BaseSize_=128;return """
        + condition
        + """;}
int main() {
 // A non-final 19200-token shard has 150 tiles. Its tail must return the bank.
 assert(release({277,128,278*128,false}));
 assert(!release({276,128,278*128,false}));
 assert(release({135,128,278*128,false}));
 // The next shard has a different absolute start and the same partial tail.
 assert(release({427,278,428*128,false}));
}
"""
    )
    exe = tmp_path / "scale_bank"
    subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_host_pipeline_order_and_first_error(tmp_path):
    import subprocess

    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    begin = source.index('extern "C" uint32_t kvip_lifetch_streaming_forward_v2(')
    entry = source[begin : source.index('\nextern "C" uint32_t aclrtlaunch_kvip_transport_snapshot', begin)]
    stub = r"""
#include <cstdint>
#include <cassert>
#include "runtime_shape.h"
#include "independent_schedule.h"
namespace pipeline_ab { constexpr uint32_t kResolveCores=3; }
int fail=0, count=0;
void* stream_expected=(void*)3;
uint32_t step(int n,void* a,void* stream) { assert(a==(void*)1);assert(stream==stream_expected);assert(++count==n);return fail==n?99:0; }
uint32_t v9_init(void*a,void*s,const rotating::RuntimeShape*){return step(1,a,s);}
uint32_t v9_li(void*a,void*s,const rotating::RuntimeShape*){return step(2,a,s);}
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t cores,void*s,void*a){assert(cores==56);return step(3,a,s);}
"""
    main = r"""
int main(){rotating::RuntimeShape shape{};shape.batch=16;shape.finalWorkers=4;shape.aicCount=28;shape.liCores=16;
for(fail=0;fail<=3;++fail){count=0;assert(kvip_lifetch_streaming_forward_v2((void*)1,(void*)2,(void*)1,stream_expected,&shape,2,56)==(fail?99:0));assert(count==(fail?fail:3));}
count=0;assert(kvip_lifetch_streaming_forward_v2((void*)1,(void*)2,(void*)1,stream_expected,&shape,2,55)!=0);assert(count==0);
count=0;shape.batch=17;assert(kvip_lifetch_streaming_forward_v2((void*)1,(void*)2,(void*)1,stream_expected,&shape,2,56)!=0);assert(count==0);return 0;}
"""
    path = tmp_path / "launch.cpp"
    path.write_text(stub + entry + main)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
            "-I" + str(ROOT / "csrc/operators/common/lifetch"),
            str(path),
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)
