# SPDX-License-Identifier: Apache-2.0
"""Check compact tail scratch against the eager metadata permutation.

Execute the production hash lookup/store/metadata commit with poisoned values.
One simulated lane executes each complete loop; this tests data dependencies,
not SIMT concurrency or Ascend cache visibility.
"""

import subprocess

import pytest

from test_recent_index import INDEXER, function


@pytest.mark.parametrize("compact", [0, 1])
def test_poisoned_values_and_high_low_transitions(tmp_path, compact):
    source = (INDEXER.parent / "shadow_exchange_parallel.h").read_text()
    native = "\n".join(function(source, marker) for marker in (
        "inline uint32_t hash(",
        "inline uint32_t get(",
        "inline bool put(",
        "inline void InitShadow(",
        "inline void CommitMetadata(",
        "inline void InitDenseShadow(",
    ))
    program = tmp_path / "tail.cpp"
    program.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __ubuf__
#define __gm__
constexpr uint32_t EMPTY=UINT32_MAX,MAP=4096,THREADS=1;
constexpr uint32_t KEYS_POS=0,VALUES_POS=MAP,KEYS_ROW=2*MAP,VALUES_ROW=3*MAP;
constexpr uint32_t CONTROL=4*MAP,STAGING=CONTROL+128;
struct {uint32_t x=0;} threadIdx;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T v){*p=v;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t value){
 uint32_t result=*p;if(result==old)*p=value;return result;
}
""" + native + r"""
int main(){
 std::mt19937 random(1516);
 constexpr uint32_t history=32768;
 std::vector<uint32_t> scratch(STAGING+4*MAP),permutation(history);
 std::vector<uint64_t> addresses(history),logical(history);
 for(unsigned round=0;round<60;++round){
  for(auto& value:scratch)value=random();
  auto poisoned=scratch;
  for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x) {
   // Use one virtual lane for the map loops, then populate scalar controls.
   // InitShadow has a 128-lane control store; reset it explicitly below.
   if(!threadIdx.x)InitShadow(scratch.data());
   scratch[CONTROL+threadIdx.x]=0;
  }
  threadIdx.x=0;
  for(uint32_t i=0;i<MAP;++i){
   assert(scratch[KEYS_POS+i]==EMPTY&&scratch[KEYS_ROW+i]==EMPTY);
#if KVIP_LOW_MISS_TAIL
   assert(scratch[VALUES_POS+i]==poisoned[VALUES_POS+i]);
   assert(scratch[VALUES_ROW+i]==poisoned[VALUES_ROW+i]);
#else
   assert(scratch[VALUES_POS+i]==EMPTY&&scratch[VALUES_ROW+i]==EMPTY);
#endif
  }
  for(uint32_t i=0;i<history;++i){
   addresses[i]=0xabc000000000ULL+65536ULL*i+round;
   logical[i]=65537ULL*i+round;
   permutation[i]=i;
   assert(get(scratch.data()+KEYS_POS,scratch.data()+VALUES_POS,i)==i);
  }
  auto originalAddresses=addresses,originalLogical=logical;
  // Alternate empty, low-demand and high-demand rounds. Repeated destinations
  // and cycles test both maps; no old round may leak through poisoned values.
  uint32_t moves=round%3==0?0:round%3==1?37:700;
  for(uint32_t i=0;i<moves;++i){
   uint32_t src=random()%history,dst=random()%1024;
   auto key=scratch.data()+KEYS_POS,val=scratch.data()+VALUES_POS;
   uint32_t left=get(key,val,src),right=get(key,val,dst);
   assert(left==permutation[src]&&right==permutation[dst]);
   assert(put(key,val,src,right)&&put(key,val,dst,left));
   assert(put(scratch.data()+KEYS_ROW,scratch.data()+VALUES_ROW,right,src));
   assert(put(scratch.data()+KEYS_ROW,scratch.data()+VALUES_ROW,left,dst));
   std::swap(permutation[src],permutation[dst]);
  }
  CommitMetadata(addresses.data(),logical.data(),scratch.data());
  for(uint32_t i=0;i<history;++i){
   assert(addresses[i]==originalAddresses[permutation[i]]);
   assert(logical[i]==originalLogical[permutation[i]]);
  }
  // A mistaken low prediction can still saturate the sparse maps. The dense
  // fallback copies only valid hash entries and must preserve the permutation.
  std::vector<uint32_t> dense(history*2,0xdeadbeef);
  InitDenseShadow(dense.data(),history,scratch.data());
  for(uint32_t i=0;i<history;++i){
   assert(dense[i]==permutation[i]);
   assert(dense[history+permutation[i]]==i);
  }
 }
}
""")
    binary = tmp_path / "tail"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        f"-DKVIP_LOW_MISS_TAIL={compact}", str(program), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_empty_phase_keeps_publication_errors_and_completion_join():
    source = (INDEXER.parent / "streaming.cpp").read_text()
    consumer = function(source, "inline void Consumer(")
    assert consumer.index("if(flags[publication].error)") < consumer.index("if(!count){")
    assert consumer.index("InitShadow") < consumer.index("for(uint32_t phase=")
    assert consumer.index("flags+kOutputCompletionFlag") < consumer.index("&& seen")
    assert consumer.index("&& seen") < consumer.index("CommitMetadata")
    assert consumer.index("PublishStreamingUpdateStatus") < consumer.index("flags[6].ready=1")
    assert "status[2]=local.GetValue(CONTROL+4)" in consumer
    producer = function(source, "inline uint32_t Producer(")
    assert producer.index("if(row->slots)") < producer.index("VF_CALL<PublishPrefetch>")
    assert producer.index("VF_CALL<PublishPrefetch>") < producer.index("flags[0].ready=1")
    finalize = function(source, "inline void FinalizeStreaming(")
    assert finalize.index("&& row->slots") < finalize.index("VF_CALL<PublishPrefetchResidency>")
    assert finalize.index("VF_CALL<PublishPrefetchResidency>") < finalize.index("updateError=*updateStatus")


def test_final_get_completion_covers_reserved_repairs_without_prefetch(tmp_path):
    source = (INDEXER.parent / "streaming.cpp").read_text()
    program = tmp_path / "residency.cpp"
    program.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>
#define __gm__
#define V9_UBMEM 0
constexpr uint32_t kLruSlots=2048,kChangeCapacity=10240,kFlagCount=16;
struct {uint32_t x;} threadIdx;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T v){*p=v;}
struct Context {uint64_t hotValid,changes,lruSlots,destinations,owners;};
struct BatchFlag {uint32_t ready=0,count=0,error=0;};
struct FinalPost {uint32_t count=0,descriptor=0,pending=0,error=0,deferred=0;};
uint32_t OwnerRow(Context*,uint32_t b){return 3-b;}
uint32_t OutputStride(uint32_t resident,Context*){return resident+512;}
""" + function(source, "inline void PublishQueryCompletion(")
        + "\n" + function(source, "inline void PublishPrefetchResidency(") + r"""
int main(){
 constexpr uint32_t batch=4,resident=16384,stride=resident+512;
 std::vector<uint32_t> valid(batch*resident),changes(batch*kChangeCapacity,UINT32_MAX);
 std::vector<uint32_t> lru(batch*kLruSlots),destinations(batch*8192);
 std::vector<FinalPost> posts(batch*4);std::vector<BatchFlag> flags(batch*kFlagCount);
 Context ctx{uint64_t(valid.data()),uint64_t(changes.data()),uint64_t(lru.data()),
   uint64_t(destinations.data()),uint64_t(posts.data())};
 for(unsigned count:{0U,1U,127U,512U,2048U}){
  std::fill(changes.begin(),changes.end(),UINT32_MAX);
  for(uint32_t b=0;b<batch;++b){
   for(uint32_t i=0;i<kLruSlots;++i)lru[b*kLruSlots+i]=(i*7)%resident;
   for(uint32_t q=0;q<4;++q){
    auto& post=posts[b*4+q];post.count=(count+3-q)/4;post.descriptor=b*8192+q*2048;
    for(uint32_t i=0;i<post.count;++i){
     uint32_t ordinal=i*4+q,slot=lru[b*kLruSlots+ordinal];
     destinations[post.descriptor+i]=OwnerRow(&ctx,b)*stride+slot;
     // Final repair recorded in the reserved prefix; no speculative Get.
     changes[b*kChangeCapacity+ordinal]=resident+ordinal;
    }
   }
  }
  auto complete=[&]{for(uint32_t q=0;q<4;++q)for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
   PublishQueryCompletion(&ctx,flags.data(),batch,resident,0,q,0,1,0,count);};
  std::fill(valid.begin(),valid.end(),0);complete();auto compact=valid;
  std::fill(valid.begin(),valid.end(),0);
  for(uint32_t b=0;b<batch;++b)for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
   PublishPrefetchResidency(&ctx,b,resident);
  complete();assert(valid==compact);
  assert(std::accumulate(valid.begin(),valid.end(),0U)==count*batch);
 }
}
""")
    binary = tmp_path / "residency"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        str(program), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
