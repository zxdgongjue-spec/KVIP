# SPDX-License-Identifier: Apache-2.0
"""Execute read-through's real dedup/descriptor code with four CPU SIMT lanes.

This validates phase ordering, query dedup, ABC precedence and bounds. It does
not emulate Ascend cache coherency or establish NPU performance.
"""

import shutil
import subprocess

from test_recent_index import INDEXER, function


def test_read_through_exact_descriptors_and_persistent_state(tmp_path):
    source = (INDEXER.parent / "read_through.h").read_text()
    functions = "\n".join(function(source, name) for name in (
        "inline void Initialize(", "inline uint32_t Find(", "inline void Query(",
    ))
    functions = (functions.replace("i += 512", "i += kTestThreads")
                 .replace("tid * 4", "tid * (kTop / kTestThreads)")
                 .replace("(tid + 1) * 4", "(tid + 1) * (kTop / kTestThreads)")
                 .replace("512 / warpSize", "kTestThreads / warpSize"))
    cpp = tmp_path / "read_through.cpp"
    cpp.write_text(r'''
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#define __ubuf__
#define __gm__
constexpr uint32_t kTestThreads=4,warpSize=1,kTop=2048,kSlots=8192,kBuckets=16384;
constexpr uint32_t kStagedRankTag=0x80000000U,kOutputMasksOffset=32,kOutputMasksPerQuery=8;
thread_local struct ThreadIndex {uint32_t x=0;} threadIdx;
class Barrier {
 std::mutex mutex;std::condition_variable ready;uint32_t arrived=0,generation=0;
public:
 void wait(){std::unique_lock<std::mutex> lock(mutex);auto old=generation;
  if(++arrived==kTestThreads){arrived=0;++generation;ready.notify_all();}
  else ready.wait(lock,[&]{return generation!=old;});}
} barrier;
void asc_syncthreads(){barrier.wait();}
void asc_threadfence(){__atomic_thread_fence(__ATOMIC_SEQ_CST);}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t old,uint32_t value){
 __atomic_compare_exchange_n(p,&old,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return old;}
void asc_atomic_or(uint32_t* p,uint32_t value){__atomic_fetch_or(p,value,__ATOMIC_SEQ_CST);}
uint32_t asc_atomic_add(uint32_t* p,uint32_t value){return __atomic_fetch_add(p,value,__ATOMIC_SEQ_CST);}
void asc_atomic_min(uint32_t* p,uint32_t value){auto old=__atomic_load_n(p,__ATOMIC_SEQ_CST);
 while(old>value&&!__atomic_compare_exchange_n(p,&old,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST)){} }
struct Context {uint64_t hotValid,tailBounds,rows,sourceBlocks,sourceOffsets,destinations,count,fallbackAddresses;};
struct Row {uint64_t addresses;};
constexpr uint32_t history=131328,resident=16384,originalStride=16896,extendedStride=originalStride+kSlots;
uint32_t OwnerRow(Context*,uint32_t){return 2;}
uint32_t OutputStride(uint32_t,Context*){return extendedStride;}
uint32_t QueryEnd(uint32_t,Context*,uint32_t){return history;}
uint32_t SparseHash(uint32_t token){return token*0x9e3779b9U;}
uint32_t StagedPrefix(uint32_t local,uint32_t* ctl){
 ctl[544+threadIdx.x]=local;asc_syncthreads();
 if(!threadIdx.x)for(uint32_t i=1;i<kTestThreads;++i)ctl[544+i]+=ctl[543+i];
 asc_syncthreads();return threadIdx.x?ctl[543+threadIdx.x]:0;
}
void StoreFallbackSource(uint32_t,uint32_t,Context* ctx,uint32_t,uint32_t out,uint32_t,
 uint64_t address,int32_t* blocks,int32_t* offsets){
 reinterpret_cast<uint64_t*>(ctx->fallbackAddresses)[out]=address;blocks[out]=out;offsets[out]=0;
}
void PrepareStagedQueryOutputs(int64_t* logical,uint32_t*,uint32_t,uint32_t,uint32_t q,
 uint32_t* selected,uint32_t*,uint32_t* output){
 for(uint32_t i=threadIdx.x;i<kTop;i+=kTestThreads)
  output[i]=selected[q*kTop+i]==UINT32_MAX?UINT32_MAX:uint32_t(logical[selected[q*kTop+i]]);
 asc_syncthreads();
}
''' + functions + r'''
template<class F>void run(F f){std::vector<std::thread> workers;
 for(uint32_t t=0;t<kTestThreads;++t)workers.emplace_back([&,t]{threadIdx.x=t;f();});
 for(auto& worker:workers)worker.join();}
int main(){
 std::mt19937 rng(417);
 for(uint32_t mode=0;mode<22;++mode){
  std::vector<uint32_t> selected(kSlots),loc(kSlots,UINT32_MAX),keys(kBuckets),ranks(kBuckets),work(kTop+64),ctl(1024);
  std::vector<uint32_t> hot(3*resident,1),bounds{0,0,130048,130432};
  std::vector<int32_t> blocks(3*kSlots,-1),offsets(3*kSlots,-1),dest(3*kSlots,-1);
  std::vector<uint64_t> addresses(history),fallback(3*kSlots);
  std::vector<int64_t> logical(history);
  for(uint32_t i=0;i<history;++i){addresses[i]=0x100000000ULL+uint64_t(i)*656;logical[i]=i^1;}
  for(uint32_t i=0;i<resident;i+=127)hot[2*resident+i]=0;
  if(mode==6){bounds[2]=512;bounds[3]=896;} // ABC takes precedence inside resident.
  if(mode==7)bounds[2]=bounds[3]=0;
  if(mode==21){bounds[2]=512;bounds[3]=1280;} // Preserve capacity-clipped ABC semantics.
  const auto savedHot=hot;const auto savedLogical=logical;const auto savedAddresses=addresses;
  std::vector<Row> rows(2);rows[1].addresses=uint64_t(addresses.data());uint32_t submitted=0;
  Context ctx{uint64_t(hot.data()),uint64_t(bounds.data()),uint64_t(rows.data()),uint64_t(blocks.data()),
   uint64_t(offsets.data()),uint64_t(dest.data()),uint64_t(&submitted),uint64_t(fallback.data())};
  for(uint32_t i=0;i<kSlots;++i){uint32_t q=i/kTop,p=i%kTop;
   if(mode==0)selected[i]=UINT32_MAX;
   else if(mode==1)selected[i]=resident+i; // False-low: full 8192 unique misses.
   else if(mode==2)selected[i]=resident+p; // Repeated Q4 set: one Get per unique ID.
   else if(mode==3)selected[i]=q==3?resident+p:p+1;
   else if(mode==4)selected[i]=p+1;
   else if(mode==5)selected[i]=bounds[2]+p%384;
   else if(mode==6)selected[i]=512+p%384;
   else{selected[i]=rng()%100<(mode-7)*10?resident+rng()%8192:rng()%resident;
    if(i%79==0)selected[i]=130048+i%384;
    if(i%127==0)selected[i]=UINT32_MAX;}
  }
  if(mode==18)selected[0]=history;
  if(mode==19)selected[3*kTop+7]=history;
  if(mode==20)for(uint32_t i=0;i<kSlots;++i)selected[i]=resident+(i%6)*16384;
  if(mode==21)for(uint32_t i=0;i<kSlots;++i)selected[i]=512+i%768;
  run([&]{Initialize(keys.data(),ranks.data(),ctl.data());});
  std::unordered_map<uint32_t,uint32_t> cold;uint32_t previous=0;
  for(uint32_t q=0;q<4;++q){
   run([&]{Query(&ctx,history,resident,656,2,1,q,originalStride,512,logical.data(),selected.data(),
    loc.data(),keys.data(),ranks.data(),work.data(),ctl.data());});
   if((mode==18&&q==0)||(mode==19&&q==3)){
    assert(ctl[1]&1U);assert(submitted==previous);break;
   }
   assert(!ctl[1]);
   for(uint32_t p=0;p<kTop;++p){uint32_t i=q*kTop+p,t=selected[i],slot=loc[i];
    if(t==UINT32_MAX){assert(slot==UINT32_MAX&&work[64+p]==UINT32_MAX);continue;}
    assert(work[64+p]==uint32_t(logical[t]));
    bool inTail=t>=bounds[2]&&t<bounds[3];
    if(inTail&&t-bounds[2]<512){assert(slot==resident+t%512);continue;}
    if(t<resident&&!inTail&&hot[2*resident+t]){assert(slot==t);continue;}
    assert(slot>=originalStride&&slot<extendedStride);
    auto inserted=cold.emplace(t,slot);
    assert(inserted.first->second==slot);
   }
   assert(ctl[0]==cold.size()&&submitted==cold.size()&&ctl[2+q]==cold.size()-previous);
   for(uint32_t j=0;j<ctl[2+q];++j){uint32_t out=kSlots+q*kTop+j;
    assert(blocks[out]==int32_t(out)&&offsets[out]==0);
    uint32_t token=uint32_t((fallback[out]-0x100000000ULL)/656);
    assert(cold.count(token)&&dest[out]==int32_t(2*extendedStride+cold.at(token)));
   }
   previous=cold.size();
  }
  if(mode==1)assert(submitted==kSlots);
  if(mode==2)assert(submitted==kTop);
  assert(hot==savedHot&&logical==savedLogical&&addresses==savedAddresses);
 }
}
''')
    binary = tmp_path / "read_through"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined", "-pthread",
        str(cpp), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_read_through_simt_address_spaces(tmp_path):
    source = (INDEXER.parent / "read_through.h").read_text()
    functions = "\n".join(function(source, name) for name in (
        "inline void Initialize(", "inline uint32_t Find(", "inline void Query(",
    ))
    cpp = tmp_path / "read_through_address_spaces.cpp"
    cpp.write_text(r'''
#include <cstdint>
#define __gm__ __attribute__((address_space(1)))
#define __ubuf__ __attribute__((address_space(3)))
constexpr uint32_t kTop=2048,kSlots=8192,kBuckets=16384,warpSize=32;
constexpr uint32_t kStagedRankTag=0x80000000U,kOutputMasksOffset=32,kOutputMasksPerQuery=8;
struct {uint32_t x;} threadIdx;
struct Context {uint64_t hotValid,tailBounds,rows,sourceBlocks,sourceOffsets,destinations,count;};
struct Row {uint64_t addresses;};
void asc_syncthreads();void asc_threadfence();
template<class T>T asc_ldcg(__gm__ T*);
template<class T>void asc_stcg(__gm__ T*,T);
uint32_t asc_atomic_cas(__ubuf__ uint32_t*,uint32_t,uint32_t);
void asc_atomic_min(__ubuf__ uint32_t*,uint32_t);
void asc_atomic_or(__ubuf__ uint32_t*,uint32_t);
uint32_t asc_atomic_add(__gm__ uint32_t*,uint32_t);
uint32_t OwnerRow(__gm__ Context*,uint32_t);
uint32_t OutputStride(uint32_t,__gm__ Context*);
uint32_t QueryEnd(uint32_t,__gm__ Context*,uint32_t);
uint32_t SparseHash(uint32_t);
uint32_t StagedPrefix(uint32_t,__ubuf__ uint32_t*);
void StoreFallbackSource(uint32_t,uint32_t,__gm__ Context*,uint32_t,uint32_t,uint32_t,uint64_t,
                         __gm__ int32_t*,__gm__ int32_t*);
void PrepareStagedQueryOutputs(__gm__ int64_t*,__ubuf__ uint32_t*,uint32_t,uint32_t,uint32_t,
                               __ubuf__ uint32_t*,__ubuf__ uint32_t*,__ubuf__ uint32_t*);
''' + functions)
    subprocess.run([
        shutil.which("clang++") or "c++", "-std=c++17", "-fsyntax-only", "-Werror", str(cpp),
    ], check=True)
