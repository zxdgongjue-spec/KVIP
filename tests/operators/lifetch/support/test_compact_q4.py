# SPDX-License-Identifier: Apache-2.0
"""Differential execution of the old Q-by-Q and new complete-Q4 resolver.

Four CPU lanes replace 512 SIMT lanes, retaining contiguous query partitions,
barrier order, atomics and native allocation/copy code. This checks logical
semantics, bounds and races between phases, not Ascend cache visibility.
"""

import subprocess

from test_recent_index import INDEXER, function


def test_compact_q4_matches_staged_resolver(tmp_path):
    kernel = INDEXER.parent
    old = function((kernel / "resolver_staged_dma2.h").read_text(), "inline void StagedQuery(")
    old = (old.replace("i+=512", "i+=kTestThreads")
           .replace("tid*4", "tid*(kTop/kTestThreads)")
           .replace("(tid+1)*4", "(tid+1)*(kTop/kTestThreads)")
           .replace("(words+511)/512", "(words+kTestThreads-1)/kTestThreads")
           .replace("544+512/warpSize-1", "544+kTestThreads/warpSize-1"))
    compact = (kernel / "resolver_compact_q4.h").read_text().replace(
        "constexpr uint32_t kCompactThreads=512;", "constexpr uint32_t kCompactThreads=kTestThreads;")
    hot = kernel.parent.parent / "lidu/op_kernel/hot_slot_selection.h"
    program = tmp_path / "compact.cpp"
    program.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(n)
#define KVIP_AUTO_ROUTE 1
constexpr uint32_t kTestThreads=4,warpSize=1,kTop=2048,kSlots=8192,kLruSlots=2048,kChangeCapacity=10240;
constexpr uint32_t kStagedRankTag=0x80000000U;
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
void asc_atomic_min(uint32_t* p,uint32_t value){auto old=__atomic_load_n(p,__ATOMIC_SEQ_CST);
 while(old>value&&!__atomic_compare_exchange_n(p,&old,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST)){} }
namespace AscendC {}
namespace Simt {uint32_t GetThreadIdx(){return threadIdx.x;}uint32_t GetThreadNum(){return kTestThreads;}}
''' + hot.read_text() + r'''
struct Context {uint64_t reservedHot,changes,changeSlots,changeCount,lruSlots,rows=0,destinations=0,sourceBlocks=0,sourceOffsets=0;};
struct Row {uint64_t addresses=0;};
using StagedContextPointer=Context*;
uint32_t OwnerRow(Context*,uint32_t b){return b;}
uint32_t OutputStride(uint32_t resident,Context*){return resident+512;}
uint32_t SparseHash(uint32_t token){return (token*0x9e3779b9U)>>19;}
uint32_t StagedPrefix(uint32_t local,uint32_t* ctl){
 ctl[544+threadIdx.x]=local;asc_syncthreads();
 if(!threadIdx.x)for(uint32_t i=1;i<kTestThreads;++i)ctl[544+i]+=ctl[543+i];
 asc_syncthreads();return threadIdx.x?ctl[543+threadIdx.x]:0;
}
''' + old + '\n' + compact + r'''
struct State {
 uint32_t resident,history,changeCount=0;
 std::vector<uint32_t> selected,loc,keys,ranks,slots,work,ctl,changes,changeSlots,src,dst,lru;
 std::vector<int32_t> reserved;std::vector<uint16_t> ids;Row row;Context ctx;
 State(uint32_t r,uint32_t h):resident(r),history(h),selected(kSlots,UINT32_MAX),loc(kSlots,UINT32_MAX),
  keys(kSlots,UINT32_MAX),ranks(kSlots,UINT32_MAX),slots(kSlots,UINT32_MAX),work(8192),ctl(8192),
  changes(kChangeCapacity,UINT32_MAX),changeSlots(kChangeCapacity,UINT32_MAX),
  src(kChangeCapacity,UINT32_MAX),dst(kChangeCapacity,UINT32_MAX),lru(kLruSlots),reserved(r,-1),ids(kTop){bind();}
 State(const State& x):resident(x.resident),history(x.history),changeCount(x.changeCount),selected(x.selected),loc(x.loc),
  keys(x.keys),ranks(x.ranks),slots(x.slots),work(x.work),ctl(x.ctl),changes(x.changes),changeSlots(x.changeSlots),
  src(x.src),dst(x.dst),lru(x.lru),reserved(x.reserved),ids(x.ids){bind();}
 void bind(){ctx={uint64_t(reserved.data()),uint64_t(changes.data()),uint64_t(changeSlots.data()),uint64_t(&changeCount),uint64_t(lru.data()),uint64_t(&row)};}
};
template<class F>void run(F f){std::vector<std::thread> workers;
 for(uint32_t t=0;t<kTestThreads;++t)workers.emplace_back([&,t]{threadIdx.x=t;f();});
 for(auto& worker:workers)worker.join();}
int main(){
 std::mt19937 rng(138);
 for(uint32_t mode=0;mode<24;++mode){
  uint32_t resident=mode==22?131072:16384,history=mode==23?(1U<<24):resident+16384;
  State base(resident,history);
  std::vector<uint32_t> shuffled(resident);std::iota(shuffled.begin(),shuffled.end(),0);
  std::shuffle(shuffled.begin(),shuffled.end(),rng);
  std::copy_n(shuffled.begin(),kLruSlots,base.lru.begin());
  for(uint32_t i=0;i<kLruSlots;++i){base.work[4096+i]=base.lru[i];base.reserved[base.lru[i]]=i;}
  std::vector<uint32_t> collisions;
  if(mode==23)for(uint32_t i=resident;i<history&&collisions.size()<kTop;++i)
   if((SparseHash(i)&8191)==7)collisions.push_back(i);
  if(mode==23)assert(collisions.size()>1024);
  for(uint32_t i=0;i<kSlots;++i){
   uint32_t q=i/kTop,pos=i%kTop,token=pos,slot=pos;
   if(mode==0){token=UINT32_MAX;slot=UINT32_MAX;}
   else if(mode==1){token=0;slot=0;}
   else if(mode==2){if(pos==0){token=resident;slot=UINT32_MAX;}}
   else if(mode==3){token=resident+i;slot=UINT32_MAX;}
   else if(mode==4){token=resident+pos;slot=UINT32_MAX;}
   else if(mode==5){if(q==3){token=resident+pos;slot=UINT32_MAX;}}
   else if(mode==6){token=(i%2)?base.lru[pos]:resident+pos;slot=token<resident?token:UINT32_MAX;}
   else if(mode==7){token=base.lru[pos];slot=UINT32_MAX;} // Invalid-hot cycles require original Index fallback.
   else if(mode==23){token=collisions[pos%collisions.size()];slot=UINT32_MAX;}
   else{
    bool miss=rng()%100<mode*4;
    token=miss?resident+rng()%8192:rng()%resident;slot=miss?UINT32_MAX:token;
    if(i%79==0){token=history-1;slot=resident+128;} // ABC hit.
    if(i%127==0){token=UINT32_MAX;slot=UINT32_MAX;}
   }
   base.selected[i]=token;base.loc[i]=slot;
   if(slot<resident)base.ctl[1024+slot/32]|=1U<<(slot%32);
  }
  uint32_t missing=0;for(uint32_t i=0;i<kSlots;++i)missing+=base.selected[i]!=UINT32_MAX&&base.loc[i]==UINT32_MAX;
  base.ctl[7]=missing<=1024?2048:missing<=2048?4096:8192;
  State oldState(base),newState(base);
  for(uint32_t q=0;q<4&&!oldState.ctl[1];++q){
   run([&]{StagedQuery(q,history,resident,656,0,0,&oldState.ctx,oldState.selected.data(),oldState.loc.data(),
    oldState.keys.data(),oldState.ranks.data(),oldState.ids.data(),oldState.slots.data(),oldState.work.data(),oldState.ctl.data());});
   uint32_t end=oldState.ctl[0],begin=end-oldState.ctl[2+q];
   for(uint32_t i=begin;i<end;++i){oldState.src[2048+i]=oldState.selected[oldState.ids[i%kTop]];oldState.dst[2048+i]=oldState.slots[i];}
  }
  run([&]{rotating::ResolveCompactQ4(&newState.ctx,history,resident,0,newState.selected.data(),newState.loc.data(),
   newState.keys.data(),newState.ranks.data(),newState.slots.data(),newState.work.data(),newState.ctl.data(),newState.src.data(),newState.dst.data());});
  if(oldState.ctl[1]||newState.ctl[1]){std::cerr<<"unexpected error mode="<<mode<<" old="<<oldState.ctl[1]<<" new="<<newState.ctl[1]<<"\n";return 1;}
  assert(oldState.selected==newState.selected);
  assert(oldState.loc==newState.loc);
  assert(oldState.src==newState.src&&oldState.dst==newState.dst);
  assert(oldState.changes==newState.changes&&oldState.changeSlots==newState.changeSlots);
  assert(oldState.changeCount==newState.changeCount);
  for(uint32_t i=0;i<6;++i)assert(oldState.ctl[i]==newState.ctl[i]);
  auto count=newState.ctl[0];std::unordered_set<uint32_t> sources,destinations;
  for(uint32_t i=0;i<count;++i){sources.insert(newState.src[2048+i]);destinations.insert(newState.dst[2048+i]);}
  assert(sources.size()==count&&destinations.size()==count);
 }
}
''')
    binary = tmp_path / "compact"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-pthread", "-fsanitize=address,undefined",
        str(program), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=120)


def test_compact_wiring_keeps_high_path_and_completion_contract():
    source = (INDEXER.parent / "streaming.cpp").read_text()
    producer = function(source, "inline uint32_t Producer(")
    assert '#include "low_miss_mode.h"' in source
    assert "compactPlan=NoPrefetchRound(ctx) && !row->slots" in producer
    assert producer.index("ctx->admissionControls),shape.batch*16)") < producer.index("compactPlan=NoPrefetchRound")
    assert producer.index("VF_CALL<ResolveCompactQ4>") < producer.index("DataCopy(output,loc,kSlots)")
    assert producer.index("DataCopy(output,loc,kSlots)") < producer.index("flags[published].ready=1")
    assert "if(!compactPlan){" in producer
    assert "VF_CALL<ResolveQueryWork>" in producer
    assert "if(profile)ticks[9]=1" in producer
    assert producer.index("flags+kOutputCompletionFlag") < producer.index("row->finalRequests=")
