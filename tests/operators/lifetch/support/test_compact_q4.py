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
    low = (kernel / "resolver_low_q4.h").read_text()
    # Each lane owns distinct control/bitmap words; the sole barrier is at exit.
    merge = function((kernel / "streaming.cpp").read_text(), "inline void MergeLowQ4Protection(")
    merge = merge.replace("asc_syncthreads();", "")
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
#include <unordered_map>
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
''' + old + '\n' + compact + '\n' + low + '\n' + merge + r'''
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
 // Initialization failures must propagate before any stale scratch access.
 // Null payloads make an accidental hash/location read fail under ASAN.
 for(uint32_t error:{1U,8U,0x80000000U}){
  std::vector<uint32_t> ctl(8192,0xdeadbeefU);ctl[1]=error;
  auto rejected=ctl;
  run([&]{rotating::ResolveLowQ4(nullptr,64,32,0,
   nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,ctl.data(),nullptr,nullptr);});
  assert(ctl==rejected);
 }
 for(uint32_t mode=0;mode<28;++mode){
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
   else if(mode==24){token=i%2?resident+i/2:pos/2;slot=UINT32_MAX;}
   else if(mode==25){token=i;slot=UINT32_MAX;} // 8192 unique self-refills.
   else if(mode==26){token=i+1==kSlots?17:resident+i;slot=UINT32_MAX;}
   else if(mode==27){token=i<4096?resident+i:i-4096;slot=UINT32_MAX;}
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

  // The post-then-Index plan retains every selected resident position. Even
  // invalid resident rows refill themselves; only cold IDs consume victims.
  State post(base);
  std::unordered_set<uint32_t> protectedRows;
  for(auto token:post.selected)if(token<resident){
   post.ctl[1024+token/32]|=1U<<(token%32);protectedRows.insert(token);
  }
  for(auto slot:post.loc)if(slot<resident)protectedRows.insert(slot);
  auto expectedLoc=post.loc;
  std::vector<uint32_t> expectedSource,expectedDest,expectedCounts(4),available;
  for(auto slot:post.lru)if(!protectedRows.count(slot))available.push_back(slot);
  std::unordered_set<uint32_t> listed(available.begin(),available.end());
  for(uint32_t slot=0;slot<resident;++slot)
   if(!protectedRows.count(slot)&&!listed.count(slot))available.push_back(slot);
  uint32_t victim=0;std::unordered_map<uint32_t,uint32_t> assigned;
  for(uint32_t q=0;q<4;++q){
   auto before=expectedSource.size();
   for(uint32_t i=q*kTop;i<(q+1)*kTop;++i){
    uint32_t token=post.selected[i];
    if(token==UINT32_MAX||expectedLoc[i]!=UINT32_MAX)continue;
    if(!assigned.count(token)){
     uint32_t slot=token<resident?token:available.at(victim++);
     assigned[token]=slot;expectedSource.push_back(token);expectedDest.push_back(slot);
    }
    expectedLoc[i]=assigned.at(token);
   }
   expectedCounts[q]=expectedSource.size()-before;
  }
  post.ctl[9]=1; // The old proof rejects invalid hot sources; the new one repairs them.
  State cached(post);
  const uint32_t words=(resident+31)/32;
  auto immutableBitmap=cached.ctl;
  std::vector<uint32_t> other(3*words);
  for(uint32_t i=0;i<words;++i){
   uint32_t bits=immutableBitmap[1024+i];
   cached.ctl[1024+i]=bits&0x11111111U;
   for(uint32_t q=0;q<3;++q)other[q*words+i]=bits&(0x22222222U<<q);
  }
  std::fill_n(cached.ctl.begin(),10,0xdeadbeefU);
  for(uint32_t lane=0;lane<512;++lane){
   threadIdx.x=lane;
   MergeLowQ4Protection(cached.ctl.data(),other.data(),words);
  }
  assert(cached.ctl[7]==kSlots);
  assert(std::equal(immutableBitmap.begin()+1024,immutableBitmap.begin()+1024+words,cached.ctl.begin()+1024));
  run([&]{rotating::ResolveCompactQ4<true>(&post.ctx,history,resident,0,post.selected.data(),post.loc.data(),
   post.keys.data(),post.ranks.data(),post.slots.data(),post.work.data(),post.ctl.data(),post.src.data(),post.dst.data());});
  run([&]{rotating::ResolveLowQ4(&cached.ctx,history,resident,0,cached.selected.data(),cached.loc.data(),
   cached.keys.data(),cached.ranks.data(),cached.slots.data(),cached.work.data(),cached.ctl.data(),cached.src.data(),cached.dst.data());});
  assert(cached.selected==post.selected&&cached.loc==post.loc);
  assert(cached.src==post.src&&cached.dst==post.dst);
  assert(cached.changes==post.changes&&cached.changeSlots==post.changeSlots);
  assert(cached.changeCount==post.changeCount);
  for(uint32_t i=0;i<10;++i)if(i!=7)assert(cached.ctl[i]==post.ctl[i]);
  assert(!post.ctl[1]&&!post.ctl[9]);assert(post.ctl[0]==expectedSource.size());
  assert(post.loc==expectedLoc);
  assert(std::equal(expectedSource.begin(),expectedSource.end(),post.src.begin()+2048));
  assert(std::equal(expectedDest.begin(),expectedDest.end(),post.dst.begin()+2048));
  assert(std::equal(expectedCounts.begin(),expectedCounts.end(),post.ctl.begin()+2));
  assert(std::unordered_set<uint32_t>(expectedSource.begin(),expectedSource.end()).size()==expectedSource.size());
  assert(std::unordered_set<uint32_t>(expectedDest.begin(),expectedDest.end()).size()==expectedDest.size());
  for(uint32_t i=0;i<expectedSource.size();++i)
   assert(expectedSource[i]==expectedDest[i]||(expectedSource[i]>=resident&&expectedDest[i]<resident));
 }
 // Marker-shaped corrupt input locations must fail before they can be
 // interpreted as handles and cause an out-of-range event publication.
 for(uint32_t bad:{0x40000000U,0x80000000U,0xc0001fffU}){
  State invalid(16384,131328);invalid.selected[0]=7;invalid.loc[0]=bad;invalid.ctl[7]=2048;
  run([&]{rotating::ResolveLowQ4(&invalid.ctx,invalid.history,invalid.resident,0,
   invalid.selected.data(),invalid.loc.data(),invalid.keys.data(),invalid.ranks.data(),
   invalid.slots.data(),invalid.work.data(),invalid.ctl.data(),invalid.src.data(),invalid.dst.data());});
  assert(invalid.ctl[1]==1);
  assert(std::all_of(invalid.src.begin(),invalid.src.end(),[](auto v){return v==UINT32_MAX;}));
  assert(std::all_of(invalid.dst.begin(),invalid.dst.end(),[](auto v){return v==UINT32_MAX;}));
 }
 // The optimized lookup must retain the old failure decision for invalid
 // tokens and an exhausted hash. No failed plan publishes destinations.
 for(uint32_t fault=0;fault<2;++fault){
  State initial(16384,131328);initial.ctl[7]=fault?2:2048;
  for(uint32_t i=0;i<kLruSlots;++i)initial.work[4096+i]=i;
  initial.selected[0]=fault?16384:initial.history;
  if(fault){initial.selected[1]=16385;initial.selected[2]=16386;}
  State original(initial),optimized(initial);
  run([&]{rotating::ResolveCompactQ4<true>(&original.ctx,original.history,original.resident,0,
   original.selected.data(),original.loc.data(),original.keys.data(),original.ranks.data(),
   original.slots.data(),original.work.data(),original.ctl.data(),original.src.data(),original.dst.data());});
  run([&]{rotating::ResolveLowQ4(&optimized.ctx,optimized.history,optimized.resident,0,
   optimized.selected.data(),optimized.loc.data(),optimized.keys.data(),optimized.ranks.data(),
   optimized.slots.data(),optimized.work.data(),optimized.ctl.data(),optimized.src.data(),optimized.dst.data());});
  assert(original.ctl[1]==optimized.ctl[1]&&optimized.ctl[1]==(fault?8U:1U));
  assert(original.dst==optimized.dst);
  assert(std::all_of(optimized.dst.begin(),optimized.dst.end(),[](auto v){return v==UINT32_MAX;}));
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
    assert "compactPlan=PostIndex || (KVIP_LOW_MISS_COMPACT_PLAN && NoPrefetchRound(ctx) && !row->slots)" in producer
    assert producer.index("ctx->admissionControls),shape.batch*16)") < producer.index("compactPlan=PostIndex")
    assert producer.index("VF_CALL<ResolveLowQ4>") < producer.index("DataCopy(output,loc,kSlots)")
    assert "VF_CALL<ResolveCompactQ4<false>>" in producer
    assert "if constexpr (PostIndex)" in producer
    assert producer.index("DataCopy(output,loc,kSlots)") < producer.index("flags[published].ready=1")
    assert "if(!compactPlan){" in producer
    assert "VF_CALL<ResolveQueryWork>" in producer
    assert "if(profile)ticks[9]=1" in producer
    # The high path still joins OutputWorker before final publication. The
    # PostIndex specialization now publishes once and returns before that
    # legacy query loop, so inspect the last (high-path) assignment here.
    assert producer.index("flags+kOutputCompletionFlag") < producer.rindex("row->finalRequests=")
