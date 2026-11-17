# SPDX-License-Identifier: Apache-2.0
"""Compare actual old/new completion functions, including feedback bytes."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_recent_index import function


REPO = Path(__file__).resolve().parents[4]


def test_post_index_completion_matches_old_state(tmp_path):
    kernel=REPO/'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    source=(kernel/'streaming.cpp').read_text()
    context=(kernel.parent/'include/context.h').read_text()
    structures=context[context.index('struct alignas(128) FinalPost'):context.index('struct alignas(128) Event')]
    old_residency=function(source,'inline void PublishStreamingResidency(')
    old_finalize=function(source,'inline void FinalizeStreaming(')
    kvaf = Path(os.environ.get("KVAF_LEGACY_SOURCE_ROOT", REPO.parent / "KVAF"))
    native = kvaf / "access_operator/demo/aiv_batch_get/include"
    if not (native / "kvaf_batch_get_device.h").is_file():
        pytest.skip("Set KVAF_LEGACY_SOURCE_ROOT to the original fused-context KVAF checkout")
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler
    write_status=function((native/'kvaf_batch_get_device.h').read_text(),'inline void WriteStatus(')
    program=tmp_path/'finalize.cpp'
    program.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(n)
#define KVIP_AUTO_ROUTE 1
#define KVIP_LOW_MISS_TAIL 1
#define KVIP_EARLY_PREFETCH_DRAIN 0
#define V9_UBMEM 0
constexpr uint32_t kMaxRounds=32,kRowStatusWords=32,kChangeCapacity=10240,kLruSlots=2048,kFlagCount=16,kPrefetch=2048;
thread_local struct {uint32_t x=0;} threadIdx;
void asc_threadfence(){}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
namespace Simt {
struct Dim3{uint32_t x,y,z;};
template<auto F,class...A>void VF_CALL(Dim3 n,A...a){for(uint32_t t=0;t<n.x;++t){threadIdx.x=t;F(a...);}}
}
enum class HardEvent{V_S};constexpr unsigned EVENT_ID0=0;
template<HardEvent>void SetFlag(unsigned){}
template<HardEvent>void WaitFlag(unsigned){}
uint64_t GetSystemCycle(){return 1;}
'''+'#include "'+str(native/'batch_get_abi.h')+'"\n'+
    '#include "'+str(kernel.parent/'include/auto_route.h')+'"\n'+structures+r'''
struct Shape {uint32_t batch=16,resident=16384,finalWorkers=4,finalPostOffset=0,tokenBytes=656;};
namespace rotating {
struct Context {
 Shape shape;
 uint64_t rows,owners,rowStatus,finalStatus,hotValid,rowOwners,outputStride,
          destinations,changes,lruSlots,autoRoute;
};
}
using rotating::Context;
struct Args{uint64_t context,flags,times,updateBindings,eventsDst;uint32_t prefetchCount,maxEvents;};
struct BatchFlag{uint32_t ready=1,count=0,error=0,reserved[29]={};};
Shape LoadRuntimeShape(Context* c){return c->shape;}
uint32_t OwnerRow(Context* c,uint32_t b){return c->rowOwners?reinterpret_cast<uint32_t*>(c->rowOwners)[b]:b;}
uint32_t OutputStride(uint32_t r,Context* c){return c->outputStride?c->outputStride:r;}
FinalPost* FinalPostAt(uint32_t o,Context* c,uint32_t i){return reinterpret_cast<FinalPost*>(c->owners+o)+i;}
namespace kvaf_device {
constexpr uint32_t kInvalidArgs=1;constexpr uint64_t kCqTimeout=8;
void CleanCacheLines(uint8_t*,uint64_t){}
'''+write_status+r'''
}
void CompletePrefetch(Context*,uint32_t,bool,uint64_t*,uint64_t*,uint64_t*){assert(false);}
void PublishPrefetchResidency(Context*,uint32_t,uint32_t){assert(false);}
KvafExecuteBatchGetKernelArgs GetArgs(Context*,uint32_t){return {};}
'''+old_residency+'\n'+(kernel/'index_cache_update.h').read_text()+
    '\ntemplate<bool ReadThrough=false,bool PostIndex=false>\n'+old_finalize+'\n'+
    '#include "'+str(kernel/'post_index_finalize.h')+'"\n'+r'''
struct Fixture {
 static constexpr uint32_t B=16,O=19,R=16384,E=10240;
 std::array<Row,B> rows{};std::array<FinalPost,B*4> posts{};
 std::array<BatchFlag,B*16> flags{};std::array<uint32_t,B> owners{},updateStatus{};
 std::array<uint32_t,B*32> info{};std::array<KvafBatchGetCoreStatus,B*2> status{};
 std::array<uint64_t,28> bindings{};std::array<uint64_t,B*512> times{};
 std::vector<uint32_t> hot,events,lru;std::vector<int32_t> destinations,changes;
 std::array<kvip_auto_route::Request,B> requests{};
 std::array<kvip_auto_route::Pending,B> pending{};
 std::array<kvip_auto_route::Feedback,O> feedback{};
 kvip_auto_route::Config config{};Context ctx{};Args args{};
 Fixture():hot(O*R),events(B*E),lru(B*2048),destinations(B*8192,-1),changes(B*E,-1){
  for(uint32_t i=0;i<hot.size();++i)hot[i]=(i%5)!=0;
  for(uint32_t b=0;b<B;++b){owners[b]=(b*7)%O;rows[b].finalPending=1;
   for(uint32_t q=0;q<4;++q){auto& p=posts[b*4+q];p.descriptor=b*8192+q*2048;p.deferred=1;p.head=99;p.nonconflicting=17;}
   for(uint32_t j=0;j<2048;++j)lru[b*2048+j]=4096+j;
   pending[b]={owners[b],1,2,1,0,0,4,131328*4,656,91,73,1,{0,0,0,0}};
   requests[b]={owners[b],1,2,4};
   auto& f=feedback[owners[b]];f.generation=1;f.step=1;f.valid=kvip_auto_route::kValid;f.necessary=50;
   std::fill(std::begin(f.reserved),std::end(f.reserved),123);
  }
  std::memset(status.data(),0xa5,sizeof(status));bind();
 }
 Fixture(const Fixture& f)=default;
 void bind(){
  config={1,O,uint64_t(requests.data()),uint64_t(feedback.data()),uint64_t(pending.data()),128,192,1,1,512,64,0,0,{0,0,0}};
  ctx={{B,R,4,0,656},uint64_t(rows.data()),uint64_t(posts.data()),uint64_t(info.data()),uint64_t(status.data()),
       uint64_t(hot.data()),uint64_t(owners.data()),R+512,uint64_t(destinations.data()),uint64_t(changes.data()),uint64_t(lru.data()),uint64_t(&config)};
  bindings[6]=uint64_t(updateStatus.data());bindings[9]=uint64_t(owners.data());bindings[16]=uint64_t(hot.data());bindings[25]=R;
  args={uint64_t(&ctx),uint64_t(flags.data()),uint64_t(times.data()),uint64_t(bindings.data()),uint64_t(events.data()),0,E};
 }
 void plan(uint32_t b,uint32_t n){
  // Model the existing init + exact Producer before every invocation.
  std::fill(changes.begin()+b*E,changes.begin()+(b+1)*E,-1);
  info[b*32]=0;rows[b].error=0;rows[b].pending=0;rows[b].finalPending=1;
  uint32_t rank=0;
  for(uint32_t q=0;q<4;++q){auto& p=posts[b*4+q];p.count=n/4+(q<n%4);
   for(uint32_t i=0;i<p.count;++i,++rank){
    const uint32_t dst=rank<7?4000+rank:4096+rank-7;
    events[b*E+2048+rank]=dst;
    destinations[p.descriptor+i]=owners[b]*(R+512)+dst;
    if(dst>=4096&&dst<6144)changes[b*E+dst-4096]=1234+rank;
    hot[owners[b]*R+dst]=0;
   }
  }
  info[b*32+1]=n;info[b*32+3]=99;info[b*32+4]=77;
 }
 void oldFinish(uint32_t b,uint64_t error,uint32_t completed){
  if(!error&&completed!=info[b*32+1])error=kvaf_device::kInvalidArgs;
  uint32_t code=uint32_t(error)?uint32_t(error):(error?kvaf_device::kInvalidArgs:0U);
  updateStatus[b]=code;
  for(uint32_t q=0;q<4;++q){auto& p=posts[b*4+q];p.pending=error?0:p.count;p.deferred=0;p.error=code;}
  if(!error&&(args.prefetchCount&0x10000000U))
   Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},R,0,&ctx,b);
  FinalizeStreaming<false,true>(&args,b,0);
 }
 void newFinish(uint32_t b,uint64_t error,uint32_t completed){
  if(!error&&completed!=info[b*32+1])error=kvaf_device::kInvalidArgs;
  uint32_t total=0;for(uint32_t q=0;q<4;++q)total+=posts[b*4+q].count;
  pipeline_ab::post_index_finalize::Finish(&args,b,total,error);
 }
 void compare(const Fixture& other)const{
  assert(!std::memcmp(rows.data(),other.rows.data(),sizeof(rows)));
  assert(!std::memcmp(posts.data(),other.posts.data(),sizeof(posts)));
  assert(!std::memcmp(status.data(),other.status.data(),sizeof(status)));
  assert(info==other.info&&hot==other.hot&&updateStatus==other.updateStatus);
  assert(events==other.events&&destinations==other.destinations&&changes==other.changes);
  assert(!std::memcmp(pending.data(),other.pending.data(),sizeof(pending)));
  assert(!std::memcmp(feedback.data(),other.feedback.data(),sizeof(feedback)));
 }
};
int main(){
 uint32_t cases=0;
 for(uint32_t parallel:{0U,1U})for(uint32_t n:{0U,7U,60U,955U,8192U})for(uint32_t failure=0;failure<13;++failure){
  Fixture base;constexpr uint32_t b=3;base.plan(b,n);base.args.prefetchCount=parallel?0x10000000U:0;
  uint64_t error=0;uint32_t completed=n;
  if(failure==1)error=16;
  if(failure==2)error=1ULL<<61;
  if(failure==3)completed=n? n-1:1;
  if(failure==4)base.rows[b].error=32;
  if(failure==5)base.info[b*32]=2;
  if(failure==6)base.rows[b].pending=1;
  if(failure==7)base.pending[b].generation=2; // new request generation
  if(failure==8)base.pending[b].step=1; // duplicate old feedback step
  if(failure==9)base.pending[b].learnable=0;
  if(failure==10){base.info[b*32+1]+=1;completed+=1;} // late raw count mismatch
  if(failure==11){base.rows[b].error=8;error=16;completed=0;}
  if(failure==12){base.bindings[25]=Fixture::R/2;error=16;} // old binding-defined invalidation span
  Fixture old(base),candidate(base);old.bind();candidate.bind();
  // Rebinding addresses must not reset configuration choices.
  old.args.prefetchCount=candidate.args.prefetchCount=base.args.prefetchCount;
  old.bindings[25]=candidate.bindings[25]=base.bindings[25];
  old.oldFinish(b,error,completed);candidate.newFinish(b,error,completed);old.compare(candidate);
  ++cases;
 }
 // Real repeated invocation: success -> new generation -> failure -> retry.
 Fixture old,candidate;constexpr uint32_t b=1;
 for(uint32_t step=0;step<4;++step){
  uint32_t n=step?7:60;
  old.plan(b,n);candidate.plan(b,n);
  old.pending[b].generation=candidate.pending[b].generation=step?2:1;
  old.pending[b].step=candidate.pending[b].step=step+2;
  uint64_t error=step==2?16:0;
  old.oldFinish(b,error,n);candidate.newFinish(b,error,n);old.compare(candidate);
 }
 std::cout<<"FINALIZE_BYTE_ORACLE_PASS cases="<<cases<<" sequence=4\n";
}
''')
    binary=tmp_path/'finalize'
    subprocess.run([compiler,'-std=c++17','-O1','-g','-fsanitize=address,undefined',str(program),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=60)
