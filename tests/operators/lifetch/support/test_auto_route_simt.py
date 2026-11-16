"""Execute the production 32-lane publisher with real CPU thread barriers."""

import subprocess

from test_auto_route import INCLUDE
from test_recent_index import function


def test_parallel_route_full_publication(tmp_path):
    source = tmp_path / "parallel-route.cpp"
    source.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdint>
#define __CCE_AICORE__ 1
#define __simt_vf__
#define __simt_callee__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
thread_local struct {uint32_t x;} threadIdx;
struct Barrier {
 std::mutex lock;std::condition_variable ready;unsigned arrived=0,generation=0;
 void wait(){std::unique_lock<std::mutex> held(lock);auto old=generation;
  if(++arrived==32){arrived=0;++generation;ready.notify_all();}
  else ready.wait(held,[&]{return generation!=old;});}
} barrier;
void asc_syncthreads(){barrier.wait();}
void asc_threadfence(){}
template<class T>T asc_ldcg(const T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
#include "auto_route_simt.h"
using namespace kvip_auto_route;
struct Context {
 uint64_t autoRoute,admissionControls,rowOwners,queryEnds;
 struct {uint32_t batch,history,tokenBytes;} shape;
};
int main(){
 for(unsigned batch:{1,2,7,8,9,16})for(unsigned fault=0;fault<5;++fault){
  std::array<Request,16> requests{};std::array<Feedback,128> feedback{};
  alignas(128) std::array<Pending,16> pending{};
  alignas(128) std::array<int32_t,64> controls{};
  std::array<uint32_t,16> owners{};std::array<uint32_t,64> ends{};
  Config config{1,128,uint64_t(requests.data()),uint64_t(feedback.data()),uint64_t(pending.data()),
   128,192,1,1,512,64,0,0,{0,0,0}};
  Context ctx{uint64_t(&config),uint64_t(controls.data()),uint64_t(owners.data()),uint64_t(ends.data()),
   {batch,UINT32_MAX,UINT32_MAX}};
  for(unsigned b=0;b<batch;++b){
   owners[b]=b+65;requests[b]={b+65,5,100,4};
   feedback[b+65]={5,99,80+b,4,0,UINT32_MAX,0,kValid,{}};
   for(unsigned q=0;q<4;++q)ends[b*4+q]=UINT32_MAX;
  }
  if(fault==1){requests[batch-1].generation=6;requests[batch-1].step=1;}
  if(fault==2){requests[batch-1].queries=0;for(unsigned q=0;q<4;++q)ends[(batch-1)*4+q]=0;}
  if(fault==3){feedback[65].necessary=8192;feedback[65].mode=1;}
  if(fault==4&&batch>1){owners[batch-1]=65;requests[batch-1].owner=65;}
  assert(Prepare(&ctx));auto expectedPending=pending;auto expectedControls=controls;auto oldFeedback=feedback;
  std::memset(pending.data(),0xa5,sizeof(pending));controls.fill(-13);
  for(unsigned row=batch;row--;){
   ParallelScratch scratch{};auto beforePending=pending;auto beforeControls=controls;
   std::vector<std::thread> threads;
   for(unsigned lane=0;lane<32;++lane)threads.emplace_back([&,lane]{threadIdx.x=lane;
    EvaluateParallel(&ctx,row,reinterpret_cast<uint32_t*>(&scratch));});
   for(auto& thread:threads)thread.join();
   assert(scratch.valid&&scratch.noPrefetch==(expectedPending[row].mode==0));
   assert(!std::memcmp(&pending[row],&expectedPending[row],sizeof(Pending)));
   for(unsigned j=0;j<16;++j)if(j!=row)
    assert(!std::memcmp(&pending[j],&beforePending[j],sizeof(Pending)));
   for(unsigned j=0;j<64;++j){bool writes=row%8==0&&j>=row*4&&j<std::min(batch,row+8)*4;
    assert(controls[j]==(writes?expectedControls[j]:beforeControls[j]));}
   assert(!std::memcmp(feedback.data(),oldFeedback.data(),sizeof(feedback)));
  }
  assert(!std::memcmp(pending.data(),expectedPending.data(),batch*sizeof(Pending)));
  assert(std::equal(controls.begin(),controls.begin()+batch*4,expectedControls.begin()));
 }
}
''')
    binary = tmp_path / "parallel-route"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-pthread", "-fsanitize=address,undefined",
        "-I", str(INCLUDE), str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_parallel_route_keeps_scalar_cache_visibility():
    source = (INCLUDE.parent / "op_kernel/lifecycle.cpp").read_text()
    init = function(source, 'void v9_init_kernel(')
    check = init.index("parallelRoute=kvip_auto_route::CanPublishRows(ctx)")
    assert init.index("reinterpret_cast<__gm__ uint8_t*>(ctx->autoRoute),128") < check
    row_load = init.index("uint32_t owner=ctx->rowOwners ?")
    refresh = init.index("reinterpret_cast<__gm__ uint8_t*>(ctx->rowOwners)+b*sizeof(uint32_t),sizeof(uint32_t)")
    assert init.index("InitializeRound(") < refresh < row_load
