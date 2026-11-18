# SPDX-License-Identifier: Apache-2.0
"""Exercise actual completion/join routing with delayed publication and errors.

The native transport is mocked; this checks ownership and publication ordering,
not Ascend cache behavior or NPU latency. ASan/UBSan cover the retained scratch
and diagnostic offsets using the production Row layout.
"""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"


@pytest.mark.parametrize("early", [0, 1])
def test_single_cq_owner_completion_error_and_diagnostic_lines(tmp_path, early):
    source = (KERNEL / "streaming.cpp").read_text()
    consumer = source.split("__aicore__ inline void Consumer", 1)[1]
    consumer = consumer.split(" TPipe pipe;", 1)[0]
    finalize = source.split("__aicore__ inline void FinalizeStreaming", 1)[1]
    finalize = finalize.split(" // Post/Index has already joined", 1)[0]
    context = (KERNEL.parent / "include/context.h").read_text()
    row = "struct alignas(128) Row" + context.split("struct alignas(128) Row", 1)[1]
    row = row.split("struct alignas(128) Event", 1)[0]
    program = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#define __aicore__
#define __gm__
#define V9_UBMEM 0
constexpr uint32_t kMaxRounds=32,kFlagCount=16;
/*ROW*/
static_assert(sizeof(Row)==1024);
static_assert(offsetof(Row,error)/128!=offsetof(Row,finalRequests)/128);
static_assert(offsetof(Row,pending)/128!=offsetof(Row,finalRequests)/128);
struct alignas(128) BatchFlag {uint32_t ready=0,count=0,error=0,reserved[29]{};};
struct Context {uint64_t rows;};
struct Args {uint64_t context,flags,times,workerStatus;uint32_t prefetchCount;};
struct Shape {uint32_t batch=16;};
Shape LoadRuntimeShape(Context*){return {};}
uint64_t now=0,errorToReturn=0;
uint32_t drains=0,releases=0,role=0;
bool nativeFlushed=false,savedFlushed=false,rowAcquired=false,pendingAcquired=false;
bool timeout=false,profile=false;
Row rows[16]{};BatchFlag flags[16*kFlagCount]{};
uint64_t times[16*512]{};uint32_t statuses[16*32]{};
uint32_t b=0;
std::function<void()> onPoll;
uint64_t GetSystemCycle(){now+=timeout?6000000001ULL:1;return now;}
namespace kvaf_device {
constexpr uint64_t kCqTimeout=8;
void CleanCacheLines(uint8_t* ptr,uint64_t length){
 auto done=flags+b*kFlagCount+7;
 auto saved=times+b*64+32;
 if(ptr==reinterpret_cast<uint8_t*>(saved)){
  assert(length==40);if(role==1)savedFlushed=true;
 }
 if(ptr==reinterpret_cast<uint8_t*>(done)){
  if(role==1){
   assert(nativeFlushed);assert(!profile||savedFlushed);++releases;
  }else if(onPoll){auto callback=onPoll;onPoll=nullptr;callback();}
 }
 if(role==2&&ptr==reinterpret_cast<uint8_t*>(rows+b))rowAcquired=true;
 if(role==2&&ptr==reinterpret_cast<uint8_t*>(&rows[b].pending))pendingAcquired=true;
}
}
void CompletePrefetch(Context*,uint32_t batch,bool spare,
 uint64_t* polls,uint64_t* drain,uint64_t* arrival){
 assert(batch==b&&spare&&drains==0);
 // No release may authorize conflicting writes before completion was read.
 assert(!flags[b*kFlagCount+7].ready);
 assert(role==(KVIP_EARLY_PREFETCH_DRAIN?1U:2U));
 ++drains;rows[b].error=errorToReturn;
 if(!errorToReturn)rows[b].pending=0;
 if(polls)*polls=7;if(drain)*drain=12;if(arrival)*arrival=GetSystemCycle();
 nativeFlushed=true;
}
#include "early_prefetch_drain.h"
void ConsumerPrefix/*CONSUMER*/}
template<bool ReadThrough=false,bool PostIndex=false,bool JoinedRow=false>
uint64_t FinalizerPrefix/*FINALIZE*/return error;}
void reset(uint32_t batch,bool profiling,uint64_t error){
 b=batch;profile=profiling;errorToReturn=error;
 now=0;drains=releases=0;role=0;timeout=false;onPoll=nullptr;
 nativeFlushed=savedFlushed=rowAcquired=pendingAcquired=false;
 std::fill_n(times,16*512,0);for(auto& f:flags)f={};for(auto& r:rows)r={};
 rows[b].pending=5;rows[b].slots=113;rows[b].finalHead=991;rows[b].finalRequests=77;
}
int main(){
 Context ctx{uint64_t(rows)};
 Args a{uint64_t(&ctx),uint64_t(flags),uint64_t(times),uint64_t(statuses),0};
 for(unsigned batch:{0U,15U})for(bool profiling:{false,true})for(uint64_t error:{0ULL,16ULL}){
  reset(batch,profiling,error);a.prefetchCount=profile?0x80000000U:0;
  // Consumer is allowed to finish before Resolver has published any Q.
  role=1;ConsumerPrefix(&a,b);
  assert(drains==unsigned(KVIP_EARLY_PREFETCH_DRAIN));
  assert(flags[b*kFlagCount+7].ready==unsigned(KVIP_EARLY_PREFETCH_DRAIN));
  for(unsigned q=0;q<4;++q)assert(!flags[b*kFlagCount+q+1].ready);
  role=2;assert(FinalizerPrefix(&a,b,0)==error);
  assert(drains==1&&flags[b*kFlagCount+7].ready);
  assert(rows[b].slots==113&&rows[b].finalHead==991&&rows[b].finalRequests==77);
  assert(rows[b].pending==(error?5U:0U));
  auto ft=times+16*128+b*128+96;
  if(profile){assert(ft[4]>=ft[0]);assert(ft[5]==7&&ft[6]==12);}
  if(KVIP_EARLY_PREFETCH_DRAIN){
   assert(releases==1&&rowAcquired&&pendingAcquired);
   if(profile){auto saved=times+b*64+32;
    assert(saved[1]>=saved[0]&&saved[1]<=ft[0]);assert(saved[4]==ft[7]);
    assert(uintptr_t(saved)/128!=uintptr_t(ft)/128);}
   // A second join is read-only: it never drains CQ or rewrites flag 7.
   assert(FinalizerPrefix(&a,b,0)==error);assert(drains==1&&releases==1);
  }
 }
#if KVIP_EARLY_PREFETCH_DRAIN
 // Finalizer may arrive first. Its acquire observes Consumer publication.
 reset(0,true,32);a.prefetchCount=0x80000000U;role=2;
 onPoll=[&]{role=1;ConsumerPrefix(&a,b);role=2;};
 assert(FinalizerPrefix(&a,b,0)==32);assert(drains==1&&releases==1);
 assert(rowAcquired&&pendingAcquired);
 // A missing owner times out; it does not steal CQ or publish success.
 reset(0,false,0);a.prefetchCount=0;role=2;timeout=true;
 assert(FinalizerPrefix(&a,b,0)==kvaf_device::kCqTimeout);
 assert(drains==0&&releases==0&&!flags[7].ready);
#endif
 // A joined Post/Index tail must not touch a prefetch CQ or require old
 // Consumer flags, even when the binary also contains early-drain support.
 for(unsigned batch:{0U,15U})for(bool profiling:{false,true}){
  reset(batch,profiling,0);a.prefetchCount=profile?0x80000000U:0;role=2;
  rows[b].pending=0;
  assert((FinalizerPrefix<false,true>(&a,b,0)==0));
  assert(drains==0&&releases==0&&!flags[b*kFlagCount+7].ready);
  assert(rows[b].slots==113&&rows[b].finalHead==991&&rows[b].finalRequests==77);
 }
}
"""
    for name, value in (("ROW", row), ("CONSUMER", consumer), ("FINALIZE", finalize)):
        program = program.replace(f"/*{name}*/", value)
    path = tmp_path / "prefetch_owner.cpp"
    path.write_text(program)
    binary = tmp_path / "prefetch_owner"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer", f"-DKVIP_EARLY_PREFETCH_DRAIN={early}",
        "-I", str(KERNEL), str(path), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
