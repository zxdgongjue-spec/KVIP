# SPDX-License-Identifier: Apache-2.0
"""Run the production synchronous adapter with address-reading native Get mocks."""

import pytest

from test_ready_batch_fetch import KERNEL, compile_and_run
from test_recent_index import function


@pytest.mark.parametrize("direct", [0, 1])
def test_direct_ranges_values_lifetime_cleaning_errors_and_reuse(tmp_path, direct):
    submit = function(
        (KERNEL / "ready_batch_device.h").read_text(),
        "__aicore__ inline ready_fetch::Submitted Submit(",
    )
    acquire = function(
        (KERNEL / "streaming.cpp").read_text(),
        "__aicore__ inline void SyncFetchDescriptors(",
    )
    source = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#define V9_SIMULATION 0
#define KVIP_READY_DIRECT_RANGES DIRECT_MODE
#include "ready_batch_fetch.h"
using namespace pipeline_ab;
constexpr uint32_t EVENT_ID0=0;
constexpr uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC=10,KVAF_AIV_BATCH_GET_STATUS_MAGIC=11;
struct Context {uint64_t sourceBlocks,sourceOffsets,destinations,fallbackAddresses;};
struct RuntimeShape {uint32_t batch=16;};
struct Queue {uint64_t initialized=1,completed_batch_count=41,poisoned=0;};
struct Profile {uint64_t magic=99,post_active_cycles=3,drain_wait_cycles=5,submitted_group_count=1;};
struct Status {uint64_t magic=11,result=0,processed_request_count=0;Profile profile;};
using KvafBatchGetQueueState=Queue;
using KvafBatchGetCoreStatus=Status;
struct KernelArgs {
 uint64_t indices,offsets,output_indices,queue_state,core_status;
};
using KvafExecuteBatchGetKernelArgs=KernelArgs;
static uint64_t clockValue=0;
static uint32_t nativeCalls=0,packCalls=0;
static bool failNative=false,badStatus=false;
static std::vector<std::array<int32_t,3>> expected;
static std::array<uint64_t,3> expectedPointers;
static std::vector<std::pair<uint64_t,uint64_t>> cleaned;
namespace kvaf_device {
constexpr uint64_t kInvalidArgs=4;
void CleanCacheLines(uint8_t* p,uint64_t n){cleaned.emplace_back(uint64_t(p),n);}
void KvafBatchGetDevice(KernelArgs a,uint32_t core,uint32_t begin,uint32_t size){
 assert(core==0&&begin==0&&size==expected.size());++nativeCalls;
 assert(a.indices==expectedPointers[0]&&a.offsets==expectedPointers[1]&&a.output_indices==expectedPointers[2]);
 const auto* indices=reinterpret_cast<const int32_t*>(a.indices);
 const auto* offsets=reinterpret_cast<const int32_t*>(a.offsets);
 const auto* destinations=reinterpret_cast<const int32_t*>(a.output_indices);
 for(uint32_t i=0;i<size;++i)assert((std::array<int32_t,3>{indices[i],offsets[i],destinations[i]}==expected[i]));
 auto* queue=reinterpret_cast<Queue*>(a.queue_state);auto* status=reinterpret_cast<Status*>(a.core_status);
 assert(!queue[0].poisoned);status->magic=badStatus?0:11;status->processed_request_count=size;
 status->result=failNative?16:0;
 if(failNative)queue[0].poisoned=1;else ++queue[0].completed_batch_count;
}
}
enum class HardEvent {V_S};
template<HardEvent>void SetFlag(uint32_t){}
template<HardEvent>void WaitFlag(uint32_t){}
void PackFetchRanges(Context* ctx,int32_t* packed,uint32_t stride,uint32_t* ranges,uint32_t count){
 ++packCalls;
 auto* source=reinterpret_cast<int32_t*>(ctx->sourceBlocks);
 auto* offset=reinterpret_cast<int32_t*>(ctx->sourceOffsets);
 auto* dest=reinterpret_cast<int32_t*>(ctx->destinations);
 for(uint32_t r=0;r<count;++r)for(uint32_t i=0;i<ranges[3*r+1];++i){
  uint32_t from=ranges[3*r]+i,to=ranges[3*r+2]+i;
  packed[to]=source[from];packed[stride+to]=offset[from];packed[2*stride+to]=dest[from];
 }
}
namespace Simt {struct Dim3{uint32_t x,y,z;};template<auto F,class... T>void VF_CALL(Dim3,T... args){F(args...);}}
""".replace("DIRECT_MODE", str(direct))
        + acquire
        + r"""
struct Adapter {
 Context* ctx;RuntimeShape shape;KernelArgs args;int32_t* packed;uint32_t* ranges;
 uint32_t query=2,stride=16*2048;bool profile=true;
 uint64_t packCycles=0,nativeCycles=0,postCycles=0,drainCycles=0,groups=0,firstNative=0,lastNative=0;
 uint64_t Invalid(){return 4;}uint64_t Now(){return ++clockValue;}
"""
        + submit
        + r"""
};
int main(){
 constexpr uint32_t capacity=16*8192,stride=16*2048;
 std::vector<int32_t> blocks(capacity),offsets(capacity),destinations(capacity),packed(3*stride,-1);
 std::vector<uint64_t> addresses(capacity);std::vector<uint32_t> scratch(3*16);
 for(uint32_t i=0;i<capacity;++i){blocks[i]=int32_t(17*i+3);offsets[i]=int32_t(i%128);destinations[i]=int32_t(29*i+7);}
 Context ctx{uint64_t(blocks.data()),uint64_t(offsets.data()),uint64_t(destinations.data()),uint64_t(addresses.data())};
 Queue queues[2];Status status;
 KernelArgs original{uint64_t(packed.data()),uint64_t(packed.data()+stride),uint64_t(packed.data()+2*stride),uint64_t(queues),uint64_t(&status)};
 Adapter io{&ctx,{},original,packed.data(),scratch.data()};
 auto run=[&](std::vector<ready_fetch::Range> ranges){
  expected.clear();cleaned.clear();uint32_t size=0;
  for(auto r:ranges){assert(r.output==size);for(uint32_t i=0;i<r.count;++i)
   expected.push_back({blocks[r.begin+i],offsets[r.begin+i],destinations[r.begin+i]});size+=r.count;}
  const bool direct=KVIP_READY_DIRECT_RANGES&&ranges.size()==1;
  expectedPointers=direct?std::array<uint64_t,3>{ctx.sourceBlocks+4ULL*ranges[0].begin,ctx.sourceOffsets+4ULL*ranges[0].begin,ctx.destinations+4ULL*ranges[0].begin}:
                         std::array<uint64_t,3>{original.indices,original.offsets,original.output_indices};
  auto before=packCalls;auto result=io.Submit(ranges.data(),uint32_t(ranges.size()),size);
  assert(packCalls-before==(direct?0U:1U));
  assert(io.args.indices==original.indices&&io.args.offsets==original.offsets&&io.args.output_indices==original.output_indices);
  for(auto r:ranges)for(auto item:{std::pair<uint64_t,uint64_t>{ctx.sourceBlocks+4ULL*r.begin,4ULL*r.count},
                                  {ctx.sourceOffsets+4ULL*r.begin,4ULL*r.count},
                                  {ctx.destinations+4ULL*r.begin,4ULL*r.count},
                                  {ctx.fallbackAddresses+8ULL*r.begin,8ULL*r.count}})
   assert(std::count(cleaned.begin(),cleaned.end(),item)==1);
  auto rangeAddress=uint64_t(scratch.data());
  assert(std::count_if(cleaned.begin(),cleaned.end(),[&](auto c){return c.first==rangeAddress;})==(direct?0:1));
  for(uint64_t address:{original.indices,original.offsets,original.output_indices})
   assert(std::count_if(cleaned.begin(),cleaned.end(),[&](auto c){return c.first==address;})==(direct?0:1));
  assert(status.profile.magic==99);
  return result;
 };
 for(bool profile:{false,true}){
  io.profile=profile;
  // Complete query, then safe/conflicting suffixes, with a packed batch between
  // direct calls. Native Get reads the actual original/packed three arrays.
  for(auto ranges:{std::vector<ready_fetch::Range>{{4096,2048,0}},
                   {{4096+19,37,0}},{{4096+19,37,0},{8192+4096,71,37}},
                   {{15*8192+4096+2047,1,0}},{{4096+56,31,0}}}){
   auto r=run(ranges);assert(!r.error&&r.completed==1);
  }
 }
#if KVIP_READY_DIRECT_RANGES
 // Malformed direct spans fail before descriptor acquisition or native work.
 auto calls=nativeCalls;auto packs=packCalls;cleaned.clear();
 for(auto r:{ready_fetch::Range{capacity,1,0},{4096+2047,2,0},{0,1,0},{4096,1,1},{4096,0,0},{4096,2,0}}){
  uint32_t size=r.count==2&&r.begin==4096?1:r.count;
  assert(io.Submit(&r,1,size).error==4);
 }
 assert(io.Submit(nullptr,0,0).error==4);
 assert(nativeCalls==calls&&packCalls==packs&&cleaned.empty());
#endif
 // Native errors keep queue poison, do not credit failed calls or reset status.
 failNative=true;auto completed=queues[0].completed_batch_count;
 auto failed=run({{4096+19,37,0}});
 assert(failed.error==16&&!failed.completed&&queues[0].poisoned&&queues[0].completed_batch_count==completed);
 // A fresh queue is required after poison; malformed completion is rejected.
 queues[0]=Queue{};failNative=false;badStatus=true;
 auto bad=run({{4096+19,37,0}});assert(bad.error==4&&!bad.completed);
}
"""
    )
    compile_and_run(tmp_path, source, "direct_ranges")
