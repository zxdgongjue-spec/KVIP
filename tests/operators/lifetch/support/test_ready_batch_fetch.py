"""Execute the production ready-batch scheduler and native adapter on CPU."""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function


ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"


def compile_and_run(tmp_path, source, name):
    path = tmp_path / f"{name}.cpp"
    path.write_text(source)
    binary = tmp_path / name
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I",
            str(KERNEL),
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_actual_scheduler_ready_conflicts_errors_and_replay(tmp_path, workers):
    source = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#define __aicore__
#include "ready_batch_fetch.h"
using namespace pipeline_ab::ready_fetch;
struct Input { uint32_t count=0,safe=0;uint64_t ready=0,prefetch=0,error=0,prefetchError=0; };
struct IO {
 uint32_t b,q,lane,workers;Input input[16];uint64_t now=0,firstCall=UINT64_MAX;
 uint32_t calls=0,failCall=0;bool poison=false,malformed=false;
 std::set<uint32_t> visited;std::vector<uint32_t> sizes;
 uint64_t Now(){now+=100000000;return now;}
 uint64_t Invalid(){return 4;}
 uint64_t Timeout(){return 8;}
 Publication Query(uint32_t row){
  assert(row%workers==lane);now+=100000000;
  const auto& x=input[row];if(now<x.ready)return {false,0,0,0,0};
  return {true,x.error,row*kSlots+q*kTop+(malformed?1:0),x.count,x.safe};
 }
 Completion Prefetch(uint32_t row){return {now>=input[row].prefetch,input[row].prefetchError};}
 Submitted Submit(const Range* ranges,uint32_t n,uint32_t size){
  assert(!poison&&n>0&&n<=b&&size>0&&size<=b*kTop);uint32_t offset=0;
  std::set<uint32_t> batchRows;
  for(uint32_t i=0;i<n;++i){auto r=ranges[i];uint32_t row=r.begin/kSlots,pos=r.begin%kTop;
   assert(row<b&&row%workers==lane&&(r.begin%kSlots)/kTop==q);
   assert(batchRows.insert(row).second&&r.output==offset);
   assert(now>=input[row].ready&&input[row].error==0);
   assert(pos+r.count<=input[row].count);
   if(pos+r.count>input[row].safe)assert(now>=input[row].prefetch&&input[row].prefetchError==0);
   for(uint32_t j=0;j<r.count;++j)assert(visited.insert(r.begin+j).second);
   offset+=r.count;
  }
  assert(offset==size);firstCall=std::min(firstCall,now);++calls;sizes.push_back(size);
  if(failCall==calls){poison=true;return {16,0};}
  now+=1000000;return {0,1};
 }
 void check(const Result& r){
  assert(!r.error&&!poison&&r.calls==calls&&r.completed==calls);
  uint32_t expected=0;for(uint32_t row=lane;row<b;row+=workers){expected+=input[row].count;
   for(uint32_t i=0;i<input[row].count;++i)assert(visited.count(row*kSlots+q*kTop+i));}
  assert(r.requests==expected&&visited.size()==expected);
 }
};
int main(){
 constexpr uint32_t workers=WORKERS;
 std::mt19937 random(91);
 for(uint32_t batch:{1U,2U,5U,16U})for(uint32_t q=0;q<4;++q)for(uint32_t lane=0;lane<workers;++lane){
  for(uint32_t replay=0;replay<12;++replay){
   IO io{batch,q,lane,workers};
   for(uint32_t row=0;row<batch;++row){auto& x=io.input[row];x.count=random()%2049;x.safe=random()%(x.count+1);
    x.ready=(random()%30)*100000000ULL;x.prefetch=x.ready+(random()%40)*100000000ULL;}
   auto result=Run(io,batch,q,lane,workers,batch*kTop);io.check(result);
  }
  IO all{batch,q,lane,workers};for(uint32_t row=0;row<batch;++row)all.input[row]={1024,512,0,0};
  auto result=Run(all,batch,q,lane,workers,batch*kTop);all.check(result);
  assert(all.calls==(lane<batch?1U:0U)); // Merge every already-ready row, not one native call per row.
  IO zero{batch,q,lane,workers};result=Run(zero,batch,q,lane,workers,batch*kTop);zero.check(result);assert(!zero.calls);
 }
 // The last row is late. Earlier safe prefixes must start before its Q flag;
 // conflicts are held until their own prefetch release, not another row's.
 IO stagger{16,2,0,workers};
 for(uint32_t row=0;row<16;row+=workers)stagger.input[row]={130,64,0,6000000000ULL};
 uint32_t last=((15/workers)*workers);stagger.input[last].ready=4000000000ULL;
 auto r=Run(stagger,16,2,0,workers,16*kTop);stagger.check(r);
 assert(stagger.firstCall<stagger.input[last].ready&&stagger.calls>=2);
 // A small ready batch is flushed after a bounded rescan; it does not wait
 // indefinitely for the 128-request target or unrelated unpublished rows.
 IO small{16,0,0,workers};small.input[0]={1,1,0,0};
 for(uint32_t row=workers;row<16;row+=workers)small.input[row].ready=4000000000ULL;
 r=Run(small,16,0,0,workers,16*kTop);small.check(r);assert(small.firstCall<4000000000ULL);
 IO invalid{1,0,0,workers};invalid.malformed=true;r=Run(invalid,1,0,0,workers,kTop);assert(r.error==4&&!invalid.calls);
 IO over{1,0,0,workers};over.input[0].count=2049;r=Run(over,1,0,0,workers,kTop);assert(r.error==4&&!over.calls);
 IO safe{1,0,0,workers};safe.input[0]={1,2,0,0};r=Run(safe,1,0,0,workers,kTop);assert(r.error==4&&!safe.calls);
 IO error{1,0,0,workers};error.input[0]={10,10,0,0,32};r=Run(error,1,0,0,workers,kTop);assert(r.error==32&&!error.calls);
 IO timeout{1,0,0,workers};timeout.input[0].ready=UINT64_MAX;r=Run(timeout,1,0,0,workers,kTop);assert(r.error==8&&!timeout.calls);
 IO conflictTimeout{1,0,0,workers};conflictTimeout.input[0]={4,2,0,UINT64_MAX};
 r=Run(conflictTimeout,1,0,0,workers,kTop);assert(r.error==8&&r.requests==2&&conflictTimeout.calls==1);
 IO conflictError{1,0,0,workers};conflictError.input[0]={4,2,0,1000000000ULL,0,64};
 r=Run(conflictError,1,0,0,workers,kTop);assert(r.error==64&&r.requests==2&&conflictError.calls==1);
 IO failed{16,0,0,workers};for(uint32_t row=0;row<16;row+=workers)failed.input[row]={200,100,0,2000000000ULL};
 failed.failCall=1;r=Run(failed,16,0,0,workers,16*kTop);assert(r.error==16&&failed.calls==1&&failed.poison);
}
""".replace("WORKERS", str(workers))
    compile_and_run(tmp_path, source, "scheduler")


@pytest.mark.parametrize("direct", [0, 1])
def test_native_adapter_queue_counts_errors_and_index_join(tmp_path, direct):
    streaming = (KERNEL / "streaming.cpp").read_text()
    entry = function(streaming, "__aicore__ inline void FetchQueryGroup(")
    source = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#define KVIP_READY_BATCH_FETCH 1
#define V9_SIMULATION 0
#include "ready_batch_fetch.h"
using namespace pipeline_ab;
constexpr uint32_t kTop=2048,kSlots=8192,kFlagCount=16,kResolveCores=3,EVENT_ID0=0;
constexpr uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC=10,KVAF_AIV_BATCH_GET_STATUS_MAGIC=11;
struct BatchFlag {uint32_t ready=0,error=0;};
struct FinalPost {uint32_t count=0,descriptor=0,nonconflicting=0,error=0,pending=0,deferred=1;};
struct RuntimeShape {uint32_t batch=16,finalWorkers=4,finalPostOffset=0,resident=16384;};
struct Context {uint64_t count=0,fallbackAddresses=0,fallbackValid=0,sourceBlocks=0,sourceOffsets=0,destinations=0;RuntimeShape shape;FinalPost posts[64];};
struct Args {uint64_t context,flags,times,packedFetch;uint32_t prefetchCount;};
struct KvafBatchGetQueueState {uint64_t initialized=1,completed_batch_count=41,poisoned=0;};
struct Profile {uint64_t magic=0,post_active_cycles=3,drain_wait_cycles=5,submitted_group_count=1;};
struct KvafBatchGetCoreStatus {uint64_t magic=11,result=0,processed_request_count=0;Profile profile;};
struct KvafExecuteBatchGetKernelArgs {
 uint64_t indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0,queue_state=0,core_status=0;
 uint32_t jettys_per_core=2,request_capacity=0,table_capacity=0,batch_get_max_inflight_total=0;
};
static KvafBatchGetQueueState queues[64];static KvafBatchGetCoreStatus statuses[64];
static uint64_t clockValue=0;static uint32_t nativeCalls=0,failCall=0,publicationCalls=0;
static BatchFlag* currentFlags;static Context* currentContext;static uint32_t currentLane,currentWorkers;
uint64_t GetSystemCycle(){return ++clockValue;}
RuntimeShape LoadRuntimeShape(Context* c){return c->shape;}
FinalPost* FinalPostAt(uint32_t,Context* c,uint32_t i){return c->posts+i;}
KvafExecuteBatchGetKernelArgs GetArgs(Context*,uint32_t pair){
 assert(pair==32+2*(2*currentWorkers+currentLane));
 KvafExecuteBatchGetKernelArgs a;a.queue_state=uint64_t(queues+pair);a.core_status=uint64_t(statuses+pair);return a;
}
namespace kvaf_device {
constexpr uint64_t kInvalidArgs=4,kCqTimeout=8;
void CleanCacheLines(uint8_t* p,uint64_t){
 // Index completion becomes visible only after both transfer phases finish.
 for(uint32_t b=currentLane;b<16;b+=currentWorkers){
  if(p==reinterpret_cast<uint8_t*>(currentFlags+b*16+6)){
   assert(nativeCalls>=2);currentFlags[b*16+6].ready=1;
  }
 }
}
void KvafBatchGetDevice(KvafExecuteBatchGetKernelArgs a,uint32_t core,uint32_t begin,uint32_t count){
 assert(core==0&&begin==0&&count>0&&a.jettys_per_core==2&&a.batch_get_max_inflight_total==2048);
 auto queue=reinterpret_cast<KvafBatchGetQueueState*>(a.queue_state);
 auto status=reinterpret_cast<KvafBatchGetCoreStatus*>(a.core_status);
 assert(!queue->poisoned);++nativeCalls;
 auto indices=reinterpret_cast<int32_t*>(a.indices);
 auto offsets=reinterpret_cast<int32_t*>(a.offsets);
 auto dst=reinterpret_cast<int32_t*>(a.output_indices);
 for(uint32_t i=0;i<count;++i){uint32_t descriptor=uint32_t(indices[i]);uint32_t b=descriptor/kSlots;
  assert(b%currentWorkers==currentLane&&descriptor%kSlots/2048==2&&offsets[i]==0&&dst[i]==indices[i]);
  uint32_t position=descriptor%2048;
  if(position>=2)assert(currentFlags[b*16+7].ready);
 }
 status->magic=11;status->processed_request_count=count;status->result=0;
 if(nativeCalls==failCall){queue->poisoned=1;status->result=16;}
 else ++queue->completed_batch_count;
 for(uint32_t b=currentLane;b<16;b+=currentWorkers)currentFlags[b*16+7].ready=1;
}
}
enum class HardEvent {V_S};
template<HardEvent>void SetFlag(uint32_t){}
template<HardEvent>void WaitFlag(uint32_t){}
void SyncFetchDescriptors(Context*,uint32_t begin,uint32_t count){assert(count&&begin+count<=16*8192);}
void PackFetchRanges(Context*,int32_t* packed,uint32_t stride,uint32_t* ranges,uint32_t n){
 for(uint32_t r=0;r<n;++r)for(uint32_t i=0;i<ranges[3*r+1];++i){
  uint32_t out=ranges[3*r+2]+i;packed[out]=ranges[3*r]+i;packed[stride+out]=0;packed[2*stride+out]=packed[out];
 }
}
void PublishQueryCompletion(Context* ctx,BatchFlag* flags,uint32_t batch,uint32_t,uint32_t,uint32_t q,
 uint32_t lane,uint32_t workers,uint64_t error,uint32_t completed){
 ++publicationCalls;assert(batch==16&&q==2&&completed==(failCall?1U:2U));
 for(uint32_t b=lane;b<batch;b+=workers){assert(flags[b*16+6].ready);
  auto& post=ctx->posts[b*4+q];post.pending=b==lane?completed:0;post.error=error;post.deferred=0;
  flags[b*16+8+q].error=error;flags[b*16+8+q].ready=1;
 }
}
namespace Simt {struct Dim3{uint32_t x,y,z;};template<auto F,class... T>void VF_CALL(Dim3,T... args){F(args...);}}
#include "ready_batch_device.h"
constexpr uint32_t kBatchGetInflight=2048;
"""
        + entry
        + r"""
int main(){
 for(uint32_t workers:{1U,2U,4U})for(bool failure:{false,true})for(bool profile:{false,true}){
  Context ctx;std::vector<BatchFlag> flags(16*16);std::vector<uint64_t> times(16*256+64*32);
  std::vector<int32_t> packed(4*workers*(9*16*2048+128),-987);
  currentContext=&ctx;currentFlags=flags.data();currentWorkers=workers;currentLane=0;
  nativeCalls=0;failCall=failure?2:0;publicationCalls=0;
  for(auto& queue:queues)queue=KvafBatchGetQueueState{};
  for(auto& status:statuses)status=KvafBatchGetCoreStatus{};
  for(uint32_t b=0;b<16;b+=workers){flags[b*16+3].ready=1;
   auto& p=ctx.posts[b*4+2];p.descriptor=b*8192+2*2048;p.count=4;p.nonconflicting=2;}
  Args a{uint64_t(&ctx),uint64_t(flags.data()),uint64_t(times.data()),uint64_t(packed.data()),
         uint32_t(0x10000000U|(profile?0x80000000U:0))};
  FetchQueryGroup(&a,2,0,workers);assert(nativeCalls==2&&publicationCalls==1);
  for(uint32_t b=0;b<16;b+=workers){assert(flags[b*16+10].ready);assert(flags[b*16+10].error==(failure?16U:0U));}
  uint32_t pair=32+4*workers;assert(queues[pair].poisoned==uint64_t(failure));
  assert(queues[pair].completed_batch_count==(failure?42U:43U));
  if(profile){auto ft=times.data()+16*128+2*24;
   assert(ft[20]==4&&ft[21]>0&&ft[22]==(failure?32U/workers:64U/workers));
   assert(ft[23]==2&&ft[12]==6&&ft[13]==10&&ft[16]==2&&ft[9]>0);
   assert(statuses[pair].profile.magic==0);
  }
 }
}
"""
    )
    compile_and_run(tmp_path, f"#define KVIP_READY_DIRECT_RANGES {direct}\n" + source, "adapter")


def test_candidate_disables_unconsumed_direct_layout_only_for_urma():
    source = (KERNEL / "output_worker.h").read_text()
    begin = source.index("#if KVIP_READY_BATCH_FETCH && !V9_UBMEM")
    end = source.index("#endif", begin)
    assert "direct=nullptr;" in source[begin:end]
    assert "PackPublicationEvents" in source[end:]
    streaming = (KERNEL / "streaming.cpp").read_text()
    assert streaming.index(
        "if(a->prefetchCount & 0x10000000U){", streaming.index("#if KVIP_READY_BATCH_FETCH")
    ) > streaming.index("error=result.error;completed=result.completed;io.Finish(result);")
