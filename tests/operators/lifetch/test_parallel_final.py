# SPDX-License-Identifier: Apache-2.0
"""Actual final worker/reducer and bridge ordering; no hardware timing claim."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch"


def test_parallel_final_worker_queue_ownership_completion_and_errors(tmp_path):
    lifecycle = (OP / "op_kernel/lifecycle.cpp").read_text()
    begin = lifecycle.index(
        "    for(uint32_t w=0;w<rt_finalWorkers;++w)", lifecycle.index("// The separate parallel kernel")
    )
    reducer = lifecycle[begin : lifecycle.index("\n#else", begin)]
    (tmp_path / "batch_get_abi.h").write_text("")
    source = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>
#define __aicore__
#define __gm__
#define __CCE_AICORE__ 1
#define V9_SIMULATION 0
#define KVAF_AIV_BATCH_GET_STATUS_MAGIC 99
#include "context.h"
using namespace rotating;
struct KvafBatchGetQueueState {uint64_t sq_producer_index=0,cq_consumer_index=0,poisoned=0,last_error=0;};
struct KvafBatchGetCoreStatus {uint64_t magic=99,result=0,completed_batch_count=1,processed_request_count=0;};
struct Args {
 uint64_t queue_state=0,core_status=0,indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0;
 uint32_t request_capacity=0,table_capacity=0,block_count=1,jettys_per_core=1;
};
using KvafExecuteBatchGetKernelArgs=Args;
std::array<KvafBatchGetQueueState,10> queues;
std::array<KvafBatchGetCoreStatus,10> statuses;
std::vector<unsigned> calls;std::set<unsigned> queueOwners,acquired,published;unsigned fault=0,badQueue=0;
namespace rotating {
FinalPost* FinalPostAt(uint32_t offset,Context* ctx,uint32_t core) {
 return reinterpret_cast<FinalPost*>(ctx->owners+offset)+core;
}
}
Args GetArgs(Context*,unsigned queue) {
 Args args;args.queue_state=uint64_t(&queues.at(queue));args.core_status=uint64_t(&statuses.at(queue));
 if(fault==4 && queue==badQueue)args.jettys_per_core=2;
 return args;
}
namespace kvaf_device {
constexpr unsigned kInvalidArgs=1;
void CleanCacheLines(uint8_t* pointer,size_t bytes){
 for(unsigned q=0;q<queues.size();++q)if(pointer==reinterpret_cast<uint8_t*>(&queues[q])) {
  assert(bytes==sizeof(queues[q]));
  if(queueOwners.count(q))published.insert(q);else acquired.insert(q);
 }
}
void KvafBatchGetDevice(Args args,unsigned core,unsigned begin,unsigned count) {
 auto* queue=reinterpret_cast<KvafBatchGetQueueState*>(args.queue_state);
 unsigned q=unsigned(queue-queues.data());assert(core==0 && args.block_count==1 && args.jettys_per_core==1);
 assert(acquired.count(q));assert(queueOwners.insert(q).second); // views never alias
 assert(args.indices==1 && args.offsets==2 && args.output_indices==3 && args.miss_count==4);
 assert(args.address_table==5 && args.valid_table==6 && args.request_capacity==2*kSlots && args.table_capacity==2*kSlots);
 calls.push_back(begin);
 auto* status=reinterpret_cast<KvafBatchGetCoreStatus*>(args.core_status);
 assert(status->magic==0); // stale successful state cannot mask a no-write native return
 if(fault==7 && q==badQueue)return;
 status->magic=KVAF_AIV_BATCH_GET_STATUS_MAGIC;
 status->result=(fault==5 && q==badQueue) || queue->poisoned?8:0;
 status->processed_request_count=count-(fault==6 && q==badQueue?1:0);
 if(status->result){queue->poisoned=1;queue->last_error=status->result;return;}
 status->completed_batch_count=++queue->sq_producer_index;queue->cq_consumer_index=queue->sq_producer_index;
}
}
#include "parallel_final.h"
int main() {
 for(fault=0;fault<12;++fault)for(unsigned safe:{0U,3U,99U}) {
  Context c{};c.shape.batch=2;c.shape.finalWorkers=4;c.shape.finalCores=8;c.shape.finalPostOffset=128;
  std::array<Row,2> rows{};std::array<uint32_t,2*kRowStatusWords> infoStorage{};
  alignas(128) std::array<uint8_t,128+8*sizeof(FinalPost)> ownerStorage{};
  c.rows=uint64_t(rows.data());c.rowStatus=uint64_t(infoStorage.data());c.owners=uint64_t(ownerStorage.data());
  c.sourceBlocks=1;c.sourceOffsets=2;c.destinations=3;c.count=4;c.fallbackAddresses=5;c.fallbackValid=6;
  badQueue=FinalQueue(2,4,0,2);calls.clear();queueOwners.clear();acquired.clear();published.clear();queues={};statuses={};
  for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w) {
   auto* post=FinalPostAt(c.shape.finalPostOffset,&c,b*4+w);
   post->count=w==3?0:7+w;post->descriptor=b*kSlots+w*16;
   post->nonconflicting=std::min(safe,post->count);post->deferred=post->count!=0;
   infoStorage[b*kRowStatusWords+1]+=post->count;
  }
  auto* bad=FinalPostAt(c.shape.finalPostOffset,&c,2);
  if(fault==1)rows[0].pending=1;
  if(fault==2)rows[0].error=4;
  if(fault==3)bad->error=4;
  if(fault==8)bad->nonconflicting=bad->count+1;
  if(fault==9)bad->deferred=0;
  if(fault==11)++infoStorage[1];
  // Reverse physical invocation order proves no sibling result is consumed
  // by a worker. Only the subsequent reducer reads all completed posts.
  for(unsigned core=8;core-->0;)ParallelFinalGet(&c,core);
  ParallelFinalGet(&c,8); // padding block is inactive
  assert(published==queueOwners); // successful and failing native calls publish queue state before reduction
  if(fault==10)bad->pending=1;
  for(unsigned b=0;b<2;++b) {
   Context* ctx=&c;auto* row=&rows[b];auto* info=infoStorage.data()+b*kRowStatusWords;
   unsigned rt_batch=2,rt_finalWorkers=4,rt_finalPostOffset=c.shape.finalPostOffset,total=0,completed=0,deferred=0;
   uint64_t error=row->error,lastHead=0,lastCq=0;
REDUCER
   if(row->pending || total!=info[1])error|=kvaf_device::kInvalidArgs;
   assert(bool(error)==(b==0 && fault!=0));
   assert(total==24);
  }
  unsigned wantedCalls=(fault==1 || fault==2)?3:(fault==3 || fault==4 || fault==8 || fault==9)?5:6;
  assert(calls.size()==wantedCalls);
  assert(!queueOwners.count(0) && !queueOwners.count(2)); // prefetch views are never reused
  if(fault==5) {
   // A failing native call publishes poison, not just a transient post error.
   // Even if a caller attempts another replay, that persistent queue rejects it.
   assert(published.count(badQueue) && queues[badQueue].poisoned && queues[badQueue].last_error);
   uint64_t head=queues[badQueue].sq_producer_index;
   bad->error=bad->pending=0;bad->deferred=1;
   queueOwners.erase(badQueue);acquired.erase(badQueue);published.erase(badQueue);
   fault=0;ParallelFinalGet(&c,2);fault=5;
   assert(bad->error && bad->pending && bad->deferred && published.count(badQueue));
   assert(queues[badQueue].sq_producer_index==head);
  }
 }
}
""".replace("REDUCER", reducer)
    cpp, binary = tmp_path / "parallel.cpp", tmp_path / "parallel"
    cpp.write_text(source)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-I{tmp_path}",
            f"-I{OP / 'include'}",
            f"-I{OP / 'op_kernel'}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("enabled", (0, 1))
def test_parallel_final_host_bridge_order_and_first_failure(tmp_path, enabled):
    source = (OP / "op_host/launch.cpp").read_text()
    entry = source[source.index('extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,') :]
    stub = r"""
#include <cstdint>
#include <cassert>
#include "runtime_shape.h"
unsigned count=0,fail=0;
uint32_t step(unsigned n,void* a,void* stream){assert(a==(void*)1 && stream==(void*)3);assert(++count==n);return fail==n?99:0;}
uint32_t v9_init(void*a,void*s,const rotating::RuntimeShape*){return step(1,a,s);}
uint32_t v9_li(void*a,void*s,const rotating::RuntimeShape*){return step(2,a,s);}
uint32_t v9_resolve(void*a,void*s,const rotating::RuntimeShape*){return step(3,a,s);}
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t b,void*a,void*s){assert(b==16 && a==(void*)2);return step(4,(void*)1,s);}
uint32_t v9_wait(void*a,void*s,const rotating::RuntimeShape*shape){assert(shape->batch==16);return step(5,a,s);}
uint32_t v9_parallel_fetch(void*a,void*s,const rotating::RuntimeShape*shape){assert(shape->finalCores==64);return step(6,a,s);}
extern "C" uint32_t aclrtlaunch_v9_fetch_update_kernel(uint32_t b,void*s,void*a,void*u){assert(b==16 && u==(void*)2);return step(V9_URMA_PARALLEL_FINAL?7:5,a,s);}
"""
    main = r"""
int main(){rotating::RuntimeShape shape{};shape.batch=16;shape.finalCores=64;
 const unsigned steps=V9_URMA_PARALLEL_FINAL?7:5;
 for(fail=0;fail<=steps;++fail){count=0;assert(kvip_lifetch_forward((void*)1,(void*)2,(void*)3,&shape)==(fail?99:0));assert(count==(fail?fail:steps));}}
"""
    cpp, binary = tmp_path / "bridge.cpp", tmp_path / "bridge"
    cpp.write_text(stub + entry + main)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            f"-DV9_URMA_PARALLEL_FINAL={enabled}",
            f"-I{OP / 'include'}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
