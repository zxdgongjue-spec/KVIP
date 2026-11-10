# SPDX-License-Identifier: Apache-2.0
"""Actual async worker code with delayed NIC completion and queue ownership."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch"


def test_async_final_delayed_completion_errors_and_queue_reuse(tmp_path):
    (tmp_path / "batch_get_abi.h").write_text("")
    source = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#define __CCE_AICORE__ 1
#define V9_URMA_PARALLEL_FINAL 1
#define V9_URMA_ASYNC_FINAL 1
#include "context.h"
using namespace rotating;
struct KvafBatchGetQueueState {
 uint64_t magic=0,initialized=0,sq_producer_index=0,cq_consumer_index=0,completed_batch_count=0,last_error=0,poisoned=0;
};
struct KvafBatchGetCoreStatus {uint64_t result=0,processed=0;};
struct KvafBatchGetDirectUrmaEntity {
 uint64_t magic=17,version=1,local_hbm_address=123,local_hbm_size=65536,local_token_id=1;
 uint64_t remote_host_ddr_address=234,remote_host_ddr_size=65536,sq_va=1,sq_db_va=2,sq_depth=32768,wqe_size=64;
 uint64_t cq_va=3,cq_db_va=4,cq_depth=4,cqe_size=64;
};
struct KvafExecuteBatchGetKernelArgs {
 uint64_t queue_state=0,core_status=0,channel_entity=0,indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0;
 uint64_t output=123;uint32_t request_capacity=0,table_capacity=0,output_rows=2048,token_bytes=2624,tokens_per_block=128;
 uint32_t block_count=1,jettys_per_core=1,batch_get_max_inflight_total=0;
};
struct Pending {unsigned begin,count;};
std::array<KvafBatchGetQueueState,10> queues;
std::array<KvafBatchGetCoreStatus,10> statuses;
std::array<KvafBatchGetDirectUrmaEntity,10> entities;
std::array<std::vector<Pending>,10> nic;
std::array<unsigned,10> safeCounts{},counts{},descriptors{},delivered{},posts{},waits{},flushes{};
unsigned phase=0,badQueue=0,fault=0,failedCq=0;
namespace rotating {
FinalPost* FinalPostAt(uint32_t offset,Context* ctx,uint32_t core) {
 return reinterpret_cast<FinalPost*>(ctx->owners+offset)+core;
}
}
KvafExecuteBatchGetKernelArgs GetArgs(Context*,unsigned q) {
 KvafExecuteBatchGetKernelArgs args;
 args.queue_state=uint64_t(&queues.at(q));args.core_status=uint64_t(&statuses.at(q));args.channel_entity=uint64_t(&entities.at(q));
 return args;
}
namespace kvaf_device {
constexpr unsigned kSuccess=0,kInvalidArgs=1,kInvalidEntity=2,kQueueFull=4,kInvalidQueue=8,kCqTimeout=16;
constexpr unsigned kEntityMagic=17,kEntityVersion=1,kQueueMagic=19,kSqDepthWqebbs=32768,kWqeBytes=64,kCqeProbeBytes=32;
void CleanCacheLines(uint8_t* pointer,size_t bytes) {
 for(unsigned q=0;q<queues.size();++q)if(pointer==reinterpret_cast<uint8_t*>(&queues[q])){
  assert(bytes==sizeof(queues[q]));++flushes[q];
 }
}
void ConfigureUbToGmCopy(){}
void PostDirectRequests(const int32_t* indices,const int32_t* out,const int32_t* offsets,const uint64_t* addresses,
 unsigned begin,unsigned count,unsigned first,unsigned width,unsigned jettys,const KvafBatchGetDirectUrmaEntity* entity,
 KvafBatchGetQueueState* queue,unsigned profile,uint64_t*,uint64_t*) {
 unsigned q=unsigned(queue-queues.data());assert(entity==&entities[q] && first==0 && jettys==1 && width==2624 && !profile);
 assert(queue->initialized && !queue->poisoned && flushes[q]>0 && count>0);
 assert(uint64_t(indices)==1 && uint64_t(offsets)==2 && uint64_t(out)==3 && uint64_t(addresses)==5);
 assert(phase==0 || phase==2); // phase 1 is Index; no final CQ is consumed there
 if(phase==0){assert(begin==descriptors[q] && count==safeCounts[q]);}
 else {assert(begin==descriptors[q]+safeCounts[q] && count==counts[q]-safeCounts[q]);}
 nic[q].push_back({begin,count});queue->sq_producer_index+=count;++posts[q];
 // Posting is deliberately NOT a payload write: only completion makes it visible.
}
uint64_t WaitForCompletion(const KvafBatchGetDirectUrmaEntity* entity,KvafBatchGetQueueState* queue,unsigned*) {
 unsigned q=unsigned(queue-queues.data());assert(phase==2 && entity==&entities[q] && !nic[q].empty());++waits[q];
 if(fault==1 && q==badQueue && ++failedCq==1)return kCqTimeout;
 delivered[q]+=nic[q].front().count;nic[q].erase(nic[q].begin());++queue->cq_consumer_index;
 return fault==9 && q==badQueue ? 32 : 0;
}
void WriteStatus(KvafBatchGetCoreStatus* status,uint64_t result,uint64_t processed,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t) {
 status->result=result;status->processed=processed;
}
}
#include "async_final.h"
int main() {
 for(unsigned scenario=0;scenario<10;++scenario)for(unsigned safe:{0U,3U,99U}) {
  fault=scenario;failedCq=0;phase=0;queues={};statuses={};entities={};nic={};
  safeCounts={};counts={};descriptors={};delivered={};posts={};waits={};flushes={};
  Context ctx{};ctx.shape.batch=2;ctx.shape.finalWorkers=4;ctx.shape.finalCores=8;ctx.shape.finalPostOffset=128;
  std::array<Row,2> rows{};std::array<uint32_t,2*kRowStatusWords> info{};
  alignas(128) std::array<uint8_t,128+8*sizeof(FinalPost)> owners{};
  ctx.rows=uint64_t(rows.data());ctx.rowStatus=uint64_t(info.data());ctx.owners=uint64_t(owners.data());
  ctx.sourceBlocks=1;ctx.sourceOffsets=2;ctx.destinations=3;ctx.count=4;ctx.fallbackAddresses=5;ctx.fallbackValid=6;
  for(unsigned b=0;b<2;++b){rows[b].pending=3;for(unsigned w=0;w<4;++w){
   unsigned q=FinalQueue(2,4,b,w);auto* post=FinalPostAt(128,&ctx,b*4+w);
   counts[q]=post->count=w==3?0:7+w;safeCounts[q]=post->nonconflicting=std::min(safe,post->count);
   descriptors[q]=post->descriptor=b*kSlots+w*16;post->deferred=post->count!=0;
  }}
  badQueue=FinalQueue(2,4,0,2);auto* bad=FinalPostAt(128,&ctx,2);
  if(fault==2){queues[badQueue].initialized=1;queues[badQueue].magic=kvaf_device::kQueueMagic;queues[badQueue].poisoned=1;}
  if(fault==3)entities[badQueue].local_hbm_address=999;
  if(fault==4)bad->nonconflicting=bad->count+1;
  if(fault==5)entities[badQueue].cq_depth=1; // legal only when at most one nonempty range
  // Each worker may submit before worker zero publishes aggregate rowStatus.
  info.fill(99);for(unsigned core=8;core-->0;)SubmitSafeFinal(&ctx,core);SubmitSafeFinal(&ctx,8);
  phase=1;for(unsigned q=0;q<10;++q){assert(waits[q]==0 && delivered[q]==0);}
  // Actual Index changes source tables, but queued WQEs already hold immutable
  // fallback addresses. Its other writes are metadata, never the KV payload.
  info.fill(0);for(auto& row:rows)row.pending=0;
  if(fault==6){rows[0].pending=1;rows[0].error=32;}
  if(fault==7 && safeCounts[badQueue])entities[badQueue].version=7;
  if(fault==8 && safeCounts[badQueue])bad->pending=2; // corrupt checkpoint count must not consume arbitrary CQEs
  phase=2;for(unsigned core=0;core<8;++core)CompleteAsyncFinal(&ctx,core);
  for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w){
   unsigned q=FinalQueue(2,4,b,w);auto* post=FinalPostAt(128,&ctx,b*4+w);
   bool invalid= b==0 && (fault==6 || (w==2 && (fault==1 || fault==2 || fault==3 || fault==4 ||
      (fault==5 && safe==3) || ((fault==7 || fault==8) && safeCounts[q]) || fault==9)));
   assert(bool(post->error)==invalid);
   if(!invalid){assert(!post->pending && !post->deferred && delivered[q]==counts[q]);assert(nic[q].empty());
    assert(posts[q]==unsigned(safeCounts[q]!=0)+unsigned(counts[q]!=safeCounts[q]));
    if(counts[q])assert(queues[q].completed_batch_count==1 && post->head==1);
   } else if(post->pending){assert(queues[q].poisoned);}
   if(fault==6 && b==0){assert(delivered[q]==safeCounts[q]);assert(nic[q].empty());}
  }
  if(fault==1 || fault==9){
   assert(queues[badQueue].poisoned && queues[badQueue].last_error);
   unsigned oldPosts=posts[badQueue];bad->pending=bad->error=0;bad->deferred=1;
   phase=0;SubmitSafeFinal(&ctx,2);assert(bad->error && posts[badQueue]==oldPosts);
  }
  if(fault==0) {
   // Init clears per-replay post state, never persistent queue cursors. Two
   // correct replays retire exactly their own range checkpoints on each Jetty.
   phase=0;
   for(unsigned core=0;core<8;++core) {
    auto* post=FinalPostAt(128,&ctx,core);post->head=0;post->deferred=post->count!=0;
    SubmitSafeFinal(&ctx,core);
   }
   phase=2;for(unsigned core=0;core<8;++core)CompleteAsyncFinal(&ctx,core);
   for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w) {
    unsigned q=FinalQueue(2,4,b,w);auto* post=FinalPostAt(128,&ctx,b*4+w);
    assert(!post->error && !post->pending && !post->deferred && nic[q].empty());
    assert(delivered[q]==2*counts[q]);
    if(counts[q])assert(queues[q].completed_batch_count==2 && post->head==2);
   }
  }
 }
}
"""
    cpp, binary = tmp_path / "async.cpp", tmp_path / "async"
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


def test_async_submit_is_after_resolver_ub_release_and_before_index():
    lifecycle = (OP / "op_kernel/lifecycle.cpp").read_text()
    resolve = lifecycle[lifecycle.index('extern "C" __global__ __aicore__ void v9_resolve_kernel') :]
    assert resolve.index("ReplicatedFinal(") < resolve.index("SubmitSafeFinal(")
    resolver = (OP / "op_kernel/resolver.h").read_text()
    assert resolver.rstrip().endswith("pipe.Destroy();\n}\n}")
    assert "#if V9_URMA_ASYNC_FINAL\n        CompleteAsyncFinal(" in lifecycle
