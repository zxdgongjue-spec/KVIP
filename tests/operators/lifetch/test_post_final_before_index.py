# SPDX-License-Identifier: Apache-2.0
"""Actual whole-range submission and delayed completion, without an NPU."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch"


def test_final_submission_joins_prefetch_then_survives_index_and_failures(tmp_path):
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
#define V9_POST_FINAL_BEFORE_INDEX 1
#define V9_URMA_SINGLE_PLANNER 1
#define V9_URMA_SKIP_FINAL_PARTITION 1
#include "context.h"
using namespace rotating;
struct KvafBatchGetQueueState {
 uint64_t magic=0,initialized=0,sq_producer_index=0,cq_consumer_index=0,completed_batch_count=0,last_error=0,poisoned=0;
};
struct KvafBatchGetCoreStatus {uint64_t result=0,processed=0;};
struct KvafBatchGetDirectUrmaEntity {
 uint64_t magic=17,version=1,local_hbm_address=123,local_hbm_size=65536,local_token_id=1;
 uint64_t remote_host_ddr_address=234,remote_host_ddr_size=65536,sq_va=1,sq_db_va=2,sq_depth=32768,wqe_size=64;
 uint64_t cq_va=3,cq_db_va=4,cq_depth=1,cqe_size=64;
};
struct KvafExecuteBatchGetKernelArgs {
 uint64_t queue_state=0,core_status=0,channel_entity=0,indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0;
 uint64_t output=123;uint32_t request_capacity=0,table_capacity=0,output_rows=2048,token_bytes=2624,tokens_per_block=128;
 uint32_t block_count=1,jettys_per_core=1,batch_get_max_inflight_total=0;
};
struct Pending {unsigned begin,count,sourceGeneration;};
std::array<KvafBatchGetQueueState,10> queues;
std::array<KvafBatchGetCoreStatus,10> statuses;
std::array<KvafBatchGetDirectUrmaEntity,10> entities;
std::array<std::vector<Pending>,10> nic;
std::array<unsigned,10> counts{},descriptors{},delivered{},posts{},waits{},flushes{};
std::array<bool,2> joined{};
unsigned phase=0,badQueue=0,fault=0,sourceGeneration=1;
namespace rotating {
FinalPost* FinalPostAt(uint32_t offset,Context* ctx,uint32_t core){
 return reinterpret_cast<FinalPost*>(ctx->owners+offset)+core;
}
}
KvafExecuteBatchGetKernelArgs GetArgs(Context*,unsigned q){
 KvafExecuteBatchGetKernelArgs args;
 args.queue_state=uint64_t(&queues.at(q));args.core_status=uint64_t(&statuses.at(q));args.channel_entity=uint64_t(&entities.at(q));
 if(fault==12 && q==badQueue)args.batch_get_max_inflight_total=2048;
 return args;
}
namespace kvaf_device {
constexpr unsigned kInvalidArgs=1,kInvalidEntity=2,kQueueFull=4,kInvalidQueue=8,kCqTimeout=16;
constexpr unsigned kEntityMagic=17,kEntityVersion=1,kQueueMagic=19,kSqDepthWqebbs=32768,kWqeBytes=64,kCqeProbeBytes=32;
void CleanCacheLines(uint8_t* pointer,size_t bytes){
 for(unsigned q=0;q<queues.size();++q)if(pointer==reinterpret_cast<uint8_t*>(&queues[q])){
  assert(bytes==sizeof(queues[q]));++flushes[q];
 }
}
void ConfigureUbToGmCopy(){}
void PostDirectRequests(const int32_t* indices,const int32_t* out,const int32_t* offsets,const uint64_t* addresses,
 unsigned begin,unsigned count,unsigned first,unsigned width,unsigned jettys,const KvafBatchGetDirectUrmaEntity* entity,
 KvafBatchGetQueueState* queue,unsigned profile,uint64_t*,uint64_t*){
 unsigned q=unsigned(queue-queues.data()),b=2;
 for(unsigned row=0;row<2;++row)for(unsigned w=0;w<4;++w)if(FinalQueue(2,4,row,w)==q)b=row;
 assert(b<2 && joined[b] && phase==0 && entity==&entities[q] && first==0 && jettys==1 && width==2624 && !profile);
 assert(queue->initialized && !queue->poisoned && flushes[q]>0 && count>0);
 assert(uint64_t(indices)==1 && uint64_t(offsets)==2 && uint64_t(out)==3 && uint64_t(addresses)==5);
 assert(begin==descriptors[q] && count==counts[q]);
 nic[q].push_back({begin,count,sourceGeneration});queue->sq_producer_index+=count;++posts[q];
}
uint64_t WaitForCompletion(const KvafBatchGetDirectUrmaEntity* entity,KvafBatchGetQueueState* queue,unsigned*){
 unsigned q=unsigned(queue-queues.data());assert(phase==2 && entity==&entities[q] && nic[q].size()==1);++waits[q];
 if(fault==1 && q==badQueue)return kCqTimeout;
 assert(nic[q].front().sourceGeneration==1);
 delivered[q]+=nic[q].front().count;nic[q].clear();++queue->cq_consumer_index;
 return fault==9 && q==badQueue?32:0;
}
void WriteStatus(KvafBatchGetCoreStatus* status,uint64_t result,uint64_t processed,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t){
 status->result=result;status->processed=processed;
}
}
void CompletePrefetch(Context* ctx,unsigned b){
 assert(phase==0 && !joined[b]);joined[b]=true;
 auto& row=reinterpret_cast<Row*>(ctx->rows)[b];
 assert(row.pending==3);
 row.error=fault==6 && !b?32:0;row.pending=row.error?1:0;
}
#include "async_final.h"
#include "post_final_before_index.h"
int main(){
 for(unsigned scenario=0;scenario<14;++scenario){
  fault=scenario;phase=0;sourceGeneration=1;queues={};statuses={};entities={};nic={};joined={};
  counts={};descriptors={};delivered={};posts={};waits={};flushes={};
  Context ctx{};ctx.shape.batch=2;ctx.shape.finalWorkers=4;ctx.shape.finalCores=8;ctx.shape.finalPostOffset=128;
  std::array<Row,2> rows{};std::array<uint32_t,2*kRowStatusWords> info{};
  alignas(128) std::array<uint8_t,128+8*sizeof(FinalPost)> owners{};
  ctx.rows=uint64_t(rows.data());ctx.rowStatus=uint64_t(info.data());ctx.owners=uint64_t(owners.data());
  ctx.sourceBlocks=1;ctx.sourceOffsets=2;ctx.destinations=3;ctx.count=4;ctx.fallbackAddresses=5;ctx.fallbackValid=6;
  for(unsigned b=0;b<2;++b){rows[b].pending=3;for(unsigned w=0;w<4;++w){
   unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
   counts[q]=p->count=w==3?0:7+w;descriptors[q]=p->descriptor=b*kSlots+w*16;
   p->nonconflicting=w==1?3:0;p->deferred=p->count!=0;
  }}
  badQueue=FinalQueue(2,4,0,2);auto* bad=FinalPostAt(128,&ctx,2);
  if(fault==2){queues[badQueue].initialized=1;queues[badQueue].magic=19;queues[badQueue].poisoned=1;}
  if(fault==3)entities[badQueue].local_hbm_address=999;
  if(fault==4)bad->descriptor=2*kSlots; // whole range would overrun descriptors
  if(fault==5)entities[badQueue].sq_depth=8; // invalid native SQ geometry
  if(fault==11)entities[badQueue].cq_depth=0;
  if(fault==13)bad->count=kSlots+1; // exceeds the resolver range contract
  SubmitFinalBeforeIndex(&ctx,1);SubmitFinalBeforeIndex(&ctx,0);SubmitFinalBeforeIndex(&ctx,2);
  for(unsigned q=0;q<10;++q)assert(waits[q]==0 && delivered[q]==0);
  if(fault==10){
   // An accidental second submission must not append a second SQ range.
   rows[0].pending=3;joined[0]=false;auto old=posts;
   SubmitFinalBeforeIndex(&ctx,0);assert(posts==old);
  }
  phase=1;sourceGeneration=2; // Index mutates metadata, never the KV payload.
  if(fault==7)entities[badQueue].version=7;
  if(fault==8)bad->pending=2; // cannot safely infer which CQEs this represents
  phase=2;for(unsigned core=8;core-->0;)CompletePostedFinal(&ctx,core);CompletePostedFinal(&ctx,8);
  for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w){
   const unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
   const bool invalid=!b && (fault==6 || (fault==10 && w<3) || (w==2 && fault!=0 && fault!=10));
   assert(bool(p->error)==invalid);
   if(!invalid){assert(!p->pending && !p->deferred && delivered[q]==counts[q] && nic[q].empty());
    assert(posts[q]==unsigned(counts[q]!=0) && waits[q]==posts[q]);
    if(counts[q])assert(queues[q].completed_batch_count==1 && p->head==1);
   }else if(p->pending){assert(queues[q].poisoned);}
   if(fault==6 && !b)assert(posts[q]==0 && waits[q]==0 && nic[q].empty());
  }
  if(fault==1 || fault==7 || fault==8 || fault==9){
   const unsigned oldPosts=posts[badQueue];assert(queues[badQueue].poisoned);
   bad->pending=bad->error=0;bad->deferred=1;rows[0].pending=3;joined[0]=false;phase=0;
   SubmitFinalBeforeIndex(&ctx,0);assert(bad->error && posts[badQueue]==oldPosts);
  }
  if(!fault){
   phase=0;sourceGeneration=1;joined={};for(auto& row:rows)row.pending=3;
   for(unsigned core=0;core<8;++core){auto* p=FinalPostAt(128,&ctx,core);p->head=0;p->deferred=p->count!=0;}
   SubmitFinalBeforeIndex(&ctx,0);SubmitFinalBeforeIndex(&ctx,1);
   phase=2;for(unsigned core=0;core<8;++core)CompletePostedFinal(&ctx,core);
   for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w){
    unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
    assert(!p->error && !p->pending && !p->deferred && nic[q].empty() && delivered[q]==2*counts[q]);
    if(counts[q])assert(queues[q].completed_batch_count==2 && p->head==2);
   }
  }
 }
}
"""
    cpp, binary = tmp_path / "post_before_index.cpp", tmp_path / "post_before_index"
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


def test_forward_posts_before_index_and_drains_afterwards(tmp_path):
    text = (OP / "op_host/launch.cpp").read_text()
    entry = text[text.index('extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,') :]
    source = r"""
#include <cstdint>
#include <cassert>
#define V9_URMA_PARALLEL_FINAL 1
#define V9_POST_FINAL_BEFORE_INDEX 1
#include "runtime_shape.h"
unsigned step=0,fail=0;
unsigned call(unsigned n){assert(++step==n);return fail==n?77:0;}
unsigned v9_init(void*,void*,const rotating::RuntimeShape*){return call(1);}
unsigned v9_li(void*,void*,const rotating::RuntimeShape*){return call(2);}
unsigned v9_resolve(void*,void*,const rotating::RuntimeShape*){return call(3);}
unsigned v9_submit_final(void*,void*,const rotating::RuntimeShape*){return call(4);}
unsigned v9_wait(void*,void*,const rotating::RuntimeShape*){assert(false);return 0;}
unsigned v9_parallel_fetch(void*,void*,const rotating::RuntimeShape*){return call(6);}
extern "C" unsigned kvop_direct_update_outputs_early(unsigned,void*,void*){return call(5);}
extern "C" unsigned aclrtlaunch_v9_fetch_update_kernel(unsigned,void*,void*,void*){return call(7);}
ENTRY
int main(){rotating::RuntimeShape shape{};
 for(fail=0;fail<=7;++fail){step=0;assert(kvip_lifetch_forward((void*)1,(void*)2,(void*)3,&shape)==(fail?77:0));
  assert(step==(fail?fail:7));}
}
""".replace("ENTRY", entry)
    cpp, binary = tmp_path / "launch.cpp", tmp_path / "launch"
    cpp.write_text(source)
    subprocess.run(["c++", "-std=c++17", f"-I{OP / 'include'}", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
