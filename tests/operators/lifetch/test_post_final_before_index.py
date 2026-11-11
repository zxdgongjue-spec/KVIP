# SPDX-License-Identifier: Apache-2.0
"""Actual whole-range submission and delayed completion, without an NPU."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch"


@pytest.mark.parametrize("merged", (False, True))
@pytest.mark.parametrize("resolve_submit", (False, True))
def test_final_submission_joins_prefetch_then_survives_index_and_failures(tmp_path, merged, resolve_submit):
    (tmp_path / "batch_get_abi.h").write_text("")
    lifecycle = (OP / "op_kernel/lifecycle.cpp").read_text()
    start = lifecycle.index("__aicore__ inline void FinishFetch(")
    finish = lifecycle[start : lifecycle.index("\n}\n#endif", start) + 2]
    start = lifecycle.index('extern "C" __global__ __aicore__ void v9_resolve_kernel(')
    resolve_entry = lifecycle[start : lifecycle.index("// Final transport", start)]
    planner = (OP.parent / "common/lifetch/index_cache_update.h").read_text()
    start = planner.index("template<uint32_t Threads>\n__simt_vf__", planner.index("// Finish the early update"))
    cleanup = planner[start:]
    source = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __global__
#define __gm__
#define __simt_vf__
#define LAUNCH_BOUND(x)
#define __CCE_AICORE__ 1
#define V9_URMA_PARALLEL_FINAL 1
#define V9_POST_FINAL_BEFORE_INDEX 1
#define V9_URMA_SINGLE_PLANNER 1
#define V9_URMA_SKIP_FINAL_PARTITION 1
#define KERNEL_TASK_TYPE_DEFAULT(x)
#define ASCEND_IS_AIV (true)
#include "context.h"
using namespace rotating;
using GM_ADDR=uint8_t*;
struct {unsigned x=0;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
void asc_threadfence(){}
namespace Simt {
struct Dim3{unsigned x,y,z;};
template<auto Fn,class... Args>void VF_CALL(Dim3 dims,Args... args){
 for(threadIdx.x=0;threadIdx.x<dims.x;++threadIdx.x)Fn(args...);threadIdx.x=0;
}
}
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
std::array<bool,2> planned{};
bool plannerUbLive=false;
unsigned phase=0,badQueue=0,fault=0,sourceGeneration=1;
unsigned currentBlock=0;uint32_t* hotValidity=nullptr;
unsigned GetBlockIdx(){return currentBlock;}
Context* GetContext(GM_ADDR bindings){return reinterpret_cast<Context*>(reinterpret_cast<uint64_t*>(bindings)[14]);}
void PublishDestinationResidency(unsigned resident,Context*,unsigned b,unsigned){
 if(!threadIdx.x)std::fill(hotValidity+b*resident,hotValidity+(b+1)*resident,1);
}
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
 assert(phase==0 && planned[b] && !plannerUbLive && !joined[b]);joined[b]=true;
 auto& row=reinterpret_cast<Row*>(ctx->rows)[b];
 assert(row.pending==3);
 row.error=fault==6 && !b?32:0;row.pending=row.error?1:0;
}
// The real planner is covered by the resolver models. Here its published
// descriptor/TPipe lifetime is the boundary exercised by the actual entry.
void ReplicatedFinal(unsigned batch,unsigned,unsigned,unsigned,unsigned,unsigned,Context* ctx){
 const unsigned b=GetBlockIdx();assert(b<batch && !planned[b] && !plannerUbLive);
 plannerUbLive=true;
 if(fault==14 && !b)reinterpret_cast<uint32_t*>(ctx->rowStatus)[0]=64;
 planned[b]=true;plannerUbLive=false;
}
#include "async_final.h"
#include "post_final_before_index.h"
RESOLVE_ENTRY
CLEANUP
FINISH
void CompleteAll(Context* ctx,uint64_t* update){
 uint64_t bindings[15]{};bindings[14]=uint64_t(ctx);
#if !V9_MERGE_FINAL_COMPLETION
 for(unsigned core=8;core-->0;)CompletePostedFinal(ctx,core);CompletePostedFinal(ctx,8);
#endif
 for(currentBlock=0;currentBlock<2;++currentBlock)
  FinishFetch(reinterpret_cast<GM_ADDR>(bindings),reinterpret_cast<GM_ADDR>(update));
}
int main(){
 for(unsigned scenario=0;scenario<15;++scenario){
  fault=scenario;phase=0;sourceGeneration=1;queues={};statuses={};entities={};nic={};joined={};
  counts={};descriptors={};delivered={};posts={};waits={};flushes={};
  planned={};plannerUbLive=false;
  Context ctx{};ctx.shape.batch=2;ctx.shape.finalWorkers=4;ctx.shape.finalCores=8;ctx.shape.finalPostOffset=128;
  ctx.shape.resident=16;
  std::array<Row,2> rows{};std::array<uint32_t,2*kRowStatusWords> info{};
  std::array<KvafBatchGetCoreStatus,4> finalStatus{};
  std::array<uint32_t,2> updateStatus{},ownerIds{0,1};std::array<uint32_t,32> valid{};hotValidity=valid.data();
  uint64_t update[28]{};update[6]=uint64_t(updateStatus.data());update[9]=uint64_t(ownerIds.data());
  update[16]=uint64_t(valid.data());update[25]=16;
  ctx.finalStatus=uint64_t(finalStatus.data());
  alignas(128) std::array<uint8_t,128+8*sizeof(FinalPost)> owners{};
  ctx.rows=uint64_t(rows.data());ctx.rowStatus=uint64_t(info.data());ctx.owners=uint64_t(owners.data());
  ctx.sourceBlocks=1;ctx.sourceOffsets=2;ctx.destinations=3;ctx.count=4;ctx.fallbackAddresses=5;ctx.fallbackValid=6;
  for(unsigned b=0;b<2;++b){rows[b].pending=3;rows[b].finalPending=1;for(unsigned w=0;w<4;++w){
   unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
   counts[q]=p->count=w==3?0:7+w;descriptors[q]=p->descriptor=b*kSlots+w*16;
   p->nonconflicting=w==1?3:0;p->deferred=p->count!=0;info[b*kRowStatusWords+1]+=p->count;
  }}
  const unsigned badWorker=V9_MERGE_FINAL_COMPLETION?0:2;
  badQueue=FinalQueue(2,4,0,badWorker);auto* bad=FinalPostAt(128,&ctx,badWorker);
  if(fault==2){queues[badQueue].initialized=1;queues[badQueue].magic=19;queues[badQueue].poisoned=1;}
  if(fault==3)entities[badQueue].local_hbm_address=999;
  if(fault==4)bad->descriptor=2*kSlots; // whole range would overrun descriptors
  if(fault==5)entities[badQueue].sq_depth=8; // invalid native SQ geometry
  if(fault==11)entities[badQueue].cq_depth=0;
  if(fault==13)bad->count=kSlots+1; // exceeds the resolver range contract
  uint64_t bindings[15]{};bindings[14]=uint64_t(&ctx);
  currentBlock=1;v9_resolve_kernel(reinterpret_cast<GM_ADDR>(bindings));
  // The merged path can post this row while a different row is unplanned.
  assert(!planned[0] && planned[1]);
  assert(posts[FinalQueue(2,4,1,0)]==unsigned(V9_RESOLVE_SUBMIT));
  currentBlock=0;v9_resolve_kernel(reinterpret_cast<GM_ADDR>(bindings));
#if !V9_RESOLVE_SUBMIT
  SubmitFinalBeforeIndex(&ctx,1);SubmitFinalBeforeIndex(&ctx,0);
#endif
  SubmitFinalBeforeIndex(&ctx,2);
  for(unsigned q=0;q<10;++q)assert(waits[q]==0 && delivered[q]==0);
  if(fault==10){
   // An accidental second submission must not append a second SQ range.
   rows[0].pending=3;joined[0]=false;auto old=posts;
   SubmitFinalBeforeIndex(&ctx,0);assert(posts==old);
  }
  phase=1;sourceGeneration=2; // Index mutates metadata, never the KV payload.
  if(fault==7)entities[badQueue].version=7;
  if(fault==8)bad->pending=2; // cannot safely infer which CQEs this represents
  phase=2;CompleteAll(&ctx,update);
  for(unsigned b=0;b<2;++b){
   const bool error=!b && fault!=0;
   assert(bool(rows[b].finalPending)==error && bool(updateStatus[b])==error);
   for(unsigned slot=0;slot<16;++slot)assert(valid[b*16+slot]==unsigned(!error));
  }
  for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w){
   const unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
   const bool invalid=!b && (fault==6 || fault==14 || (fault==10 && w<3) || (w==badWorker && fault!=0 && fault!=10));
   assert(bool(p->error)==invalid);
   if(!invalid){assert(!p->pending && !p->deferred && delivered[q]==counts[q] && nic[q].empty());
    assert(posts[q]==unsigned(counts[q]!=0) && waits[q]==posts[q]);
    if(counts[q])assert(queues[q].completed_batch_count==1 && p->head==1);
   }else if(p->pending){assert(queues[q].poisoned);}
   if((fault==6 || fault==14) && !b)assert(posts[q]==0 && waits[q]==0 && nic[q].empty());
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
   phase=2;CompleteAll(&ctx,update);
   for(unsigned b=0;b<2;++b)for(unsigned w=0;w<4;++w){
    unsigned q=FinalQueue(2,4,b,w);auto* p=FinalPostAt(128,&ctx,b*4+w);
    assert(!p->error && !p->pending && !p->deferred && nic[q].empty() && delivered[q]==2*counts[q]);
    if(counts[q])assert(queues[q].completed_batch_count==2 && p->head==2);
   }
  }
 }
}
""".replace("CLEANUP", cleanup)
        .replace("FINISH", finish)
        .replace("RESOLVE_ENTRY", resolve_entry)
    )
    cpp, binary = tmp_path / "post_before_index.cpp", tmp_path / "post_before_index"
    cpp.write_text(source)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DV9_MERGE_FINAL_COMPLETION={int(merged)}",
            f"-DV9_RESOLVE_SUBMIT={int(resolve_submit)}",
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


@pytest.mark.parametrize("merged", (False, True))
@pytest.mark.parametrize("resolve_submit", (False, True))
def test_forward_posts_before_index_and_drains_afterwards(tmp_path, merged, resolve_submit):
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
unsigned v9_submit_final(void*,void*,const rotating::RuntimeShape*){assert(!V9_RESOLVE_SUBMIT);return call(4);}
unsigned v9_wait(void*,void*,const rotating::RuntimeShape*){assert(false);return 0;}
unsigned v9_parallel_fetch(void*,void*,const rotating::RuntimeShape*){assert(!V9_MERGE_FINAL_COMPLETION);return call(6-V9_RESOLVE_SUBMIT);}
extern "C" unsigned kvop_direct_update_outputs_early(unsigned,void*,void*){return call(5-V9_RESOLVE_SUBMIT);}
extern "C" unsigned aclrtlaunch_v9_fetch_update_kernel(unsigned,void*,void*,void*){return call((V9_MERGE_FINAL_COMPLETION?6:7)-V9_RESOLVE_SUBMIT);}
ENTRY
int main(){rotating::RuntimeShape shape{};const unsigned steps=(V9_MERGE_FINAL_COMPLETION?6:7)-V9_RESOLVE_SUBMIT;
 for(fail=0;fail<=steps;++fail){step=0;assert(kvip_lifetch_forward((void*)1,(void*)2,(void*)3,&shape)==(fail?77:0));
  assert(step==(fail?fail:steps));}
}
""".replace("ENTRY", entry)
    cpp, binary = tmp_path / "launch.cpp", tmp_path / "launch"
    cpp.write_text(source)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            f"-DV9_MERGE_FINAL_COMPLETION={int(merged)}",
            f"-DV9_RESOLVE_SUBMIT={int(resolve_submit)}",
            f"-I{OP / 'include'}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize(
    "enabled,single_planner,post_before,valid",
    ((0, 0, 0, True), (1, 1, 1, True), (1, 0, 1, False), (1, 1, 0, False), (2, 1, 1, False)),
)
def test_resolve_submit_native_contract_and_getter(tmp_path, enabled, single_planner, post_before, valid):
    (tmp_path / "batch_get_abi.h").touch()
    text = (OP / "op_host/launch.cpp").read_text()
    getter = next(line for line in text.splitlines() if "uint32_t v9_resolve_submit(" in line)
    source = tmp_path / "contract.cpp"
    source.write_text(
        '#include "context.h"\n#include <cassert>\n'
        + getter
        + "\nint main(){assert(v9_resolve_submit(nullptr)==V9_RESOLVE_SUBMIT);}\n"
    )
    binary = tmp_path / "contract"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-DV9_URMA_PARALLEL_FINAL=1",
            f"-DV9_RESOLVE_SUBMIT={enabled}",
            f"-DV9_URMA_SINGLE_PLANNER={single_planner}",
            f"-DV9_POST_FINAL_BEFORE_INDEX={post_before}",
            f"-I{tmp_path}",
            f"-I{OP / 'include'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if valid:
        assert result.returncode == 0, result.stderr
        subprocess.run([str(binary)], check=True)
    else:
        assert result.returncode != 0
        assert "Resolve submission" in result.stderr
