# SPDX-License-Identifier: Apache-2.0
"""Type-check the complete tail and execute its real descriptor packing.

KVAF's CPU device model supplies the native API, while the small Ascend host
stubs below supply TPipe and cross-core declarations. This is not a substitute
for CANN compilation, SyncAll validation, or device cache-coherence testing.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_collective_tail_uses_batch_scheduled_mixed_vector_kernel():
    repo = Path(__file__).resolve().parents[4]
    kernel = repo / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    source = (kernel / "streaming.cpp").read_text()
    entry = source.split("void v9_streaming_resolve_update(GM_ADDR arg)", 1)[1]
    assert "__schedmode__(1) void v9_streaming_resolve_update" in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)" in entry
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)" in entry
    assert entry.index("state->active==1") < entry.index("post_index_tail::Run")
    tail = (kernel / "post_index_tail.h").read_text()
    # Init's route decision is immutable until every AIV has selected the
    # collective branch; a transient zero lets late AIVs miss the barrier.
    assert "if (i != 14) words[i] = 0" in tail
    assert "FinalizeStreaming<false, true>" in tail
    transport = (kernel / "post_index_transport.h").read_text()
    assert "RunBatchGetValidation(" not in transport
    assert "ValidateBatchGetRequests" not in transport
    assert "args.tokens_per_block = 1" in tail
    assert tail.index("post_index_transport::Preflight(args") < tail.index("Simt::VF_CALL<Pack>")
    assert "OwnerRow(ctx" not in tail
    assert "16384 * 4" not in tail
    assert "prefix[i % batch + 1] += uint32_t(records[i].count)" in tail
    assert "ApplyPlan<false>" in tail and "ApplyPlan<true>" in tail
    # OutputWorker's old diagnostic slots overlap the per-core scratch. Only
    # initialize the new ABI after those writers finish at the first barrier.
    assert tail.index("OutputWorker(a") < tail.index("SyncAll();") < tail.index("trace[2] = 0x50495831U")
    assert tail.count("RowError(records,") == 1  # one final join after every CQ


def test_whole_tail_header_and_q4_pack(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    kvaf = Path(os.environ.get("KVAF_LEGACY_SOURCE_ROOT", repo.parent / "ubmem-pr" / "KVAF-urma-check"))
    native = kvaf / "access_operator/demo/aiv_batch_get"
    model = kvaf / "tests/aiv_model"
    if not (model / "kernel_operator.h").is_file():
        pytest.skip("Set KVAF_LEGACY_SOURCE_ROOT to the original fused-context KVAF checkout")
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler
    source = tmp_path / "post_index_tail.cpp"
    source.write_text(r'''
#include "batch_get_model.h"
#include <cstddef>
#include <array>
#include <random>
#define KVIP_OUTPUT_WORKER 1
#include "pipeline_types.h"
#include "runtime_shape.h"
using namespace AscendC;
using namespace pipeline_ab;
using namespace rotating;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
void asc_atomic_or(uint32_t* p,uint32_t value){*p|=value;}
void asc_atomic_add(uint32_t* p,uint32_t value){*p+=value;}
enum class HardEvent {V_S};
template<HardEvent>void SetFlag(uint32_t){}
template<HardEvent>void WaitFlag(uint32_t){}
constexpr uint32_t EVENT_ID0=0,kSlots=8192,kTop=2048,kRowStatusWords=32;
enum class TPosition {VECCALC};
template<class T>struct LocalTensor {
 T* p; T* GetPhyAddr(){return p;} T GetValue(uint32_t i){return p[i];}
};
template<TPosition>struct TBuf {
 std::vector<uint64_t> storage;
 template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(storage.data())};}
};
struct TPipe {
 template<TPosition P>void InitBuffer(TBuf<P>& buf,uint32_t bytes){buf.storage.resize((bytes+7)/8);}
 void Destroy(){}
};
void SyncAll(){}
struct Context {
 RuntimeShape shape;
 uint64_t sourceBlocks,sourceOffsets,destinations,rowStatus,fallbackAddresses,fallbackValid,rows,rowOwners;
};
struct alignas(64) Row {uint64_t entity,queue,addresses,output;uint32_t slots,requests,suppressed,error;};
RuntimeShape LoadRuntimeShape(Context* ctx){return ctx->shape;}
// Negative control: scalar code must never call the SIMT-only owner helper.
uint32_t OwnerRow(Context*,uint32_t)=delete;
struct FinalPost {uint32_t count,pending,error,deferred,descriptor;};
FinalPost finalPosts[4]{};
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t index){return finalPosts+index;}
KvafExecuteBatchGetKernelArgs nativeArgs{};
KvafExecuteBatchGetKernelArgs GetArgs(Context*,uint32_t){return nativeArgs;}
template<bool>uint32_t Producer(Args*,uint32_t){return 0;}
void OutputWorker(Args*,uint32_t){}
void PublishStreamingUpdateStatus(uint32_t*,uint32_t,uint32_t){}
void PublishStreamingResidency(uint32_t,uint32_t,Context*,uint32_t){}
template<bool=false,bool=false>void FinalizeStreaming(Args*,uint32_t,uint32_t){}
#include "post_index_tail.h"

int main(){
 using namespace pipeline_ab::post_index_tail;
 static_assert(offsetof(Record,active)==14*sizeof(uint64_t));
 static_assert(sizeof(Record)==128 && alignof(Record)==128);
 std::mt19937 random(1117);
 constexpr uint32_t batch=16,cores=64;
 std::vector<int32_t> blocks(batch*kSlots),offsets(batch*kSlots),destinations(batch*kSlots);
 std::vector<int32_t> packed(batch*9*kSlots,-1);
 std::vector<uint64_t> addresses(batch*kSlots);
 std::vector<uint8_t> valid(batch*kSlots,1);
 constexpr uint64_t remoteBase=0x10000000,remoteBytes=uint64_t(batch)*kSlots*656;
 constexpr uint32_t outputRows=batch*kSlots+8192;
 constexpr uint64_t localBytes=uint64_t(outputRows)*656;
 for(uint32_t i=0;i<batch*kSlots;++i){blocks[i]=i;offsets[i]=0;destinations[i]=i+8192;addresses[i]=remoteBase+uint64_t(i)*656;}

 Context ctx{};
 ctx.sourceBlocks=reinterpret_cast<uint64_t>(blocks.data());
 ctx.sourceOffsets=reinterpret_cast<uint64_t>(offsets.data());
 ctx.destinations=reinterpret_cast<uint64_t>(destinations.data());
 ctx.fallbackAddresses=reinterpret_cast<uint64_t>(addresses.data());
 ctx.fallbackValid=reinterpret_cast<uint64_t>(valid.data());
 const std::array<uint32_t,4> special[]={{0,0,0,0},{2048,2048,2048,2048},
   {0,0,0,2048},{0,0,0,1},{1,0,17,0},{0,2048,0,2048},{3,5,11,17}};
 for(uint32_t repeat=0;repeat<32;++repeat){
  for(uint32_t b=0;b<batch;++b){
   uint32_t n[4];
   for(uint32_t q=0;q<4;++q)n[q]=repeat<7?special[repeat][q]:random()%2049;
   uint32_t total=n[0]+n[1]+n[2]+n[3];
   auto out=packed.data()+b*3*kSlots;
   for(uint32_t lanes:{1U,2U,(cores-1-b)/batch+1,64U}){
    std::fill_n(out,3*kSlots,-1);
   // Reverse scheduling ensures correctness does not depend on AIV order.
   for(uint32_t lane=lanes;lane--;){
    uint32_t begin=uint64_t(total)*lane/lanes,end=uint64_t(total)*(lane+1)/lanes;
    assert(end-begin<=(total+lanes-1)/lanes);
    uint32_t error=0;
    Simt::VF_CALL<Pack>(Simt::Dim3{256,1,1},&ctx,b,begin,end-begin,n[0],n[1],n[2],n[3],out,
      batch*kSlots,outputRows,656,remoteBase,remoteBytes,localBytes,&error);
    assert(!error);
   }
   uint32_t cursor=0;
   for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<n[q];++i,++cursor){
    auto src=b*kSlots+q*kTop+i;
    assert(out[cursor]==blocks[src]);
    assert(out[kSlots+cursor]==offsets[src]);
    assert(out[2*kSlots+cursor]==destinations[src]);
   }
   for(uint32_t i=total;i<kSlots;++i){
    assert(out[i]==-1 && out[kSlots+i]==-1 && out[2*kSlots+i]==-1);
   }
   }
  }
 }
 auto check=[&](uint64_t remote=remoteBytes,uint64_t local=localBytes){
  uint32_t error=0;
  std::fill(packed.begin(),packed.end(),-1);
  Simt::VF_CALL<Pack>(Simt::Dim3{256,1,1},&ctx,0,0,1,1,0,0,0,packed.data(),
    batch*kSlots,outputRows,656,remoteBase,remote,local,&error);
  if(error)assert(packed[0]==-1&&packed[kSlots]==-1&&packed[2*kSlots]==-1);
  return error;
 };
 assert(!check());
 blocks[0]=-1;assert(check()==kvaf_device::kInvalidLookup);
 blocks[0]=batch*kSlots;assert(check()==kvaf_device::kInvalidLookup);blocks[0]=0;
 valid[0]=0;assert(check()==kvaf_device::kInvalidLookup);valid[0]=1;
 offsets[0]=1;assert(check()==kvaf_device::kInvalidLookup);
 offsets[0]=-1;assert(check()==kvaf_device::kInvalidLookup);offsets[0]=0;
 destinations[0]=-1;assert(check()==kvaf_device::kInvalidOutput);
 destinations[0]=outputRows;assert(check()==kvaf_device::kInvalidOutput);destinations[0]=8192;
 assert(check(remoteBytes,655)==kvaf_device::kInvalidOutput);
 addresses[0]=remoteBase-1;assert(check()==kvaf_device::kInvalidLookup);
 // A frozen per-token address at the very end is legal; validating a full
 // 128-token block here would incorrectly reject this legitimate read.
 addresses[0]=remoteBase+remoteBytes-656;assert(!check());
 ++addresses[0];assert(check()==kvaf_device::kInvalidLookup);
 addresses[0]=UINT64_MAX;assert(check()==kvaf_device::kInvalidLookup);
 addresses[0]=remoteBase;assert(check(655)==kvaf_device::kInvalidLookup);

 // Execute the production Run error path, with otherwise usable descriptors
 // and native entities. A pre-existing row error or a speculative write must
 // stop before any doorbell, even when resolver status reports success.
 Row row{};ctx.rows=reinterpret_cast<uint64_t>(&row);
 ctx.shape.batch=1;ctx.shape.aicCount=1;ctx.shape.resident=2048;ctx.shape.history=4096;
 uint32_t status[32]{};ctx.rowStatus=reinterpret_cast<uint64_t>(status);
 std::array<Record,2> records{};
 std::array<uint64_t,28> update{};
 std::vector<uint32_t> eventSources(10240),eventDestinations(10240);
 std::vector<uint8_t> sq(32768*64),cq(8*64),output(localBytes);
 uint32_t sqDoorbell=0,cqDoorbell=0;int32_t missCount=2;
 KvafBatchGetQueueState queue{};KvafBatchGetCoreStatus nativeStatus{};
 KvafBatchGetDirectUrmaEntity entity{};
 entity.magic=kvaf_device::kEntityMagic;entity.version=kvaf_device::kEntityVersion;
 entity.remote_host_ddr_address=remoteBase;entity.remote_host_ddr_size=remoteBytes;
 entity.local_hbm_address=reinterpret_cast<uint64_t>(output.data());entity.local_hbm_size=localBytes;
 entity.local_token_id=1;entity.sq_va=reinterpret_cast<uint64_t>(sq.data());
 entity.sq_db_va=reinterpret_cast<uint64_t>(&sqDoorbell);entity.sq_depth=32768;entity.wqe_size=64;
 entity.cq_va=reinterpret_cast<uint64_t>(cq.data());entity.cq_db_va=reinterpret_cast<uint64_t>(&cqDoorbell);
 entity.cq_depth=8;entity.cqe_size=64;
 nativeArgs.output=entity.local_hbm_address;nativeArgs.output_rows=outputRows;
 nativeArgs.token_bytes=656;nativeArgs.miss_count=reinterpret_cast<uint64_t>(&missCount);
 nativeArgs.channel_entity=reinterpret_cast<uint64_t>(&entity);
 nativeArgs.queue_state=reinterpret_cast<uint64_t>(&queue);nativeArgs.core_status=reinterpret_cast<uint64_t>(&nativeStatus);
 Args args{};args.context=reinterpret_cast<uint64_t>(&ctx);args.postIndex=reinterpret_cast<uint64_t>(records.data());
 args.packedFetch=reinterpret_cast<uint64_t>(packed.data());args.maxEvents=10240;
 args.eventsSrc=reinterpret_cast<uint64_t>(eventSources.data());args.eventsDst=reinterpret_cast<uint64_t>(eventDestinations.data());
 args.updateBindings=reinterpret_cast<uint64_t>(update.data());
 for(uint32_t fault:{1U,2U}){
  row={};row.error=fault==1?0x4000:0;row.slots=fault==2?1:0;
  records={};records[0].active=1;status[0]=0;status[1]=2;
  for(uint32_t q=0;q<4;++q){finalPosts[q]={};finalPosts[q].descriptor=q*kTop;}
  finalPosts[0].count=2;
  eventSources[2048]=2048;eventSources[2049]=2049;
  eventDestinations[2048]=1;eventDestinations[2049]=2;
  auto before=aiv_model::testDoorbells;
  Run(&args,0);
  assert(aiv_model::testDoorbells==before);
  assert(records[0].error==(fault==1?0x4000:kvaf_device::kInvalidArgs));
  assert(queue.sq_producer_index==0&&!queue.poisoned);
 }
 std::puts("POST_INDEX_TAIL_TYPES_PACK_PASS");
}
''')
    source.write_text(source.read_text().replace('#include "batch_get_model.h"', r'''
#include "kernel_operator.h"
#include "kvaf_batch_get_device.h"
#include <cassert>
#include <cstdio>
#include <vector>
namespace aiv_model {unsigned testDoorbells=0;void Poll(void*){} void Doorbell(uint32_t,uint32_t*){++testDoorbells;}}
'''))
    binary = tmp_path / "post_index_tail"
    subprocess.run([
        compiler, "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer", "-I", str(model),
        "-I", str(native / "include"),
        "-I", str(repo / "csrc/operators/lightning_indexer_with_fetch/op_kernel"),
        "-I", str(repo / "csrc/operators/lightning_indexer_with_fetch/include"),
        str(source), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "POST_INDEX_TAIL_TYPES_PACK_PASS" in result.stdout
