# SPDX-License-Identifier: Apache-2.0
"""Run reused Post/CQ helpers against the original KVAF CPU WQE model."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_original_blocking_baseline_and_deferred_native_helpers(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    kvaf = Path(os.environ.get("KVAF_LEGACY_SOURCE_ROOT", repo.parent / "ubmem-pr/KVAF-urma-check"))
    if not (kvaf / "tests/batch_get_kernel_model_test.cc").is_file():
        pytest.skip("Set KVAF_LEGACY_SOURCE_ROOT to the original fused-context checkout")
    dest = tmp_path / "kvaf"
    native = Path("access_operator/demo/aiv_batch_get")
    for part in (native / "include", native / "kernel", Path("tests/aiv_model")):
        shutil.copytree(kvaf / part, dest / part)
    source = (kvaf / "tests/batch_get_kernel_model_test.cc").read_text()
    source = source.replace('"../access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"',
                            '"' + str(dest / native / "kernel/execute_batch_get_aiv.asc") + '"')
    # The original model only receives CQ cache probes. The new API also
    # flushes status/queue lines, which have no hardware side effect in this model.
    prefix, poll = source.split("void Poll(void* address)", 1)
    poll, suffix = poll.split("}  // namespace aiv_model", 1)
    assert poll.count("  CHECK(false);") == 1
    # Schedule a late or erroneous CQE for one chosen FIFO checkpoint while
    # preserving the original native wait/doorbell and CQ owner-bit handling.
    prefix += "uint64_t inject_checkpoint=UINT64_MAX, inject_polls=0; bool inject_error=false;\n"
    prefix += "uintptr_t status_base=0; unsigned header_cleans=0, profile_cleans=0;\n"
    poll = poll.replace("{", """{
  auto addr=reinterpret_cast<uintptr_t>(address);
  if(status_base && addr>=status_base && addr-status_base<sizeof(KvafBatchGetCoreStatus)) {
    if(addr-status_base<offsetof(KvafBatchGetCoreStatus,profile)) ++header_cleans;
    else ++profile_cleans;
    return;
  }""",1)
    probe = "    if (queue.timeout || queue.checkpoints.empty() ||"
    assert poll.count(probe) == 1 and poll.count("queue.error ?") == 1
    poll = poll.replace(probe, "    if (queue.cq_producer == inject_checkpoint && inject_polls) "
                               "{ --inject_polls; return; }\n" + probe)
    poll = poll.replace("queue.error ?", "(queue.error || (queue.cq_producer==inject_checkpoint && inject_error)) ?")
    source = prefix + "void Poll(void* address)" + poll.replace("  CHECK(false);", "  return;") + "}  // namespace aiv_model" + suffix
    source = source.replace("int main() {", "int original_blocking_checks() {")
    source = source.rstrip()[:-1] + "return 0;\n}\n"
    source += r'''
#include "post_index_transport.h"
using namespace pipeline_ab::post_index_transport;
int main(){
  CHECK(original_blocking_checks()==0); // Unmodified original baseline cases.
  for(uint32_t count:{1U,127U,128U,129U,2048U}){
    Fixture f(4096,1,1,0);bool blocking=false;State state;
    CHECK(Preflight(f.args,31,count,blocking)==0&&!blocking);
    auto calls=f.queues[0].completed_batch_count;
    CHECK(Post(f.args,31,count,state)==0);
    CHECK(state.pending==1&&f.queues[0].cq_consumer_index==0);
    CHECK(aiv_model::hardware[0].producer==count&&aiv_model::hardware[0].completed==0);
    CHECK(aiv_model::hardware[0].checkpoints.size()==1);
    // Index may now reuse all UB and mutate source metadata: WQEs already
    // carry final addresses, and native Post has consumed all hard events.
    for(auto& a:model_events)for(auto& b:a)for(bool e:b)CHECK(!e);
    std::fill_n(aiv_model::ub,sizeof(aiv_model::ub),0xAB);
    std::fill(f.addresses.begin(),f.addresses.end(),0);
    CHECK(Wait(state)==0&&state.completed==count&&!state.pending);
    CHECK(f.queues[0].completed_batch_count==calls+1);
    for(uint32_t i=0;i<f.capacity;++i)for(uint32_t byte=0;byte<16;++byte)
      CHECK(f.output[f.rows[i]*16+byte]==(i>=31&&i<31+count?f.remote[f.indices[i]*16+byte]:0));
    CHECK(Wait(state)==0&&f.queues[0].cq_consumer_index==1);
  }
  { // Queue ownership and completion indices survive successive graph rounds.
    Fixture f(4096,1,1,0);
    for(unsigned round=0;round<4;++round){State state;
      bool blocking=false;CHECK(Preflight(f.args,0,2048,blocking)==0&&!blocking);
      CHECK(Post(f.args,0,2048,state)==0);CHECK(Wait(state)==0);
      CHECK(f.queues[0].sq_producer_index==(round+1)*2048);
      CHECK(f.queues[0].cq_consumer_index==round+1);
    }
  }
  { // Respect old max-inflight; fallback keeps its original bounded scheduler.
    Fixture f(4096,1,1,1024);bool blocking=false;State state;
    CHECK(Preflight(f.args,0,2048,blocking)==0&&blocking);
    CHECK(aiv_model::hardware[0].producer==0);
    CHECK(GetBlocking(f.args,0,2048,state)==0&&state.completed==2048);
    CHECK(f.statuses[0].profile.peak_submitted_not_drained<=1024);
  }
  {Fixture f(4096,1,1,1);bool blocking=false;
    CHECK(Preflight(f.args,0,2048,blocking)==kInvalidArgs);
  }
  { // A depth-one CQ cannot use the old two-group bounded fallback. Reject
    // collectively before the first WQE, rather than poison a half-posted queue.
    Fixture f(4096,1,1,1024);bool blocking=false;
    f.entities[0].cq_depth=1;
    CHECK(Preflight(f.args,0,2048,blocking)==kInvalidEntity);
    CHECK(!blocking&&aiv_model::hardware[0].producer==0);
    CHECK(aiv_model::hardware[0].checkpoints.empty());
    CHECK(!f.queues[0].poisoned&&f.queues[0].sq_producer_index==0);
  }
  {Fixture f(4096,1,1,0);bool blocking=false;f.queues[0].poisoned=1;
    CHECK(Preflight(f.args,0,2048,blocking)==kInvalidQueue);
  }
  for(bool timeout:{false,true}){
    Fixture f(4096,1,1,0);State state;
    aiv_model::hardware[0].timeout=timeout;aiv_model::hardware[0].error=!timeout;
    bool blocking=false;CHECK(Preflight(f.args,0,2048,blocking)==0&&!blocking);
    CHECK(Post(f.args,0,2048,state)==0);
    CHECK(Wait(state)==(timeout?kCqTimeout:kCqError));
    CHECK(f.queues[0].poisoned&&f.queues[0].last_error);
    CHECK(Preflight(f.args,0,1,blocking)==kInvalidQueue);
  }
  { // HIGH RetirePair keeps complete public header and never flushes its untouched profile.
    Fixture f(64,1,1,0);State first,second;
    alignas(64) KvafBatchGetCoreStatus status{};
    std::memset(&status.profile,0xa5,sizeof(status.profile));
    auto oldProfile=status.profile;
    f.args.core_status=reinterpret_cast<uint64_t>(&status);
    CHECK(Post(f.args,0,5,first)==0 && Post(f.args,5,13,second)==0);
    aiv_model::status_base=f.args.core_status;
    uint64_t stats=UINT64_MAX;
    CHECK(RetirePair(first,second,&stats)==0 && stats==2);
    aiv_model::status_base=0;
    CHECK(aiv_model::header_cleans==2 && aiv_model::profile_cleans==0);
    CHECK(std::memcmp(&oldProfile,&status.profile,sizeof(status.profile))==0);
    CHECK(status.magic==KVAF_AIV_BATCH_GET_STATUS_MAGIC && status.processed_request_count==13);
    CHECK(status.result==0 && status.completed_batch_count==2 && status.sq_producer_index==18 &&
          status.cq_consumer_index==2 && status.failed_request_index==UINT64_MAX);
  }
  for(unsigned target:{0U,1U})for(unsigned fault:{0U,1U,2U,3U}){
    Fixture f(64,1,1,0);State first,second;
    // Unequal prefix/suffix counts expose an incorrect attribution of a late CQE.
    CHECK(Post(f.args,0,5,first)==0 && Post(f.args,5,13,second)==0);
    aiv_model::inject_checkpoint=target;
    aiv_model::inject_polls=fault==1 ? kMaxCqPolls : fault==2 ? 2*kMaxCqPolls : 0;
    aiv_model::inject_error=fault==3;
    const uint64_t expected=fault==0 ? kSuccess : fault==3 ? kCqError : kCqTimeout;
    uint64_t statistics=0;
    CHECK(RetirePair(first,second,&statistics)==expected);
    CHECK((statistics>>32)==((fault==1 || fault==2) ? FinalPoll::maxPauseCycles : 0));
    uint32_t expectedProbes=fault==1 ? kMaxCqPolls+2 :
                            fault==2 ? 2*kMaxCqPolls+target : 2;
    CHECK(uint32_t(statistics)==expectedProbes);
    CHECK(f.statuses[0].result==expected && f.queues[0].last_error==expected);
    CHECK(bool(f.queues[0].poisoned)==bool(fault));
    const uint64_t retired=fault==2 ? target : 2;
    CHECK(f.queues[0].cq_consumer_index==retired);
    CHECK(aiv_model::hardware[0].cq_consumer==retired);
    CHECK(aiv_model::hardware[0].checkpoints.size()==2-retired);
    CHECK(first.pending==(retired==0) && second.pending==(retired<2));
    CHECK(first.completed==(retired>=1 && !(fault==3 && target==0) ? 5u : 0u));
    CHECK(second.completed==(retired==2 && !(fault==3 && target==1) ? 13u : 0u));
    bool blocking=false;CHECK(Preflight(f.args,0,1,blocking)==(fault ? kInvalidQueue : kSuccess));
  }
  // Deliver after an explicit number of empty probes, straddling the native
  // doorbell wrap. Verify both paths consume exactly one real CQE each.
  for(bool paced : {false,true}){
    for(uint32_t misses : {0U,3U,4U,8U,17U,6000U}){
      Fixture f(64,1,1,0); State state;
      auto& queue=f.queues[0]; auto& hardware=aiv_model::hardware[0];
      constexpr uint64_t cursor=(1ULL<<22)-1;
      queue.magic=kQueueMagic;queue.initialized=1;
      queue.cq_consumer_index=hardware.cq_producer=hardware.cq_consumer=cursor;
      // Before the CQE arrives, owner still belongs to the previous epoch.
      hardware.cq[(cursor%8)*16]=uint32_t((cursor/8)&1U)<<2;
      aiv_model::inject_checkpoint=cursor;aiv_model::inject_polls=misses;aiv_model::inject_error=false;
      CHECK(Post(f.args,0,2,state)==0);
      const uint64_t cycleBefore=aiv_model::cycle;
      if(paced){
        uint64_t stats=UINT64_MAX;State empty{};
        CHECK(RetirePair(state,empty,&stats)==0);
        uint32_t delay=0;
        for(uint32_t n=4;n<=misses;n++)delay+=n<8?128:n<16?256:512;
        CHECK((stats>>32)==std::min(delay,FinalPoll::maxPauseCycles));
        CHECK(uint32_t(stats)==misses+1);
        if(misses<4)CHECK(aiv_model::cycle==cycleBefore);
      }else CHECK(Wait(state)==0);
      CHECK(hardware.cq_db==0 && hardware.cq_consumer==cursor+1 && queue.cq_consumer_index==cursor+1);
      CHECK(state.completed==2 && !state.pending && !aiv_model::inject_polls);
    }
  }
  aiv_model::inject_checkpoint=UINT64_MAX;
  {KvafExecuteBatchGetKernelArgs args{};State state;bool blocking=false;
    CHECK(Preflight(args,0,0,blocking)==0&&!blocking);
    CHECK(Post(args,0,0,state)==0&&Wait(state)==0&&GetBlocking(args,0,0,state)==0);
  }
  std::cout<<"EXISTING_POST_CQ_BASELINE_AND_PHASE_PASS\n";
}
'''
    program = tmp_path / "legacy_model.cpp"
    program.write_text(source)
    binary = tmp_path / "legacy_model"
    subprocess.run([
        shutil.which("clang++") or "c++", "-std=c++17", "-O1", "-g",
        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
        "-I", str(dest / "tests/aiv_model"), "-I", str(dest / native / "include"),
        "-I", str(repo / "csrc/operators/lightning_indexer_with_fetch/op_kernel"),
        str(program), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "EXISTING_POST_CQ_BASELINE_AND_PHASE_PASS" in result.stdout
