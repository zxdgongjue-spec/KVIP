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
