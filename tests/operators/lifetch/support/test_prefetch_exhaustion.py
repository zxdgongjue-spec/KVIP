# SPDX-License-Identifier: Apache-2.0
"""Execute the native skip guard and a multi-epoch transport model on CPU."""

import subprocess

from test_recent_index import COMMON, INDEXER, function


def test_exhaustion_never_discards_an_admissible_request(tmp_path):
    prefetch = (INDEXER.parent / "prefetch.h").read_text()
    guard = function(prefetch, "        if(kvip_prefetch::Exhausted(")
    source = tmp_path / "prefetch_exhaustion.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#define __aicore__
#include "prefetch_exhaustion.h"
#include "prefetch_budget.h"
#include "admission_policy.h"
struct State {
 std::array<uint32_t,256> words{};
 uint32_t GetValue(uint32_t offset) const {return words.at(offset);}
};
struct Run {
 std::vector<State> states;
 std::vector<std::vector<uint32_t>> admitted;
 std::vector<uint32_t> scanned,skipped;
 std::array<bool,32> complete{};
};
Run service(std::vector<State> initial, bool skip, uint32_t window, bool carry) {
 Run result{initial,std::vector<std::vector<uint32_t>>(initial.size()),{}, {},{}};
 for(uint32_t epoch=0;epoch<32;++epoch) {
  if(epoch)assert(result.complete[epoch-1]);
  uint32_t exhaustedRows=0,scannedRows=0;
  for(uint32_t b=0;b<result.states.size();++b) {
   auto& state=result.states[b];
   if(state.GetValue(11))continue; // Existing error propagation precedes skip.
   if(skip) {
"""
        + guard
        + r"""
   }
   ++scannedRows;
   uint32_t allowance=rotating::PrefetchAllowance(state.words[8],state.words[9],
       state.words[208],state.words[209],state.words[210],
       kvip_admission::WindowAllowance(window,epoch,state.words[9],carry));
   // Distinct candidate IDs also reveal a skipped epoch with nonzero capacity.
   uint32_t candidates=(epoch*137+b*53)%641;
   uint32_t count=std::min(allowance,candidates);
   for(uint32_t i=0;i<count;++i)result.admitted[b].push_back(epoch*1024+i);
   state.words[8]+=count;state.words[9]+=count;
   if(count)++state.words[211]; // Outstanding completions cannot be discarded.
  }
  // Executed even if every row is exhausted/error: next lane can proceed.
  result.skipped.push_back(exhaustedRows);
  result.scanned.push_back(scannedRows);
  result.complete[epoch]=true;
 }
 return result;
}
int main() {
 using kvip_prefetch::Exhausted;
 std::mt19937 random(191);
 for(uint32_t value:{0U,1U,511U,512U,8192U,UINT32_MAX}) {
  assert(Exhausted(value,0,value,512,2048));
  assert(Exhausted(0,value,8192,value,2048));
  assert(Exhausted(0,value,8192,512,value));
 }
 assert(!Exhausted(8191,511,8192,512,512));
 // The window allowance may be zero temporarily; future carry can recover.
 assert(kvip_admission::WindowAllowance(64,1,128,true)==0);
 assert(!Exhausted(128,128,8192,512,2048));
 assert(kvip_admission::WindowAllowance(64,2,128,true)==64);
 for(unsigned trial=0;trial<50000;++trial) {
  uint32_t slots=random(),requests=random(),sl=random(),rl=random(),ql=random();
  bool exhausted=Exhausted(slots,requests,sl,rl,ql);
  for(uint32_t window:{1U,64U,128U,480U,UINT32_MAX}) {
   auto allowance=rotating::PrefetchAllowance(slots,requests,sl,rl,ql,window);
   assert(exhausted==(allowance==0));
  }
 }
 uint64_t skipped=0;
 for(unsigned trial=0;trial<3000;++trial) {
  std::vector<State> states(1+random()%16);
  for(auto& state:states) {
   state.words[208]=random()%2049;
   state.words[209]=random()%2049;
   state.words[210]=random()%2049;
   // Test valid and saturated/over-limit states without unsigned subtraction.
   state.words[8]=random()%513;state.words[9]=random()%513;
   state.words[211]=random()%9;
   state.words[11]=(random()%7==0)?0x20000:0;
  }
  uint32_t window=1+random()%480;bool carry=bool(random()%2);
  auto reference=service(states,false,window,carry);
  auto optimized=service(states,true,window,carry);
  assert(reference.admitted==optimized.admitted);
  assert(reference.complete==optimized.complete);
  for(uint32_t b=0;b<states.size();++b)
   assert(reference.states[b].words==optimized.states[b].words);
  for(uint32_t epoch=0;epoch<32;++epoch) {
   assert(optimized.complete[epoch]);
   assert(optimized.scanned[epoch]+optimized.skipped[epoch]==reference.scanned[epoch]);
   skipped+=optimized.skipped[epoch];
  }
  // Exact final repair sees the same prefetch IDs and outstanding count.
  for(uint32_t b=0;b<states.size();++b)for(uint32_t selected=0;selected<32768;selected+=293) {
   bool before=std::find(reference.admitted[b].begin(),reference.admitted[b].end(),selected)!=reference.admitted[b].end();
   bool after=std::find(optimized.admitted[b].begin(),optimized.admitted[b].end(),selected)!=optimized.admitted[b].end();
   assert(before==after);
  }
 }
 assert(skipped>0);
}
"""
    )
    binary = tmp_path / "prefetch_exhaustion"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
         "-I", str(INDEXER.parents[1] / "include"), "-I", str(COMMON),
         str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_guard_preserves_service_publication_and_compiles_out():
    source = (INDEXER.parent / "prefetch.h").read_text()
    source = "\n".join(line for line in source.splitlines()
                       if not line.lstrip().startswith("#include"))
    outputs = []
    for enabled in (0, 1):
        result = subprocess.run(
            ["c++", "-E", "-P", "-x", "c++", "-DV9_UBMEM=0",
             "-DKVIP_ADAPTIVE_ADMISSION=1",
             f"-DKVIP_SKIP_EXHAUSTED_PREFETCH={enabled}", "-"],
            input=source, text=True, capture_output=True, check=True,
        )
        outputs.append(result.stdout)
    assert "kvip_prefetch::Exhausted" not in outputs[0]
    assert "exhaustedRows" not in outputs[0]
    enabled = outputs[1]
    guard = enabled.index("if(kvip_prefetch::Exhausted(")
    assert enabled.index("postInput.SetValue(64,0)") < guard
    assert enabled.index("if(state.GetValue(11))continue;") < guard
    assert guard < enabled.index("uint32_t sequence=state.GetValue(246)+1;")
    assert enabled.index("event->skipped=exhaustedRows;") < enabled.index("event->complete=1;")
