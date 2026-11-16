# SPDX-License-Identifier: Apache-2.0
"""CPU checks for one uniform mode and stable runtime control storage."""

import subprocess

import pytest

from test_recent_index import INDEXER, function


@pytest.mark.parametrize("enabled,ubmem", [(0, 0), (1, 0), (1, 1)])
def test_runtime_mode_changes_without_changing_control_address(tmp_path, enabled, ubmem):
    source = tmp_path / "mode.cpp"
    source.write_text(r'''
#include <array>
#include <cassert>
#include <cstdint>
#define __aicore__
#define __gm__
#include "low_miss_mode.h"
struct Context { uint64_t admissionControls; struct { uint32_t batch; } shape; };
int main() {
    std::array<int32_t,64> control{};
    Context ctx{uint64_t(control.data()), {16}};
    assert(!rotating::NoPrefetchRound(&ctx)); // Invalid window sizes.
    for(unsigned b=0;b<16;++b)control[4*b+1]=64;
    constexpr bool expected=KVIP_LOW_MISS_SKIP && !V9_UBMEM;
    assert(rotating::NoPrefetchRound(&ctx)==expected);
    for(unsigned b=0;b<16;++b) {
        control[4*b]=512; assert(!rotating::NoPrefetchRound(&ctx));
        control[4*b]=0; assert(rotating::NoPrefetchRound(&ctx)==expected);
    }
    // Invalid controls must retain the existing error-reporting path.
    for(auto invalid : {-1,2049}) {
        control[0]=invalid; assert(!rotating::NoPrefetchRound(&ctx));
    }
    control[0]=0; control[3]=2; assert(!rotating::NoPrefetchRound(&ctx));
    control[3]=0; control[2]=65536; assert(!rotating::NoPrefetchRound(&ctx));
    control[2]=0; control[1]=481; assert(!rotating::NoPrefetchRound(&ctx));
    control[1]=64; ctx.shape.batch=0; assert(!rotating::NoPrefetchRound(&ctx));
    ctx.shape.batch=17; assert(!rotating::NoPrefetchRound(&ctx));
    ctx.shape.batch=16; ctx.admissionControls=0; assert(!rotating::NoPrefetchRound(&ctx));
}
''')
    binary = tmp_path / "mode"
    subprocess.run([
        "c++", "-std=c++17", "-fsanitize=address,undefined", "-O1",
        "-DKVIP_ADAPTIVE_ADMISSION=1", f"-DKVIP_LOW_MISS_SKIP={enabled}",
        f"-DV9_UBMEM={ubmem}", "-I", str(INDEXER.parents[1] / "include"),
        str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_skipped_window_removes_both_sides_of_cross_core_handshake(tmp_path):
    body = function(
        (INDEXER / "indexer.h").read_text(),
        "__aicore__ inline void QuantLightningIndexerKernel<QLIT>::FinishBaseBlock(",
    ).replace("QuantLightningIndexerKernel<QLIT>::", "")
    source = tmp_path / "handshake.cpp"
    source.write_text(r'''
#include <cassert>
#include <initializer_list>
#include <cstdint>
#define __aicore__
#define ASCEND_IS_AIC (isCube)
#define ASCEND_IS_AIV (!isCube)
namespace QLICommon {
struct RunInfo { uint32_t s2Idx; bool isLastS2InnerLoop; };
struct ConstInfo { static constexpr int QLI_SYNC_MODE4=4; };
}
constexpr int PIPE_FIX=1,PIPE_V=2;
unsigned waits=0,signals=0;
template<int,int> void CrossCoreWaitFlag(int id){assert(id==8);++waits;}
template<int,int> void CrossCoreSetFlag(int id){assert(id==8);++signals;}
namespace rotating {
int EpochAt(unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned tile,unsigned) {
 return tile==128 ? 0 : -1;
}
}
struct Model {
 bool noPrefetchRound_=false,isCube=false,rotationServed_=false;
 uint32_t resident_=16384,aiCoreIdx=0;
 struct { unsigned batch=16,history=131328,resident=16384,splitQ=1,rounds=8,serviceAics=4; } runtimeShape_;
 struct {
  unsigned seed=0,progress=0,fetch=0,final=0;
  void ProcessResidentTopK(const QLICommon::RunInfo&){++seed;}
  void PrepareRotationProgress(){++progress;}
  void RotateFetch(unsigned){++fetch;}
  void ProcessTopK(const QLICommon::RunInfo&){++final;}
 } vectorService;
''' + body + r'''
};
int main() {
 for(bool low : {false,true}) {
  waits=signals=0;
  Model cube,vec; cube.isCube=true;
  cube.noPrefetchRound_=vec.noPrefetchRound_=low;
  for(unsigned tile=0;tile<1026;++tile){
   QLICommon::RunInfo info{tile,tile==1025};
   cube.FinishBaseBlock(info); vec.FinishBaseBlock(info);
  }
  assert(waits==signals && waits==(low?0U:1U));
  assert(vec.vectorService.final==1);
  assert(cube.vectorService.final==0);
  assert(vec.vectorService.seed==(low?0U:1U));
  assert(vec.vectorService.progress==(low?0U:1U));
  assert(vec.vectorService.fetch==(low?0U:1U));
 }
}
''')
    binary = tmp_path / "handshake"
    subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined", "-O1",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
