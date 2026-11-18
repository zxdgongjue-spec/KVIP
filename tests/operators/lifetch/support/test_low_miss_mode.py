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


def test_li_reads_current_auto_decision_only_after_init(tmp_path):
    kernel = INDEXER.parent
    indexer = (kernel / "indexer.cpp").read_text()
    li = function(indexer, "__aicore__ inline bool LiNoPrefetch(")
    launch = (kernel.parent / "op_host/launch.cpp").read_text()
    direct = function(launch, 'extern "C" uint32_t v9_li(')
    combined = function(launch, 'extern "C" uint32_t kvip_lifetch_streaming_forward(')
    source = tmp_path / "li_mode.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#define __aicore__
#define __gm__
#define KVIP_OUTPUT_WORKER 1
#define KVIP_ADAPTIVE_ADMISSION 1
#define KVIP_LOW_MISS_SKIP 1
#define KVIP_AUTO_ROUTE 1
#define KVIP_LOW_MISS_TAIL 1
#define KVIP_LOW_MISS_DIRECT_INDEX 1
#define KVIP_SELECTED_METADATA 1
#define KVIP_EARLY_FETCH_READY 1
#define KVIP_READY_BATCH_FETCH 1
#define KVIP_LOW_MISS_POST_INDEX 1
#include "pipeline_types.h"
#include "low_miss_mode.h"
namespace rotating {
struct RuntimeShape { uint32_t batch=16, resident=16384, aicCount=32, finalWorkers=4; };
struct Context { uint64_t autoRoute=1, admissionControls=0; RuntimeShape shape; };
}
namespace kvaf_device {
static uint8_t* last;
void CleanCacheLines(uint8_t* p, uint32_t) { last=p; }
}
''' + li + r'''
int stage=0, initResult=0;
uint32_t v9_init(void*, void*, const rotating::RuntimeShape*) { assert(stage++==0); return initResult; }
uint32_t LaunchLi(void*,void*,const rotating::RuntimeShape*,uint32_t afterInit) {
    assert(afterInit == uint32_t(stage==1)); ++stage; return 0;
}
uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*) {
    assert(stage++==2); return 0;
}
''' + direct + combined + r'''
int main(){
    uint64_t bindings[16]{}, records[16]{};
    int32_t controls[64]{};
    pipeline_ab::Args args{};
    rotating::Context ctx;
    ctx.admissionControls = uint64_t(controls);
    args.postIndex = uint64_t(records);
    bindings[15] = uint64_t(&args);
    for(unsigned i=0;i<16;++i) controls[i*4+1]=64;
    auto& mode=records[pipeline_ab::kPostIndexRouteWord];
    for(uint64_t value : {uint64_t(1),uint64_t(0),uint64_t(1)}) {
        mode=value;
        assert(LiNoPrefetch(&ctx,bindings,true)==(value==1));
        assert(kvaf_device::last==reinterpret_cast<uint8_t*>(&mode));
    }
    // Standalone hot LI can receive manual rows changed since a previous init.
    controls[60]=512;
    assert(!LiNoPrefetch(&ctx,nullptr,false));
    assert(kvaf_device::last==reinterpret_cast<uint8_t*>(controls));
    mode=pipeline_ab::kReadAdmissionRows;
    assert(!LiNoPrefetch(&ctx,bindings,true));
    controls[60]=0; assert(LiNoPrefetch(&ctx,bindings,true));
    controls[61]=0; assert(!LiNoPrefetch(&ctx,bindings,true));
    controls[61]=64;
    args.postIndex=0; assert(LiNoPrefetch(&ctx,bindings,true));
    bindings[15]=0; assert(LiNoPrefetch(&ctx,bindings,true));
    ctx.admissionControls=0; assert(!LiNoPrefetch(&ctx,bindings,true));
    assert(v9_li(&ctx,&ctx,&ctx.shape)==0 && stage==1);
    stage=0;
    assert(kvip_lifetch_streaming_forward(&ctx,&ctx,&ctx,&ctx,&ctx.shape,4)==0 && stage==3);
    stage=0; initResult=7;
    assert(kvip_lifetch_streaming_forward(&ctx,&ctx,&ctx,&ctx,&ctx.shape,4)==7 && stage==1);
}
''')
    binary = tmp_path / "li_mode"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        "-include", "initializer_list", "-I", str(kernel), "-I", str(kernel.parent / "include"),
        str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
