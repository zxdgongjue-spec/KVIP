"""Reproduce a real middle-window TopK + prefetch A/B in the CANN simulator.

Both cases use the prepared production methods and FastService, including WQE
construction and submission. The simulator writes the doorbell into memory;
network DMA/CQ completion is outside this LI-window submission measurement.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

KERNEL = r'''
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "indexer_types.h"
#include "topk.h"
#include "prefetch.h"
#define KVIP_OUTPUT_WORKER 0
#include "pipeline_types.h"
#include "incremental_vector_helpers.h"
using namespace AscendC;
class Adapter { public:
rotating::RuntimeShape runtimeShape_{};
uint32_t trunkLen_=8192,traceCalls_=0;uint64_t traceExtraWait_=0;
GM_ADDR rotationContext_=nullptr;GM_ADDR streamingWorkspace_=nullptr;uint32_t blockId_=0;
GlobalTensor<uint16_t> scoreGm;
LocalTensor<uint16_t> mrgValueLocal_,scoreOutLocal_;
LocalTensor<uint32_t> indicesOutLocal_,topkSharedTmpLocal_;
topk::LITopk<uint16_t> topkOp_;
__aicore__ inline uint32_t Min(uint32_t a,uint32_t b){return a<b?a:b;}
__aicore__ inline uint32_t Max(uint32_t a,uint32_t b){return a>b?a:b;}
__aicore__ inline void PrepareRotationProgress() {
 PipeBarrier<PIPE_ALL>();
 rotating::PrepareProgress(reinterpret_cast<__ubuf__ uint32_t*>(scoreOutLocal_.GetPhyAddr()),(runtimeShape_.resident/128/64)*64);
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}
#include "actual_methods.h"
};
#endif
extern "C" __global__ __aicore__ void window_model(GM_ADDR context,GM_ADDR timing){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
 TPipe pipe;Adapter a;TBuf<TPosition::VECCALC> input,ids,values,old,scratch,map,meta,liLive;
 a.rotationContext_=context;a.runtimeShape_=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(context));
 pipe.InitBuffer(liLive,90624);a.topkOp_.Init(2048,8192);
 pipe.InitBuffer(input,20480);pipe.InitBuffer(ids,8448);pipe.InitBuffer(values,4096);
 pipe.InitBuffer(scratch,a.topkOp_.GetSharedTmpBufferSize());pipe.InitBuffer(map,16384);pipe.InitBuffer(old,8192);pipe.InitBuffer(meta,1024);
 a.incrementalMetadata_=meta.Get<uint64_t>();a.mrgValueLocal_=input.Get<uint16_t>();a.indicesOutLocal_=ids.Get<uint32_t>();
 a.scoreOutLocal_=values.Get<uint16_t>();a.incrementalOld_=old.Get<uint32_t>();a.incrementalNew_=map.Get<uint16_t>();a.topkSharedTmpLocal_=scratch.Get<uint32_t>();
 a.topkOp_.InitBuffers(a.topkSharedTmpLocal_,a.indicesOutLocal_);
 pipe.InitBuffer(a.incrementalInputBuf_,a.kIncrementalInputBytes);a.incrementalInput_=a.incrementalInputBuf_.Get<uint16_t>();a.incrementalPipeline_=true;
 PipeBarrier<PIPE_ALL>();uint64_t t0=GetSystemCycle();
 a.AdvanceWindow(1,8192);
 PipeBarrier<PIPE_ALL>();uint64_t t1=GetSystemCycle();
 rotating::FastService<true>(2,131072,16384,8,1,656,256,240,1,23808,context,1,0,0,a.topkSharedTmpLocal_,a.mrgValueLocal_,a.indicesOutLocal_,8192);
 PipeBarrier<PIPE_ALL>();uint64_t t2=GetSystemCycle();
 auto result=reinterpret_cast<__gm__ uint64_t*>(timing);result[0]=t0;result[1]=t1;result[2]=t2;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(result),64);
 pipe.Destroy();
 }
#endif
}
''' 
CMAKE = r'''
cmake_minimum_required(VERSION 3.16)
project(window_model LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Target")
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/cann CACHE PATH "CANN")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
set(OP "@OP@")
set(KVAF "@KVAF@")
ascendc_library(window_kernel STATIC kernel.cpp)
ascendc_include_directories(window_kernel PRIVATE ${OP}/op_kernel ${OP}/op_kernel/indexer ${OP}/include ${KVAF} ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
ascendc_compile_options(window_kernel PRIVATE -O2 -DV9_SIMULATION=1 --cce-auto-sync=off -mllvm -user-reserved-ub-size=223232)
add_executable(window_model main.cpp)
target_include_directories(window_model PRIVATE ${OP}/include ${KVAF})
target_link_libraries(window_model PRIVATE window_kernel host_intf_pub ascendcl)
target_link_options(window_model PRIVATE -Wl,--allow-shlib-undefined)
''' 


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kvaf-root', required=True, type=Path)
    parser.add_argument('--cann-root', type=Path, default=Path('/usr/local/Ascend/cann'))
    parser.add_argument('--soc', default='Ascend950PR_9599')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    results = {}
    for label in ('a', 'b'):
        source = output / f'{label}-source'
        command = [sys.executable, str(here / 'prepare.py'), str(source), '--topk-prefetch']
        if label == 'a':
            command.append('--baseline')
        subprocess.run(command, check=True)
        op = source / 'csrc/operators/lightning_indexer_with_fetch'
        case = output / label
        case.mkdir()
        methods = (op / 'op_kernel/indexer/incremental_methods.h').read_text()
        methods = methods[:methods.index('__aicore__ inline void FinalIncremental')]
        (case / 'actual_methods.h').write_text(methods)
        (case / 'kernel.cpp').write_text(KERNEL)
        (case / 'main.cpp').write_text((here / 'window_model_host.cpp').read_text())
        (case / 'CMakeLists.txt').write_text(CMAKE.replace('@OP@', str(op)).replace('@KVAF@', str(args.kvaf_root.resolve() / 'access_operator/demo/aiv_batch_get/include')))
        if args.prepare_only:
            continue
        with (case / 'build.log').open('w') as log:
            subprocess.run(['cmake', '-S', str(case), '-B', str(case / 'build'), '-DRUN_MODE=sim', f'-DSOC_VERSION={args.soc}', f'-DASCEND_CANN_PACKAGE_PATH={args.cann_root}'], check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(['cmake', '--build', str(case / 'build'), '-j8'], check=True, stdout=log, stderr=subprocess.STDOUT)
        run = case / 'trace'
        run.mkdir()
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = str(args.cann_root / 'tools/simulator' / args.soc / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        print(f'RUNNING {label}: exact TopK + prefetch; log={case / "model.log"}', flush=True)
        with (case / 'model.log').open('w') as log:
            subprocess.run([str(case / 'build/window_model')], cwd=run, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        log = (case / 'model.log').read_text()
        print('\n'.join(line for line in log.splitlines() if line.startswith(('PASS', 'CYCLES'))), flush=True)
        match = re.search(r'CYCLES (.*)', log)
        if not match or 'PASS unchanged progress' not in log:
            raise RuntimeError(f'Incomplete model checks: {case}')
        results[label] = {name: int(value) for name, value in re.findall(r'(\w+)=(\d+)', match[1])}
    if results:
        (output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        a, b = results['a']['total'], results['b']['total']
        print(f'TopK + prefetch cycles: A={a} B={b}; reduction={(a-b)/a*100:.2f}%', flush=True)
        if b >= a:
            raise RuntimeError('Fusion did not beat serial window processing')
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
