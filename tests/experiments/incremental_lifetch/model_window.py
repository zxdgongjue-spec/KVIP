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
 uint64_t options=reinterpret_cast<__gm__ uint64_t*>(timing)[7];
 TPipe pipe;Adapter a;TBuf<TPosition::VECCALC> input,ids,values,old,scratch,map,meta,liLive;
 a.rotationContext_=context;a.runtimeShape_=rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(context));
 pipe.InitBuffer(liLive,90624);a.topkOp_.Init(2048,8192);
 pipe.InitBuffer(input,20480);pipe.InitBuffer(ids,8448);pipe.InitBuffer(values,4096);
 pipe.InitBuffer(scratch,a.topkOp_.GetSharedTmpBufferSize());pipe.InitBuffer(map,16384);pipe.InitBuffer(old,8192);pipe.InitBuffer(meta,1024);
 a.incrementalMetadata_=meta.Get<uint64_t>();a.mrgValueLocal_=input.Get<uint16_t>();a.indicesOutLocal_=ids.Get<uint32_t>();
 a.scoreOutLocal_=values.Get<uint16_t>();a.incrementalOld_=old.Get<uint32_t>();a.incrementalNew_=map.Get<uint16_t>();a.topkSharedTmpLocal_=scratch.Get<uint32_t>();
 a.topkOp_.InitBuffers(a.topkSharedTmpLocal_,a.indicesOutLocal_);
 pipe.InitBuffer(a.incrementalInputBuf_,a.kIncrementalInputBytes);a.incrementalInput_=a.incrementalInputBuf_.Get<uint16_t>();a.incrementalPipeline_=(options&2)==0;
 PipeBarrier<PIPE_ALL>();uint64_t t0=GetSystemCycle();
 a.AdvanceWindow(1,8192);
 PipeBarrier<PIPE_ALL>();uint64_t t1=GetSystemCycle();
 if(!(options&1))rotating::FastService<true>(2,131072,16384,8,1,656,256,240,1,23808,context,1,0,0,a.topkSharedTmpLocal_,a.mrgValueLocal_,a.indicesOutLocal_,8192);
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
    parser.add_argument('--profile-selection', action='store_true', help='Measure per-Q check/pack; tagged B must stay below 1800 cycles including bitmap loads')
    parser.add_argument('--tagged-miss', action='store_true', help='Prefix-tagged hot residency and admission-only bitmap through actual window/submit')
    parser.add_argument('--bitmap-miss', action='store_true', help='Indexed true-miss A vs maintained bitmap B; validate initialization and admission updates')
    parser.add_argument('--ranked-miss', action='store_true', help='Cold floor vs true-miss floor through the real window/submit path')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-b-topk-cycles', type=int, default=36000,
                        help='B incremental window must be below this cycle count (20 us at 1.8 GHz)')
    parser.add_argument('--stress', action='store_true', help='Also check B with sparse/dense candidates, unequal progress and no preloading')
    args = parser.parse_args()
    if args.tagged_miss: args.bitmap_miss=True
    if args.bitmap_miss: args.ranked_miss=True
    if args.profile_selection and not args.bitmap_miss:
        parser.error('--profile-selection requires --bitmap-miss or --tagged-miss')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    results = {}
    for label in ('a', 'b'):
        source = output / f'{label}-source'
        command = [sys.executable, str(here / 'prepare.py'), str(source), '--topk-prefetch']
        if label == 'a':
            command.append('--baseline')
        if args.ranked_miss:
            command = [sys.executable, str(here / 'prepare_ranked.py'), str(source), '--minimum-half']
            if label == 'b' or args.bitmap_miss: command.append('--true-miss')
            if label == 'b' and args.bitmap_miss: command.append('--tagged-miss' if args.tagged_miss else '--bitmap-miss')
        subprocess.run(command, check=True)
        op = source / 'csrc/operators/lightning_indexer_with_fetch'
        case = output / label
        case.mkdir()
        methods = (op / 'op_kernel/indexer/incremental_methods.h').read_text()
        methods = methods[:methods.index('__aicore__ inline void FinalIncremental')]
        if args.tagged_miss and label == 'b':
            # Prefix tagging is validated separately; this model starts from saved prefix state.
            methods=methods[methods.index('bool incrementalIdsPending_'):]
        if args.profile_selection:
            methods=methods.replace('rankTrace_=trace;', 'rankTrace_=true;')
        (case / 'actual_methods.h').write_text(methods)
        kernel = KERNEL
        if (op / 'op_kernel/indexer/window_fast_vector.h').exists():
            kernel = kernel.replace('#include "incremental_vector_helpers.h"', '#include "incremental_vector_helpers.h"\n#include "window_fast_vector.h"')
        if args.ranked_miss:
            kernel = kernel.replace('#include "window_fast_vector.h"', '#include "window_fast_vector.h"\n#include "recent_seed.h"')
        if args.bitmap_miss and label == 'b':
            kernel=kernel.replace(' PipeBarrier<PIPE_ALL>();uint64_t t0=',''' if(!reinterpret_cast<__gm__ uint64_t*>(timing)[6]) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(context);
 uint64_t begin=GetSystemCycle();
 for(uint32_t b=0;b<2;++b){
 auto covered=reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(ctx->incremental,2,131072,b));
 kvip_prefetch_rank::InitializeCovered(a.topkSharedTmpLocal_,covered,
 reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+b*16384,
 reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*16384,16384,131072);
 Simt::VF_CALL<rotating::InitializeTailMappings>(Simt::Dim3{256,1,1},131072,16384,ctx,b);
 }
 PipeBarrier<PIPE_ALL>();auto stamps=reinterpret_cast<__gm__ uint64_t*>(timing);
 stamps[3]=begin;stamps[4]=GetSystemCycle();stamps[6]=1;
 }
 PipeBarrier<PIPE_ALL>();uint64_t t0=''')
        if args.tagged_miss and label == 'b':
            begin=kernel.index(' kvip_prefetch_rank::InitializeCovered(')
            end=kernel.index(' Simt::VF_CALL<rotating::InitializeTailMappings>',begin)
            kernel=kernel[:begin]+kernel[end:]
        (case / 'kernel.cpp').write_text(kernel)
        host=(here / 'window_model_host.cpp').read_text()
        if args.bitmap_miss:
            host=host.replace('(B*512+B*(4*K*6+1024))/8','(B*512+B*(4*K*6+1024)+B*H/8)/8')
            host=host.replace('uint32_t mapped=*wanted[b].begin();mapping[b*H+mapped]=R;wanted[b].erase(mapped);', 'wanted[b].insert(17);wanted[b].insert(33);')
            host=host.replace('r.slots=1;r.requests=1;', 'r.slots=0;r.requests=0;')
            host=host.replace('lru[b*2048+1+i]', 'lru[b*2048+i]')
            host=host.replace(' void* contextD=upload',''' std::vector<uint32_t> hot(B*R,1),reserved(B*R,UINT32_MAX),bounds(B*2);
 for(uint32_t b=0;b<B;++b){hot[b*R+17]=0;reserved[b*R+33]=0;bounds[b*2]=H-384;bounds[b*2+1]=H;}
 ctx.hotValid=(uint64_t)upload(hot);ctx.reservedHot=(uint64_t)upload(reserved);ctx.tailBounds=(uint64_t)upload(bounds);ctx.outputStride=R+384;
 void* contextD=upload''')
            if label=='b':
                host=host.replace(' auto rr=read<rotating::Row>',''' auto finalMap=read<uint32_t>(mapD,B*H);
 auto bitmapBase=reinterpret_cast<uint32_t*>(result.data())+(B*512+B*(4*K*6+1024))/4;
 for(uint32_t b=0;b<B;++b)for(uint32_t t=0;t<H;++t){
 bool expected=(t<R&&hot[b*R+t]&&int32_t(reserved[b*R+t])<0)||finalMap[b*H+t]!=UINT32_MAX;
 bool actual=(bitmapBase[b*H/32+t/32]>>(t%32))&1U;
 if(actual!=expected)throw std::runtime_error("covered bitmap token="+std::to_string(t));}
 printf("PASS covered bitmap: hot validity, reserved, ABC tail and actual WQE admissions; init cycles=%llu\\n",(unsigned long long)(times[4]-times[3]));
 auto rr=read<rotating::Row>''')
        if args.tagged_miss and label == 'b':
            host=host.replace(' void* privateD=upload(privateState);', '''
 for(uint32_t q=0;q<Q;++q){auto p=reinterpret_cast<uint8_t*>(privateState.data())+B*512+(q/4)*(4*K*6+1024);
 auto ids=reinterpret_cast<uint32_t*>(p+4*K*2)+(q%4)*K;
 for(uint32_t i=0;i<K;++i)if(ids[i]<R&&ids[i]!=17&&ids[i]!=33)ids[i]|=0x80000000U;}
 void* privateD=upload(privateState);''')
            host=host.replace('uint32_t id=ids[i];uint16_t value=', '''uint32_t tag=ids[i],id=tag&0x3fffffffU;
 if(bool(tag&0x80000000U)!=(id<R&&id!=17&&id!=33))throw std::runtime_error("lost residency tag");
 uint16_t value=''')
            host=host.replace('bool expected=(t<R&&hot[b*R+t]&&int32_t(reserved[b*R+t])<0)||finalMap[b*H+t]!=UINT32_MAX;',
                              'bool expected=finalMap[b*H+t]>=R&&finalMap[b*H+t]<R+8192;')
            host=host.replace('PASS covered bitmap: hot validity, reserved, ABC tail and actual WQE admissions;',
                              'PASS admission-only bitmap and hot tags: hot validity, reserved, ABC tail and actual WQE admissions;')
        if args.profile_selection:
            anchor=' // Same progress and TopK state:'
            host=host.replace(anchor, ''' for(uint32_t q=0;q<Q;++q)printf("PACK q=%u calls=%llu check_pack_cycles=%llu rank_emit_cycles=%llu\\n",q,(unsigned long long)result[q*16+7],(unsigned long long)result[q*16+5],(unsigned long long)result[q*16+6]);
'''+anchor)
        (case / 'main.cpp').write_text(host)
        (case / 'CMakeLists.txt').write_text(CMAKE.replace('@OP@', str(op)).replace('@KVAF@', str(args.kvaf_root.resolve() / 'access_operator/demo/aiv_batch_get/include')))
        if args.prepare_only:
            continue
        with (case / 'build.log').open('w') as log:
            subprocess.run(['cmake', '-S', str(case), '-B', str(case / 'build'), '-DRUN_MODE=sim', f'-DSOC_VERSION={args.soc}', f'-DASCEND_CANN_PACKAGE_PATH={args.cann_root}'], check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(['cmake', '--build', str(case / 'build'), '-j8'], check=True, stdout=log, stderr=subprocess.STDOUT)
        run = case / 'trace'
        run.mkdir()
        env = dict(os.environ)
        simulator=args.cann_root/'tools/simulator'/args.soc/'lib'
        if not simulator.exists(): simulator=args.cann_root/'aarch64-linux/simulator/dav_3510/lib'
        env['LD_LIBRARY_PATH'] = str(simulator) + ':' + env.get('LD_LIBRARY_PATH', '')
        print(f'RUNNING {label}: exact TopK + prefetch; log={case / "model.log"}', flush=True)
        with (case / 'model.log').open('w') as log:
            subprocess.run([str(case / 'build/window_model')], cwd=run, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        log = (case / 'model.log').read_text()
        print('\n'.join(line for line in log.splitlines() if line.startswith(('PASS', 'CYCLES', 'PACK'))), flush=True)
        match = re.search(r'CYCLES (.*)', log)
        if not match or 'PASS unchanged progress' not in log:
            raise RuntimeError(f'Incomplete model checks: {case}')
        results[label] = {name: int(value) for name, value in re.findall(r'(\w+)=(\d+)', match[1])}
        if args.profile_selection:
            pack=[int(value) for value in re.findall(r'PACK q=\d+ calls=1 check_pack_cycles=(\d+)',log)]
            if len(pack)!=8: raise RuntimeError('Missing per-Q selection measurements')
            results[label]['check_pack_max']=max(pack)
            if args.tagged_miss and label=='b' and max(pack)>=1800:
                raise RuntimeError(f'Tagged check/pack exceeds 1 us: {max(pack)} cycles')
    if args.stress and not args.prepare_only:
        case = output / 'b'
        for label, extra in (('stress', []), ('stress-no-preload', ['--no-preload'])):
            trace = case / label
            trace.mkdir()
            with (case / f'{label}.log').open('w') as log:
                subprocess.run([str(case / 'build/window_model'), '--stress', *extra],
                               cwd=trace, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            print('\n'.join(line for line in (case / f'{label}.log').read_text().splitlines()
                            if line.startswith(('PASS', 'CYCLES', 'PACK'))), flush=True)
    if results:
        (output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        a, b = results['a']['total'], results['b']['total']
        print(f'TopK + prefetch cycles: A={a} B={b}; reduction={(a-b)/a*100:.2f}%', flush=True)
        if not args.tagged_miss and results['b']['topk'] >= args.max_b_topk_cycles:
            raise RuntimeError(f"B incremental window exceeded target: {results['b']['topk']} cycles")
        if b >= a:
            raise RuntimeError('Fusion did not beat serial window processing')
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
