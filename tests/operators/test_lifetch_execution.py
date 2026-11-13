"""CPU regressions for the unified LIFetch execution module."""
import ctypes
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import execution as init_transport
import execution as replay_execution
import execution as ubmem_smoke
import kvip.adapters.kvaf
import kvip.runtime.build
import numpy as np
import pytest
import torch
from execution import BenchPool, FusedReplayState, runner_options
from kvip.adapters.kvaf import UbmemLease
from kvip.operators.lifetch.forward import LIFetchForward
from kvip.operators.lifetch.tiling import RuntimeShape
from kvip.operators.lifetch.workspace import UbmemQueryInputs, UbmemWorkspace, device_words
from kvip.operators.lifetch.workspace import ubmem_rotation_spec as rotation_spec
from kvip.operators.lifetch_ubmem.build import UbmemGeometry
from reporting import timing_summary
from test_lifetch_workloads import reference
from workloads import ReplayPagePlan, assemble_round, configuration

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('transport,workers', [('urma', 40), ('ubmem', 2)])
def test_pool_puts_once_and_retains_tables_until_all_cases_complete(tmp_path, monkeypatch, transport, workers):
    events = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            assert kwargs['npu_core_count'] == workers
            events.append('create')

        def link_to(self, server):
            events.append('link')

        def initialize(self, size, slot):
            events.append('initialize')

        def batch_put(self, keys, source, offsets, lengths):
            events.append(('put', keys))
            assert len(keys) <= source.shape[0]
            assert offsets == [i * 128 * 656 for i in keys]

        def prepare_access_table(self, keys):
            events.append(('table', len(keys)))
            return keys

        def access_table_view(self, table, **kwargs):
            return SimpleNamespace(addresses=torch.arange(len(table)), valid=torch.ones(len(table)))

        def release_access_table(self, table):
            events.append(('release_table', len(table)))

        def shutdown(self):
            events.append('shutdown')

    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(NpuRuntime=Runtime))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None), raising=False)
    pool = BenchPool(SimpleNamespace(batches=(1, 2), history=256, transport=transport,
                                    kvaf_root=tmp_path, kvaf_server='test:1'))
    first = pool.borrow(torch, torch.device('cpu'), 1, 256, 2)
    original = first.cpu_payload.clone()
    with pytest.raises(ValueError):
        pool.borrow(torch, torch.device('cpu'), 2, 256, 2)
    with pytest.raises(RuntimeError):
        pool.close()
    pool.release()
    for batch in (2, 1, 2):
        item = pool.borrow(torch, torch.device('cpu'), batch, 256, 2)
        assert item.runtime is first.runtime
        assert len(item.device_table.addresses) == batch * 2
        assert torch.equal(item.cpu_payload[:1], original)
        pool.release()
    assert len([event for event in events if isinstance(event, tuple) and event[0] == 'put']) == 1
    assert 'shutdown' not in events
    pool.close()
    assert events.count('create') == events.count('shutdown') == 1
    assert sorted(event[1] for event in events if isinstance(event, tuple) and event[0] == 'release_table') == [2, 4]


def test_native_payload_check_uses_device_reference_and_detects_corruption():
    from execution import NativeMtp
    native = NativeMtp.__new__(NativeMtp)
    native.torch = torch
    native.batch, native.hot, native.token_bytes, native.source_tokens = 1, 2048, 656, 2048
    selected = torch.arange(2048).repeat(1, 4, 1)
    native.destinations = selected.clone()
    native.forward = torch.arange(2048).reshape(1, -1)
    native.counts = torch.tensor([0, 0])
    reference = torch.ones((1, 2048, 656), dtype=torch.uint8)
    native.output = reference.clone().reshape(-1, 656)
    # Deliberately different CPU data proves validation uses the retained reference.
    native.owner = SimpleNamespace(session=SimpleNamespace(reference=reference, cpu_payload=torch.zeros_like(reference)))
    assert native.verify(selected) == [0, 0]
    native.output[100, 655] = 0
    with pytest.raises(ValueError, match='payload_mismatch'):
        native.verify(selected)
    assert reference[0, 100, 655] == 1


def test_put_chunks_keep_one_registered_source_and_global_offsets(tmp_path, monkeypatch):
    calls = []
    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def link_to(self, *args):
            pass
        def initialize(self, *args):
            pass
        def batch_put(self, keys, source, offsets, lengths):
            calls.append((source.data_ptr(), keys, offsets, lengths))
        def prepare_access_table(self, keys):
            return keys
        def access_table_view(self, table, **kwargs):
            return SimpleNamespace(addresses=torch.arange(len(table)), valid=torch.ones(len(table)))
        def release_access_table(self, table):
            pass
        def shutdown(self):
            pass
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(NpuRuntime=Runtime))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None), raising=False)
    pool = BenchPool(SimpleNamespace(batches=(2,), history=16384, transport='urma',
                                    kvaf_root=tmp_path, kvaf_server='test:1'))
    pool.borrow(torch, torch.device('cpu'), 2, 16384, 2)
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert [key for _, keys, _, _ in calls for key in keys] == list(range(256))
    for _, keys, offsets, lengths in calls:
        assert offsets == [key * 128 * 656 for key in keys]
        assert lengths == [128 * 656] * len(keys)
    import numpy as np
    from workloads import payload
    for row in range(2):
        assert np.array_equal(pool.cpu_payload[row].numpy(), payload(row, np.arange(16384)))
    pool.release()
    pool.close()


@pytest.mark.parametrize("prepare", [False, True])
def test_dense_shadow_against_swap_reference(tmp_path, prepare):
    root = Path(__file__).resolve().parents[2]
    header = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/shadow_exchange_parallel.h').read_text()
    header = header.replace('THREADS=512', 'THREADS=1').replace('CHUNK=512', 'CHUNK=1')
    begin = header.index('__simt_callee__ inline uint32_t prefix(')
    end = header.index('\n__simt_vf__', begin)
    header = header[:begin] + '__simt_callee__ inline uint32_t prefix(uint32_t, __ubuf__ uint32_t*) { return 0; }' + header[end:]
    header = header.replace('uint32_t extras=ctl[32+15]', 'uint32_t extras=out')
    code = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#include <numeric>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {uint32_t x=0;} threadIdx;
namespace pipeline_ab { constexpr uint32_t kMapSize=4096,kHistory=131072,kResident=16384; }
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t a,uint32_t b){uint32_t old=*p;if(old==a)*p=b;return old;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
void asc_syncthreads(){} void asc_threadfence(){}
''' + header + r'''
int main(){using namespace pipeline_ab;
 constexpr uint32_t H=16384,R=8192,N=5000;
 std::vector<uint32_t> m(SCRATCH_WORDS+4*MAP),dense(H*6),valid(R,1),scale(H),source(N),dest(N),reference(H),inverse(H);
 std::vector<uint64_t> keys(H*16),addr(H),logical(H);
 std::iota(reference.begin(),reference.end(),0);std::iota(inverse.begin(),inverse.end(),0);
 for(uint32_t i=0;i<H;i++){scale[i]=i;addr[i]=i+100000;logical[i]=i+200000;for(int j=0;j<16;j++)keys[16*i+j]=16*i+j;}
 for(uint32_t i=0;i<N;i++){source[i]=8192+i;dest[i]=i;}
 // Exercise a swap whose source moved during a previous batch.
 source[3000]=reference[0];dest[3000]=6000;
 InitShadow(m.data());
 for(uint32_t i=0;i<N;i++){
  if(i==1500)InitDenseShadow(dense.data(),H,m.data());
  if(i<1500)ApplyBatch(source.data(),dest.data(),i,1,keys.data(),scale.data(),valid.data(),R,m.data(),nullptr,nullptr,nullptr,H);
  else ApplyDenseBatch(source.data(),dest.data(),i,1,keys.data(),scale.data(),valid.data(),R,dense.data(),H,m.data());
  assert(m[CONTROL]==0);uint32_t p=inverse[source[i]],d=dest[i],a=reference[p],b=reference[d];
  reference[d]=a;reference[p]=b;inverse[a]=d;inverse[b]=p;
 }
 CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);
 for(uint32_t i=0;i<H;i++){
  assert(dense[i]==reference[i] && dense[H+reference[i]]==i);
  assert(scale[i]==reference[i] && keys[i*16]==uint64_t(reference[i])*16);
  assert(addr[i]==uint64_t(reference[i])+100000 && logical[i]==uint64_t(reference[i])+200000);
 }
}
'''
    if prepare:
        code = code.replace(',nullptr,nullptr,nullptr,H)', ')')
        code = code.replace('ApplyBatch(source.data()', 'ApplyBatch<true>(source.data()').replace('ApplyDenseBatch(source.data()', 'ApplyDenseBatch<true>(source.data()')
        code = code.replace('R,m.data());', 'R,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);')
        code = code.replace('R,dense.data(),H,m.data());', 'R,dense.data(),H,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H));')
        code = code.replace('CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);',
                            'CommitPreparedMetadata(addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H,m.data(),dense.data(),true);')
        code = code.replace('assert(m[CONTROL]==0);', 'assert(m[CONTROL]==0);assert(addr[0]==100000 && logical[0]==200000);')
    src = tmp_path / 'dense.cpp'
    exe = tmp_path / 'dense'
    src.write_text(code)
    subprocess.run(['c++', '-std=c++17', '-O2', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("use_dense", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_dense_shadow_parallel_batch_and_cross_batch_chain(tmp_path, prepare, use_dense, optimized):
    """Run the real vector body with a 32-lane host model of one warp."""
    root = Path(__file__).resolve().parents[2]
    header = (root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/shadow_exchange_parallel.h').read_text()
    header = header.replace('THREADS=512', 'THREADS=32').replace('CHUNK=512', 'CHUNK=32')
    # Keep full metadata staging capacity even when shrinking CHUNK below MAP.
    # A one-warp model has one partial sum instead of sixteen.
    header = header.replace('lane<16?scan[lane]:0', 'lane<1?scan[lane]:0')
    header = header.replace('ctl[32+15]', 'ctl[32]')
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <condition_variable>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct ThreadIdx {uint32_t x;}; thread_local ThreadIdx threadIdx{};
struct Barrier {
 std::mutex m; std::condition_variable cv; unsigned n=0,gen=0;
 void wait(){std::unique_lock<std::mutex> l(m);unsigned g=gen;if(++n==32){n=0;++gen;cv.notify_all();}else cv.wait(l,[&]{return gen!=g;});}
} barrier;
uint32_t lanes[32];
void asc_syncthreads(){barrier.wait();}
void asc_threadfence(){__atomic_thread_fence(__ATOMIC_SEQ_CST);}
uint32_t asc_shfl_up(uint32_t v,uint32_t d){lanes[threadIdx.x]=v;barrier.wait();uint32_t r=threadIdx.x>=d?lanes[threadIdx.x-d]:0;barrier.wait();return r;}
template<class T>T asc_ldcg(T* p){return __atomic_load_n(p,__ATOMIC_SEQ_CST);}
template<class T>void asc_stcg(T* p,T v){__atomic_store_n(p,v,__ATOMIC_SEQ_CST);}
uint32_t asc_atomic_cas(uint32_t* p,uint32_t a,uint32_t b){__atomic_compare_exchange_n(p,&a,b,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return a;}
void asc_atomic_or(uint32_t* p,uint32_t v){__atomic_fetch_or(p,v,__ATOMIC_SEQ_CST);}
namespace pipeline_ab { constexpr uint32_t kMapSize=4096,kHistory=131072,kResident=16384; }
''' + header + r'''
template<class F> void launch(F f){std::vector<std::thread> ts;for(unsigned i=0;i<32;i++)ts.emplace_back([&,i]{threadIdx.x=i;f();});for(auto &t:ts)t.join();}
int main(){using namespace pipeline_ab;
 constexpr uint32_t H=512,R=256,N=96;
 std::vector<uint32_t> m(SCRATCH_WORDS+4*MAP),dense(H*6),valid(R,1),scale(H),src(N),dst(N),ref(H),inverse(H);
 std::vector<uint64_t> keys(H*16),addr(H),logical(H);
 std::iota(ref.begin(),ref.end(),0);std::iota(inverse.begin(),inverse.end(),0);
 for(uint32_t i=0;i<H;i++){scale[i]=i;addr[i]=i+1000;logical[i]=i+2000;for(int j=0;j<16;j++)keys[16*i+j]=16*i+j;}
 for(uint32_t i=0;i<N;i++){src[i]=256+i;dst[i]=i;}
 src[32]=0;dst[33]=0; // moved source and repeated destination across batches
 src[64]=65;src[65]=64; // two-row cycle within the third batch
 launch([&]{InitShadow(m.data());});
 for(uint32_t phase=0;phase<3;phase++){
  uint32_t begin=phase*32;
  if(phase==1)launch([&]{InitDenseShadow(dense.data(),H,m.data());});
  if(!phase)launch([&]{ApplyBatch(src.data(),dst.data(),begin,32,keys.data(),scale.data(),valid.data(),R,m.data(),nullptr,nullptr,nullptr,H);});
  else launch([&]{ApplyDenseBatch(src.data(),dst.data(),begin,32,keys.data(),scale.data(),valid.data(),R,dense.data(),H,m.data());});
  assert(!m[CONTROL]);
  uint32_t first=0;
  if(phase==2){uint32_t a=ref[64],b=ref[65];ref[64]=b;ref[65]=a;inverse[a]=65;inverse[b]=64;first=2;}
  for(uint32_t j=first;j<32;j++){uint32_t i=begin+j,p=inverse[src[i]],d=dst[i],a=ref[p],b=ref[d];ref[d]=a;ref[p]=b;inverse[a]=d;inverse[b]=p;}
 }
 launch([&]{CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);});
 for(uint32_t i=0;i<H;i++){
  assert(dense[i]==ref[i] && dense[H+ref[i]]==i);
  assert(scale[i]==ref[i] && keys[i*16]==uint64_t(ref[i])*16);
  assert(addr[i]==uint64_t(ref[i])+1000 && logical[i]==uint64_t(ref[i])+2000);
 }
}
'''
    if not use_dense:
        source = source.replace('if(phase==1)launch([&]{InitDenseShadow(dense.data(),H,m.data());});', '')
        source = source.replace('if(!phase)launch', 'if(true)launch')
        source = source.replace('CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);', 'CommitMetadata(addr.data(),logical.data(),m.data());')
        source = source.replace('assert(dense[i]==ref[i] && dense[H+ref[i]]==i);', 'assert(get(m.data()+KEYS_POS,m.data()+VALUES_POS,i)==ref[i]);')
    if prepare:
        source = source.replace(',nullptr,nullptr,nullptr,H)', ')')
        source = source.replace('ApplyBatch(src.data()', 'ApplyBatch<true>(src.data()').replace('ApplyDenseBatch(src.data()', 'ApplyDenseBatch<true>(src.data()')
        source = source.replace('R,m.data());', 'R,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);')
        source = source.replace('R,dense.data(),H,m.data());', 'R,dense.data(),H,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H));')
        target = ('CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);' if use_dense else 'CommitMetadata(addr.data(),logical.data(),m.data());')
        source = source.replace(target, 'CommitPreparedMetadata(addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H,m.data(),dense.data(),' + str(use_dense).lower() + ');')
        source = source.replace('assert(!m[CONTROL]);', 'assert(!m[CONTROL]);for(unsigned i=0;i<H;i++){assert(addr[i]==i+1000 && logical[i]==i+2000);}')
    if optimized:
        source = '#define KVIP_FUSE_QUERY_WORK 1\n' + source
        # Sparse prepared values are indexed by MAP buckets, unlike dense rows.
        source = source.replace('H=512,R=256,N=96', 'H=8192,R=4096,N=96')
        if prepare and use_dense:
            source = source.replace('InitDenseShadow(dense.data(),H,m.data());',
                'ExpandPreparedMetadata(reinterpret_cast<uint64_t*>(dense.data()+2*H),H,m.data());InitDenseShadow(dense.data(),H,m.data());')
        elif prepare:
            # Model the production MTE2 staging before sparse metadata commit.
            source = source.replace('launch([&]{CommitPreparedMetadata(',
                'std::copy_n(reinterpret_cast<uint64_t*>(dense.data()+2*H),MAP,reinterpret_cast<uint64_t*>(m.data()+STAGING));'
                'std::copy_n(reinterpret_cast<uint64_t*>(dense.data()+2*H)+H,MAP,reinterpret_cast<uint64_t*>(m.data()+STAGING)+MAP);'
                'launch([&]{CommitPreparedMetadata(')
            source = '#include <algorithm>\n' + source
    src, exe = tmp_path / 'parallel.cpp', tmp_path / 'parallel'
    src.write_text(source)
    subprocess.run(['c++', '-std=c++17', '-O2', '-pthread', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=45)


def test_benchmark_options_preserve_workload_and_do_not_profile():
    args = configuration(['bench', '--batches', '1,2,4,8,16', '--seed', '93',
                          '--miss-percent', '15', '--iterations', '7'])
    worker = runner_options(args)
    assert worker.batches == '1,2,4,8,16'
    assert worker.seed == 93 and worker.miss_percent == 15
    assert worker.iterations == 7 and worker.trace_rounds == 0
    assert worker.q_workers == 2
    assert not hasattr(worker, "streaming")  # No retired pipeline selector.


def test_diagnostic_repeats_follow_iterations():
    args = configuration(['diagnose', '--iterations', '7', '--stages', 'update'])
    worker = runner_options(args)
    assert worker.trace_rounds == 7
    assert worker.stages == ('update',)


def test_real_never_falls_back_to_synthetic():
    import pytest
    args = configuration(['bench', '--input', 'real', '--trace-root', '/trace',
                          '--group', '10', '--round-range', '51:52'])
    with pytest.raises(ValueError, match='continuous replay'):
        runner_options(args)


def test_worker_dependency_environment_is_explicit_and_local(tmp_path):
    import os

    from execution import ROOT, worker_environment
    package = tmp_path / 'build-npu/python/kvaf/__init__.py'
    package.parent.mkdir(parents=True)
    package.write_text('')
    inherited = {'PYTHONPATH': '/previous/kvaf', 'ASCEND_HOME_PATH': '/cann'}
    environment = worker_environment(tmp_path, inherited)
    assert environment['PYTHONPATH'].split(os.pathsep)[:2] == [
        str(ROOT / 'python'), str(tmp_path / 'build-npu/python')]
    assert environment['ASCEND_HOME_PATH'] == '/cann'
    assert inherited['PYTHONPATH'] == '/previous/kvaf'


def test_missing_selected_kvaf_does_not_fall_back(tmp_path):
    import pytest
    from execution import worker_environment
    with pytest.raises(FileNotFoundError, match='Selected KVAF'):
        worker_environment(tmp_path, {'PYTHONPATH': '/other/working/kvaf'})


def test_supervisor_passes_selected_environment_to_real_child(tmp_path):
    import json
    import os
    import sys

    from execution import supervise
    observed = tmp_path / 'observed.json'
    marker = tmp_path / 'worker-result.json'
    program = (
        'import json, os, pathlib, sys; '
        'pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(selected=os.environ["KVIP_TEST_SELECTED"]))); '
        'pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(exit_code=0)))'
    )
    environment = dict(os.environ, KVIP_TEST_SELECTED='explicit-root')
    code = supervise([sys.executable, '-c', program, str(observed), str(marker)], tmp_path,
                     environment=environment)
    assert code == 0
    assert json.loads(observed.read_text()) == {'selected': 'explicit-root'}


def test_fetch_only_is_automatic_only_for_level_three_fetch():
    for arguments, expected in [
        (['bench'], False),
        (['diagnose', '--level', '2'], False),
        (['diagnose', '--level', '3', '--stages', 'update'], False),
        (['diagnose', '--level', '3', '--stages', 'fetch'], True),
        (['diagnose', '--level', '3'], True),
    ]:
        worker = runner_options(configuration(arguments))
        assert worker.diagnostic_fetch_only is expected
        assert not hasattr(worker, "fetch_only_compare")  # No path bypasses full-graph timing.


def test_bench_matrix_rejects_invalid_percentages():
    import pytest
    for value in ('nan', 'inf', '3,nan', '3,101', '-1,3', '3,3.0', '3,'):
        with pytest.raises(SystemExit):
            configuration(['bench', f'--miss-percent={value}'])


def test_matrix_dispatch_collects_twenty_pairs_without_npu(tmp_path, monkeypatch):
    import json

    import execution as execution
    from reporting import atomic_json, timing_summary

    def launch(job, directory):
        directory.mkdir(parents=True)
        assert job.miss_percent == (3, 5, 10, 15)
        for miss in job.miss_percent:
            for batch in job.batches:
                atomic_json(directory / f'miss{miss}' / f'bs{batch}' / 'report.json', dict(
                    batch=batch, miss_percent=miss, correctness='PASS', timing=dict(
                        native_chain=timing_summary([100, 110]),
                        prototype_with_update=timing_summary([80, 90]))))
        atomic_json(directory / 'worker-result.json', dict(exit_code=0))
        return 0

    monkeypatch.setattr(execution, 'launch', launch)
    for transport in ('urma', 'ubmem'):
        output = tmp_path / transport
        config = configuration(['bench', '--transport', transport, '--batches', '1,2,4,8,16',
                                '--miss-percent', '3,5,10,15', '--kvaf-server', 'test:1',
                                '--output', str(output)])
        assert execution.dispatch(config) == 0
        summary, = output.rglob('summary.json')
        rows = json.loads(summary.read_text())['cases']
        assert len(rows) == 20
        assert [(row['miss_percent'], row['batch']) for row in rows] == [
            (miss, bs) for miss in (3, 5, 10, 15) for bs in (1, 2, 4, 8, 16)]
        assert {(row['batch'], row['miss_percent']) for row in rows} == {
            (bs, miss) for bs in (1, 2, 4, 8, 16) for miss in (3, 5, 10, 15)}
        assert all(row['transport'] == transport and row['accepted'] for row in rows)
        assert all(row['serial_mean_us'] == 105 and row['fused_mean_us'] == 85 for row in rows)
        assert 'Serial p50/p99' in summary.with_name('summary.log').read_text()


def test_nested_miss_progress_reaches_supervised_worker(tmp_path):
    from execution import progress
    (tmp_path / 'config.json').write_text('{}')
    child = tmp_path / 'miss15'
    child.mkdir()
    progress(child, 'BS=8: paired benchmark')
    assert (child / 'stage.log').read_text().strip() == 'BS=8: paired benchmark'
    assert (tmp_path / 'stage.log').read_text().strip() == 'miss15: BS=8: paired benchmark'


@pytest.mark.parametrize('show_details', [False, True])
@pytest.mark.parametrize('exit_code', [0, 1])
def test_supervisor_details_saved_and_failures_visible(tmp_path, capsys, show_details, exit_code):
    import sys

    from execution import supervise

    (tmp_path / 'cases.log').write_text('DETAIL_COUNTERS prefetch=128\n')
    program = (
        'import json, pathlib, sys; '
        'pathlib.Path(sys.argv[1]).write_text(json.dumps({"exit_code": int(sys.argv[2])})); '
        'print("FAILURE_REASON" if int(sys.argv[2]) else "worker detail"); '
        'sys.exit(int(sys.argv[2]))'
    )
    code = supervise([sys.executable, '-c', program,
                      str(tmp_path / 'worker-result.json'), str(exit_code)],
                     tmp_path, show_case_details=show_details)
    output = capsys.readouterr().out
    assert code == exit_code
    assert ('DETAIL_COUNTERS' in output) == show_details
    assert 'DETAIL_COUNTERS' in (tmp_path / 'summary.log').read_text()
    if exit_code:
        assert 'FAILURE_REASON' in output


@pytest.mark.parametrize('transport', ['urma', 'ubmem'])
def test_initialization_acquires_and_releases_selected_lease(monkeypatch, tmp_path, transport):
    class Tensor(np.ndarray):
        def cpu(self):
            return self
        def to(self, device):
            return self
        def contiguous(self):
            return self
    def tensor(value, **kwargs):
        return np.asarray(value, dtype=kwargs.get('dtype')).view(Tensor)
    stored = {}
    def put(keys, source, offsets, sizes):
        assert keys == [0] and offsets == [0] and sizes == [128 * 656]
        stored['bytes'] = source.copy().reshape(-1)
    def get(addresses, valid, indices, offsets, count, destinations, output, stream, flag):
        assert count.tolist() == [128]
        assert np.array_equal(offsets, np.arange(128) * 656)
        for offset, destination in zip(offsets, destinations):
            output[destination] = stored['bytes'][offset:offset + 656]
    runtime = SimpleNamespace(link_to=Mock(), initialize=Mock(), shutdown=Mock(), initialized=True,
                              batch_put=Mock(side_effect=put), batch_get=Mock(side_effect=get),
                              prepare_access_table=Mock(return_value='table'), release_access_table=Mock(),
                              access_table_view=Mock(return_value=SimpleNamespace(addresses='addresses', valid='valid')))
    lease = SimpleNamespace(check_stream=Mock(), close=Mock(), region=Mock())
    constructor = Mock(return_value=runtime)
    kvaf = SimpleNamespace(__file__=str(tmp_path / 'build-npu/python/kvaf/__init__.py'), NpuRuntime=constructor)
    monkeypatch.setitem(sys.modules, 'kvaf', kvaf)
    monkeypatch.setitem(sys.modules, 'acl', SimpleNamespace(get_soc_name=lambda: 'Ascend950PR_9599'))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    stream = SimpleNamespace(synchronize=Mock())
    torch = SimpleNamespace(npu=SimpleNamespace(set_device=Mock(), current_stream=lambda device: stream),
                            device=lambda value: value, uint8=np.uint8, int32=np.int32,
                            zeros=lambda shape, **kwargs: tensor(np.zeros(shape, dtype=kwargs['dtype'])),
                            from_numpy=tensor, tensor=tensor, equal=np.array_equal,
                            arange=lambda n, **kwargs: tensor(np.arange(n), dtype=kwargs.get('dtype')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(kvip.adapters.kvaf, 'KvafGetLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.adapters.kvaf, 'UbmemLease', Mock(return_value=lease))
    monkeypatch.setattr(kvip.runtime.build, 'kvaf_build_paths', lambda path: ('cann', 'headers'))
    monkeypatch.setattr(kvip.runtime.build, 'build_libraries', lambda **kwargs: ('library',))
    kernels = tmp_path / 'build-npu/bin'
    kernels.mkdir(parents=True)
    for name in ('kvaf_execute_batch_get_kernels.o', 'kvaf_execute_batch_put_kernels.o'):
        (kernels / name).touch()
    assert init_transport.initialize_transport(transport, ['--device', '5', '--kvaf-server', 'host:1',
                                           '--kvaf-root', str(tmp_path)]) == 0
    runtime.link_to.assert_called_once_with('host:1')
    runtime.initialize.assert_called_once()
    lease.check_stream.assert_called_once()
    lease.close.assert_called_once()
    runtime.shutdown.assert_called_once()
    runtime.release_access_table.assert_called_once_with('table')
    runtime.batch_get.assert_called_once()
    assert constructor.call_args.kwargs.get('transport', 'urma') == transport
    if transport == 'ubmem':
        lease.region.assert_called_once()
        assert lease.region.call_args.args[0].shape == (128, 656)
    else:
        lease.region.assert_not_called()


def cpu_state(tmp_path):
    inputs = assemble_round([reference(tmp_path)])
    pages = ReplayPagePlan([inputs])
    state = FusedReplayState.__new__(FusedReplayState)
    state.torch, state.device, state.pages, state.initial = torch, torch.device('cpu'), pages, inputs
    state.batch, state.history, state.resident, state.stride, state.token_bytes = 1, 4096, 2048, 2176, 656
    state.logical_addresses = np.arange(4096, dtype=np.int64)[None] * 656 + 0x100000
    state.storage = SimpleNamespace(
        logical_ids=torch.arange(4096).repeat(2, 1), keys=torch.zeros((2, 4096, 128), dtype=torch.uint8),
        scales=torch.ones((2, 4096)), hot_valid=torch.ones((2, 2048), dtype=torch.int32),
        addresses=torch.zeros((2, 4096), dtype=torch.int64))
    state.workspace = SimpleNamespace(inputs=SimpleNamespace(
        query=torch.zeros((4, 2, 128), dtype=torch.uint8),
        weights=torch.zeros((4, 2), dtype=torch.bfloat16), scale=torch.zeros((4, 2)),
        query_ends=torch.zeros(4, dtype=torch.int32)), tail_bounds=torch.zeros((1, 2), dtype=torch.int32))
    state.output = torch.zeros((2, 2176, 656), dtype=torch.uint8)
    state.rotating = SimpleNamespace(lru_slots=torch.empty((1, 2048), dtype=torch.int32),
                                    reserved_hot=torch.empty((1, 2048), dtype=torch.int32))
    state.policies = [SimpleNamespace(slots=np.arange(2048, dtype=np.int32)[None])]
    return state, inputs


def test_prepare_imports_bits_and_retains_current_state(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.storage.logical_ids[1, [0, 3000]] = state.storage.logical_ids[1, [3000, 0]]
    state.storage.hot_valid[1, 0] = 0
    inputs.weights[:] = 0xBF80  # -1 BF16, not float value 49024.
    state.prepare(inputs)
    assert torch.all(state.workspace.inputs.weights == -1)
    assert state.storage.scales[1, 0] == 1095
    assert state.storage.logical_ids[1, 0] == 3000 and state.storage.hot_valid[1, 0] == 0
    assert state.storage.addresses[1, 0] == 0x100000 + 3000 * 656
    assert state.rotating.lru_slots[0, -1] == 2047
    ids = np.arange(3968, 4096)
    np.testing.assert_array_equal(state.output[1, 2048:].numpy(), state.pages.token_payload(0, ids))


def test_completion_error_stops_before_reading_outputs(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.update = SimpleNamespace(status=torch.tensor([7]))
    with pytest.raises(RuntimeError, match='completion failed'):
        state.validate(inputs)


def valid_index_state(tmp_path):
    state, inputs = cpu_state(tmp_path)
    state.prepare(inputs)
    layout = state.storage.logical_ids[1].numpy()
    state.upload(state.output[1, :state.resident], state.pages.token_payload(0, layout[:state.resident]))
    return state, inputs


def test_all_index_fields_and_hot_bytes_checked_before_next_prepare(tmp_path):
    state, inputs = valid_index_state(tmp_path)
    state.validate_index(inputs, state.output[1:].numpy())
    # A non-first key column outside the selected TopK must still be checked.
    state.storage.keys[1, 3000, 127] ^= 1
    with pytest.raises(RuntimeError, match='Index keys mismatch'):
        state.validate_index(inputs, state.output[1:].numpy())


@pytest.mark.parametrize('field', ['scales', 'addresses', 'logical_ids', 'hot_valid', 'output'])
def test_persistent_state_corruption_is_fatal(tmp_path, field):
    state, inputs = valid_index_state(tmp_path)
    if field == 'output':
        state.output[1, 1, 655] ^= 1
    elif field == 'logical_ids':
        state.storage.logical_ids[1, 1] = 0  # Duplicate identity, not a permutation.
    elif field == 'hot_valid':
        state.storage.hot_valid[1, 1] = 2
    else:
        getattr(state.storage, field)[1, 3000] += 1
    with pytest.raises(RuntimeError, match='Replay'):
        state.validate_index(inputs, state.output[1:].numpy())


def test_urma_replay_allocates_complete_scratch_before_binding(monkeypatch, tmp_path):
    from kvip.adapters import kvaf
    from kvip.operators.lifetch import binding, workspace
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry

    real_workspace = workspace.LIFetchWorkspace
    monkeypatch.setattr(workspace, 'LIFetchWorkspace',
                        lambda torch, **kwargs: real_workspace(torch, pin_memory=False, **kwargs))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda device: None), raising=False)
    lease = SimpleNamespace(sizes={'status': 1536}, close=lambda: closed.append(True))
    closed = []
    monkeypatch.setattr(kvaf, 'KvafGetLease', lambda *args: lease)
    monkeypatch.setattr(kvaf, 'KvafQueueView', lambda *args: None)
    shape = li_geometry(2, 'auto', 8, 24)
    expected = workspace.rotation_spec(2, 12288, 4096, final_aivs(2, 'auto', 24),
                                      shape['intermediate_fetches'],
                                      shape['service_aivs_per_window'], 1536)

    class BindingReached(Exception):
        pass

    def inspect_binding(*args, scratch, **kwargs):
        assert kwargs['output_stride'] - kwargs['resident'] >= 384
        assert kwargs['tail_bounds'].shape == (2, 2)
        assert set(scratch) == set(expected)
        for name, (size, dtype) in expected.items():
            assert tuple(scratch[name].shape) == size
            assert scratch[name].dtype == getattr(torch, dtype)
            assert torch.count_nonzero(scratch[name]) == 0
        raise BindingReached

    monkeypatch.setattr(binding, 'LIFetchBinding', inspect_binding)
    from correctness import stateful_inputs
    initial = next(iter(stateful_inputs(tmp_path)))
    with pytest.raises(BindingReached):
        FusedReplayState(torch, SimpleNamespace(transport='urma'),
                         SimpleNamespace(addresses=None, valid=None), None, initial, None, None,
                         device=torch.device('cpu'), aics=24, windows=8,
                         prefetch_tokens=0, soc='Ascend950PR_9599')
    assert closed == [True]


def setup_run(monkeypatch, *, mode='bench'):
    initial = SimpleNamespace(geometry={'batch': 1}, identities=((0, 'request', 0),))
    ref = SimpleNamespace(manifest_path='/source/manifest.json')
    class Sequence:
        rounds = ((ref,),)
        def __iter__(self):
            return iter([initial])
    sequence = Sequence()
    monkeypatch.setattr(replay_execution, 'ReplaySequence', lambda *args: sequence)
    monkeypatch.setattr(replay_execution, 'ReplayPagePlan', lambda seq: SimpleNamespace(page_count=32))
    return SimpleNamespace(trace_root='/trace', group=10, round_range=(1, 1), mode=mode)


def test_real_pair_saves_serial_before_fused_failure(tmp_path, monkeypatch):
    config = setup_run(monkeypatch)
    calls = []
    def run(config, directory, sequence, pages, *, fused):
        calls.append(fused)
        if fused:
            persisted = json.loads((tmp_path / 'completed_timings.json').read_text())
            assert persisted['execution'] == 'INCOMPLETE'
            assert persisted['timing']['native_chain']['mean_us'] == 20
            raise RuntimeError('fused failed')
        return {'all_rounds': timing_summary([20, 20]), 'correctness': 'PASS'}
    monkeypatch.setattr(replay_execution, 'run_path', run)
    with pytest.raises(RuntimeError, match='fused failed'):
        replay_execution.run_real(config, tmp_path)
    assert calls == [False, True]
    assert not (tmp_path / 'report.json').exists()


def test_real_numerical_failure_keeps_paired_timing(tmp_path, monkeypatch):
    config = setup_run(monkeypatch)
    def run(config, directory, sequence, pages, *, fused):
        return {'all_rounds': timing_summary([10 if fused else 20]),
                'correctness': 'FAIL' if fused else 'PASS'}
    monkeypatch.setattr(replay_execution, 'run_path', run)
    assert replay_execution.run_real(config, tmp_path) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['correctness'] == 'FAIL' and report['execution'] == 'COMPLETED'
    assert report['timing']['prototype_with_update']['mean_us'] == 10
    assert report['sequence']['serial']['correctness'] == 'PASS'


@pytest.mark.parametrize('width', [656, 2624])
def test_existing_native_mtp_supports_group_payload_width(monkeypatch, width):
    import torch
    from execution import NativeMtp
    calls = []
    runtime = SimpleNamespace(batch_get=lambda *args: calls.append(args))
    device_table = SimpleNamespace(addresses=torch.ones(16, dtype=torch.int64), valid=torch.ones(16, dtype=torch.uint8))
    session = SimpleNamespace(cpu_payload=torch.from_numpy(np.zeros((1, 2048, width), dtype=np.uint8)),
                              runtime=runtime, device_table=device_table, batch_get_attempted=False)
    mapping = torch.full((1, 2048), -1, dtype=torch.int32)
    mapping[0, :128] = torch.arange(128, dtype=torch.int32)
    owner = SimpleNamespace(batch=1, resident_budget=128, cache_map=mapping, session=session, lease=None)
    native = NativeMtp(torch, torch.device('cpu'), owner, {}, source_tokens=2048, padded_tokens=2048,
                       token_bytes=width)
    assert native.output.shape == (128, width)
    native.offsets.fill_(656)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_stream=lambda: None), raising=False)
    native.fetch()
    assert len(calls) == 1 and torch.all(calls[0][3] == width)
    assert calls[0][6].shape == (128, width)


def test_native_li_keeps_synthetic_contract_and_supports_real_causal_q4():
    from execution import native_li
    calls = []
    def invoke(**kwargs):
        calls.append(kwargs)
        return np.zeros((1, 4, 2048), dtype=np.int32)
    fixture = dict(query=None, key=None, native_inputs={'weights': None, 'query_scale': None, 'key_scale': None},
                   li_query_lengths=None, actual_key=None, block_table=None)
    backend = SimpleNamespace(npu_quant_lightning_indexer=invoke)
    native_li(backend, fixture)
    native_li(backend, dict(fixture, sparse_mode=3))
    assert [call['sparse_mode'] for call in calls] == [0, 3]
    assert calls[0]['layout_query'] == calls[1]['layout_query'] == 'BSND'


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def data_ptr(self):
        return self.values.ctypes.data

    def numel(self):
        return self.values.size

    def to(self, device):
        return self

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def copy_(self, other):
        self.values[...] = other.values


def binding_fixture(transport="urma", batch=1, history=131072, resident=16384, heads=32):
    stream = SimpleNamespace(npu_stream=303)
    torch = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        uint64=np.uint64,
        zeros=Mock(side_effect=lambda shape, dtype, device: Tensor(np.zeros(shape, dtype=dtype))),
        empty=Mock(side_effect=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype))),
        empty_like=lambda value: Tensor(np.empty_like(value.values)),
        tensor=lambda data, dtype, device: Tensor(np.asarray(data, dtype=dtype)),
        from_numpy=Tensor,
        npu=SimpleNamespace(current_stream=Mock(return_value=stream)),
    )
    rotating = SimpleNamespace(
        torch=torch,
        device=5,
        prefetch_limit=512,
        runtime_shape=RuntimeShape(
            batch=batch, history=history, resident=resident, indexHeads=heads, aicCount=32, finalWorkers=4
        ),
        context=Tensor(np.zeros(320, dtype=np.uint8)),
        hot_valid=Tensor(np.zeros(32, dtype=np.int32)),
        arguments=[Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(15)],
        library=SimpleNamespace(
            kvip_lifetch_streaming_forward=Mock(return_value=0),
            kvip_streaming_resolver_cores=Mock(return_value=3),
            kvip_streaming_flag_count=Mock(return_value=16),
        ),
    )
    if transport == "ubmem":
        rotating.lease = SimpleNamespace(check_stream=Mock(return_value=stream))
    update = SimpleNamespace(
        device=5,
        bindings=Tensor(np.zeros(28, dtype=np.uint64)),
        storage=[Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(4)],
    )
    return rotating, update


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
def test_public_forward_only_launches_retained_workspace(transport):
    rotating, update = binding_fixture(transport)
    forward = LIFetchForward(rotating, update)
    allocations = (rotating.torch.zeros.call_count, rotating.torch.empty.call_count)
    pointer = forward.workspace.data_ptr()
    for _ in range(2):
        forward()
        args = rotating.library.kvip_lifetch_streaming_forward.call_args.args
        assert args[:4] == (rotating.bindings.data_ptr(), update.bindings.data_ptr(), pointer, 303)
        assert args[5] == 2
        assert ctypes.cast(args[4], ctypes.POINTER(RuntimeShape)).contents.batch == 1
    assert allocations == (rotating.torch.zeros.call_count, rotating.torch.empty.call_count)
    if transport == "ubmem":
        rotating.lease.check_stream.side_effect = RuntimeError("released lease")
        with pytest.raises(RuntimeError, match="released lease"):
            forward()
        assert rotating.library.kvip_lifetch_streaming_forward.call_count == 2
    rotating.library.kvip_lifetch_streaming_forward.return_value = 42
    if transport == "ubmem":
        rotating.lease.check_stream.side_effect = None
    with pytest.raises(RuntimeError, match="42"):
        forward()


@pytest.mark.parametrize("batch", (0, 17, 32))
def test_batch_limit_fails_before_device_allocation(batch):
    rotating, update = binding_fixture(batch=batch)
    with pytest.raises(ValueError, match="BS1..16"):
        LIFetchForward(rotating, update)
    rotating.torch.zeros.assert_not_called()
    rotating.torch.empty.assert_not_called()


@pytest.mark.parametrize("history,resident,heads", ((33024, 8192, 64), (210944, 131072, 32), (65536, 2048, 64)))
def test_runtime_workspace_scales_without_recompilation(history, resident, heads):
    rotating, update = binding_fixture(history=history, resident=resident, heads=heads)
    rotating.prefetch_limit = 2048
    forward = LIFetchForward(rotating, update)
    assert forward.query_bitmaps.values.shape == (1, 4, resident // 32)
    assert forward.events_src.values.shape == (1, 10240)
    assert forward.dense_shadow.values.shape == (1, 3 * history)
    assert np.frombuffer(forward.workspace.values, dtype="<u4", count=1, offset=96)[0] & 0xFFFF == 2048


def test_device_and_aiv_budget_are_checked():
    rotating, update = binding_fixture(batch=16)
    update.device = 6
    with pytest.raises(ValueError, match="same device"):
        LIFetchForward(rotating, update)
    update.device = 5
    rotating.runtime_shape.aicCount = 24
    with pytest.raises(ValueError, match="AIV budget"):
        LIFetchForward(rotating, update)
    rotating.torch.empty.assert_not_called()


def test_layer_groups_share_only_ephemeral_storage():
    first, update1 = binding_fixture()
    second, update2 = binding_fixture()
    pool = {}
    first.streaming_scratch = second.streaming_scratch = pool
    a, b = LIFetchForward(first, update1), LIFetchForward(second, update2)
    assert a.dense_shadow is b.dense_shadow
    assert a.packed_fetch is b.packed_fetch
    assert a.flags is b.flags
    assert a.workspace.data_ptr() != b.workspace.data_ptr()
    assert a.rotating.context.data_ptr() != b.rotating.context.data_ptr()
    second.torch.empty.assert_not_called()


def test_device_failure_invalidates_only_failed_owner(tmp_path):
    """Execute the retained production error handler after removing old update."""
    import subprocess
    from pathlib import Path

    header = Path(__file__).resolve().parents[2] / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/index_cache_update.h'
    source = tmp_path / 'failure.cpp'
    source.write_text(r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct { unsigned x; } threadIdx;
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T value){*p=value;}
void asc_threadfence(){}
''' + f'#include "{header}"\n' + r'''
int main(){
 uint64_t args[28]={};uint32_t status[2]={0,0},owners[2]={2,0};
 std::vector<uint32_t> valid(3*1024,1);
 args[6]=(uintptr_t)status;args[9]=(uintptr_t)owners;
 args[16]=(uintptr_t)valid.data();args[25]=1024;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,0,0);
 for(auto v:valid)assert(v==1);
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,0,9);
 assert(status[0]==4 && status[1]==0);
 for(unsigned i=0;i<valid.size();++i)assert(valid[i]==(i<2048?1U:0U));
 status[1]=2;
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)CompleteEarlyUpdate<256>(args,1,0);
 assert(status[1]==2);
 for(unsigned i=0;i<1024;++i)assert(valid[i]==0 && valid[1024+i]==1);
}
''')
    binary = tmp_path / 'failure'
    subprocess.run(['c++', '-std=c++17', '-fsanitize=address,undefined', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_host_copy_and_launch_models():
    import os
    import tempfile
    root = ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem"
    tests = ROOT / "tests/operators/lifetch_models"
    compiler = os.environ.get("CXX", "c++")
    common = {
        "SIM_HISTORY": 33024,
        "KVIP_STREAMING_UBMEM": 1,
        "KVIP_OUTPUT_WORKER": 1,
        "SIM_RESIDENT": 8192,
        "INDEX_HEADS": 64,
        "SIM_Q": 4,
        "V9_AIC_COUNT": 32,
        "V9_SPLIT_Q": 1,
        "V9_INTERMEDIATE_FETCHES": 8,
        "V9_FINAL_AIVS": 0,
        "V9_SERVICE_AICS": 0,
        "V9_FINAL_REPLICATED": 1,
        "V9_DEDICATED_SERVICE": 0,
        "V9_UBPIPE_DEPTH": 8,
        "V9_UBPIPE_ROWS": 4,
        "V9_WINDOW_UBPIPE_DEPTH": 8,
        "V9_USER_UB_BYTES": 223232,
    }
    with tempfile.TemporaryDirectory(prefix="kvip-ubmem-model-") as directory:
        for batch, resident, width in (
            (1, 8192, 656),
            (1, 8192, 2624),
            (2, 8192, 656),
            (2, 8192, 2624),
            (1, 2048, 2624),
            (32, 2048, 656),
        ):
            definitions = common | {
                "SIM_BATCH": batch,
                "SIM_RESIDENT": resident,
                "TOKEN_BYTES": width,
                "V9_SPLIT_Q": int(2 * batch <= common["V9_AIC_COUNT"]),
            }
            options = [
                compiler,
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                *[f"-D{k}={v}" for k, v in definitions.items()],
                f"-I{root / 'include'}",
                f"-I{root / 'op_kernel/copy'}",
            ]
            for name, sources in (
                ("copy", [tests / "copy_model.cpp"]),
                ("abi", [tests / "abi_test.cpp", root / "op_host/launch.cpp"]),
            ):
                binary = Path(directory) / f"{name}-{batch}-{width}"
                subprocess.run([*options, *map(str, sources), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)
    print("Static ABI and real copy-header models passed B1/B2 and B32; 2048/8192 hot slots, 656/2624-byte rows")


def lease_fixture(token_bytes=656):
    device = SimpleNamespace(type="npu", index=1)
    output = SimpleNamespace(
        device=device, data_ptr=lambda: 0x800000, numel=lambda: 128 * token_bytes, element_size=lambda: 1
    )
    stream = SimpleNamespace(npu_stream=17)
    slot_bytes = (128 * token_bytes + 4095) // 4096 * 4096
    context = dict(
        abi_version=1,
        lease_id=9,
        device_id=1,
        region_base=0x100000,
        mapped_base=0x400000,
        region_bytes=slot_bytes * 4,
        slot_bytes=slot_bytes,
        output_address=output.data_ptr(),
        output_bytes=output.numel(),
        token_bytes=token_bytes,
        tokens_per_block=128,
    )
    runtime = SimpleNamespace(
        transport="ubmem",
        acquire_ubmem_fused_context=Mock(return_value=context),
        release_ubmem_fused_context=Mock(),
        retain_ubmem_fused_graph=Mock(),
    )
    torch = SimpleNamespace(npu=SimpleNamespace(current_stream=Mock(return_value=stream), synchronize=Mock()))
    return torch, runtime, output, stream


@pytest.mark.parametrize("token_bytes", (656, 2624))
def test_padded_kvaf_slots_and_failed_release_retain_owners(token_bytes):
    torch, runtime, output, stream = lease_fixture(token_bytes)
    lease = UbmemLease(torch, runtime, output, stream)
    binding = object()
    lease.bindings.append(binding)
    graph = object()
    lease.retain_graph(graph)
    runtime.release_ubmem_fused_context.side_effect = RuntimeError("device fence failed")
    with pytest.raises(RuntimeError, match="fence"):
        lease.close()
    assert not lease.closed and lease.bindings == [binding] and lease.graphs == [graph]
    assert lease.outputs
    runtime.release_ubmem_fused_context.side_effect = None
    lease.close()
    assert lease.closed and not lease.outputs and not lease.bindings and not lease.graphs


def test_invalid_native_lease_descriptor_releases_acquired_lease():
    torch, runtime, output, stream = lease_fixture()
    context = runtime.acquire_ubmem_fused_context.return_value
    context["device_id"] = 7
    with pytest.raises(ValueError, match="ABI/device"):
        UbmemLease(torch, runtime, output, stream)
    runtime.release_ubmem_fused_context.assert_called_once_with(context, stream)


def test_peer_and_device_mapping_bases_are_kept_distinct():
    torch, runtime, output, stream = lease_fixture()

    class Wrapped:
        def __init__(self, array):
            self.array = array

        def to(self, device):
            return self.array

    torch.from_numpy = Wrapped
    lease = UbmemLease(torch, runtime, output, stream)
    words = lease.region(output).view("<u8")
    assert words[2] == 0x100000  # addresses supplied by CPU service
    assert words[3] == 0x400000  # actual imported NPU mapping
    assert words[4] == runtime.acquire_ubmem_fused_context.return_value["region_bytes"]


def test_capture_stream_only_allowed_inside_fenced_scope():
    torch, runtime, output, stream = lease_fixture()
    lease = UbmemLease(torch, runtime, output, stream)
    capture = SimpleNamespace(npu_stream=23)
    with lease.capture_scope():
        torch.npu.current_stream.return_value = capture
        assert lease.check_stream() is capture
    with pytest.raises(RuntimeError, match="execution stream"):
        lease.check_stream()
    torch.npu.current_stream.return_value = stream
    torch.npu.synchronize.side_effect = [None, RuntimeError("failed fence")]
    with pytest.raises(RuntimeError, match="failed fence"):
        with lease.capture_scope():
            pass
    with pytest.raises(RuntimeError, match="unknown"):
        lease.close()
    runtime.release_ubmem_fused_context.assert_not_called()


def test_static_shape_and_ubmem_scratch_contract():
    for batch in (1, 2):
        for width in (656, 2624):
            shape = UbmemGeometry(batch, 33024, 8192, 64, width, 32, 8, "Ascend950DT_9582")
            assert set(shape.options("/cann")) == {"RUN_MODE", "ASCEND_CANN_PACKAGE_PATH", "SOC_VERSION"}
            scratch = rotation_spec(batch, shape.history, shape.resident, shape.aic_count, shape.intermediate_fetches)
            assert scratch["ubmem_status"] == ((batch, 16), "int64")
            assert "final_status" not in scratch and "prefetch_status" not in scratch
    with pytest.raises(ValueError, match="cold tiles"):
        UbmemGeometry(2, 8448, 8192, 64, 656, 32, 8, "Ascend950DT_9582")


@pytest.mark.parametrize("batch,resident", [(1, 2048), (16, 2048), (24, 16384), (32, 2048)])
def test_extended_ubmem_geometry(batch, resident):
    if batch > 16:
        with pytest.raises(ValueError, match="BS1..16"):
            UbmemGeometry(batch, 131072, resident, 32, 656, 32, 8, "Ascend950PR_9599")
        return
    shape = UbmemGeometry(batch, 131072, resident, 32, 656, 32, 8, "Ascend950PR_9599")
    spec = rotation_spec(batch, shape.history, resident, shape.aic_count, shape.intermediate_fetches)
    assert spec["mapping"][0] == (batch, shape.history)
    assert spec["reserved_hot"][0] == (batch, resident)


def test_shared_workspace_keeps_ubmem_transport_specific_status():
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry
    from kvip.operators.lifetch.workspace import rotation_spec as urma_spec

    for batch in (1, 8, 16, 32):
        geometry = li_geometry(batch, "auto", 8, 32)
        original = urma_spec(batch, 131072, 16384, final_aivs(batch, "auto", 32),
                             geometry["intermediate_fetches"], geometry["service_aivs_per_window"], 1216)
        ubmem = rotation_spec(batch, 131072, 16384, 32, 8)
        assert set(original) - {"final_status"} == set(ubmem) - {"ubmem_status"}
        assert all(ubmem[name] == spec for name, spec in original.items() if name != "final_status")
        assert ubmem["ubmem_status"] == ((batch, 16), "int64")


def test_ubmem_supports_sixteen_requests_and_many_windows():
    shape = UbmemGeometry(16, 131072, 16384, 32, 656, 32, 32, "Ascend950PR_9599")
    assert shape.options("/cann")["SOC_VERSION"] == "Ascend950PR_9599"


def test_workspace_initializes_with_npu_supported_zero_dtypes():
    def zeros(size, *, dtype, device):
        if dtype == np.uint64:
            raise RuntimeError("aclnnInplaceZero does not support DT_UINT64")
        return np.zeros(size, dtype=dtype)

    torch = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        uint8=np.uint8,
        uint64=np.uint64,
        zeros=zeros,
        full=lambda size, value, *, dtype, device: np.full(size, value, dtype=dtype),
        empty=lambda size, *, dtype, device: np.empty(size, dtype=dtype),
    )
    with patch("kvip.operators.lifetch.workspace.UbmemQueryInputs"):
        workspace = UbmemWorkspace(torch, batch=2, history=33024, resident=8192, aics=32, windows=8, device="npu:0")
    status = workspace.scratch["ubmem_status"]
    assert status.shape == (2, 16) and status.itemsize == 8
    assert status.tobytes() == bytes(2 * 128)
    workspace.inputs.block_table = SimpleNamespace(shape=(2, 33024 // 128))
    storage = SimpleNamespace(history=33024, keys=object(), scales=object())
    assert len(workspace.arguments(storage)) == 14
    assert not hasattr(workspace, "index_plan")
    assert not hasattr(workspace, "index_work")


def test_device_abi_words_preserve_all_pointer_bits_in_signed_storage():
    words = [0, 17, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]

    def from_numpy(values):
        assert values.dtype == np.int64
        return SimpleNamespace(to=lambda device: (device, values.copy()))

    device, values = device_words(SimpleNamespace(from_numpy=from_numpy), words, "npu:1")
    assert device == "npu:1" and values.shape == (len(words),)
    assert values.tobytes() == np.asarray(words, dtype="<u8").tobytes()
    assert values.view(np.uint64).tolist() == words


def test_shared_index_update_abi_preserves_transport_descriptors():
    from kvip.operators.lifetch.index_update import _update_pointers

    def tensor(address):
        return SimpleNamespace(data_ptr=lambda: address)

    rotating = SimpleNamespace(
        change_slots=tensor(5), changes=tensor(6), rows=tensor(9), row_status=tensor(11),
        final_posts=tensor(12), final_workers=15, hot_valid=tensor(17),
        arguments=[None] * 9 + [tensor(19)], locations=tensor(20), change_count=tensor(27),
    )
    addresses = tensor(3)
    addresses.stride = lambda dim: 8
    query = SimpleNamespace(unpack_indices=tensor(21), output_valid=tensor(22))
    prefix = (tensor(1), tensor(2), addresses, tensor(4), tensor(10), rotating, tensor(7), tensor(18))
    for statuses, expected in (((13, 14, 16), (13, 14, 16)), ((113, 114, 128), (113, 114, 128))):
        values = _update_pointers(*prefix, tensor(statuses[0]), tensor(statuses[1]), statuses[2],
                                  tensor(28), query, tensor(23), tensor(24), 25, 26)
        assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                          expected[0], expected[1], 15, expected[2], 17, 18,
                          19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    values = _update_pointers(*prefix, tensor(13), tensor(14), 16, tensor(28), None, None, None, 25, 26)
    assert values[18:24] == [0] * 6


def test_service_update_arguments_use_matching_layer_group_buffers():
    from kvip.adapters.vllm.selector import index_update_arguments

    storage = SimpleNamespace(keys=object(), scales=object(), addresses=object(), logical_ids=object())
    workspace = SimpleNamespace(inputs=SimpleNamespace(row_owners=object()))
    policy = SimpleNamespace(fused_errors=[object(), object()], fused_outputs=[object(), object()])
    rotating = object()
    values = index_update_arguments(policy, workspace, storage, rotating, 1)
    assert values == dict(keys=storage.keys, scales=storage.scales, addresses=storage.addresses,
                          logical_ids=storage.logical_ids, row_owners=workspace.inputs.row_owners,
                          rotating=rotating, status=policy.fused_errors[1], query_inputs=workspace.inputs,
                          outputs=policy.fused_outputs[1])


def test_unique_active_owners_accepts_repeated_padding_without_device_read():
    inputs = object.__new__(UbmemQueryInputs)
    with patch("kvip.adapters.vllm.inputs.QueryInputs.prepare", return_value=None) as original:
        inputs.prepare([0, 0], query_owners=[0, 0])
        inputs.prepare([4, 0], query_owners=[1, 1])
        assert original.call_count == 2
        with pytest.raises(ValueError, match="distinct"):
            inputs.prepare([4, 1], query_owners=[1, 1])
        assert original.call_count == 2


@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_public_selector_dispatches_transport(transport):
    import kvip

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch = SimpleNamespace(inference_mode=InferenceMode)
    policy = object()
    config = SimpleNamespace(transport=transport)
    with patch("kvip.adapters.vllm.binding._bind_selectors") as urma, \
         patch("kvip.adapters.vllm.binding._bind_ubmem_selectors") as ubmem:
        kvip.bind_selectors(torch, policy, config)
        (ubmem if transport == "ubmem" else urma).assert_called_once_with(torch, policy, config)
        (urma if transport == "ubmem" else ubmem).assert_not_called()


@pytest.mark.parametrize("percent", [10, 30, 50])
def test_controlled_fixture_matches_urma(percent):
    from execution import controlled_fixture

    fixture = controlled_fixture(8192, 4096, 32, percent)
    order = fixture["order"]
    assert fixture["expected"].shape == (4, 2048)
    assert fixture["native_miss_union"] > 0
    assert len(set(order)) == len(order) == 8192
    assert len(fixture["info"]["final_misses_per_query"]) == 4


def test_shared_q4_fixture_topk_and_lru_match_urma():
    from execution import controlled_fixture, layer_payload
    from kvip.cache.lru import SlotLRU

    fixture = controlled_fixture(8192, 4096, 32, 10)
    scores, order = fixture["scores"], fixture["order"]
    expected = fixture["expected"]
    physical_scores = scores[:, order]
    physical = np.stack([np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in physical_scores])
    assert np.array_equal(np.sort(physical, axis=-1), expected)
    union = np.unique(physical[physical >= 4096])
    assert len(union) == fixture["native_miss_union"]
    policy = SlotLRU()
    policy.resize(4096, 4096)
    values = physical.reshape(-1)
    policy.commit(values[values < 4096])
    selected = policy.slots[0, :2048]
    assert len(set(selected)) == 2048 and np.all((selected >= 0) & (selected < 4096))
    assert layer_payload(1, np.arange(3), 656).shape == (3, 656)
    assert layer_payload(1, np.arange(3), 2624).shape == (3, 2624)
    np.testing.assert_array_equal(layer_payload(1, np.arange(3), 2624)[:, :656], layer_payload(1, np.arange(3), 656))


def test_mixed_query_counts_cover_padding_and_query_lengths():
    from execution import mixed_query_counts

    assert mixed_query_counts(2) == [1, 2]
    counts = mixed_query_counts(16)
    assert counts[:4] == [1, 2, 3, 4]
    assert counts[-1] == 0
    assert set(counts) == {0, 1, 2, 3, 4}


def test_main_keeps_cases_after_numerical_failure_and_returns_nonzero(tmp_path, monkeypatch):
    import torch
    build = tmp_path / 'build-npu'
    library = tmp_path / 'kernel.so'
    library.write_bytes(b'test artifact')
    monkeypatch.setitem(sys.modules, 'kvaf', SimpleNamespace(__file__=str(build / 'python/kvaf/__init__.py')))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True, set_device=lambda device: None),
                        raising=False)
    monkeypatch.setattr(ubmem_smoke, 'kvaf_cann', lambda build: tmp_path)
    monkeypatch.setattr(ubmem_smoke, 'query_device_resources', lambda *args: {'effective_aic': 32})
    monkeypatch.setattr(ubmem_smoke, 'build_ubmem_library', lambda **kwargs: library)
    calls = []
    def check(*args, **kwargs):
        assert args[4].intermediate_fetches == 4
        batch = args[4].batch
        calls.append(batch)
        return batch != 1
    monkeypatch.setattr(ubmem_smoke, 'run_ubmem_case', check)
    monkeypatch.setitem(sys.modules, 'acl', SimpleNamespace(get_soc_name=lambda: 'Ascend950PR_9599'))
    config = configuration(['bench', '--transport', 'ubmem', '--kvaf-root', str(tmp_path),
                            '--kvaf-server', '127.0.0.1:1234', '--batches', '1,2',
                            '--intermediate-fetches', '4'])
    (tmp_path / 'results').mkdir()
    status = ubmem_smoke.run_ubmem(config, tmp_path / 'results')
    assert calls == [1, 2] and status == 1


def test_real_fetch_owner_injects_session_without_loading_another_module(monkeypatch):
    from execution import NativeFetchSession, RealFetchSession

    class Session:
        pass

    observed = []
    monkeypatch.setattr(RealFetchSession, '_prepare',
                        lambda owner, *args: observed.append(owner.session_factory))
    owner = RealFetchSession.open(None, 'cpu', batch=1, history=4096, resident=2048,
                                 server='host:1', session_factory=Session)
    assert observed == [Session]
    assert owner.geometry['output_bytes'] == (2048 + 4 * 2048) * 656
    RealFetchSession.open(None, 'cpu', batch=1, history=4096, resident=2048, server='host:1')
    assert observed[-1] is NativeFetchSession


def test_real_fetch_serializes_each_leased_queue(monkeypatch, tmp_path):
    from execution import RealFetchSession

    owner = RealFetchSession(torch, 'cpu', 1, 4096, 2048)
    owner.lease = dict(worker_count=2, jettys_per_core=2,
                       device_channel_entity=4096, device_queue_state=8192)
    owner.ddr_block_addresses = torch.zeros(32, dtype=torch.int64)
    owner.ddr_valid = torch.ones(32, dtype=torch.int32)
    owner.output = torch.zeros((1, 1, 656), dtype=torch.uint8)
    sizes = dict(args=64, status=32, entity=128, queue=256)
    library = SimpleNamespace(**{
        f'refactor_fetch_{name}_bytes': Mock(return_value=size)
        for name, size in sizes.items()
    })
    packed = []

    def pack(output, size, fields, count):
        assert size == sizes['args'] and count == 17
        packed.append(list(fields))
        ctypes.memset(output, len(packed), size)
        return 0

    library.refactor_fetch_pack_args = Mock(side_effect=pack)
    monkeypatch.setattr(init_transport.ctypes, 'CDLL', lambda _: library)
    synchronize = Mock()
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=synchronize), raising=False)
    args = owner.build_device_args(tmp_path / 'abi.so')
    assert args.shape == (4, 64)
    assert args[:, 0].tolist() == [1, 2, 3, 4]
    assert [fields[8] for fields in packed] == [4096 + i * 128 for i in range(4)]
    assert [fields[9] for fields in packed] == [8192 + i * 256 for i in range(4)]
    assert [fields[7] for fields in packed] == [owner.core_status.data_ptr() + i * 32 for i in range(4)]
    assert all(fields[10:] == [32, 8192, owner.geometry['output_tokens'], 656, 128, 1, 1]
               for fields in packed)
    assert owner.abi_sizes == sizes
    synchronize.assert_called_once()
    owner.in_flight = True
    with pytest.raises(ValueError, match='idle live lease'):
        owner.build_device_args(tmp_path / 'abi.so')
