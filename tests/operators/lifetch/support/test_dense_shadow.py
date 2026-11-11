"""Exercise the production dense fallback on large, chained exchanges."""
import subprocess

import pytest
from pathlib import Path


@pytest.mark.parametrize("prepare", [False, True])
def test_dense_shadow_against_swap_reference(tmp_path, prepare):
    root = Path(__file__).resolve().parents[4]
    header = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/shadow_exchange_parallel.h').read_text()
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
  if(i<1500)ApplyBatch(source.data(),dest.data(),i,1,keys.data(),scale.data(),valid.data(),R,m.data());
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
def test_dense_shadow_parallel_batch_and_cross_batch_chain(tmp_path, prepare, use_dense):
    """Run the real vector body with a 32-lane host model of one warp."""
    root = Path(__file__).resolve().parents[4]
    header = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/shadow_exchange_parallel.h').read_text()
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
  if(!phase)launch([&]{ApplyBatch(src.data(),dst.data(),begin,32,keys.data(),scale.data(),valid.data(),R,m.data());});
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
        source = source.replace('ApplyBatch(src.data()', 'ApplyBatch<true>(src.data()').replace('ApplyDenseBatch(src.data()', 'ApplyDenseBatch<true>(src.data()')
        source = source.replace('R,m.data());', 'R,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);')
        source = source.replace('R,dense.data(),H,m.data());', 'R,dense.data(),H,m.data(),addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H));')
        target = ('CommitDenseMetadata(addr.data(),logical.data(),dense.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H);' if use_dense else 'CommitMetadata(addr.data(),logical.data(),m.data());')
        source = source.replace(target, 'CommitPreparedMetadata(addr.data(),logical.data(),reinterpret_cast<uint64_t*>(dense.data()+2*H),H,m.data(),dense.data(),' + str(use_dense).lower() + ');')
        source = source.replace('assert(!m[CONTROL]);', 'assert(!m[CONTROL]);for(unsigned i=0;i<H;i++){assert(addr[i]==i+1000 && logical[i]==i+2000);}')
    src, exe = tmp_path / 'parallel.cpp', tmp_path / 'parallel'
    src.write_text(source)
    subprocess.run(['c++', '-std=c++17', '-O2', '-pthread', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=45)
