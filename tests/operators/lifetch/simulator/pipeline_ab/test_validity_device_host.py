"""Host check of the exact inline hot-slot invalidation used by the kernel."""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
HEADER = REPO / 'csrc/operators/lightning_indexer_with_fetch_streaming/op_kernel/shadow_exchange_parallel.h'


def test_actual_invalidate_freed_hot(tmp_path):
    source, executable = tmp_path / 'validity.cpp', tmp_path / 'validity'
    text = HEADER.read_text()
    pos = text.index('inline void InvalidateFreedHot(')
    start = text.rfind('\n', 0, pos) + 1
    end = text.index('\n', pos)
    helper = text[start:end]
    source.write_text(r'''
#include <cstdint>
#include <cassert>
#define __simt_callee__
#define __gm__
template<class T> void asc_stcg(T* p,T v) {*p=v;}
''' + helper + r'''
int main(){
 uint32_t valid[]={1,1,1,1};
 InvalidateFreedHot(valid,2,4);
 assert(valid[0]==1 && valid[1]==1 && valid[2]==0 && valid[3]==1);
 InvalidateFreedHot(valid,4,4);
 assert(valid[3]==1);
}
''')
    subprocess.run(['c++', '-std=c++17', '-O2', str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_actual_batch_invalidates_freed_hot_source(tmp_path):
    """Single-lane adaptation exercises the actual batch code for one swap."""
    source, executable = tmp_path / 'batch.cpp', tmp_path / 'batch'
    header = HEADER.read_text().replace('THREADS=512', 'THREADS=1')
    begin = header.index('__simt_callee__ inline uint32_t prefix(')
    end = header.index('\n__simt_vf__', begin)
    header = header[:begin] + '__simt_callee__ inline uint32_t prefix(uint32_t, __ubuf__ uint32_t*) { return 0; }' + header[end:]
    header = header.replace('uint32_t extras=ctl[32+15]', 'uint32_t extras=out')
    prefix = r'''
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
'''
    main = r'''
int main(){using namespace pipeline_ab;
 std::vector<uint32_t> m(SCRATCH_WORDS),valid(4,1),scale(16),source={6,2},dest={0,1};
 std::vector<uint64_t> keys(16*16);std::iota(scale.begin(),scale.end(),0);std::iota(keys.begin(),keys.end(),0);
 InitShadow(m.data());
 ApplyBatch(source.data(),dest.data(),0,1,keys.data(),scale.data(),valid.data(),4,m.data());
 assert(valid[0]==1 && valid[2]==1);
 ApplyBatch(source.data(),dest.data(),1,1,keys.data(),scale.data(),valid.data(),4,m.data());
 assert(m[CONTROL]==0 && valid[0]==1 && valid[1]==1 && valid[2]==0 && valid[3]==1);
 assert(get(m.data()+KEYS_POS,m.data()+VALUES_POS,1)==2);
 assert(get(m.data()+KEYS_POS,m.data()+VALUES_POS,2)==1);
}
'''
    source.write_text(prefix + header + main)
    subprocess.run(['c++', '-std=c++17', '-O2', str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=10)
