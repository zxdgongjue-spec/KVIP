"""Verify residency publication follows the non-contiguous per-Q post ranges."""
import subprocess
from pathlib import Path


def test_streaming_residency_reads_each_post_range(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_fetch/op_kernel/lifecycle.cpp').read_text()
    start = source.index('inline void PublishStreamingResidency(')
    end = source.index('\n}', start) + 2
    function = source[start:end]
    code = r'''
#include <cstdint>
#include <vector>
#include <cassert>
#define __gm__
struct {unsigned x=0;} threadIdx;
constexpr unsigned kSlots=8192,kTop=2048,kChangeCapacity=10240,kLruSlots=2048;
struct alignas(128) FinalPost {uint32_t count,pending,error,deferred,descriptor,nonconflicting;uint64_t head;};
struct Context {uint64_t hotValid,destinations,changes,lruSlots,owners;};
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_threadfence(){}
unsigned OwnerRow(Context*,unsigned b){return b;}
unsigned OutputStride(unsigned r,Context*){return r;}
''' + function + r'''
int main(){
 std::vector<unsigned> valid(4096),lru(2048);std::vector<int> dst(8192,-1),changes(10240,-1);
 std::vector<FinalPost> posts(4);posts[0].count=2;posts[0].descriptor=0;posts[1].count=1;posts[1].descriptor=2048;
 dst[0]=7;dst[1]=8;dst[2048]=23;lru[0]=42;changes[0]=123;
 Context c{uint64_t(valid.data()),uint64_t(dst.data()),uint64_t(changes.data()),uint64_t(lru.data()),uint64_t(posts.data())};
 for(unsigned tid=0;tid<256;++tid){threadIdx.x=tid;PublishStreamingResidency(4096,0,&c,0);}
 assert(valid[7] && valid[8] && valid[23] && valid[42]);assert(!valid[0] && !valid[22]);
}
'''
    cpp = tmp_path / 'residency.cpp'
    cpp.write_text(code)
    exe = tmp_path / 'residency'
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
