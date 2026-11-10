"""Exercise the actual UB classifier's scalar semantics on the host.

This is not an AIV race/performance test. Each simulated lane executes in
sequence; independent query bitmaps are intentionally not shared.
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_actual_classifier_boundaries(tmp_path):
    source = tmp_path / 'classifier.cpp'
    exe = tmp_path / 'classifier'
    source.write_text(r'''
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx;
void asc_syncthreads() {}
template<class T>T asc_ldcg(T* p) { return *p; }
void asc_atomic_or(uint32_t* p,uint32_t v) { *p|=v; }
namespace rotating {
constexpr uint32_t kTop=2048,kSlots=8192,kLruSlots=2048;
struct Context { uint64_t mapping,lruSlots; };
uint32_t ResolvePhysicalSlot(uint32_t resident,uint32_t* lru,uint32_t slot) {
 if(lru && uint32_t(slot-resident)<kLruSlots)return lru[slot-resident];
 return slot>=resident+kSlots && slot<0x80000000U ? slot-kSlots : slot;
}
}
#include "staged_topk_classify.h"
int main() {
 using namespace rotating;
 constexpr uint32_t r=16384,h=32768,half=8192;
 std::vector<uint32_t> map(h,UINT32_MAX),lru(2048),idx(kTop,UINT32_MAX),out(kTop,0),valid(r,1),bits(r/32),error(1);
 std::vector<int32_t> reserved(r,-1);
 for(uint32_t i=0;i<2048;++i)lru[i]=i+10;
 Context c{uint64_t(map.data()),uint64_t(lru.data())};
 idx[0]=3;idx[1]=8193;idx[2]=7;idx[3]=100;idx[4]=20000;idx[5]=20001;idx[6]=h;
 reserved[7]=0;valid[100]=0;map[7]=r;map[100]=r+1;map[20000]=r+2;
 for(uint32_t lane=0;lane<512;++lane){threadIdx.x=lane;ClassifyTopkHalf(h,r,0,3,0,half,8193,8194,true,true,true,&c,idx.data(),out.data(),valid.data(),reserved.data(),bits.data(),error.data());}
 for(uint32_t lane=0;lane<512;++lane){threadIdx.x=lane;ClassifyTopkHalf(h,r,0,3,half,half,8193,8194,true,true,true,&c,idx.data(),out.data(),valid.data()+half,reserved.data()+half,bits.data(),error.data());}
 assert(out[0]==3 && out[1]==UINT32_MAX && out[2]==10 && out[3]==11 && out[4]==12 && out[5]==UINT32_MAX && out[6]==UINT32_MAX);
 assert(bits[0]==((1U<<3)|(1U<<10)|(1U<<11)|(1U<<12)) && error[0]==1);
 // Empty prefetch must not accept a stale mapping.
 idx[0]=20000;out[0]=0;error[0]=0;threadIdx.x=0;
 ClassifyTopkHalf(h,r,0,0,0,half,0,0,true,true,false,&c,idx.data(),out.data(),valid.data(),reserved.data(),bits.data(),error.data());
 assert(out[0]==UINT32_MAX);
 // The second request has distinct mapping/LRU storage; offset errors must
 // not be masked by identical data in adjacent batch rows.
 std::vector<uint32_t> map2(2*h,UINT32_MAX),lru2(2*2048),bits2(r/32),out2(kTop,UINT32_MAX);
 map2[h+20000]=r; lru2[2048]=91; lru2[0]=37;
 Context c2{uint64_t(map2.data()),uint64_t(lru2.data())};idx[0]=20000;threadIdx.x=0;
 ClassifyTopkHalf(h,r,1,1,0,half,0,0,true,true,false,&c2,idx.data(),out2.data(),valid.data(),reserved.data(),bits2.data(),error.data());
 assert(out2[0]==91 && (bits2[91/32]&(1U<<(91%32))));
}
''')
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(HERE.parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming/op_kernel'), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
