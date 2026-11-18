# SPDX-License-Identifier: Apache-2.0
"""Exercise the production transfer templates with mixed cycles across independent AIVs."""

import subprocess
from pathlib import Path


def test_four_index_fields_across_ub_and_spill(tmp_path):
    kernel = Path(__file__).resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    source = tmp_path / "transfer.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __simt_vf__
#define __simt_callee__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct { uint32_t x; } threadIdx;
template<class T> T asc_ldcg(T *p) { return *p; }
template<class T> void asc_stcg(T *p, T x) { *p=x; }
void asc_threadfence() {}
void asc_syncthreads() {}
void asc_atomic_or(uint32_t *p, uint32_t x) { *p|=x; }
namespace pipeline_ab {
constexpr uint32_t kPrefetch=2048, kSlots=8192, THREADS=512, MAP=4096, EMPTY=UINT32_MAX,
                   KEYS_POS=0, VALUES_POS=MAP, CONTROL=4*MAP;
uint32_t prefix(uint32_t, uint32_t*);
uint32_t get(uint32_t*,uint32_t*,uint32_t);
}
#include "high_index_workspace.h"
using namespace pipeline_ab::high_post_index;

void check(uint32_t pairs, uint32_t longer, uint32_t cores) {
    const uint32_t n = 2*pairs + longer + 11; // include untouched guard rows
    std::vector<uint32_t> ids(n), sourceAt(n), scale(n);
    std::iota(ids.begin(), ids.end(), 0);
    sourceAt=ids;
    std::mt19937 rng(778);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<uint64_t> keys(n*16), address(n), logical(n);
    for(uint32_t i=0;i<n;++i){
        scale[i]=3*i+11; address[i]=10000000000ULL+16*i; logical[i]=99999999999ULL-i;
        for(uint32_t f=0;f<16;++f) keys[16*i+f]=20000000000ULL+16*i+f;
    }
    std::vector<Edge> edges(kCapacity);
    for(uint32_t i=0;i<pairs;++i){
        auto a=ids[2*i],b=ids[2*i+1];
        edges[i]={a,b}; sourceAt[a]=b; sourceAt[b]=a;
    }
    for(uint32_t i=0;i<longer;++i){
        auto a=ids[2*pairs+i], b=ids[2*pairs+(i+1)%longer];
        edges[kCapacity-1-i]={a,b}; sourceAt[b]=a;
    }
    std::vector<Tuple> spill(longer);
    std::vector<std::vector<Tuple>> ub(cores);
    // All gathers finish before any scatter. Reverse core traversal on scatter
    // so a cycle crossing lanes catches accidental reads of already updated tuples.
    for(int phase=0;phase<2;++phase) for(uint32_t j=0;j<cores;++j){
        uint32_t c=phase?cores-1-j:j;
        uint32_t p0=pairs*c/cores,p1=pairs*(c+1)/cores;
        uint32_t s0=longer*c/cores,s1=longer*(c+1)/cores;
        bool spilled=s1-s0>kLocalTuples;
        if(!phase&&!spilled) ub[c].resize(s1-s0);
        for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x){
            auto run = phase ? (spilled ? Transfer<true,true>:Transfer<true,false>)
                             : (spilled ? Transfer<false,true>:Transfer<false,false>);
            run(edges.data(),p0,p1,s0,s1,keys.data(),scale.data(),address.data(),
                logical.data(),spilled?spill.data():nullptr,spilled?nullptr:ub[c].data());
        }
    }
    for(uint32_t i=0;i<n;++i){uint32_t s=sourceAt[i];
        assert(scale[i]==3*s+11 && address[i]==10000000000ULL+16*s && logical[i]==99999999999ULL-s);
        for(uint32_t f=0;f<16;++f) assert(keys[16*i+f]==20000000000ULL+16*s+f);
    }
}
int main(){check(0,0,4);check(5000,0,64);check(17,219,7);
           check(19,2*kLocalTuples+1,2);check(0,2*kLocalTuples+2,2);}
""")
    binary = tmp_path / "transfer"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(kernel), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_every_cycle_edge_has_one_owner_in_skewed_batches(tmp_path):
    kernel = Path(__file__).resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    source = tmp_path / "partition.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __aicore__
#include "high_index_partition.h"
using namespace pipeline_ab::high_post_index;

int main(){
    std::mt19937 rng(12);
    for(uint32_t attempt=0;attempt<300;++attempt){
        unsigned batch=1+rng()%16, cores=32*(1+rng()%2);
        struct Row {std::vector<int> pairs,singles;};
        std::vector<Row> rows(batch);
        uint32_t total=0;
        for(unsigned b=0;b<batch;++b){
            // Empty/rejected rows, one very heavy row and both odd/even boundaries.
            uint32_t p=(b==0?rng()%5000:rng()%30), s=rng()%43;
            if((b+attempt)%4==0) p=s=0;
            rows[b].pairs.resize(p);rows[b].singles.resize(s);total+=p*2+s;
        }
        for(uint32_t c=0;c<cores;++c){
            uint32_t l=total*c/cores,r=total*(c+1)/cores,base=0,moved=0;
            for(auto& row:rows){
                uint32_t p=row.pairs.size(), count=2*p+row.singles.size();
                Segment seg{};
                if(PartitionRow(l,r,base,count,p,seg)){
                    assert(seg.p0<=seg.p1 && seg.p1<=p &&
                           seg.s0<=seg.s1 && seg.s1<=row.singles.size());
                    for(uint32_t j=seg.p0;j<seg.p1;++j) assert(++row.pairs[j]==1);
                    for(uint32_t j=seg.s0;j<seg.s1;++j) assert(++row.singles[j]==1);
                    moved+=2*(seg.p1-seg.p0)+seg.s1-seg.s0;
                }
                base+=count;
            }
            assert(moved+1>=r-l && moved<=r-l+1);
        }
        for(auto& row:rows){
            assert(std::all_of(row.pairs.begin(),row.pairs.end(),[](int n){return n==1;}));
            assert(std::all_of(row.singles.begin(),row.singles.end(),[](int n){return n==1;}));
        }
    }
}
""")
    binary = tmp_path / "partition"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(kernel), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
