# SPDX-License-Identifier: Apache-2.0
"""Compare the production cold-chain edge encoder with ordered in-place swaps."""

import subprocess
from pathlib import Path


def test_repeated_victims_and_mixed_cycles(tmp_path):
    kernel = Path(__file__).resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    source = tmp_path / "cold_chain.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#define __gm__
#define __ubuf__
#define __simt_callee__
constexpr uint32_t kCapacity = 20480;
struct Edge {uint32_t src, dst;};
template<class T> void asc_stcg(T* dst,T value){*dst=value;}
#include "high_index_cold_chain.h"

void check(std::mt19937& rng, uint32_t phases, uint32_t hot, uint32_t maxPerPhase) {
    const uint32_t history = hot + phases * maxPerPhase;
    std::vector<uint32_t> oracle(history), cold(history-hot), first(hot), last(hot), touched(hot);
    std::vector<bool> seen(hot), written(history);
    std::vector<Edge> edges(kCapacity);
    std::iota(oracle.begin(), oracle.end(), 0);
    std::iota(cold.begin(), cold.end(), hot);
    std::shuffle(cold.begin(),cold.end(),rng);
    uint32_t nextCold=0, unique=0, repeats=0;
    for(uint32_t phase=0;phase<phases;++phase){
        std::vector<uint32_t> slots(hot);
        std::iota(slots.begin(),slots.end(),0);
        std::shuffle(slots.begin(),slots.end(),rng);
        uint32_t count = rng()%(std::min(hot,maxPerPhase)+1);
        for(uint32_t i=0;i<count;++i){
            uint32_t dst=slots[i], src=cold[nextCold++];
            std::swap(oracle[src],oracle[dst]);
            bool fresh=!seen[dst];
            VisitColdSlot(dst,src,fresh,first.data(),last.data(),touched.data(),
                          edges.data(),unique,repeats);
            fresh ? ++unique : ++repeats;
            seen[dst]=true;
        }
    }
    uint32_t pairs=0,singles=repeats;
    for(uint32_t j=0;j<unique;++j){
        uint32_t dst=touched[j];
        CloseColdSlot(dst,first[dst],last[dst],edges.data(),pairs,singles);
        if(first[dst]==last[dst]) ++pairs;
        else singles+=2;
    }
    assert(2*pairs+singles==nextCold+unique);
    std::vector<uint32_t> actual(history);
    std::iota(actual.begin(),actual.end(),0);
    auto assign=[&](uint32_t dst,uint32_t src){
        assert(dst<history && src<history && !written[dst]); written[dst]=true; actual[dst]=src;
    };
    for(uint32_t j=0;j<pairs;++j){
        const auto& e=edges[j]; assign(e.dst,e.src);assign(e.src,e.dst);
    }
    for(uint32_t j=0;j<singles;++j){
        const auto& e=edges[kCapacity-1-j];assign(e.dst,e.src);
    }
    assert(actual==oracle);
}
int main(){
    std::mt19937 rng(42);
    check(rng,0,16,16);
    for(unsigned trial=0;trial<2000;++trial)check(rng,1+rng()%10,1+rng()%127,1+rng()%96);
    for(unsigned trial=0;trial<32;++trial)check(rng,16,1024,512);
}
""")
    binary = tmp_path / "cold_chain"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(kernel), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
