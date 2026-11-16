# SPDX-License-Identifier: Apache-2.0
"""Execute direct Index exchanges against an independent permutation oracle.

The CPU executes the production field-copy and proof predicates. The resolver
model deliberately generates cross-Q duplicates and late hot hits; it checks
the eligibility assumptions, not Ascend memory-ordering performance.
"""

import subprocess

from test_recent_index import INDEXER, function


def test_native_direct_index_and_all_query_proof(tmp_path):
    kernel = INDEXER.parent
    predicate = function(
        (kernel / "resolver_staged_dma2.h").read_text(),
        "inline bool StagedDirectIndexSource(",
    )
    program = tmp_path / "direct-index.cpp"
    program.write_text(r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
constexpr uint32_t THREADS=1,SET=2048,SRCSET=0,DSTSET=SET,CONTROL=2*SET;
struct {uint32_t x=0;} threadIdx;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
void asc_atomic_or(uint32_t* p,uint32_t value){*p|=value;}
#include "disjoint_index.h"
""" + predicate + r"""
int main(){
 using namespace pipeline_ab;
 constexpr uint32_t resident=16384,history=32768,top=2048,queries=4;
 std::mt19937 random(317);
 for(uint32_t round=0;round<30;++round){
  std::vector<uint32_t> selected(top*queries,UINT32_MAX),locations(top*queries,UINT32_MAX);
  std::vector<uint32_t> sources,destinations,counts(queries),mapping(history,UINT32_MAX);
  std::vector<uint64_t> keys(history*16),addresses(history),logical(history);
  std::vector<uint32_t> scales(history),permutation(history),hotValid(resident,1);
  std::vector<uint64_t> kv(resident);
  for(uint32_t i=0;i<history;++i){
   permutation[i]=i;scales[i]=random();addresses[i]=(uint64_t(random())<<32)|random();
   logical[i]=131073ULL*i+round;
   for(uint32_t j=0;j<16;++j)keys[i*16+j]=0x800000000000ULL+17ULL*i+j;
   if(i<resident){mapping[i]=i;kv[i]=logical[i];}
  }
  // Include empty, one miss, Q4-only misses, maximal distinct misses, and
  // large repeated lists that must consume only one destination per source.
  for(uint32_t q=0;q<queries;++q)for(uint32_t i=0;i<top;++i){
   uint32_t token=round%5==0?i:round%5==1?(i?i:resident+1):
       round%5==2?(q==3?resident+i:i):round%5==3?resident+q*top+i:
       (i%5==0?resident+random()%700:random()%resident);
   selected[q*top+i]=token;locations[q*top+i]=mapping[token];
  }
  // A current ABC hit is neither a missing cold row nor a new hot destination.
  if(round%3==0){selected[0]=history-1;locations[0]=resident+128;}
  bool eligible=true;
  for(uint32_t i=0;i<selected.size();++i)
   eligible &= StagedDirectIndexSource(selected[i],locations[i],resident,history);
  assert(eligible);
  // Q4 invalid-hot sources defeat the proof even when Q1..Q3 were all safe.
  auto q4=3*top+77,oldToken=selected[q4],oldLocation=locations[q4];
  selected[q4]=17;locations[q4]=UINT32_MAX;
  bool allQProof=true;
  for(uint32_t i=0;i<selected.size();++i)
   allQProof &= StagedDirectIndexSource(selected[i],locations[i],resident,history);
  assert(!allQProof);
  selected[q4]=oldToken;locations[q4]=oldLocation;

  std::unordered_set<uint32_t> protectedHot;
  for(auto slot:locations)if(slot<resident)protectedHot.insert(slot);
  uint32_t next=0;
  std::unordered_map<uint32_t,uint32_t> resolved;
  for(uint32_t q=0;q<queries;++q){
   auto before=sources.size();
   for(uint32_t i=q*top;i<(q+1)*top;++i){
    if(locations[i]!=UINT32_MAX)continue;
    auto token=selected[i];
    if(!resolved.count(token)){
     while(protectedHot.count(next))++next;
     assert(next<resident);resolved[token]=next;sources.push_back(token);destinations.push_back(next);
     protectedHot.insert(next++);
    }
    locations[i]=resolved.at(token);
   }
   counts[q]=uint32_t(sources.size()-before);
  }
  assert(std::unordered_set<uint32_t>(sources.begin(),sources.end()).size()==sources.size());
  assert(std::unordered_set<uint32_t>(destinations.begin(),destinations.end()).size()==destinations.size());
  auto oldKeys=keys,oldAddresses=addresses,oldLogical=logical;auto oldScales=scales;
  uint32_t begin=0;std::vector<uint32_t> work(CONTROL+1,99);
  for(uint32_t q=0;q<queries;++q){
   ApplyDisjointIndex(sources.data(),destinations.data(),begin,counts[q],resident,history,
    keys.data(),scales.data(),work.data());assert(!work[CONTROL]);
   // Descriptor publication must continue to see immutable original tables.
   assert(addresses==oldAddresses&&logical==oldLogical);
   for(uint32_t i=begin;i<begin+counts[q];++i){
    std::swap(permutation[sources[i]],permutation[destinations[i]]);
    kv[destinations[i]]=oldLogical[sources[i]];
   }
   begin+=counts[q];
  }
  CommitDisjointMetadata(sources.data(),destinations.data(),0,begin,addresses.data(),logical.data());
  for(uint32_t i=0;i<history;++i){
   assert(scales[i]==oldScales[permutation[i]]);
   assert(addresses[i]==oldAddresses[permutation[i]]&&logical[i]==oldLogical[permutation[i]]);
   for(uint32_t j=0;j<16;++j)assert(keys[i*16+j]==oldKeys[permutation[i]*16+j]);
  }
  for(uint32_t i=0;i<selected.size();++i)if(locations[i]<resident)
   assert(kv[locations[i]]==oldLogical[selected[i]]);
  // Neither empty phases nor direct Index swaps publish KV residency.
  assert(std::all_of(hotValid.begin(),hotValid.end(),[](auto v){return v==1;}));
 }
 assert(!StagedDirectIndexSource(history,UINT32_MAX,resident,history));
 assert(StagedDirectIndexSource(UINT32_MAX,UINT32_MAX,resident,history));
 // Malformed ranges fail before any key/scale mutation.
 for(auto pair:{std::array<uint32_t,2>{0,1},{resident, resident},{history,1}}){
  std::vector<uint64_t> keys(history*16,91);std::vector<uint32_t> scales(history,73);
  std::vector<uint32_t> work(CONTROL+1);
  ApplyDisjointIndex(&pair[0],&pair[1],0,1,resident,history,keys.data(),scales.data(),work.data());
  assert(work[CONTROL]&&std::all_of(keys.begin(),keys.end(),[](auto v){return v==91;})&&
         std::all_of(scales.begin(),scales.end(),[](auto v){return v==73;}));
 }
}
""")
    binary = tmp_path / "direct-index"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        "-I", str(kernel), str(program), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_direct_consumer_keeps_join_and_error_contract():
    source = (INDEXER.parent / "streaming.cpp").read_text()
    producer = function(source, "inline uint32_t Producer(")
    consumer = function(source, "inline void Consumer(")
    assert producer.index("VF_CALL<StagedInitializeReady<true>>") < producer.index("reserved[0]=")
    assert "!row->slots && !ctl.GetValue(9)" in producer
    assert "VF_CALL<StagedInitializeReady<false>>" in producer
    assert consumer.index("flags[kRawQueryFlag].error") < consumer.index("reserved[0]==1")
    assert consumer.index("flags+kOutputCompletionFlag") < consumer.index("VF_CALL<CommitDisjointMetadata>")
    assert consumer.index("if(flags[publication].error)") < consumer.index("VF_CALL<ApplyDisjointIndex>")
    assert "if(!directIndex){" in consumer
    assert "if(profile)ticks[3]+=count;" in consumer
