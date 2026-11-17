# SPDX-License-Identifier: Apache-2.0
"""Execute production post-Index exchanges against a full permutation oracle.

This tests field copies, sharding and malformed-pair rejection on the CPU.
Q4 dedup and unique victim allocation are exercised by test_compact_q4; Ascend
cross-kernel/cache ordering and Get/CQ completion still require the NPU replay.
"""

import subprocess

from test_recent_index import INDEXER


def test_parallel_post_index_preserves_complete_rows_and_self_refills(tmp_path):
    kernel = INDEXER.parent
    header = (kernel / "post_index_update.h").read_text().replace(
        "constexpr uint32_t kThreads = 512;", "constexpr uint32_t kThreads = 1;"
    )
    source = tmp_path / "post_index_update.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(n)
struct {uint32_t x=0;} threadIdx;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T value){*p=value;}
void asc_atomic_or(uint32_t* p,uint32_t value){*p|=value;}
''' + header + r'''
int main(){using namespace post_index;
 constexpr uint32_t R=16384,H=32768;
 std::mt19937 random(638);
 for(uint32_t generation=0;generation<5;++generation){
  std::vector<uint64_t> keys(H*16),addresses(H),logical(H),kv(R);
  std::vector<uint32_t> scales(H),hotValid(R,1),order(R);
  for(uint32_t i=0;i<H;++i){
   logical[i]=uint64_t(generation)*H+i;addresses[i]=logical[i]*656+128;
   scales[i]=random();for(uint32_t j=0;j<16;++j)keys[i*16+j]=logical[i]*17+j;
   if(i<R)kv[i]=logical[i];
  }
  std::iota(order.begin(),order.end(),0);
  // Every round promotes real misses; subsequent low rounds and a general
  // high-path swap consume the state left by the preceding round.
  for(uint32_t round=0;round<8;++round){
   std::shuffle(order.begin(),order.end(),random);
   uint32_t n=round==0?0:round==1?8192:round==2?8192:round==3?1:1217;
   std::vector<uint32_t> src(n),dst(n),permutation(H);
   std::iota(permutation.begin(),permutation.end(),0);
   for(uint32_t i=0;i<n;++i){
    dst[i]=order[i];src[i]=round==2||i%7==0?dst[i]:R+i;
    if(src[i]==dst[i]){hotValid[dst[i]]=0;kv[dst[i]]=UINT64_MAX;}
    std::swap(permutation[src[i]],permutation[dst[i]]);
   }
   assert(std::unordered_set<uint32_t>(src.begin(),src.end()).size()==n);
   assert(std::unordered_set<uint32_t>(dst.begin(),dst.end()).size()==n);
   auto oldKeys=keys,oldAddresses=addresses,oldLogical=logical,oldKv=kv;
   auto oldScales=scales,oldValid=hotValid;
   std::vector<uint64_t> postedAddresses(n),tndLogical(n);
   for(uint32_t i=0;i<n;++i){postedAddresses[i]=addresses[src[i]];tndLogical[i]=logical[src[i]];}
   for(uint32_t lanes:{1U,2U,4U,64U}){
    keys=oldKeys;scales=oldScales;addresses=oldAddresses;logical=oldLogical;
    std::vector<uint32_t> work(16384);
    for(uint32_t lane=0;lane<lanes;++lane){
     ValidatePlan(src.data(),dst.data(),0,n,R,H,lane,lanes,work.data());assert(!work[0]);
    }
    // Reverse shard order proves no shard depends on another shard's writes.
    for(uint32_t lane=lanes;lane--;)
     ApplyPlan(src.data(),dst.data(),0,n,R,H,keys.data(),scales.data(),addresses.data(),logical.data(),lane,lanes,work.data());
    assert(hotValid==oldValid&&kv==oldKv); // Index cannot declare CQ success.
    for(uint32_t i=0;i<H;++i){
     assert(scales[i]==oldScales[permutation[i]]);
     assert(addresses[i]==oldAddresses[permutation[i]]);
     assert(logical[i]==oldLogical[permutation[i]]);
     for(uint32_t j=0;j<16;++j)assert(keys[i*16+j]==oldKeys[permutation[i]*16+j]);
    }
    for(uint32_t i=0;i<n;++i){
     assert(postedAddresses[i]==oldAddresses[src[i]]);
     assert(logical[dst[i]]==tndLogical[i]);
    }
   }
   // Successful CQ drain is a separate later phase, including self-refills.
   for(uint32_t i=0;i<n;++i){kv[dst[i]]=tndLogical[i];hotValid[dst[i]]=1;}
   for(uint32_t i=0;i<n;++i)assert(kv[dst[i]]==logical[dst[i]]);
   if(round==4){
    // A high-route permutation can leave arbitrary physical Index ordering;
    // the next low invocation uses those current rows, not stale logical IDs.
    uint32_t a=117,b=R+101;
    for(uint32_t j=0;j<16;++j)std::swap(keys[a*16+j],keys[b*16+j]);
    std::swap(scales[a],scales[b]);std::swap(addresses[a],addresses[b]);
    std::swap(logical[a],logical[b]);kv[a]=logical[a];
   }
  }
 }
 // Unsupported hot-to-hot dependencies must be rejected before all posting.
 uint32_t work[4],src,dst;
 for(auto pair:std::vector<std::pair<uint32_t,uint32_t>>{{0,1},{H,0},{R,R}}){
  src=pair.first;dst=pair.second;ValidatePlan(&src,&dst,0,1,R,H,0,1,work);assert(work[0]);
 }
 src=3;dst=3;ValidatePlan(&src,&dst,0,1,R,H,0,1,work);assert(!work[0]);
 ValidatePlan(&src,&dst,0,8193,R,H,0,1,work);assert(work[0]);
 ValidatePlan(&src,&dst,0,1,R,H,0,0,work);assert(work[0]);
 ValidatePlan(&src,&dst,0,1,R,H,1,1,work);assert(work[0]);
}
''')
    binary = tmp_path / "post_index_update"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)
