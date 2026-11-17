# SPDX-License-Identifier: Apache-2.0
"""Execute production post-Index exchanges against a full permutation oracle.

This tests field copies, sharding and malformed-pair rejection on the CPU.
Q4 dedup and unique victim allocation are exercised by test_compact_q4; Ascend
cross-kernel/cache ordering and Get/CQ completion still require the NPU replay.
"""

import subprocess

import pytest

from test_recent_index import INDEXER


@pytest.mark.parametrize("threads", [16, 512])
def test_parallel_post_index_preserves_complete_rows_and_self_refills(tmp_path, threads):
    kernel = INDEXER.parent
    header = (kernel / "post_index_update.h").read_text().replace(
        "constexpr uint32_t kThreads = 512;", f"constexpr uint32_t kThreads = {threads};"
    )
    source = tmp_path / "post_index_update.cpp"
    source.write_text(r'''
#include <algorithm>
#include <array>
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
struct LoadRange {uintptr_t begin,end;};
std::vector<LoadRange> leaderOnly;
std::vector<uint64_t> metadataReads;
template<class T>void Watch(T* p,size_t n){
 leaderOnly.push_back({reinterpret_cast<uintptr_t>(p),reinterpret_cast<uintptr_t>(p+n)});
 metadataReads.push_back(0);
}
template<class T>T asc_ldcg(T* p){
 uintptr_t address=reinterpret_cast<uintptr_t>(p);
 for(size_t i=0;i<leaderOnly.size();++i)if(address>=leaderOnly[i].begin&&address<leaderOnly[i].end){
  assert(threadIdx.x%16==0);++metadataReads[i];
 }
 return *p;
}
// Execute each group's leader first and record its shuffle values. Followers
// consume the same call sequence, including self/error skips and later events.
std::array<std::vector<uint32_t>,32> broadcasts;
size_t shufflePosition=0;
void BeginLane(){
 shufflePosition=0;
 if(threadIdx.x%16==0)broadcasts[threadIdx.x/16].clear();
}
uint32_t asc_shfl(uint32_t value,int source,int width){
 assert(source==0&&width==16);
 auto& history=broadcasts[threadIdx.x/16];
 if(threadIdx.x%16==0)history.push_back(value);
 assert(shufflePosition<history.size());
 return history[shufflePosition++];
}
template<class T>void asc_stcg(T* p,T value){*p=value;}
void asc_atomic_or(uint32_t* p,uint32_t value){*p|=value;}
void asc_atomic_add(uint32_t* p,uint32_t value){*p+=value;}
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
     for(threadIdx.x=0;threadIdx.x<kThreads;++threadIdx.x)
      ValidatePlan(src.data(),dst.data(),0,n,R,H,lane,lanes,work.data());
     assert(!work[0]);
    }
    // Reverse shard order proves no shard depends on another shard's writes.
    uint32_t prefix[]={0,n};
    std::vector<uint64_t> records(lanes*16);
    for(uint32_t lane=lanes;lane--;)for(threadIdx.x=0;threadIdx.x<kThreads;++threadIdx.x){
     BeginLane();
     ApplyPlan<false>(src.data(),dst.data(),0,n,H,prefix,1,n*lane/lanes,n*(lane+1)/lanes,
       nullptr,records.data(),lanes,keys.data(),scales.data(),addresses.data(),logical.data(),H,nullptr);
    }
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
 threadIdx.x=0;
 uint32_t work[4],src,dst;
 for(auto pair:std::vector<std::pair<uint32_t,uint32_t>>{{0,1},{H,0},{R,R}}){
  src=pair.first;dst=pair.second;ValidatePlan(&src,&dst,0,1,R,H,0,1,work);assert(work[0]);
 }
 src=3;dst=3;ValidatePlan(&src,&dst,0,1,R,H,0,1,work);assert(!work[0]);
 ValidatePlan(&src,&dst,0,8193,R,H,0,1,work);assert(work[0]);
 ValidatePlan(&src,&dst,0,1,R,H,0,0,work);assert(work[0]);
 ValidatePlan(&src,&dst,0,1,R,H,1,1,work);assert(work[0]);
 // Uneven requests and empty rows must spread across all AIVs. Physical
 // owners are deliberately permuted, and the DDR address stride differs.
 constexpr uint32_t BH=512,BR=128,AS=544,STRIDE=144,OFFSET=5;
 for(uint32_t batch:{1U,3U,16U})for(uint32_t cores:{16U,32U,64U}){
  std::vector<uint32_t> prefix(batch+1),owners(batch),s(batch*STRIDE),d(batch*STRIDE);
  std::vector<uint64_t> records(cores*16),keys(batch*BH*16),addr(batch*AS),ids(batch*BH);
  std::vector<uint32_t> scales(batch*BH);
  for(uint32_t row=0;row<batch;++row){
   owners[row]=batch-1-row;
   uint32_t n=batch==1?111:(row%4==0?0:row%4==1?111:row%4==2?1:17);
   prefix[row+1]=prefix[row]+n;
   for(uint32_t i=0;i<n;++i){d[row*STRIDE+OFFSET+i]=i;s[row*STRIDE+OFFSET+i]=(row%3==0||i%7==0)?i:BR+i;}
  }
  for(uint32_t row=0;row<batch;++row)for(uint32_t i=0;i<BH;++i){
   ids[row*BH+i]=row*BH+i;scales[row*BH+i]=(row*BH+i)*13;
   for(uint32_t f=0;f<16;++f)keys[(row*BH+i)*16+f]=(row*BH+i)*17+f;
  }
  for(uint32_t i=0;i<addr.size();++i)addr[i]=0x100000+uint64_t(i)*656;
  const auto initialKeys=keys,initialAddr=addr,initialIds=ids;const auto initialScales=scales;
  for(bool failed:{false,true}){
   keys=initialKeys;addr=initialAddr;ids=initialIds;scales=initialScales;
   std::fill(records.begin(),records.end(),0);
   const uint32_t badRow=batch==1?0:1;
   if(failed)records[(badRow+batch<cores?badRow+batch:badRow)*16]=4;
   auto expectedKeys=keys,expectedAddr=addr,expectedIds=ids;auto expectedScales=scales;
   uint32_t expectedMoved=0;
   for(uint32_t row=0;row<batch;++row){
    if(failed&&row==badRow)continue;
    for(uint32_t i=0;i<prefix[row+1]-prefix[row];++i){
     uint32_t a=s[row*STRIDE+OFFSET+i],b=d[row*STRIDE+OFFSET+i];
     if(a!=b)++expectedMoved;
     for(uint32_t f=0;f<16;++f)std::swap(expectedKeys[(owners[row]*BH+a)*16+f],expectedKeys[(owners[row]*BH+b)*16+f]);
     std::swap(expectedAddr[owners[row]*AS+a],expectedAddr[owners[row]*AS+b]);
     std::swap(expectedIds[owners[row]*BH+a],expectedIds[owners[row]*BH+b]);
     std::swap(expectedScales[owners[row]*BH+a],expectedScales[owners[row]*BH+b]);
    }
   }
   for(bool profiled:{false,true}){
    keys=initialKeys;addr=initialAddr;ids=initialIds;scales=initialScales;
    uint32_t sumMoved=0,minAssigned=UINT32_MAX,maxAssigned=0;
    leaderOnly.clear();metadataReads.clear();
    Watch(s.data(),s.size());Watch(d.data(),d.size());Watch(prefix.data(),prefix.size());
    Watch(owners.data(),owners.size());Watch(records.data(),records.size());
    for(uint32_t core=cores;core--;){
     uint32_t first=prefix.back()*core/cores,end=prefix.back()*(core+1)/cores,counter=0;
     minAssigned=std::min(minAssigned,end-first);maxAssigned=std::max(maxAssigned,end-first);
     // Execute either one or the real 32 independent 16-thread groups.
     // Only each group's lane 0 changes scalar metadata and adds its count;
     // thread 0 initializes the diagnostic counter before other groups run.
     for(threadIdx.x=0;threadIdx.x<kThreads;++threadIdx.x){
      BeginLane();
      if(profiled)ApplyPlan<true>(s.data(),d.data(),OFFSET,STRIDE,BH,prefix.data(),batch,first,end,
        owners.data(),records.data(),cores,keys.data(),scales.data(),addr.data(),ids.data(),AS,&counter);
      else ApplyPlan<false>(s.data(),d.data(),OFFSET,STRIDE,BH,prefix.data(),batch,first,end,
        owners.data(),records.data(),cores,keys.data(),scales.data(),addr.data(),ids.data(),AS,nullptr);
     }
     assert(counter<=end-first);sumMoved+=counter;
    }
    uint32_t validEvents=prefix.back()-(failed?prefix[badRow+1]-prefix[badRow]:0);
    assert(metadataReads[0]==validEvents&&metadataReads[1]==validEvents);
    leaderOnly.clear();metadataReads.clear();
    assert(maxAssigned-minAssigned<=1);
    assert(!profiled||sumMoved==expectedMoved);
    assert(keys==expectedKeys&&addr==expectedAddr&&ids==expectedIds&&scales==expectedScales);
   }
  }
 }
}
''')
    binary = tmp_path / "post_index_update"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)
