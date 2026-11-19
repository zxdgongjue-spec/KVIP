# SPDX-License-Identifier: Apache-2.0
"""Execute the original descriptor packing for both graph routes.

One virtual lane runs the complete partition. This checks descriptor order,
validation and publication counters, not SIMT concurrency or cache visibility.
"""

import subprocess

import pytest

from test_recent_index import INDEXER, function


@pytest.mark.parametrize("compact", [0, 1])
def test_original_packing_matches_reference_in_both_routes(tmp_path, compact):
    native = function(
        (INDEXER.parent / "staged_urma_pack.h").read_text(),
        "inline void PackStagedUrmaQuery(",
    )
    # Keep all data/control operations; only collapse the SIMT partition width.
    native = native.replace("511", "0").replace("512", "1")
    program = tmp_path / "pack.cpp"
    program.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>
#define __ubuf__
#define __gm__
constexpr uint32_t kTop=2048,kSlots=8192,warpSize=1;
struct {uint32_t x=0;} threadIdx;
uint32_t prefixCalls=0,reservedLoads=0;
uintptr_t reservedBegin=0,reservedEnd=0;
void asc_syncthreads(){}
void asc_threadfence(){}
template<class T> T asc_ldcg(T* p){
 const auto address=uintptr_t(p);
 if(address>=reservedBegin&&address<reservedEnd)++reservedLoads;
 return *p;
}
template<class T> void asc_stcg(T* p,T v){*p=v;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
uint32_t asc_atomic_add(uint32_t* p,uint32_t v){auto old=*p;*p+=v;return old;}
uint32_t StagedPrefix(uint32_t v,uint32_t* ctl){++prefixCalls;ctl[544]=v;return 0;}
struct Row {uint64_t addresses;};
struct Context {
 uint64_t rows,reservedHot,sourceBlocks,sourceOffsets,destinations,count,fallbackAddresses;
};
using StagedContextPointer=Context*;
uint32_t OwnerRow(Context*,uint32_t b){return 4-b;}
uint32_t OutputStride(uint32_t resident,Context*){return resident+512;}
void StoreFallbackSource(uint32_t history,uint32_t bytes,Context* ctx,uint32_t batch,
 uint32_t output,uint32_t token,uint64_t address,int32_t* blocks,int32_t* offsets){
 if(ctx->fallbackAddresses){
  reinterpret_cast<uint64_t*>(ctx->fallbackAddresses)[output]=address;
  blocks[output]=output;offsets[output]=0;
 }else{blocks[output]=batch*(history/128)+token/128;offsets[output]=(token%128)*bytes;}
}
""" + native + r"""
template<class... Args>void dispatch(bool low,uint32_t used,Args... args){
 PackStagedUrmaQuery(args...);
}
int main(){
 std::mt19937 rng(1704);
 constexpr uint32_t history=131072,resident=16384,batch=2,query=3,capacity=4*kSlots;
 std::vector<uint32_t> selected(kTop),slots(kSlots),ctl(1024);
 std::vector<uint16_t> ids(kTop);
 std::vector<int32_t> reserved(4*resident),blocks(capacity),offsets(capacity),destinations(capacity);
 std::vector<uint64_t> addresses(history),fallback(capacity);
 std::vector<Row> rows(4);rows[batch].addresses=uint64_t(addresses.data());
 uint32_t submitted=0;
 Context ctx{uint64_t(rows.data()),uint64_t(reserved.data()),uint64_t(blocks.data()),
  uint64_t(offsets.data()),uint64_t(destinations.data()),uint64_t(&submitted),0};
 reservedBegin=uintptr_t(reserved.data());reservedEnd=reservedBegin+reserved.size()*4;
 for(uint32_t i=0;i<history;++i)addresses[i]=0x123400000000ULL+uint64_t(i)*656;
 for(uint32_t i=0;i<kTop;++i){ids[i]=(i*31)%kTop;selected[i]=(i*59)%history;}
 for(auto& slot:slots)slot=rng()%resident;
 for(auto& value:reserved)value=rng()%3?-1:int32_t(rng()%2048);
 for(bool low:{false,true})for(uint32_t used:{0U,1U,31U,2048U})for(uint32_t count:{0U,1U,31U,511U,512U,513U,2048U})
 for(uint32_t begin:{0U,1793U,4096U})for(bool useFallback:{false,true})for(bool useDirect:{false,true}){
  constexpr uint32_t stride=8192,start=19;
  std::vector<int32_t> direct(6*stride+64,0);
  direct[6*stride]=start;direct[6*stride+8]=start+7;
  std::fill(ctl.begin(),ctl.end(),0);std::fill(blocks.begin(),blocks.end(),-7);
  std::fill(offsets.begin(),offsets.end(),-7);std::fill(destinations.begin(),destinations.end(),-7);
  std::fill(fallback.begin(),fallback.end(),0);submitted=13;prefixCalls=reservedLoads=0;
  ctx.fallbackAddresses=useFallback?uint64_t(fallback.data()):0;
  std::vector<uint32_t> order;uint32_t safe=0;
  for(uint32_t phase=0;phase<2;++phase)for(uint32_t i=0;i<count;++i){
   int32_t ordinal=reserved[batch*resident+slots[begin+i]];
   bool conflict=ordinal>=0&&uint32_t(ordinal)<used;
   if(uint32_t(conflict)==phase){order.push_back(i);if(!phase)++safe;}
  }
  dispatch(low,used,history,resident,656,batch,query,used,begin,count,&ctx,
   selected.data(),ids.data(),slots.data(),ctl.data(),useDirect?direct.data():nullptr,stride);
  assert(!ctl[1]&&ctl[16]==count&&ctl[17]==safe&&submitted==13+count);
  if(!count){assert(!prefixCalls&&!reservedLoads&&!ctl[18]&&!ctl[19]);continue;}
  assert(prefixCalls==1&&reservedLoads==2*count);
  for(uint32_t position=0;position<count;++position){
   uint32_t rank=begin+order[position],token=selected[ids[rank%kTop]],slot=slots[rank];
   uint32_t output=batch*kSlots+query*kTop+position;
   int32_t block=useFallback?int32_t(output):int32_t(batch*(history/128)+token/128);
   int32_t offset=useFallback?0:int32_t((token%128)*656);
   int32_t destination=OwnerRow(&ctx,batch)*(resident+512)+slot;
   assert(blocks[output]==block&&offsets[output]==offset&&destinations[output]==destination);
   if(useFallback)assert(fallback[output]==addresses[token]);
   if(useDirect){
    bool conflict=position>=safe;uint32_t pos=conflict?start+7+position-safe:start+position;
    uint32_t base=conflict?3*stride:0;
    assert(direct[base+pos]==block&&direct[base+stride+pos]==offset&&direct[base+2*stride+pos]==destination);
   }
  }
  if(useDirect){assert(direct[6*stride]==start+safe);assert(direct[6*stride+8]==start+7+count-safe);}
  for(uint32_t i=0;i<capacity;++i)if(i<batch*kSlots+query*kTop||i>=batch*kSlots+query*kTop+count)
   assert(blocks[i]==-7&&offsets[i]==-7&&destinations[i]==-7);
 }
 // Invalid slots/tokens and an overflowing direct reservation still fail.
 for(uint32_t used:{0U,7U})for(uint32_t fault=0;fault<3;++fault){
  std::fill(ctl.begin(),ctl.end(),0);submitted=0;
  std::vector<int32_t> direct(6*4096+64,0);
  const auto savedSlot=slots[0];const auto savedToken=selected[ids[0]];
  if(fault==0)slots[0]=resident;
  if(fault==1)selected[ids[0]]=history;
  if(fault==2)direct[6*4096]=4096;
  dispatch(true,used,history,resident,656,batch,query,used,0,32,&ctx,
   selected.data(),ids.data(),slots.data(),ctl.data(),direct.data(),4096);
  assert(ctl[1]&(fault==0?4U:fault==1?1U:16U));assert(!submitted);
  slots[0]=savedSlot;selected[ids[0]]=savedToken;
 }
}
""")
    binary = tmp_path / "pack"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
        f"-DKVIP_LOW_MISS_TAIL={compact}", str(program), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_register_publication_matches_original(tmp_path):
    """Drive every lane of one Q. The independent prefix oracle supplies the collective."""
    kernel = INDEXER.parent
    old = function((kernel / "staged_urma_pack.h").read_text(), "inline void PackStagedUrmaQuery(")
    fast = function((kernel / "output_worker.h").read_text(), "inline void PackSmallPublicationEvents(")
    source = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>
#define __gm__
#define __ubuf__
constexpr unsigned kSlots=8192,kTop=2048,warpSize=32;
struct {unsigned x;} threadIdx;
struct Row {uint64_t addresses;};
struct Context {uint64_t rows,reservedHot,sourceBlocks,sourceOffsets,destinations,count,fallbackAddresses;};
using StagedContextPointer=Context*;
uint32_t OwnerRow(Context*,unsigned b){return 3-b;}
uint32_t OutputStride(unsigned r,Context*){return r+64;}
void asc_syncthreads(){} void asc_threadfence(){}
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
uint32_t asc_atomic_add(uint32_t* p,uint32_t v){auto old=*p;*p+=v;return old;}
unsigned before[513], safe[512];
uint32_t StagedPrefix(unsigned local,unsigned* ctl){
 assert(local==safe[threadIdx.x]); ctl[544+512/warpSize-1]=before[512];return before[threadIdx.x];
}
template<class... A>void StoreFallbackSource(unsigned history,unsigned bytes,Context* c,unsigned b,unsigned pos,
 unsigned token,uint64_t addr,int32_t* blocks,int32_t* offsets){
 if(c->fallbackAddresses){((uint64_t*)c->fallbackAddresses)[pos]=addr;blocks[pos]=pos;offsets[pos]=0;}
 else{blocks[pos]=b*(history/128)+token/128;offsets[pos]=(token%128)*bytes;}
}
""" + old + fast + r"""
int main(){
 constexpr unsigned b=1,q=2,base=b*kSlots+q*kTop,res=1024,h=4096,raw=7,used=16;
 for(unsigned n:{0,1,31,32,33,97,258,511,512,513})
 for(unsigned kind:{0,1,2,3,4})for(bool fallback:{false,true}){
  std::vector<uint32_t> token(n+raw),slot(n+raw);
  std::vector<uint16_t> ids(n);std::iota(ids.begin(),ids.end(),0U);
  std::vector<int> ordinal((b+1)*res,-1);std::vector<uint64_t> addresses(h);
  for(unsigned i=0;i<n;i++){token[raw+i]=(i*31)%h;slot[raw+i]=(i*3)%res;
   ordinal[b*res+slot[raw+i]]=i%3==0?1:i%3==1?-1:18;}
  if(n && (kind&1))slot[raw+n/2]=res;
  if(n && (kind&2))token[raw+n-1]=h;
  std::iota(addresses.begin(),addresses.end(),10000);Row rows[b+1]{};rows[b].addresses=uint64_t(addresses.data());
  unsigned totals[2]={19,19};std::vector<unsigned> controls[2];
  std::vector<int32_t> indices[2],offsets[2],dest[2];std::vector<uint64_t> addr[2];
  for(unsigned arm=0;arm<2;arm++){
   auto& ctl=controls[arm];ctl.assign(640,0x33);ctl[1]=kind==4 ? 8 : 0;
   indices[arm].assign(base+kTop,-1);offsets[arm]=dest[arm]=indices[arm];addr[arm].assign(base+kTop,47);
   Context c{uint64_t(rows),uint64_t(ordinal.data()),uint64_t(indices[arm].data()),uint64_t(offsets[arm].data()),
             uint64_t(dest[arm].data()),uint64_t(&totals[arm]),fallback?uint64_t(addr[arm].data()):0};
   unsigned per=(n+511)/512;
   for(unsigned i=0;i<512;i++){
    safe[i]=0;
    for(unsigned j=i*per;j<(i+1)*per && j<n;j++)
     safe[i]+=slot[raw+j]<res && (ordinal[b*res+slot[raw+j]]<0 || unsigned(ordinal[b*res+slot[raw+j]])>=used);
    before[i+1]=before[i]+safe[i];
   }
   // StagedPrefix is supplied by the oracle; run lane zero after peers so
   // its final status publication observes all errors at the block barrier.
   for(unsigned i=512;i--;){
    threadIdx.x=i;
    if(arm && n && n<=512)
     PackSmallPublicationEvents(&c,h,res,656,b,q,used,raw,n,token.data(),slot.data(),ctl.data());
    else
     PackStagedUrmaQuery(h,res,656,b,q,used,0,n,&c,token.data()+raw,ids.data(),slot.data()+raw,ctl.data());
   }
  }
  assert(indices[0]==indices[1] && offsets[0]==offsets[1] && dest[0]==dest[1] && addr[0]==addr[1]);
  assert(totals[0]==totals[1]);
  for(unsigned i:{1,16,17,18,19})assert(controls[0][i]==controls[1][i]);
 }
}
"""
    program = tmp_path / "register_pack.cpp"
    program.write_text(source)
    binary = tmp_path / "register_pack"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    str(program), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
