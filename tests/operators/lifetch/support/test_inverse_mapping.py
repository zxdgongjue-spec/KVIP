# SPDX-License-Identifier: Apache-2.0
"""Execute the native inverse lookup and tail initializer against a dense oracle."""

import subprocess

from test_recent_index import COMMON, INDEXER, function


def test_native_inverse_mapping_multiround_and_publication(tmp_path):
    metadata = (COMMON / "cache_metadata.h").read_text()
    prefetch = (INDEXER.parent / "prefetch.h").read_text()
    tail = function(prefetch, "inline void InitializeTailMappings(")
    source = tmp_path / "inverse_mapping.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __simt_callee__
#define __gm__
struct {uint32_t x;} threadIdx;
uint64_t fences=0;
inline void asc_threadfence(){++fences;}
template<class T> T asc_ldcg(const T* p){return *p;}
template<class T> void asc_stcg(T* p,T value){*p=value;}
struct Context {
 uint64_t mapping=0,changes=0,lruSlots=0,tailBounds=0,outputStride=0,changeCount=0;
 uint32_t prefetchLimit=512;
};
namespace rotating {
constexpr uint32_t kSlots=8192,kLruSlots=2048,kChangeCapacity=10240;
"""
        + function(metadata, "__simt_callee__ inline uint32_t OutputStride(uint32_t rt_resident,__gm__ Context* ctx)").replace("#endif\n", "")
        + function(metadata, "__simt_callee__ inline void RecordChange(")
        + function(metadata, "__simt_callee__ inline uint32_t ResolvePhysicalSlot(")
        + '\n#include "inverse_mapping.h"\n#define KVIP_INVERSE_MAPPING 0\n'
        + tail.replace("InitializeTailMappings", "InitializeDenseTail")
        + "\n#undef KVIP_INVERSE_MAPPING\n#define KVIP_INVERSE_MAPPING 1\n"
        + tail
        + r"""
}
using namespace rotating;
uint32_t readMapping(uint32_t h,uint32_t r,uint32_t b,Context* ctx,uint32_t* map,uint32_t token) {
 const auto view=LoadPrefetchMappingView(h,r,b,ctx,map);
 return ReadPrefetchMapping(view,token);
}
void initTail(bool inverse,uint32_t h,uint32_t r,Context& ctx,uint32_t b) {
 for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x) {
  if(inverse)InitializeTailMappings(h,r,&ctx,b);
  else InitializeDenseTail(h,r,&ctx,b);
 }
}
int main(){
 constexpr uint32_t H=32768,R=4096,B=3;
 std::mt19937 random(371);
 std::vector<uint32_t> dense(B*H),dirty(B*H),changes(B*kChangeCapacity),lru(B*kLruSlots);
 std::array<uint32_t,B*2> tail{};std::array<uint32_t,B> counts{};
 for(auto& x:dirty)x=random();
 for(auto& x:lru)x=random()%R;
 Context ctx{reinterpret_cast<uint64_t>(dirty.data()),reinterpret_cast<uint64_t>(changes.data()),
  reinterpret_cast<uint64_t>(lru.data()),reinterpret_cast<uint64_t>(tail.data()),R+512,
  reinterpret_cast<uint64_t>(counts.data()),512};
 uint64_t checks=0;
 auto check=[&](uint32_t b){
  for(uint32_t token=0;token<H;++token) {
   uint32_t actual=readMapping(H,R,b,&ctx,dirty.data()+b*H,token);
   assert(actual==dense[b*H+token]);
   assert(ResolvePhysicalSlot(R,lru.data()+b*kLruSlots,actual)==
          ResolvePhysicalSlot(R,lru.data()+b*kLruSlots,dense[b*H+token]));
  }
  assert(readMapping(H,R,b,&ctx,dirty.data()+b*H,H)==UINT32_MAX);
  ++checks;
 };
 for(uint32_t round=0;round<100;++round){
  // Per-round DMA reset, including a dirty suffix from previous final repairs.
  std::fill(changes.begin(),changes.end(),UINT32_MAX);
  std::fill(dense.begin(),dense.end(),UINT32_MAX);
  ctx.outputStride=round%7==0?0:R+(round%3==0?128:512);
  for(uint32_t b=0;b<B;++b){
   uint32_t capacity=OutputStride(R,&ctx)-R;
   tail[2*b]=round%5==0?H-64:random()%H;
   tail[2*b+1]=tail[2*b]+random()%768;
   Context reference=ctx;reference.mapping=reinterpret_cast<uint64_t>(dense.data());
   initTail(false,H,R,reference,b);
   initTail(true,H,R,ctx,b);
   assert(counts[b]==2048); // Still published even though no map initialization.
   check(b);
   std::vector<uint32_t> tokens(H);std::iota(tokens.begin(),tokens.end(),0);
   std::shuffle(tokens.begin(),tokens.end(),random);
   uint32_t limit=round%9==0?0:(round%9==1?2048:random()%2049);
   uint32_t ordinal=0;
   for(uint32_t token:tokens){
    if(ordinal==limit)break;
    // The existing cold predicate excludes current ABC locations.
    if(token>=tail[2*b] && token<tail[2*b+1] && token-tail[2*b]<capacity)continue;
    // Actual publication order. No lookup is allowed until map + inverse have
    // both been released by the event fence. Model a visible map first: the
    // previous inverse cannot claim a new token, even if its ordinal is reused.
    asc_stcg(dirty.data()+b*H+token,R+ordinal);
    assert(readMapping(H,R,b,&ctx,dirty.data()+b*H,token)==UINT32_MAX);
    RecordChange(R,&ctx,b,R+ordinal,token);
    asc_threadfence(); // Event.complete publication / next serialized epoch.
    dense[b*H+token]=R+ordinal;
    ++ordinal;
    if(ordinal%256==0)check(b);
   }
   check(b);
  }
  // Final resolver can replace any inverse entry. All native map readers have
  // finished here; next invocation must reset this prefix before any lookup.
  for(auto& x:changes)x=random()%H;
 }
 assert(checks>1000 && fences>0);
 // Same token and ordinal in the previous round must not look current merely
 // because its stale direct map matches: the new inverse starts invalid.
 std::fill(changes.begin(),changes.end(),UINT32_MAX);ctx.tailBounds=0;ctx.outputStride=0;
 dirty[37]=R+12;
 assert(readMapping(H,R,0,&ctx,dirty.data(),37)==UINT32_MAX);
 RecordChange(R,&ctx,0,R+12,37);asc_threadfence();
 assert(readMapping(H,R,0,&ctx,dirty.data(),37)==R+12);
 changes[12]=99;dirty[99]=R+12;
 assert(readMapping(H,R,0,&ctx,dirty.data(),37)==UINT32_MAX);
 assert(readMapping(H,R,0,&ctx,dirty.data(),99)==R+12);
 for(uint32_t raw:{0U,R-1,R+2048,R+8192,R+8192+511,UINT32_MAX,0x80000000U}){
  dirty[47]=raw;assert(readMapping(H,R,0,&ctx,dirty.data(),47)==UINT32_MAX);
 }
 // Runtime fallback must agree with init. Extended/legacy interfaces retain
 // the dense table; they cannot use an inverse that only records 2048 slots.
 for(unsigned fallback=0;fallback<3;++fallback){
  Context legacy=ctx;
  if(fallback==0)legacy.changes=0;
  if(fallback==1)legacy.lruSlots=0;
  if(fallback==2)legacy.prefetchLimit=2049;
  assert(!KVIP_INVERSE_MAPPING_ACTIVE(&legacy));
  for(uint32_t raw:{R+2048,R+8192,UINT32_MAX}){
   dirty[47]=raw;assert(readMapping(H,R,0,&legacy,dirty.data(),47)==raw);
  }
 }
 // Explicit history clipping and circular ABC address checks at service size.
 std::vector<uint32_t> serviceMap(131328,UINT32_MAX);ctx.tailBounds=reinterpret_cast<uint64_t>(tail.data());
 ctx.outputStride=16384+512;tail[0]=131200;tail[1]=131800;
 assert(readMapping(131328,16384,0,&ctx,serviceMap.data(),131327)==
        16384+8192+131327%512);
 assert(readMapping(131328,16384,0,&ctx,serviceMap.data(),131328)==UINT32_MAX);
}
"""
    )
    binary = tmp_path / "inverse_mapping"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
         "-I", str(INDEXER.parents[1] / "include"), str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_inverse_mapping_compiles_out_and_keeps_publication_order():
    prefetch = (INDEXER.parent / "prefetch.h").read_text()
    native = (INDEXER.parents[1] / "include/inverse_mapping.h").read_text()
    init = (INDEXER.parent / "lifecycle.cpp").read_text()
    assert "if(!KVIP_INVERSE_MAPPING_ACTIVE(ctx))" in init
    assert "if(KVIP_INVERSE_MAPPING_ACTIVE(ctx))return;" in prefetch
    assert "if (!KVIP_INVERSE_MAPPING_ACTIVE(ctx))" in native
    for marker in ("inline void BatchBuild(", "inline void PrepareCandidateWqes("):
        body = function(prefetch, marker)
        assert body.index("asc_stcg(map+token,slot)") < body.index("RecordChange(")
        assert body.index("RecordChange(") < body.rindex("asc_threadfence()")
    files = ("prefetch.h", "lifecycle.cpp", "staged_selected_metadata.h",
             "staged_topk_classify.h", "resolver_staged_dma2.h")
    for name in files:
        source = (INDEXER.parent / name).read_text()
        source = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#include"))
        result = subprocess.run(
            ["c++", "-E", "-P", "-x", "c++", "-D__CCE_AICORE__=1",
             "-DV9_UBMEM=0", "-DKVIP_INVERSE_MAPPING=0", "-"],
            input=source, text=True, capture_output=True, check=True,
        )
        assert "ReadPrefetchMapping" not in result.stdout
        assert "KVIP_INVERSE_MAPPING_ACTIVE" not in result.stdout
