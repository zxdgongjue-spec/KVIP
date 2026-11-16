"""Execute the production selected-only classifier against the existing path."""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function


ROOT = Path(__file__).resolve().parents[4]
OPERATOR = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel"


@pytest.mark.parametrize("selected,ubmem", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_selected_metadata_keeps_default_and_ubmem_driver_paths(tmp_path, selected, ubmem):
    vector = (OPERATOR / "indexer/indexer_vector.h").read_text()
    begin = function(vector, "__aicore__ inline void QLIVector<QLIT>::BeginFinalMetadata(")
    emit = function(vector, "__aicore__ inline void QLIVector<QLIT>::EmitFinalClassification(")
    source = tmp_path / "driver.cpp"
    source.write_text(begin + "\n" + emit)
    processed = subprocess.check_output(
        [
            "c++",
            "-E",
            "-P",
            f"-DKVIP_SELECTED_METADATA={selected}",
            f"-DV9_UBMEM={ubmem}",
            str(source),
        ],
        text=True,
    )
    begin = function(processed, "__aicore__ inline void QLIVector<QLIT>::BeginFinalMetadata(")
    emit = function(processed, "__aicore__ inline void QLIVector<QLIT>::EmitFinalClassification(")
    active = bool(selected and not ubmem)
    assert ("ClassifySelectedMetadata" in emit) == active
    assert ("ClassifyTopkHalf" in emit) != active
    assert ("DataCopy(streamingValid_" in begin) != active
    # Both branches still join mapping publication before classification and
    # publish the same output, protection bits, and error after the same merge.
    classify = emit.index("ClassifySelectedMetadata" if active else "ClassifyTopkHalf")
    assert emit.index("if(events[i].complete)break") < classify
    assert classify < emit.index("MergeClassifyBitmap") < emit.index("DataCopy(output,result,2048)")
    assert emit.index("DataCopy(output,result,2048)") < emit.index("errors[query*32]=bitmap.GetValue")


def test_selected_metadata_matches_all_resident_chunks(tmp_path):
    resolve = function(
        (ROOT / "csrc/operators/common/lifetch/cache_metadata.h").read_text(),
        "__simt_callee__ inline uint32_t ResolvePhysicalSlot(",
    )
    source = tmp_path / "selected_metadata.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#define __simt_vf__
#define __simt_callee__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
void asc_syncthreads(){}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
uintptr_t validBegin=0,validEnd=0,reservedBegin=0,reservedEnd=0;
uint32_t validLoads=0,reservedLoads=0;
bool selectedPath=false;
std::set<uintptr_t> allowedValid,allowedReserved;
template<class T> T asc_ldcg(T* p){
 uintptr_t address=uintptr_t(p);
 if(selectedPath && address>=validBegin && address<validEnd){
  assert(allowedValid.count(address));++validLoads;
 }
 if(selectedPath && address>=reservedBegin && address<reservedEnd){
  assert(allowedReserved.count(address));++reservedLoads;
 }
 return *p;
}
namespace rotating {
constexpr uint32_t kTop=2048,kLruSlots=2048,kSlots=8192;
struct Context {uint64_t mapping,lruSlots,hotValid,reservedHot;};
uint32_t OutputStride(uint32_t resident,Context*){return resident+512;}
"""
        + resolve
        + '\n}\n#include "staged_topk_classify.h"\n#include "staged_selected_metadata.h"\n'
        + r"""
int main(){using namespace rotating;std::mt19937 rng(451);
 for(uint32_t resident:{2048U,2176U,8192U,16384U,33024U,131072U}){
  uint32_t history=resident+10240,words=(resident+31)/32;
  std::vector<uint32_t> valid(4*resident),mapping(2*history),lru(2*kLruSlots),indices(kTop);
  std::vector<int32_t> reserved(2*resident);
  validBegin=uintptr_t(valid.data());validEnd=validBegin+valid.size()*4;
  reservedBegin=uintptr_t(reserved.data());reservedEnd=reservedBegin+reserved.size()*4;
  // Different owners and packed rows must not alias their metadata.
  for(auto& value:valid)value=rng()%3;
  for(auto& value:reserved)value=rng()%3 ? -1 : int32_t(rng()%2048);
  for(auto& value:lru)value=rng()%resident;
  for(uint32_t i=0;i<mapping.size();++i){
   switch(i%5){
    case 0:mapping[i]=i%resident;break;
    case 1:mapping[i]=resident+i%2048;break; // Reserved rank -> actual LRU slot.
    case 2:mapping[i]=resident+kSlots+i%384;break; // ABC encoded slot.
    case 3:mapping[i]=0x80000000U+(i%2048);break; // Missing-rank marker.
    default:mapping[i]=UINT32_MAX;
   }
  }
  for(uint32_t flags=0;flags<8;++flags)for(uint32_t used:{0U,1U,2048U})
  for(uint32_t batch:{0U,1U})for(uint32_t mode=0;mode<4;++mode){
   uint32_t owner=3-batch;bool hasValid=flags&1,hasReserved=flags&2,hasTail=flags&4;
   uint32_t tailBegin=resident-128,tailEnd=resident+256;
   for(uint32_t i=0;i<kTop;++i){
    indices[i]=mode==1?UINT32_MAX:mode==2?resident+(i*31)%10240:
      mode==3?(i*7)%resident:(i*997)%history;
   }
   if(mode==0){indices[0]=UINT32_MAX;indices[1]=history;indices[2]=history+1;
    indices[3]=0x80000000U;indices[4]=resident-1;indices[5]=tailBegin;
    indices[6]=tailEnd;indices[7]=resident;indices[8]=0;indices[9]=0;}
   Context ctx{uint64_t(mapping.data()),uint64_t(lru.data()),
     hasValid?uint64_t(valid.data()):0,hasReserved?uint64_t(reserved.data()):0};
   // Execute two successive states; there is no retained metadata snapshot.
   for(uint32_t replay=0;replay<2;++replay){
    for(uint32_t i=0;i<resident;i+=17){valid[owner*resident+i]^=1;reserved[batch*resident+i]=replay?-1:2047;}
    const auto validBefore=valid;const auto reservedBefore=reserved;
    const auto mappingBefore=mapping;const auto lruBefore=lru;
    allowedValid.clear();allowedReserved.clear();
    for(uint32_t token:indices)if(token<resident && !(hasTail&&token>=tailBegin&&token<tailEnd)){
     if(hasValid)allowedValid.insert(uintptr_t(valid.data()+owner*resident+token));
     if(hasReserved && (!hasValid||valid[owner*resident+token]))
      allowedReserved.insert(uintptr_t(reserved.data()+batch*resident+token));
    }
    std::vector<uint32_t> locations(kTop+2,0xdddddddd),bits(4106,0);
    locations.front()=locations.back()=0x12345678;bits.front()=bits.back()=0x12345678;
    validLoads=reservedLoads=0;selectedPath=true;
    for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
     ClassifySelectedMetadata(history,resident,batch,owner,used,tailBegin,tailEnd,hasTail,&ctx,
       indices.data(),locations.data()+1,bits.data()+1,bits.data()+4097);
    selectedPath=false;
    assert(validLoads<=kTop&&reservedLoads<=kTop);
    if(mode==1||mode==2)assert(!validLoads&&!reservedLoads);
    for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)MergeClassifyBitmap(bits.data()+1,resident);
    for(uint32_t chunk:{4096U,8192U}){
     std::vector<uint32_t> expected(kTop,0xcccccccc),expectedBits(4104,0);
     for(uint32_t base=0;base<resident;base+=chunk){
      uint32_t count=std::min(chunk,resident-base);
      for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)
       ClassifyTopkHalf(history,resident,batch,used,base,count,tailBegin,tailEnd,
        hasValid,hasReserved,hasTail,&ctx,indices.data(),expected.data(),
        valid.data()+owner*resident+base,reserved.data()+batch*resident+base,
        expectedBits.data(),expectedBits.data()+4096);
     }
     for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)MergeClassifyBitmap(expectedBits.data(),resident);
     assert(std::equal(expected.begin(),expected.end(),locations.begin()+1));
     assert(std::equal(expectedBits.begin(),expectedBits.end(),bits.begin()+1));
     assert(expectedBits[4096]==uint32_t(mode==0));
    }
    assert(locations.front()==0x12345678&&locations.back()==0x12345678);
    assert(bits.front()==0x12345678&&bits.back()==0x12345678);
    assert(valid==validBefore&&reserved==reservedBefore&&mapping==mappingBefore&&lru==lruBefore);
   }
  }
 }
}
"""
    )
    binary = tmp_path / "selected_metadata"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-I",
            str(OPERATOR),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_no_prefetch_preserves_candidate_hits_and_q4_protection(tmp_path):
    resolve = function(
        (ROOT / "csrc/operators/common/lifetch/cache_metadata.h").read_text(),
        "__simt_callee__ inline uint32_t ResolvePhysicalSlot(",
    )
    source = tmp_path / "no_prefetch_metadata.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __simt_vf__
#define __simt_callee__
#define __aicore__
#define __ubuf__
#define __gm__
#define LAUNCH_BOUND(x)
#define KVIP_LOW_MISS_TAIL 1
struct {uint32_t x;} threadIdx;
void asc_syncthreads(){}
void asc_atomic_or(uint32_t* p,uint32_t v){*p|=v;}
bool forbidReserved=false;
uintptr_t reservedBegin=0,reservedEnd=0,mappingBegin=0,mappingEnd=0;
template<class T>T asc_ldcg(T* p){
 assert(!forbidReserved || uintptr_t(p)<reservedBegin || uintptr_t(p)>=reservedEnd);
 assert(!forbidReserved || uintptr_t(p)<mappingBegin || uintptr_t(p)>=mappingEnd);
 return *p;
}
namespace rotating {
constexpr uint32_t kTop=2048,kLruSlots=2048,kSlots=8192;
struct Context {uint64_t mapping,lruSlots,hotValid,reservedHot;};
uint32_t tailCapacity=256;
uint32_t OutputStride(uint32_t resident,Context*){return resident+tailCapacity;}
''' + resolve + '\n}\n#include "staged_topk_classify.h"\n#include "staged_selected_metadata.h"\n' + r'''
int main(){using namespace rotating;
 constexpr uint32_t R=4096,H=8192,B=1,O=2,T=256;
 std::vector<uint32_t> valid(3*R,1),map(2*H,UINT32_MAX),lru(2*kLruSlots);
 std::vector<int32_t> reserved(2*R,-1);
 for(uint32_t i=0;i<kLruSlots;++i){lru[B*kLruSlots+i]=i*2;reserved[B*R+i*2]=i;}
 reservedBegin=uintptr_t(reserved.data());reservedEnd=reservedBegin+reserved.size()*4;
 mappingBegin=uintptr_t(map.data());mappingEnd=mappingBegin+map.size()*4;
 valid[O*R+2]=0; // An invalid candidate remains a true miss.
 const uint32_t begin=R-128,end=R+128;
 for(uint32_t token=begin;token<end;++token)map[B*H+token]=R+kSlots+token%T;
 Context ctx{uint64_t(map.data()),uint64_t(lru.data()),uint64_t(valid.data()),uint64_t(reserved.data())};
 std::vector<uint32_t> selected(4*kTop),locations(4*kTop),unionBits((R+31)/32);
 // All candidate slots are selected somewhere in Q4, including last-query-only hits.
 for(uint32_t q=0;q<4;++q)for(uint32_t i=0;i<kTop;++i)
   selected[q*kTop+i]=q==3?2*i:q==0?H-1:(i*17+q)%H;
 selected[0]=UINT32_MAX;selected[1]=H;selected[2]=2;
 for(uint32_t capacity:{0U,128U,256U})for(bool low:{false,true})for(uint32_t used:{0U,1U}){
  tailCapacity=capacity;
  // Poison all uninitialized locations, as sparse init intentionally retains
  // the previous round's map. Only the high path consumes initialized entries.
  std::fill(map.begin(),map.end(),low&&!used?uint32_t(R+kSlots+99):UINT32_MAX);
  for(uint32_t token=begin;token<end&&token-begin<capacity;++token)
   map[B*H+token]=R+kSlots+token%capacity;
  std::fill(unionBits.begin(),unionBits.end(),0);
  forbidReserved=low && !used;
  for(uint32_t q=0;q<4;++q){
   std::vector<uint32_t> bits(4104,0);
   for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x){
    if(low&&!used)
     ClassifySelectedMetadata<true>(H,R,B,O,used,begin,end,true,&ctx,selected.data()+q*kTop,
       locations.data()+q*kTop,bits.data(),bits.data()+4096);
    else
     ClassifySelectedMetadata<false>(H,R,B,O,used,begin,end,true,&ctx,selected.data()+q*kTop,
       locations.data()+q*kTop,bits.data(),bits.data()+4096);
   }
   for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)MergeClassifyBitmap(bits.data(),R);
   assert(bits[4096]==uint32_t(q==0));
   for(uint32_t i=0;i<kTop;++i){
    uint32_t token=selected[q*kTop+i],expected=UINT32_MAX;
    if(token<H){
     bool tail=token>=begin && token<end;
     if(token<R && !tail && valid[O*R+token] &&
        ((low&&!used)||reserved[B*R+token]<0))expected=token;
     else if(low&&!used){
      if(tail&&token-begin<capacity)expected=R+token%capacity;
     }else expected=ResolvePhysicalSlot(R,lru.data()+B*kLruSlots,map[B*H+token]);
    }
    assert(locations[q*kTop+i]==expected);
    if(expected<R)assert(bits[expected/32]&(1U<<(expected%32)));
   }
   for(uint32_t w=0;w<unionBits.size();++w)unionBits[w]|=bits[w];
  }
  // Every known Q4 hot output is protected before any query can choose a victim.
  for(uint32_t slot:locations)if(slot<R)assert(unionBits[slot/32]&(1U<<(slot%32)));
  if(low&&!used){
   assert(locations[2]==UINT32_MAX);
   assert(locations[3*kTop+32]==64);
   assert(unionBits[64/32]&(1U<<(64%32)));
  }else assert(locations[3*kTop+32]==UINT32_MAX);
 }
}
''')
    binary = tmp_path / "no_prefetch_metadata"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
        "-I", str(OPERATOR), str(source), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
