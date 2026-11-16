"""Execute round initialization with deferred DMA and poisoned prior rounds."""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function


ROOT = Path(__file__).resolve().parents[4]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch"


@pytest.mark.parametrize("compact,ubmem", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("sparse", [0, 1])
def test_round_init_dma_coverage_and_streaming_prefix(tmp_path, compact, ubmem, sparse):
    lifecycle = (OP / "op_kernel/lifecycle.cpp").read_text()
    streaming = (OP / "op_kernel/streaming.cpp").read_text()
    prefetch = (OP / "op_kernel/prefetch.h").read_text()
    metadata = subprocess.check_output(
        [
            "c++",
            "-E",
            "-P",
            "-x",
            "c++",
            "-DKVIP_FUSE_QUERY_WORK=0",
            str(ROOT / "csrc/operators/common/lifetch/cache_metadata.h"),
        ],
        text=True,
    )
    # Transport fields are irrelevant to this helper; retain the current full
    # status size so its profile suffix is also covered by the DMA model.
    (tmp_path / "batch_get_abi.h").write_text("struct KvafBatchGetCoreStatus { char storage[1216]; };\n")
    source = tmp_path / "compact_init.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#define __CCE_AICORE__ 1
#define __aicore__
#define __simt_vf__
#define __simt_callee__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
#define KVIP_OUTPUT_WORKER 1
#define KVIP_FUSE_QUERY_WORK 0
#include "context.h"
#include "pipeline_types.h"
#include "round_init.h"
using namespace rotating;
using pipeline_ab::kPrefetch;
constexpr uint32_t kLruSlots=2048,kChangeCapacity=10240,EVENT_ID0=0;
struct {uint32_t x;} threadIdx;
void asc_syncthreads(){}
void asc_threadfence(){}
uint32_t* changesBase=nullptr;uint32_t changesSize=0;
template<class T>T asc_ldcg(T* p){
 uintptr_t a=uintptr_t(p),begin=uintptr_t(changesBase),end=begin+changesSize*4;
 // All real streaming readers exercised below must stay inside each row's
 // prefix, even when the previous round left valid-looking suffix records.
 if(a>=begin&&a<end)assert(((a-begin)/4)%kChangeCapacity<kLruSlots);
 return *p;
}
template<class T>void asc_stcg(T* p,T v){*p=v;}
"""
        + function(metadata, "__simt_callee__ inline uint32_t OwnerRow(")
        + "\n"
        + function(metadata, "__simt_callee__ inline uint32_t OutputStride(")
        + "\n"
        + function(prefetch, "inline void InitializeTailMappings(")
        + "\n"
        + function(streaming, "inline void PublishPrefetchResidency(")
        + "\n"
        + function(streaming, "__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishPrefetch(")
        + "\n"
        + function(streaming, "inline void PublishQ(")
        + r"""
constexpr uint32_t guard=0xa5a5a5a5,poison=0x12345678;
struct Region {
 std::string name;uint32_t n;std::vector<uint32_t> memory,expected,writes;
 Region(std::string label,uint32_t count):name(label),n(count),memory(count+16,guard),expected(count,poison),writes(count,0){std::fill_n(data(),n,poison);}
 uint32_t* data(){return memory.data()+8;}
 uint64_t address(){return uint64_t(data());}
 void mark(uint32_t begin,uint32_t count,uint32_t value){assert(begin+count<=n);std::fill_n(expected.begin()+begin,count,value);}
 void check(){assert(std::equal(expected.begin(),expected.end(),data()));
  for(uint32_t i=0;i<8;++i)assert(memory[i]==guard&&memory[n+8+i]==guard);}
};
std::vector<std::unique_ptr<Region>> regions;
Region& make(const char* name,uint32_t n){regions.emplace_back(new Region(name,n));return *regions.back();}
Region& locate(uint32_t* p,uint32_t count){
 for(auto& r:regions)if(uintptr_t(p)>=r->address()&&uintptr_t(p)+uint64_t(count)*4<=r->address()+uint64_t(r->n)*4)return *r;
 assert(false);return *regions[0];
}
std::vector<std::function<void()>> mte2,mte3;
void finish(std::vector<std::function<void()>>& work){for(auto& op:work)op();work.clear();}
bool credits[5][8]{};
enum class HardEvent {V_MTE3,MTE3_V,MTE3_S,MTE2_MTE3,MTE3_MTE2};
template<HardEvent E>void SetFlag(uint32_t id){assert(!credits[unsigned(E)][id]);credits[unsigned(E)][id]=true;}
template<HardEvent E>void WaitFlag(uint32_t id){
 assert(credits[unsigned(E)][id]);credits[unsigned(E)][id]=false;
 if(E==HardEvent::MTE3_V||E==HardEvent::MTE3_S||E==HardEvent::MTE3_MTE2)finish(mte3);
 if(E==HardEvent::MTE2_MTE3)finish(mte2);
}
template<class T>struct LocalTensor {
 T* p=nullptr;uint32_t n=0;
};
template<class T>struct GlobalTensor {
 T* p=nullptr;void SetGlobalBuffer(T* value){p=value;}
 GlobalTensor operator[](uint32_t offset)const{return {p+offset};}
};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {
 std::vector<uint32_t> data;
 template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data()),uint32_t(data.size()*4/sizeof(T))};}
};
uint32_t maxUbBytes=0;
struct TPipe {
 template<class T>void InitBuffer(T& b,uint32_t bytes){assert(bytes<=65536);maxUbBytes=std::max(maxUbBytes,bytes);b.data.resize(bytes/4);}
 void Destroy(){assert(mte2.empty()&&mte3.empty());for(auto& c:credits)for(bool v:c)assert(!v);}
};
template<class T>void Duplicate(LocalTensor<T> dst,T value,uint32_t count){
 assert(mte3.empty()&&count<=dst.n);std::fill_n(dst.p,count,value);
}
uint32_t dmaCalls=0;
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,uint32_t count){
 assert(count<=src.n&&count%8==0&&mte2.empty());++dmaCalls;
 Region& region=locate(reinterpret_cast<uint32_t*>(dst.p),count);
 mte3.push_back([=,&region]{
  std::copy_n(src.p,count,dst.p);uint32_t begin=uint32_t(dst.p-region.data());
  for(uint32_t i=0;i<count;++i)++region.writes[begin+i];
 });
}
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t count){
 assert(mte3.empty()&&count<=dst.n&&count%8==0);locate(reinterpret_cast<uint32_t*>(src.p),count);
 mte2.push_back([=]{std::copy_n(src.p,count,dst.p);});
}
namespace kvaf_device {void CleanCacheLines(uint8_t*,uint32_t count){assert(count==4);}}
"""
        + function(lifecycle, "__aicore__ inline void FillRoundWords(")
        + "\n"
        + function(lifecycle, "__aicore__ inline void InitializeRound(")
        + r"""
int main(){
 const bool compact=KVIP_COMPACT_INIT&&!V9_UBMEM;
 assert(kvip_round_init::kFillWords==(compact?16384U:4096U));
 for(uint32_t batch:{1U,2U,16U})for(bool streaming:{false,true}){
  uint32_t history=batch==1?33024:batch==2?131328:210944,rounds=batch==1?0:8,lanes=4,fw=2;
  uint32_t finalWorkers=4,postOffset=batch*128+batch*finalWorkers*kFinalStateWords*4;
  regions.clear();Context ctx{};pipeline_ab::Args args{};
  auto& rows=make("rows",batch*sizeof(Row)/4);ctx.rows=rows.address();
  auto& progress=make("progress",batch*4*sizeof(Progress)/4);ctx.progress=progress.address();
  auto& owners=make("owners",(postOffset+batch*finalWorkers*sizeof(FinalPost))/4);ctx.owners=owners.address();
  auto& status=make("rowStatus",batch*kRowStatusWords);ctx.rowStatus=status.address();
  auto& finalStatus=make("finalStatus",batch*2*sizeof(KvafBatchGetCoreStatus)/4);ctx.finalStatus=finalStatus.address();
  auto& events=make("events",(rounds?rounds:1)*lanes*sizeof(Event)/4);ctx.events=events.address();
  auto& count=make("count",1);ctx.count=count.address();
  auto& mapping=make("mapping",batch*history);ctx.mapping=mapping.address();
  auto& changes=make("changes",batch*kChangeCapacity);ctx.changes=changes.address();
  auto& slots=make("changeSlots",batch*kChangeCapacity);ctx.changeSlots=slots.address();
  auto& lru=make("lru",batch*kLruSlots);ctx.lruSlots=lru.address();
  auto& changeCount=make("changeCount",batch);ctx.changeCount=changeCount.address();
  auto& valid=make("hotValid",batch*16384);ctx.hotValid=valid.address();
  auto& tails=make("tailBounds",batch*2);ctx.tailBounds=tails.address();ctx.outputStride=16384+384;
  for(uint32_t i=0;i<lru.n;++i)lru.data()[i]=lru.expected[i]=(i%2048)*2;
  for(uint32_t b=0;b<batch;++b){tails.data()[b*2]=tails.expected[b*2]=history-128;tails.data()[b*2+1]=tails.expected[b*2+1]=history;}
  auto& flags=make("flags",batch*pipeline_ab::kFlagCount*sizeof(pipeline_ab::BatchFlag)/4);args.flags=flags.address();
  auto& workerStatus=make("workerStatus",batch*32);args.workerStatus=workerStatus.address();
  auto& errors=make("queryErrors",batch*128);args.queryErrors=errors.address();
  uint32_t ds=batch*2048,meta=(3*batch+31)&~31U;
  auto& packed=make("packedFetch",4*fw*(3*ds+meta)+4*fw*(6*ds+64));args.packedFetch=packed.address();args.prefetchCount=fw<<16;
  for(uint32_t replay=0;replay<3;++replay){
   bool noPrefetch=replay==1;
   bool clearMapping=kvip_round_init::ClearMapping(streaming,noPrefetch);
   assert(clearMapping==!(KVIP_LOW_MISS_SPARSE_INIT&&!V9_UBMEM&&streaming&&noPrefetch));
   if(noPrefetch)for(uint32_t i=0;i<mapping.n;++i)
    mapping.data()[i]=i%2?16384+3:16384+kSlots+42; // Stale prefetch and ABC tags.
   // High -> low -> high: skipped stale state must be cleared again as soon
   // as prefetch is re-enabled, including after a wrong low prediction.
   // Every omitted suffix remains a valid-looking positive token, never -1.
   for(auto& r:regions){std::copy_n(r->data(),r->n,r->expected.begin());std::fill(r->writes.begin(),r->writes.end(),0);}
   for(uint32_t b=0;b<batch;++b){
    uint32_t clear=kvip_round_init::ChangeWords(streaming,kLruSlots,kChangeCapacity);
    assert(clear==(compact&&streaming?kLruSlots:kChangeCapacity));
    changes.mark(b*kChangeCapacity,clear,UINT32_MAX);
    std::copy_n(lru.data()+b*kLruSlots,kLruSlots,slots.expected.begin()+b*kChangeCapacity);
   }
   for(auto* r:{&rows,&progress,&owners,&status,&finalStatus,&events})r->mark(0,r->n,0);
   count.mark(0,1,0);if(clearMapping)mapping.mark(0,mapping.n,UINT32_MAX);
   if(streaming){
    for(auto* r:{&flags,&workerStatus,&errors})r->mark(0,r->n,0);
    for(uint32_t g=0;g<4*fw;++g)packed.mark(4*fw*(3*ds+meta)+g*(6*ds+64)+6*ds,64,0);
   }
   dmaCalls=0;
   for(uint32_t b=batch;b--;)InitializeRound(batch,history,rounds,lanes,finalWorkers,postOffset,&ctx,b,streaming?&args:nullptr,noPrefetch);
   assert(maxUbBytes==kvip_round_init::kFillWords*4);
   for(auto& r:regions)r->check();
   for(auto* r:{&rows,&progress,&owners,&status,&finalStatus,&events})for(auto n:r->writes)assert(n==1);
   for(auto n:mapping.writes)assert(n==uint32_t(clearMapping));
   for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<kChangeCapacity;++i){
    assert(changes.writes[b*kChangeCapacity+i]==uint32_t(!(compact&&streaming)||i<kLruSlots));
    assert(slots.writes[b*kChangeCapacity+i]==uint32_t(i<kLruSlots));
   }
   for(uint32_t b=0;b<batch;++b){
    for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)InitializeTailMappings(history,16384,&ctx,b);
    assert(changeCount.data()[b]==kLruSlots);
    for(uint32_t t=0;t<history;++t)assert(mapping.data()[b*history+t]==
      (t>=history-128?16384+kSlots+t%384:mapping.expected[b*history+t]));
    changes.data()[b*kChangeCapacity]=7;changes.data()[b*kChangeCapacity+kLruSlots-1]=9;
    // Holes deliberately remain below a positive changeCount; streaming
    // residency and prefetch publication may not interpret those records.
    changeCount.data()[b]=kChangeCapacity;
    std::fill_n(changes.data()+b*kChangeCapacity+kLruSlots,kSlots,1234+replay);
   }
   changesBase=changes.data();changesSize=changes.n;
   std::fill_n(valid.data(),valid.n,0);
   for(uint32_t b=0;b<batch;++b){
    for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)PublishPrefetchResidency(&ctx,b,16384);
    for(uint32_t i=0;i<16384;++i)assert(valid.data()[b*16384+i]==uint32_t(i==0||i==4094));
    std::vector<uint32_t> outSource(kChangeCapacity,poison),outDest(kChangeCapacity,poison);
    uint32_t prefetchCount=replay?0:kLruSlots;
    for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishPrefetch(changes.data()+b*kChangeCapacity,
      slots.data()+b*kChangeCapacity,outSource.data(),outDest.data(),prefetchCount);
    for(uint32_t i=0;i<kChangeCapacity;++i){
     assert(outSource[i]==(i<prefetchCount?changes.data()[b*kChangeCapacity+i]:poison));
     assert(outDest[i]==(i<prefetchCount?slots.data()[b*kChangeCapacity+i]:poison));
    }
    std::vector<uint32_t> selected(kSlots),finalSlots(kSlots);std::vector<uint16_t> ids(kTop);
    for(uint32_t i=0;i<kSlots;++i){selected[i]=9000+i;finalSlots[i]=i;}
    for(uint32_t i=0;i<kTop;++i)ids[i]=uint16_t(100+i);
    uint32_t added=replay?0:2;
    for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)PublishQ(selected.data(),ids.data(),finalSlots.data(),outSource.data(),outDest.data(),0,added);
    for(uint32_t i=0;i<added;++i){assert(outSource[kLruSlots+i]==selected[ids[i]]);assert(outDest[kLruSlots+i]==finalSlots[i]);}
    assert(outSource[kLruSlots+added]==poison); // No suffix scan via changeCount.
   }
   changesBase=nullptr;changesSize=0;
  }
 }
}
"""
    )
    binary = tmp_path / "compact_init"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DKVIP_COMPACT_INIT={compact}",
            f"-DV9_UBMEM={ubmem}",
            f"-DKVIP_LOW_MISS_SPARSE_INIT={sparse}",
            "-DKVIP_AUTO_ROUTE=1",
            "-DKVIP_SELECTED_METADATA=1",
            "-DKVIP_LOW_MISS_TAIL=1",
            "-DKVIP_LOW_MISS_SKIP=1",
            "-DKVIP_STREAMING_LI=1",
            "-I",
            str(tmp_path),
            "-I",
            str(OP / "include"),
            "-I",
            str(OP / "op_kernel"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
