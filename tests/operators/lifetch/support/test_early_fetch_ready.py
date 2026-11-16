"""Run real publisher/join code with deferred output DMA and eager Get readers.

The transport mock consumes descriptors at the exact ready publication. Output
DMA reads UB only at its completion event, so moving the completion flag or
allowing metadata commit too early fails independently of wall-clock timing.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('ubmem,early', [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize('ready_batch', [0, 1])
@pytest.mark.parametrize('low_tail', [0, 1])
def test_fetch_ready_precedes_output_dma_but_not_metadata_join(tmp_path, ubmem, early, ready_batch, low_tail):
    root = Path(__file__).resolve().parents[4]
    kernel_dir = root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    worker = (kernel_dir / 'output_worker.h').read_text()
    worker = worker[worker.index('__aicore__ inline uint32_t AwaitPublication'):]
    streaming = (kernel_dir / 'streaming.cpp').read_text()
    publish = streaming[streaming.index('__aicore__ inline void PublishQueryOutputs'):]
    publish = publish[:publish.index('// The producer and Fetch AIV')]
    producer_join = streaming.split(' uint64_t publishWait=GetSystemCycle();', 1)[1].split(' status[0]=', 1)[0]
    consumer_join = streaming.split(' if(!error){uint64_t start=GetSystemCycle();while(true)', 1)[1].split('\n if(!error\n', 1)[0]
    input_error = streaming.split(' if(inputError){', 1)[1].split('\n TPipe pipe;', 1)[0]
    producer_done = streaming.split(' if(!kEarlyFetchReady){flags[5].error=status[0];', 1)[1].split('\n}', 1)[0]
    lifecycle = (kernel_dir / 'lifecycle.cpp').read_text()
    initialize_flags = re.search(r'FillRoundWords\(streaming->flags.*?;', lifecycle, re.S).group(0)
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
#define KVIP_OUTPUT_WORKER 1
#include "pipeline_types.h"
#include "publication_flags.h"
#include "low_miss_mode.h"
using namespace pipeline_ab;
using namespace rotating;
constexpr uint32_t kSlots=8192,kTop=2048,kLruSlots=2048;
constexpr uint32_t kOutputMasksOffset=32,kOutputMasksPerQuery=8;
struct Shape {uint32_t batch=2,history=131328,resident=16384,tokenBytes=656,finalWorkers=4,finalPostOffset=0;};
struct Context {
 uint64_t selected=0,locations=0,rows=0,owners=0,rowStatus=0,autoRoute=0,admissionControls=0;
 Shape shape{};
};
using StagedContextPointer=Context*;
struct Row {uint32_t slots=0;};
struct alignas(128) FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;uint64_t head=0;};
template<class T> struct LocalTensor {
 T* p; T GetValue(unsigned i)const{return p[i];} void SetValue(unsigned i,T v){p[i]=v;}
 T* GetPhyAddr()const{return p;} LocalTensor operator[](unsigned i)const{return {p+i};}
 template<class U>LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(p)};}
};
template<class T>struct GlobalTensor {T* p=nullptr;void SetGlobalBuffer(T* v){p=v;}GlobalTensor operator[](unsigned i)const{return {p+i};}};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {std::vector<uint32_t> data;template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}};
struct TPipe {template<class T>void InitBuffer(T& b,unsigned bytes){b.data.resize(bytes/4);}void Destroy(){}};
std::vector<std::function<void()>> dma;
enum class HardEvent {V_MTE2,MTE2_V,V_S,MTE3_V,V_MTE3,MTE3_S};
constexpr int EVENT_ID0=0;
BatchFlag flags[2*kFlagCount];FinalPost posts[8];
uint32_t activeBatch=0,currentQuery=0,initialError=0,packErrorQuery=99,missingRaw=99;
bool outputEnabled=true,descriptorSynced[4]{},postSynced[4]{},outputDone[4]{};
bool noPrefetchMode=false;
uint32_t usedSlots=0;
int32_t visibleControls[8]{},publishedControls[8]{};
uint32_t controlRefreshes=0;
uint32_t fetchReads=0,outputStarts=0,outputCopies=0,metadataCommits=0;
uint64_t clockValue=0;bool timeoutClock=false;
uint32_t logicalOut[8*kTop]{},slotOut[8*kTop]{};
uint32_t queryCount[4]{2,0,1,2};
void finish_dma(){for(auto& op:dma)op();dma.clear();}
template<HardEvent E>void SetFlag(int){}
template<HardEvent E>void WaitFlag(int){
 if constexpr(E==HardEvent::MTE3_S || E==HardEvent::MTE3_V)finish_dma();
}
template<class T>void Duplicate(LocalTensor<T> a,T value,unsigned n){std::fill_n(a.p,n,value);}
template<class T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,unsigned n){std::copy_n(src.p,n,dst.p);}
template<class T>void DataCopy(GlobalTensor<T> dst,LocalTensor<T> src,unsigned n){
 assert(n==kTop);unsigned q=currentQuery;
 assert(!flags[activeBatch*kFlagCount+kOutputsDoneFlag].ready || !kEarlyFetchReady);
 // The earliest legal Get observes the publication while TND output is stale.
 if(kEarlyFetchReady)assert(flags[activeBatch*kFlagCount+q+1].ready);
 else assert(!flags[activeBatch*kFlagCount+q+1].ready);
 ++outputStarts;
 dma.emplace_back([=]{std::copy_n(src.p,n,dst.p);++outputCopies;});
}
uint64_t GetSystemCycle(){clockValue+=timeoutClock?6000000001ULL:1;return clockValue;}
Shape LoadRuntimeShape(Context*){return {};}
template<class T>uint32_t* FinalPtr(LocalTensor<T> t){return reinterpret_cast<uint32_t*>(t.p);}
void CopyContextSnapshot(LocalTensor<uint32_t> dst,Context* src){std::memcpy(dst.p,src,sizeof(*src));}
namespace Simt {struct Dim3{int x,y,z;};template<auto F,class... A>void VF_CALL(Dim3,A... args){F(args...);}}
void MergeQueryBitmaps(uint32_t*,uint32_t*,unsigned,int64_t*,int64_t*,unsigned,unsigned,uint32_t*,bool){}
void PreparePublicationOutputs(int64_t*,uint32_t*,uint32_t,uint32_t b,uint32_t q,uint32_t*,uint32_t* work){
 currentQuery=q;assert(!flags[b*kFlagCount+kOutputsDoneFlag].ready || !kEarlyFetchReady);
 std::fill_n(work,8,0);work[0]=1U<<(b*4+q);
 for(unsigned i=0;i<kTop;++i)work[64+i]=100000+b*10000+q*2048+i;
}
void PackPublicationEvents(Context*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t q,uint32_t used,uint32_t,
 uint32_t count,uint32_t*,uint32_t*,uint32_t*,uint16_t*,uint32_t*,uint32_t* ctl,int32_t* direct,uint32_t){
 assert((direct==nullptr)==bool(KVIP_READY_BATCH_FETCH && !V9_UBMEM));
 assert(q==currentQuery);ctl[16]=count;ctl[17]=count;
 if(q==packErrorQuery)ctl[1]=16;
}
void SyncFetchDescriptors(Context*,uint32_t begin,uint32_t){descriptorSynced[(begin%kSlots)/kTop]=true;}
namespace kvaf_device {
void CleanCacheLines(uint8_t* ptr,uint64_t){
 if(ptr==reinterpret_cast<uint8_t*>(visibleControls)) {
  std::copy_n(publishedControls,8,visibleControls);
  ++controlRefreshes;
 }
 for(unsigned q=0;q<4;++q){
  if(ptr==reinterpret_cast<uint8_t*>(posts+activeBatch*4+q))postSynced[q]=true;
  auto f=flags+activeBatch*kFlagCount+q+1;
  if(ptr==reinterpret_cast<uint8_t*>(f) && f->ready && !f->error){
   auto p=posts+activeBatch*4+q;
   assert(postSynced[q] && descriptorSynced[q]);
   assert(p->count==queryCount[q] && p->descriptor==activeBatch*kSlots+q*kTop);
   assert(p->nonconflicting==p->count && !p->error);
   // Mock a ready transport consumer, not an expected copy of writer logic.
   assert(dma.empty());
   if(kEarlyFetchReady)assert(slotOut[(activeBatch*4+q)*kTop]==0xdeadbeefU);
   else if(outputEnabled)assert(slotOut[(activeBatch*4+q)*kTop]==activeBatch*10000+q*kTop);
   ++fetchReads;
  }
 }
 if(ptr==reinterpret_cast<uint8_t*>(flags+activeBatch*kFlagCount+kOutputsDoneFlag) &&
    kEarlyFetchReady && flags[activeBatch*kFlagCount+kOutputsDoneFlag].ready){
  assert(dma.empty());
  if(!flags[activeBatch*kFlagCount+kOutputsDoneFlag].error && outputEnabled)assert(outputCopies==8);
 }
}
}
#define LIFETCH_CLEAN_CACHE_LINES kvaf_device::CleanCacheLines
#include "FINAL_POST_HEADER"
/*PUBLISH_OUTPUTS*/
/*OUTPUT_WORKER*/
uint32_t ProducerInputError(Args* a,Context* ctx,uint32_t b,uint32_t inputError){
 auto flags=reinterpret_cast<BatchFlag*>(a->flags)+b*kFlagCount;
 if(inputError){/*PRODUCER_ERROR*/
 return 0;
}
uint32_t JoinProducer(BatchFlag* flags){
 uint32_t control[2]{};LocalTensor<uint32_t> ctl{control};
 uint64_t publishWait=GetSystemCycle();/*PRODUCER_JOIN*/
 return ctl.GetValue(1);
}
uint32_t JoinConsumer(BatchFlag* flags){
 uint32_t error=0;
 if(!error){uint64_t start=GetSystemCycle();while(true)/*CONSUMER_JOIN*/
 if(!error)++metadataCommits;
 return error;
}
uint32_t ProducerDone(BatchFlag* flags,uint32_t error){
 uint32_t usefulPrefetch=0;
 uint32_t status[]{error};
 if(!kEarlyFetchReady){flags[5].error=status[0];/*PRODUCER_DONE*/
}
void FillRoundWords(uint64_t dst,uint32_t count,LocalTensor<uint32_t>){std::fill_n(reinterpret_cast<uint32_t*>(dst),count,0);}
void InitializeFlags(Args* streaming,uint32_t b){LocalTensor<uint32_t> local{nullptr};/*INITIALIZE_FLAGS*/}
void reset(){
 for(auto& f:flags)f={};for(auto& p:posts)p={};
 std::fill_n(logicalOut,8*kTop,0xdeadbeefU);std::fill_n(slotOut,8*kTop,0xdeadbeefU);
 std::fill_n(descriptorSynced,4,false);std::fill_n(postSynced,4,false);
 fetchReads=outputStarts=outputCopies=metadataCommits=0;clockValue=0;
 initialError=0;packErrorQuery=missingRaw=99;outputEnabled=true;timeoutClock=false;dma.clear();
 controlRefreshes=0;
}
void runWorker(unsigned b){
 activeBatch=b;
 std::vector<uint32_t> selected(2*kSlots),locations(2*kSlots),events(2*10240),status(64);
 for(unsigned i=0;i<locations.size();++i)locations[i]=(i/kSlots)*10000+i%kSlots;
 Row rows[2]{};rows[b].slots=usedSlots;
 Context ctx{uint64_t(selected.data()),uint64_t(locations.data()),uint64_t(rows),uint64_t(posts),uint64_t(status.data())};
 ctx.autoRoute=KVIP_AUTO_ROUTE;ctx.admissionControls=uint64_t(visibleControls);
 for(unsigned row=0;row<2;++row) {
  publishedControls[row*4]=noPrefetchMode?0:512;
  publishedControls[row*4+1]=64;
  publishedControls[row*4+2]=publishedControls[row*4+3]=0;
 }
 std::copy_n(publishedControls,8,visibleControls);
 if(KVIP_AUTO_ROUTE)for(unsigned row=0;row<2;++row)
  visibleControls[row*4]=noPrefetchMode?512:0; // Previous invocation chose the opposite route.
 uint64_t bindings[25]{};uint32_t owners[]{0,1};std::vector<int64_t> logical(2*131328);
 int64_t unpack[8]{},valid[8]{};
 bindings[3]=uint64_t(logical.data());bindings[9]=uint64_t(owners);bindings[24]=131328;
 if(outputEnabled){bindings[20]=uint64_t(unpack);bindings[21]=uint64_t(valid);bindings[22]=uint64_t(logicalOut);bindings[23]=uint64_t(slotOut);}
 std::vector<int32_t> packed(4*(9*2*kTop+32+64));
 Args args{};args.context=uint64_t(&ctx);args.flags=uint64_t(flags);args.updateBindings=uint64_t(bindings);
 args.eventsSrc=args.eventsDst=uint64_t(events.data());args.maxEvents=10240;args.prefetchCount=1U<<16;args.packedFetch=uint64_t(packed.data());
 auto f=flags+b*kFlagCount;f[0].ready=1;f[0].error=initialError;
 for(unsigned q=0;q<4;++q){f[kRawQueryFlag+q].ready=q!=missingRaw;f[kRawQueryFlag+q].count=queryCount[q];}
 OutputWorker(&args,b);
}
int main(){
 static_assert(kFlagCount==16);
 static_assert(kEarlyFetchReady==bool(KVIP_EARLY_FETCH_READY && !V9_UBMEM));
 for(bool low:{false,true})for(unsigned used:{0U,8U})for(unsigned b:{0U,1U}){
  noPrefetchMode=low;usedSlots=used;
  reset();runWorker(b);assert(fetchReads==4 && outputCopies==8 && dma.empty());
  assert(controlRefreshes==0); // Publication no longer reads routing controls.
  for(unsigned q=0;q<4;++q){assert(flags[b*kFlagCount+q+1].ready);assert(slotOut[(b*4+q)*kTop]==b*10000+q*kTop);}
  assert(!JoinProducer(flags+b*kFlagCount));assert(!JoinConsumer(flags+b*kFlagCount));assert(metadataCommits==1);
  auto error=flags[b*kFlagCount+kOutputCompletionFlag].error;ProducerDone(flags+b*kFlagCount,77);
  assert(!kEarlyFetchReady || flags[b*kFlagCount+kOutputCompletionFlag].error==error);
 }
 // Q-ready alone cannot authorize metadata commit or producer finalization.
 if(kEarlyFetchReady){
  reset();flags[4].ready=1;timeoutClock=true;
  assert(JoinConsumer(flags)==8 && metadataCommits==0);assert(JoinProducer(flags)==8);
  flags[5].ready=1;flags[5].error=32;
  assert(JoinConsumer(flags)==32 && metadataCommits==0);assert(JoinProducer(flags)==32);
 }
 reset();initialError=2;runWorker(0);assert(fetchReads==0 && outputCopies==0);
 assert(JoinProducer(flags)==2 && JoinConsumer(flags)==2 && metadataCommits==0);
 // A later query packing error must drain successful prior outputs, wake every
 // remaining Get reader, and fail the output-completion join.
 reset();packErrorQuery=1;runWorker(0);assert(fetchReads==1 && outputCopies==2 && dma.empty());
 for(unsigned q=1;q<4;++q)assert(flags[q+1].ready && flags[q+1].error==16);
 assert(JoinConsumer(flags)==16 && !metadataCommits);
 reset();missingRaw=1;timeoutClock=true;runWorker(0);
 assert(fetchReads==1 && outputCopies==2);assert(JoinConsumer(flags)==8 && !metadataCommits);
 reset();outputEnabled=false;runWorker(0);assert(fetchReads==4 && outputCopies==0);assert(!JoinConsumer(flags));
 // Producer input failure must leave the early output-done cache line solely
 // owned by OutputWorker. Initial raw/Q readers still get a terminal error.
 reset();activeBatch=0;uint32_t status[32]{};Context ctx{};ctx.rowStatus=uint64_t(status);
 Args args{};args.flags=uint64_t(flags);ProducerInputError(&args,&ctx,0,64);
 assert(flags[0].ready && flags[4].ready && flags[15].ready);
 assert(kEarlyFetchReady ? !flags[5].ready : flags[5].ready);
 // The actual lifecycle flag fill includes the reused slot and clears stale
 // success/error from the prior invocation, without touching the other batch.
 flags[5].ready=1;flags[5].error=123;flags[kFlagCount+5].ready=1;
 InitializeFlags(&args,0);assert(!flags[5].ready && !flags[5].error && flags[kFlagCount+5].ready);
}
'''
    for name, replacement in {
        'FINAL_POST_HEADER': str(root / 'csrc/operators/common/lifetch/final_post.h'),
        'PUBLISH_OUTPUTS': publish,
        'OUTPUT_WORKER': worker,
        'PRODUCER_ERROR': input_error,
        'PRODUCER_JOIN': producer_join,
        'CONSUMER_JOIN': consumer_join,
        'PRODUCER_DONE': producer_done,
        'INITIALIZE_FLAGS': initialize_flags,
    }.items():
        marker = name if name == 'FINAL_POST_HEADER' else f'/*{name}*/'
        program = program.replace(marker, replacement)
    source, exe = tmp_path / 'publication.cpp', tmp_path / 'publication'
    source.write_text(program)
    subprocess.run([
        shutil.which('clang++') or 'c++', '-std=c++17', '-O1',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        f'-DV9_UBMEM={ubmem}', f'-DKVIP_EARLY_FETCH_READY={early}',
        f'-DKVIP_READY_BATCH_FETCH={ready_batch}',
        f'-DKVIP_LOW_MISS_TAIL={low_tail}',
        '-DKVIP_LOW_MISS_SKIP=1', '-DKVIP_ADAPTIVE_ADMISSION=1',
        f'-DKVIP_AUTO_ROUTE={int(bool(low_tail and early and ready_batch and not ubmem))}',
        '-I', str(kernel_dir), '-I', str(kernel_dir.parent / 'include'),
        str(source), '-o', str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True, timeout=20)
