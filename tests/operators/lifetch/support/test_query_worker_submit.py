"""Execute the production per-Q batching controller with host queue stubs."""
import subprocess
from pathlib import Path


def test_query_worker_batches_kvaf_calls_and_preserves_conflicts(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    begin = source.index('__aicore__ inline void FetchQueryGroup(')
    end = source.index('__aicore__ inline void FinalizeStreaming', begin)
    body = source[begin:end]
    code = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __gm__
#define V9_SIMULATION 0
constexpr uint32_t kFlagCount=12,kTop=2048,kSlots=8192,kBatchGetInflight=2048;
constexpr uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC=123,KVAF_AIV_BATCH_GET_PROFILE_MAGIC=456;
struct BatchFlag {uint32_t ready=0,count=0,error=0,reserved[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;};
struct KvafBatchGetQueueState {uint32_t x=0;};
struct Profile {uint64_t magic=0,post_active_cycles=0,drain_wait_cycles=0,submitted_group_count=0;};
struct KvafBatchGetCoreStatus {uint64_t magic=123,processed_request_count=0,result=0,completed_batch_count=0; Profile profile;};
struct DeviceArgs {uint64_t indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0,queue_state=0,core_status=0;
 uint32_t request_capacity=0,table_capacity=0,batch_get_max_inflight_total=0,jettys_per_core=1;};
struct Context {uint64_t sourceBlocks,sourceOffsets,destinations,count,fallbackAddresses=0,fallbackValid=0;};
struct Shape {uint32_t batch=4,finalWorkers=4,finalPostOffset=0,resident=16384;};
struct Args {uint64_t context=0,flags=0,times=0,packedFetch=0;uint32_t prefetchCount=0;};
Shape LoadRuntimeShape(Context*){return {};}
FinalPost posts[16];DeviceArgs deviceArgs[20];std::vector<std::vector<int32_t>> calls,offsetCalls,destCalls;
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t i){return posts+i;}
unsigned FinalQueue(unsigned batch,unsigned workers,unsigned row,unsigned q){return q==0?row*2+1:batch*2+row*(workers-1)+q-1;}
DeviceArgs GetArgs(Context*,unsigned i){return deviceArgs[i];}
void SyncFetchDescriptors(Context*,unsigned,unsigned){}
uint64_t GetSystemCycle(){return 100;}
enum class HardEvent {V_S};template<HardEvent>void SetFlag(int){} template<HardEvent>void WaitFlag(int){} constexpr int EVENT_ID0=0;
void PackFetchRanges(Context* c,int32_t* p,uint32_t stride,uint32_t* ranges,uint32_t rows){
 for(unsigned r=0;r<rows;r++){auto begin=ranges[3*r],count=ranges[3*r+1],out=ranges[3*r+2];
 for(unsigned i=0;i<count;i++){p[out+i]=((int32_t*)c->sourceBlocks)[begin+i];p[stride+out+i]=((int32_t*)c->sourceOffsets)[begin+i];p[2*stride+out+i]=((int32_t*)c->destinations)[begin+i];}}}
void PublishQueryCompletion(Context*,BatchFlag*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t){}
namespace Simt {struct Dim3{int x,y,z;};template<auto f,class... T>void VF_CALL(Dim3,T... a){f(a...);}}
namespace kvaf_device {
constexpr uint64_t kCqTimeout=6,kInvalidArgs=1;
void CleanCacheLines(uint8_t*,uint64_t){}
void WriteStatus(KvafBatchGetCoreStatus* s,uint64_t result,uint64_t processed,uint64_t completed,uint64_t,uint64_t,uint64_t,uint64_t){s->magic=123;s->result=result;s->processed_request_count=processed;s->completed_batch_count=completed;}
void KvafBatchGetDevice(DeviceArgs a,unsigned,unsigned begin,unsigned n){
 assert(a.batch_get_max_inflight_total==2048 && a.jettys_per_core==2);auto status=(KvafBatchGetCoreStatus*)a.core_status;
 calls.emplace_back((int32_t*)a.indices+begin,(int32_t*)a.indices+begin+n);
 offsetCalls.emplace_back((int32_t*)a.offsets+begin,(int32_t*)a.offsets+begin+n);
 destCalls.emplace_back((int32_t*)a.output_indices+begin,(int32_t*)a.output_indices+begin+n);
 status->magic=123;status->result=0;status->processed_request_count=n;status->completed_batch_count=1;
}
}
''' + body + r'''
int main(){
 std::vector<int32_t> src(4*kSlots),off(4*kSlots),dst(4*kSlots),packed(4*2*(9*4*kTop+32+64));
 for(unsigned i=0;i<src.size();i++){src[i]=i;off[i]=i+1;dst[i]=i+2;}
 Context c{uint64_t(src.data()),uint64_t(off.data()),uint64_t(dst.data()),0};
 BatchFlag flags[4*kFlagCount];Args a{uint64_t(&c),uint64_t(flags),0,uint64_t(packed.data()),0};
 KvafBatchGetQueueState queues[20];KvafBatchGetCoreStatus statuses[20];
 for(int i=0;i<20;i++){deviceArgs[i].queue_state=uint64_t(&queues[i]);deviceArgs[i].core_status=uint64_t(&statuses[i]);}
 for(unsigned b=0;b<4;b++){flags[b*kFlagCount+1].ready=1;flags[b*kFlagCount+7].ready=1;
  auto &p=posts[b*4];p.count=2;p.nonconflicting=1;p.descriptor=b*kSlots;}
 FetchQueryGroup(&a,0,0,2,false);
 assert(calls.size()==2);assert((calls[0]==std::vector<int32_t>{0,16384}));
 assert((calls[1]==std::vector<int32_t>{1,16385}));
 assert((offsetCalls[0]==std::vector<int32_t>{1,16385}));
 assert((destCalls[0]==std::vector<int32_t>{2,16386}));
 for(unsigned b: {0U,2U})assert(flags[b*kFlagCount+8].ready && !flags[b*kFlagCount+8].error && !posts[b*4].error);
 assert(!flags[1*kFlagCount+8].ready && !flags[3*kFlagCount+8].ready);
 FetchQueryGroup(&a,0,1,2,false);assert(calls.size()==4);
 assert((calls[2]==std::vector<int32_t>{8192,24576}));
 assert((calls[3]==std::vector<int32_t>{8193,24577}));
 // Direct producer output consumes the same descriptors without invoking packing.
 calls.clear();offsetCalls.clear();destCalls.clear();
 auto direct=packed.data()+8*(3*4*kTop+32);unsigned stride=4*kTop;
 for(unsigned phase=0;phase<2;phase++){
  auto out=direct+phase*3*stride;
  for(unsigned i=0;i<2;i++){unsigned id=i*16384+phase;out[i]=id;out[stride+i]=id+1;out[2*stride+i]=id+2;}
  direct[6*stride+phase*8]=2;
 }
 FetchQueryGroup(&a,0,0,2,true);
 assert(calls.size()==2 && (calls[0]==std::vector<int32_t>{0,16384}) && (calls[1]==std::vector<int32_t>{1,16385}));
 assert((offsetCalls[1]==std::vector<int32_t>{2,16386}));
 assert((destCalls[1]==std::vector<int32_t>{3,16387}));
 // Mismatched publication counters fail closed.
 direct[6*stride]=3;calls.clear();FetchQueryGroup(&a,0,0,2,true);assert(calls.empty());
 // Invalid producer descriptors must fail closed without calling KVAF.
 posts[0].nonconflicting=3;calls.clear();FetchQueryGroup(&a,0,0,2,false);
 assert(calls.empty() && flags[8].error==1 && flags[2*kFlagCount+8].error==1);
 // More workers than requests do not touch a queue or flag.
 calls.clear();FetchQueryGroup(&a,0,4,8);assert(calls.empty());
}
'''
    cpp, exe = tmp_path / 'query.cpp', tmp_path / 'query'
    cpp.write_text(code)
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_prefetch_does_not_reset_owned_final_status():
    root = Path(__file__).resolve().parents[4]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    helper = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/staged_fetch_worker.h').read_text()
    assert 'CompletePrefetch(ctx,b,true,profile?' in kernel
    assert 'if(initializeSpare)' in helper
    assert 'pair=2*shape.batch+2*(q*workers+lane)' in kernel
    assert 'args.jettys_per_core=2' in kernel
    worker = kernel.split('__aicore__ inline void FetchQueryGroup(', 1)[1].split(
        '__aicore__ inline void FinalizeStreaming', 1)[0]
    assert worker.count('reinterpret_cast<__gm__ uint8_t*>(queue),2*sizeof(*queue)') == 2


def test_two_jetty_pairs_do_not_overlap_prefetch_or_each_other():
    for batch in (1, 2, 4, 8, 16):
        for workers in (1, 2, 4):
            prefetch = {2 * b for b in range(batch)}
            pairs = [2 * batch + 2 * (q * workers + lane)
                     for q in range(4) for lane in range(min(workers, batch))]
            views = [v for pair in pairs for v in (pair, pair + 1)]
            capacity = max(batch * 5, 2 * batch + 8 * workers)
            assert len(views) == len(set(views))
            assert not prefetch.intersection(views)
            assert max(views) < capacity
            assert (capacity + 1) // 2 <= 48


def test_standalone_dispatch_reuses_fused_worker():
    root = Path(__file__).resolve().parents[4]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    entry = kernel.split('void v9_query_fetch_only(', 1)[1]
    assert 'FetchQueryGroup(reinterpret_cast<__gm__ Args*>(arg),q,lane,workers,skipEmptyConflict)' in entry
    assert 'uint32_t workers=2' in entry
    assert 'query==4?8U:2U' in entry


def test_native_reference_uses_kvaf_default_partition():
    root = Path(__file__).resolve().parents[4]
    kernel = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    entry = kernel.split('void v9_native_fetch_only(', 1)[1]
    assert 'args.block_count=cores;args.jettys_per_core=2' in entry
    assert 'kvaf_device::KvafBatchGetDevice(args,core)' in entry
    assert 'query==4?8U:2U' in entry
    assert entry.count('reinterpret_cast<__gm__ uint8_t*>(queues),2*sizeof(*queues)') == 2
    assert entry.count('reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status)') == 2
