"""Execute the real safe-submit body against deterministic host queue stubs."""
import subprocess
from pathlib import Path


def test_safe_ranges_are_posted_for_all_requests_before_any_cq_drain(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    start = source.index('__aicore__ inline void SubmitSafeQuery(')
    end = source.index('\n}', start) + 2
    body = source[start:end]
    code = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __gm__
#define V9_SIMULATION 0
constexpr unsigned kFlagCount=12,kSlots=8192,kTop=2048;
constexpr uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC=123;
struct BatchFlag {uint32_t ready=0,count=0,error=0,reserved[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;};
struct KvafBatchGetQueueState {uint64_t pos=0;};
struct KvafBatchGetDirectUrmaEntity {uint32_t sq_depth=32768,cq_depth=1024;};
struct KvafBatchGetCoreStatus {uint64_t magic=123,processed_request_count=0,result=0;};
struct DeviceArgs {uint64_t indices=0,offsets=0,output_indices=0,miss_count=0,address_table=0,valid_table=0;
 uint64_t queue_state=0,channel_entity=0,core_status=0;uint32_t request_capacity=0,table_capacity=0;
 uint32_t batch_get_max_inflight_total=0,jettys_per_core=1,token_bytes=656;};
struct Context {uint64_t sourceBlocks=1,sourceOffsets=1,destinations=1,count=1,fallbackAddresses=1,fallbackValid=1;};
struct Shape {uint32_t batch=2,finalWorkers=4,finalPostOffset=0;};
struct Args {uint64_t context=0,flags=0,times=0;uint32_t prefetchCount=0;};
Shape LoadRuntimeShape(Context*){return {};}
FinalPost posts[8];DeviceArgs deviceArgs[10];std::vector<int> event_log;
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t i){return posts+i;}
unsigned FinalQueue(unsigned batch,unsigned workers,unsigned row,unsigned q){return q==0?row*2+1:batch*2+row*(workers-1)+q-1;}
DeviceArgs GetArgs(Context*,unsigned i){return deviceArgs[i];}
void SyncFetchDescriptors(Context*,unsigned,unsigned){}
uint64_t GetSystemCycle(){return 100;}
namespace kvaf_device {
constexpr uint64_t kCqTimeout=1,kInvalidArgs=2;
void CleanCacheLines(uint8_t*,uint64_t){}
void KvafBatchGetDevice(DeviceArgs args,unsigned,unsigned,unsigned){auto s=(KvafBatchGetCoreStatus*)args.core_status;s->magic=123;s->result=0;s->processed_request_count=0;}
void ConfigureUbToGmCopy(){}
void PostDirectRequests(int*,int*,int*,uint64_t*,unsigned begin,unsigned count,unsigned,unsigned,unsigned,
 KvafBatchGetDirectUrmaEntity*,KvafBatchGetQueueState* queue,unsigned,uint64_t*,uint64_t*){
 event_log.push_back(begin);queue->pos+=count;}
}
''' + body + r'''
int main(){
 Context ctx;BatchFlag flags[24];Args a{uint64_t(&ctx),uint64_t(flags),0,0};
 KvafBatchGetQueueState queues[10];KvafBatchGetDirectUrmaEntity entities[10];KvafBatchGetCoreStatus statuses[10];
 for(int i=0;i<10;++i){deviceArgs[i].queue_state=uint64_t(&queues[i]);deviceArgs[i].channel_entity=uint64_t(&entities[i]);deviceArgs[i].core_status=uint64_t(&statuses[i]);}
 for(int b=0;b<2;++b){flags[b*12+1].ready=1;posts[b*4].count=posts[b*4].nonconflicting=3;posts[b*4].descriptor=b*8192;}
 SubmitSafeQuery(&a,0,0);SubmitSafeQuery(&a,1,0);
 assert((event_log==std::vector<int>{0,8192}));
 assert(flags[8].reserved[0]==1 && flags[20].reserved[0]==1);
 assert(flags[8].ready==0 && flags[20].ready==0);
 event_log.clear();deviceArgs[1].batch_get_max_inflight_total=64;
 SubmitSafeQuery(&a,0,0);
 assert(event_log.empty() && flags[8].reserved[0]==2);
}
'''
    cpp = tmp_path / 'query.cpp'
    cpp.write_text(code)
    exe = tmp_path / 'query'
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
