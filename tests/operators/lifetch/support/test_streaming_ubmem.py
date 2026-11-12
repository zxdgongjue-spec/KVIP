"""Execute the actual UBmem Q worker with deterministic transport/clock stubs."""
from pathlib import Path
import subprocess


def test_query_copy_completion_and_failure_publication(tmp_path):
    root = Path(__file__).resolve().parents[4]
    header = root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/staged_ubmem_fetch.h'
    sdk = tmp_path / 'ubmem'
    sdk.mkdir()
    (sdk / 'fused_batch_get_device.h').write_text('')
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __gm__
constexpr uint32_t kFlagCount=16,kResolveCores=3,kSlots=8192,kTop=2048;
constexpr uint32_t kUbmemInvalidRegion=1,kUbmemIncomplete=2,kUbmemInvalidAddress=4;
struct BatchFlag {uint32_t ready=0,count=0,error=0,pad[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;uint64_t head=0;};
struct Shape {uint32_t batch=2,resident=16384,finalPostOffset=0,tokenBytes=656;};
struct Context {uint64_t fallbackAddresses=1,destinations=2,rowOwners=0,outputStride=0;};
struct Args {uint64_t context,flags,times=0;uint32_t prefetchCount=0;};
struct Region {uint64_t peerBase=1,mappedBase=2,bytes=3,outputBase=4,outputBytes=5;};
Region region;FinalPost posts[8];BatchFlag* flags;FinalPost* awaited=nullptr;
bool fail=false,invalid=false;uint32_t calls=0,waits=0;uint64_t cycle=0;
Shape LoadRuntimeShape(Context*){return {};}
Region* UbmemRegionAt(Context*){return &region;}
bool UbmemRegionValid(Region*,uint32_t){return !invalid;}
FinalPost* FinalPostAt(uint32_t,Context*,uint32_t i){return posts+i;}
uint64_t GetSystemCycle(){return ++cycle;}
void SyncFetchDescriptors(Context*,uint32_t,uint32_t){}
struct TPipe {template<class T>void InitBuffer(T&,unsigned){} void Destroy(){}};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {template<class T>int Get(){return 0;}};
enum class HardEvent {V_S};constexpr int EVENT_ID0=0;
template<HardEvent>void SetFlag(int){} template<HardEvent>void WaitFlag(int){}
void PublishQueryCompletion(Context*,BatchFlag* f,uint32_t batch,uint32_t,uint32_t,uint32_t q,
 uint32_t lane,uint32_t workers,uint64_t error,uint32_t){
 for(unsigned b=lane;b<batch;b+=workers){f[b*kFlagCount+8+q].ready=1;f[b*kFlagCount+8+q].error=error;}}
namespace Simt {struct Dim3 {int x,y,z;};template<auto f,class... A>void VF_CALL(Dim3,A... a){f(a...);}}
namespace kvaf_device {
void CleanCacheLines(uint8_t* ptr,uint64_t){
 if(ptr==(uint8_t*)awaited && !awaited->head){++waits;if(waits==3)awaited->head=1;}}
struct UbmemBatchGetDeviceResult{uint32_t scheduled=0,completed=0,error=0;};
struct UbmemBatchGetDeviceArgs{uint64_t* addresses;int32_t* output_indices;
 uint64_t peer_base,mapped_base,region_bytes,output_base,output_bytes;
 uint32_t begin,count,owner_base,resident,worker,workers,token_bytes;};
UbmemBatchGetDeviceResult KvafBatchGetDevice(UbmemBatchGetDeviceArgs a,int){
 assert(!awaited || awaited->head);assert(a.workers==1 && a.worker==0);
 assert(a.token_bytes==656 && a.count==2);++calls;
 return {a.count,fail?0:a.count,fail?4U:0U};}
}
''' + f'\n#include "{header}"\n' + r'''
void reset(){calls=waits=0;fail=invalid=false;awaited=nullptr;for(auto &p:posts)p={};}
int main(){
 Context c;BatchFlag f[32];flags=f;Args a{uint64_t(&c),uint64_t(f)};
 for(auto &x:f)x.ready=1;
 reset();posts[1].count=2;posts[1].descriptor=kTop;awaited=posts;
 FetchQueryGroup(&a,1,0,2);
 assert(waits==3 && calls==1 && posts[1].head && posts[1].pending==2 && !f[9].error);
 // Empty Q still releases successors and acknowledges completion.
 reset();posts[2].descriptor=2*kTop;posts[1].head=1;
 FetchQueryGroup(&a,2,0,2);assert(!calls && posts[2].head && f[10].ready);
 // Transport failure is propagated; incomplete data is not marked successful.
 reset();posts[0].count=2;fail=true;FetchQueryGroup(&a,0,0,2);
 assert(calls==1 && posts[0].head && posts[0].pending==0 && f[8].error);
 // Invalid descriptors never reach transport, but wake the next Q.
 reset();posts[0].descriptor=1;FetchQueryGroup(&a,0,0,2);
 assert(!calls && posts[0].head && f[8].error);
 // A spare worker must not access a request/region/queue.
 reset();FetchQueryGroup(&a,0,2,4);assert(!calls);
}
'''
    cpp = tmp_path / 'worker.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'worker'
    subprocess.run(['c++', '-std=c++17', '-I', str(tmp_path), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=5)
