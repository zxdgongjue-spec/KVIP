"""Execute the actual UBmem Q worker with deterministic transport/clock stubs."""
from pathlib import Path
import subprocess


def test_query_copy_completion_and_failure_publication(tmp_path):
    root = Path(__file__).resolve().parents[4]
    header = root / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_ubmem_fetch.h'
    sdk = tmp_path / 'ubmem'
    sdk.mkdir()
    (sdk / 'fused_batch_get_device.h').write_text('')
    source = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(x)
#define __ubuf__
struct {unsigned x=0;} threadIdx;
struct {unsigned x=1;} blockDim;
void asc_syncthreads(){}
template<class T>T asc_ldcg(T* p){return *p;}
void asc_atomic_or(unsigned* p,unsigned v){*p|=v;}
void asc_atomic_add(unsigned* p,unsigned v){*p+=v;}
#define __gm__
constexpr uint32_t kFlagCount=16,kResolveCores=3,kSlots=8192,kTop=2048;
constexpr uint32_t kUbmemInvalidRegion=1,kUbmemIncomplete=2,kUbmemInvalidAddress=4;
struct BatchFlag {uint32_t ready=0,count=0,error=0,pad[29]={};};
struct FinalPost {uint32_t count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0;uint64_t head=0;};
struct Shape {uint32_t batch=2,resident=16384,finalPostOffset=0,tokenBytes=656;};
struct Context {uint64_t owners=0,fallbackAddresses=1,destinations=2,rowOwners=0,outputStride=0;};
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
template<class T>struct LocalTensor {T* p=nullptr;T GetValue(unsigned i){return p[i];} T* GetPhyAddr(){return p;} LocalTensor operator[](unsigned i){return {p+i};}};
struct TPipe {template<class T>void InitBuffer(T& b,unsigned n){b.data.resize(n/4);} void Destroy(){}};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {std::vector<uint32_t> data;template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}};
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
UbmemBatchGetDeviceResult KvafBatchGetDevice(UbmemBatchGetDeviceArgs a,LocalTensor<uint8_t>){
 if(awaited && !awaited->head)assert(a.begin==kTop);assert(a.workers==1 && a.worker==0);
 assert(a.token_bytes==656 && a.count>0);++calls;
 return {a.count,fail?0:a.count,fail?4U:0U};}
}
''' + f'\n#include "{header}"\n' + r'''
void reset(){calls=waits=0;fail=invalid=false;awaited=nullptr;for(auto &p:posts)p={};}
int main(){
 Context c;c.owners=uint64_t(posts);int32_t dest[kSlots*2]{};c.destinations=uint64_t(dest);BatchFlag f[32];flags=f;Args a{uint64_t(&c),uint64_t(f)};
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
 // Disjoint Q2 starts without waiting for Q1.
 reset();a.prefetchCount=kUbmemOverlapFetch;posts[0].count=1;dest[0]=10;
 posts[1].count=2;posts[1].descriptor=kTop;dest[kTop]=20;dest[kTop+1]=22;
 awaited=posts;FetchQueryGroup(&a,1,0,2);assert(calls==1 && waits==0 && posts[1].pending==2);
 // Safe prefix is copied first; only the adjacent-sector suffix waits.
 reset();posts[0].count=1;dest[0]=10;posts[1].count=2;posts[1].descriptor=kTop;
 dest[kTop]=20;dest[kTop+1]=11;awaited=posts;
 FetchQueryGroup(&a,1,0,2);assert(calls==2 && waits==3 && posts[1].pending==2);
 // Empty Q2 must not wait on Q1.
 reset();posts[1].descriptor=kTop;awaited=posts;
 FetchQueryGroup(&a,1,0,2);assert(!calls && !waits && posts[1].head);
 // Q3 must also account for Q1 when Q2 is empty.
 reset();posts[0].count=1;dest[0]=10;posts[1].descriptor=kTop;
 posts[2].count=1;posts[2].descriptor=2*kTop;dest[2*kTop]=11;awaited=posts;
 FetchQueryGroup(&a,2,0,2);assert(calls==1 && waits==3 && posts[2].pending==1);
 // A spare worker must not access a request/region/queue.
 reset();FetchQueryGroup(&a,0,2,4);assert(!calls);
}
'''
    cpp = tmp_path / 'worker.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'worker'
    subprocess.run(['c++', '-std=c++17', '-I', str(tmp_path), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=5)
