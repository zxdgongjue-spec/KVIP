#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstddef>
#include <vector>
#include "api.h"
#include "context.h"
namespace {std::vector<unsigned> calls, blocks;unsigned fail=0;}
#define LAUNCH(name,number) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##name##_kernel(uint32_t blockCount,void*,void*) { \
 calls.push_back(number);blocks.push_back(blockCount);return number==fail?71:0;}
#define LAUNCH_WIDTH(width,name,number) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##width##_##name##_kernel(uint32_t blockCount,void*,void*) { \
 calls.push_back(number);blocks.push_back(blockCount);return number==fail?71:0;}
LAUNCH_WIDTH(656,init,1)
LAUNCH_WIDTH(656,li,2)
LAUNCH_WIDTH(656,resolve,3)
LAUNCH_WIDTH(656,fetch,8)
LAUNCH_WIDTH(656,complete,4)
LAUNCH_WIDTH(2624,init,11)
LAUNCH_WIDTH(2624,li,12)
LAUNCH_WIDTH(2624,resolve,13)
LAUNCH_WIDTH(2624,fetch,18)
LAUNCH_WIDTH(2624,complete,14)
LAUNCH(direct_update_owned,5)
LAUNCH(direct_update_outputs,6)
LAUNCH(direct_apply_owned,7)
#define LI_PHASE(width,phase) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##width##_li_kernel_##phase(uint32_t n,void*,void*) { calls.push_back(2);blocks.push_back(n);return 0; }
LI_PHASE(656,1) LI_PHASE(656,2) LI_PHASE(656,3)
LI_PHASE(2624,1) LI_PHASE(2624,2) LI_PHASE(2624,3)
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update_656(uint32_t n,void*,void*) { calls.push_back(9);blocks.push_back(n);return fail==9?71:0; }
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update_2624(uint32_t n,void*,void*) { calls.push_back(19);blocks.push_back(n);return fail==19?71:0; }
extern "C" uint32_t aclrtlaunch_v9_counter_calibration_656(uint32_t,void*,void*) { return 0; }
int main() {
    using namespace rotating;
    static_assert(sizeof(Context)==344 && sizeof(Row)==1024 && sizeof(UbmemRegion)==128);
    static_assert(offsetof(Row,error)==11*4 && offsetof(Row,pending)==211*4 && offsetof(Row,finalPending)==248*4);
    static_assert(offsetof(Context,shape)==256);
    static_assert(offsetof(Context,thresholdBounds)==336);
    static_assert(offsetof(Context,ubmemRegion)==128 && offsetof(Context,ubmemStatus)==136 && offsetof(Context,lruSlots)==144);
    assert(kvip_ubmem_abi_version()==4 && kvip_ubmem_context_bytes()==344 && kvip_ubmem_region_bytes()==128);
    RuntimeShape shape{};
    shape.batch=SIM_BATCH;shape.history=SIM_HISTORY;shape.resident=SIM_RESIDENT;
    shape.indexHeads=INDEX_HEADS;shape.aicCount=V9_AIC_COUNT;shape.splitQ=V9_SPLIT_Q;
    shape.rounds=V9_INTERMEDIATE_FETCHES;shape.tokenBytes=TOKEN_BYTES;
    assert(PrepareShape(&shape)==0);
    assert(shape.liCores==shape.batch*(1+shape.splitQ));
    assert(shape.serviceLanes==2*shape.serviceAics);
    assert(shape.finalPostOffset==shape.batch*128+shape.finalCores*288*4);
    shape.tokenBytes=656;
    void* pointer=reinterpret_cast<void*>(1);
    if(shape.batch>16){
        assert(kvip_lifetch_streaming_forward(pointer,pointer,pointer,pointer,&shape,2)!=0);
        assert(calls.empty());
    }else{
        assert(kvip_lifetch_streaming_forward(pointer,pointer,pointer,pointer,&shape,2)==0);
        assert((calls==std::vector<unsigned>{1,2,9}));
        assert((blocks==std::vector<unsigned>{shape.batch,shape.liCores,shape.batch*3+8}));
        calls.clear();blocks.clear();shape.tokenBytes=2624;
        assert(kvip_lifetch_streaming_forward(pointer,pointer,pointer,pointer,&shape,2)==0);
        assert((calls==std::vector<unsigned>{11,12,19}));
        for(unsigned f: {1U,2U,9U}){
            calls.clear();shape.tokenBytes=656;fail=f;
            assert(kvip_lifetch_streaming_forward(pointer,pointer,pointer,pointer,&shape,2)==71);
            assert(calls.back()==fail);
        }
        calls.clear();assert(kvip_lifetch_streaming_forward(nullptr,pointer,pointer,pointer,&shape,2)!=0 && calls.empty());
    }
    for(uint32_t e=1;e<shape.rounds;++e)assert(RuntimeTrigger(shape.history,shape.resident,shape.rounds,e)>
        RuntimeTrigger(shape.history,shape.resident,shape.rounds,e-1));
    assert(RuntimeTrigger(shape.history,shape.resident,shape.rounds,0)==shape.resident/128);
    for(uint32_t tile=0;tile<=shape.history/128+1;++tile) {
        bool expected=false;
        for(uint32_t e=0;e<shape.rounds;++e)expected |= RuntimeTrigger(shape.history,shape.resident,shape.rounds,e)==tile;
        assert(IsTrigger(shape.history,shape.resident,shape.rounds,tile)==expected);
    }
}
