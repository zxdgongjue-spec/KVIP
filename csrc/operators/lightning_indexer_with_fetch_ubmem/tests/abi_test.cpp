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
LAUNCH(init,1)
LAUNCH(li,2)
LAUNCH(resolve,3)
LAUNCH(complete,4)
LAUNCH(direct_update_owned,5)
LAUNCH(direct_update_outputs,6)
LAUNCH(direct_apply_owned,7)
int main() {
    using namespace rotating;
    static_assert(sizeof(Context)==336 && sizeof(Row)==1024 && sizeof(UbmemRegion)==128);
    static_assert(offsetof(Row,error)==11*4 && offsetof(Row,pending)==211*4 && offsetof(Row,finalPending)==248*4);
    static_assert(offsetof(Context,shape)==256);
    static_assert(offsetof(Context,ubmemRegion)==128 && offsetof(Context,ubmemStatus)==136 && offsetof(Context,lruSlots)==144);
    assert(kvip_ubmem_abi_version()==3 && kvip_ubmem_context_bytes()==336 && kvip_ubmem_region_bytes()==128);
    RuntimeShape shape{};
    shape.batch=SIM_BATCH;shape.history=SIM_HISTORY;shape.resident=SIM_RESIDENT;
    shape.indexHeads=INDEX_HEADS;shape.aicCount=V9_AIC_COUNT;shape.splitQ=V9_SPLIT_Q;
    shape.rounds=V9_INTERMEDIATE_FETCHES;shape.tokenBytes=TOKEN_BYTES;
    assert(PrepareShape(&shape)==0);
    assert(shape.liCores==kLiCores && shape.rounds==kRounds && shape.serviceLanes==kServiceLanes);
    assert(shape.finalWorkers==kFinalWorkers && shape.finalPostOffset==kFinalPostOffset);
    void* pointer=reinterpret_cast<void*>(1);
    assert(kvip_ubmem_forward(pointer,pointer,pointer,&shape)==0);
    assert((calls==std::vector<unsigned>{1,2,3,4,5,6,7}));
    assert((blocks==std::vector<unsigned>{shape.batch,shape.liCores,shape.finalCores,shape.batch,shape.batch,shape.batch,shape.batch*4}));
    calls.clear();fail=4;
    assert(kvip_ubmem_forward(pointer,pointer,pointer,&shape)==71);
    assert((calls==std::vector<unsigned>{1,2,3,4}));
    calls.clear();assert(kvip_ubmem_forward(nullptr,pointer,pointer,&shape)!=0 && calls.empty());
    for(uint32_t e=1;e<kRounds;++e)assert(Trigger(e)>Trigger(e-1));
    assert(Trigger(0)==SIM_RESIDENT/128);
    if(kRounds>1)assert(Trigger(kRounds-1)==SIM_HISTORY/128);
    for(uint32_t tile=0;tile<=SIM_HISTORY/128+1;++tile) {
        bool expected=false;
        for(uint32_t e=0;e<kRounds;++e)expected |= Trigger(e)==tile;
        assert(IsTrigger(tile)==expected);
    }
}
