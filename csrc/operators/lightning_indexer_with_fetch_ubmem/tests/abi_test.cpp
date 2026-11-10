#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstddef>
#include <vector>
#include "api.h"
#include "context.h"
namespace {std::vector<unsigned> calls,blocks;unsigned fail=0;void* expectedBindings;void* expectedUpdate;void* expectedStream;}
#define LAUNCH(name,number) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##name##_kernel(uint32_t count,void*,void*) { \
 calls.push_back(number);blocks.push_back(count);return number==fail?71:0;}
LAUNCH(init,1)
LAUNCH(li,2)
LAUNCH(resolve,3)
LAUNCH(complete,4)
LAUNCH(direct_update_owned,5)
LAUNCH(direct_update_outputs,6)
LAUNCH(direct_apply_owned,7)
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_kernel(uint32_t count,void* stream,void* bindings,void* update) {
    assert(bindings==expectedBindings && update==expectedUpdate && stream==expectedStream);
    calls.push_back(8);blocks.push_back(count);return fail==8?71:0;
}
int main() {
    using namespace rotating;
    static_assert(sizeof(Context)==256 && sizeof(Row)==1024 && sizeof(UbmemRegion)==128);
    static_assert(offsetof(Row,mappingState)==0);
    static_assert(offsetof(Row,error)==11*4 && offsetof(Row,pending)==211*4 && offsetof(Row,finalPending)==248*4);
    static_assert(offsetof(Context,ubmemRegion)==128 && offsetof(Context,ubmemStatus)==136 && offsetof(Context,lruSlots)==144);
    assert(kvip_ubmem_abi_version()==2 && kvip_ubmem_context_bytes()==256 && kvip_ubmem_region_bytes()==128);
    assert(v9_final_copy_stages()==V9_UBMEM_FINAL_COPY_STAGES);
    assert(v9_tagged_map()==V9_UBMEM_TAGGED_MAP);
    assert(v9_ubpipe_depth()==(V9_UBMEM_GROUPED_COPY?V9_UBMEM_FINAL_COPY_STAGES:8));
    assert(v9_window_ubpipe_depth()==(V9_UBMEM_GROUPED_COPY?4:8));
    void* pointer=reinterpret_cast<void*>(1);
    auto update=reinterpret_cast<void*>(2),stream=reinterpret_cast<void*>(3);
    expectedBindings=pointer;expectedUpdate=update;expectedStream=stream;
    assert(kvip_ubmem_forward(pointer,update,stream)==0);
    assert((calls==std::vector<unsigned>{1,2,3,8,6,7}));
    assert((blocks==std::vector<unsigned>{kInitCores,kLaunchCores,kFinalCores,SIM_BATCH,kOutputCores,SIM_BATCH*4}));
    calls.clear();blocks.clear();
    assert(kvip_ubmem_update_plan(pointer,pointer)==0);
    assert(kvip_ubmem_update_outputs(pointer,pointer)==0);
    assert(kvip_ubmem_update_apply(pointer,pointer)==0);
    assert((calls==std::vector<unsigned>{5,6,7}));
    assert((blocks==std::vector<unsigned>{SIM_BATCH,kOutputCores,SIM_BATCH*4}));
    calls.clear();blocks.clear();
    assert(v9_fetch(pointer,stream)==0);
    assert(kvip_ubmem_complete_plan(pointer,update,stream)==0);
    assert((calls==std::vector<unsigned>{4,8}) && (blocks==std::vector<unsigned>{SIM_BATCH,SIM_BATCH}));
    calls.clear();fail=8;
    assert(kvip_ubmem_forward(pointer,update,stream)==71);
    assert((calls==std::vector<unsigned>{1,2,3,8}));
    calls.clear();fail=6;
    assert(kvip_ubmem_forward(pointer,update,stream)==71);
    assert((calls==std::vector<unsigned>{1,2,3,8,6}));
    calls.clear();assert(kvip_ubmem_forward(nullptr,pointer,pointer)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(nullptr,update,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(pointer,nullptr,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(pointer,update,nullptr)!=0 && calls.empty());
    for(uint32_t e=1;e<kRounds;++e)assert(Trigger(e)>Trigger(e-1));
    assert(Trigger(kRounds-1)==SIM_HISTORY/128);
}
