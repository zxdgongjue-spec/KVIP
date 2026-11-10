#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <vector>
#include "api.h"
#include "context.h"
namespace {std::vector<unsigned> calls,blocks;unsigned fail=0;void* expectedBindings;void* expectedUpdate;void* expectedStream;}
#define LAUNCH(name,number) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##name##_kernel(uint32_t count,void*,void*) { \
 calls.push_back(number);blocks.push_back(count);return number==fail?71:0;}
LAUNCH(init,1)
LAUNCH(li,2)
LAUNCH(init_li,12)
LAUNCH(resolve,3)
LAUNCH(complete,4)
LAUNCH(direct_update_owned,5)
LAUNCH(direct_update_outputs,6)
LAUNCH(direct_apply_owned,7)
extern "C" uint32_t aclrtlaunch_kvip_ubmem_resolve_plan_kernel(uint32_t count,void* stream,void* bindings) {
    assert(bindings==expectedBindings && stream==expectedStream);
    calls.push_back(10);blocks.push_back(count);return fail==10?71:0;
}
extern "C" uint32_t aclrtlaunch_kvip_ubmem_resolve_copy_kernel(uint32_t count,void* stream,void* bindings) {
    assert(bindings==expectedBindings && stream==expectedStream);
    calls.push_back(11);blocks.push_back(count);return fail==11?71:0;
}
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_kernel(uint32_t count,void* stream,void* bindings,void* update) {
    assert(bindings==expectedBindings && update==expectedUpdate && stream==expectedStream);
    calls.push_back(8);blocks.push_back(count);return fail==8?71:0;
}
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_outputs_kernel(uint32_t count,void* stream,void* bindings,void* update) {
    assert(bindings==expectedBindings && update==expectedUpdate && stream==expectedStream);
    calls.push_back(9);blocks.push_back(count);return fail==9?71:0;
}
int main() {
    using namespace rotating;
    static_assert(sizeof(Context)==256 && sizeof(Row)==1024 && sizeof(UbmemRegion)==128);
    static_assert(offsetof(Row,mappingState)==0);
    static_assert(offsetof(Row,error)==11*4 && offsetof(Row,pending)==211*4 && offsetof(Row,finalPending)==248*4);
    static_assert(offsetof(Context,ubmemRegion)==128 && offsetof(Context,ubmemStatus)==136 && offsetof(Context,lruSlots)==144);
    assert(kvip_ubmem_abi_version()==2 && kvip_ubmem_context_bytes()==256 && kvip_ubmem_region_bytes()==128);
    assert(v9_final_copy_stages()==V9_UBMEM_FINAL_COPY_STAGES);
    assert(v9_final_copy_rows()==V9_UBMEM_FINAL_COPY_ROWS);
    assert(v9_ordered_pair()==V9_UBMEM_ORDERED_PAIR);
    assert(v9_tagged_map()==V9_UBMEM_TAGGED_MAP);
    assert(v9_merge_outputs()==V9_UBMEM_MERGE_OUTPUTS);
    assert(v9_single_plan()==V9_UBMEM_SINGLE_PLAN);
    assert(v9_staged_outputs()==V9_STAGED_OUTPUTS);
    assert(v9_fused_init()==V9_UBMEM_FUSED_INIT);
    assert(v9_ubpipe_depth()==(V9_UBMEM_GROUPED_COPY?V9_UBMEM_FINAL_COPY_STAGES:8));
    assert(v9_ubpipe_rows()==(TOKEN_BYTES==656?4:(V9_UBMEM_GROUPED_COPY?V9_UBMEM_FINAL_COPY_ROWS:1)));
    assert(v9_window_ubpipe_depth()==(V9_UBMEM_GROUPED_COPY?4:8));
    assert(v9_window_ubpipe_rows()==(TOKEN_BYTES==656?8:(V9_UBMEM_GROUPED_COPY?2:1)));
    void* pointer=reinterpret_cast<void*>(1);
    auto update=reinterpret_cast<void*>(2),stream=reinterpret_cast<void*>(3);
    expectedBindings=pointer;expectedUpdate=update;expectedStream=stream;
    assert(kvip_ubmem_forward(pointer,update,stream)==0);
    const unsigned planCall=V9_UBMEM_MERGE_OUTPUTS?9:8;
    std::vector<unsigned> forwardCalls,forwardBlocks;
    if(V9_UBMEM_FUSED_INIT){forwardCalls={12};forwardBlocks={kLaunchCores};}
    else {forwardCalls={1,2};forwardBlocks={kInitCores,kLaunchCores};}
    if(V9_UBMEM_SINGLE_PLAN){forwardCalls.push_back(10);forwardBlocks.push_back(SIM_BATCH);}
    forwardCalls.push_back(V9_UBMEM_SINGLE_PLAN?11:3);forwardBlocks.push_back(kFinalCores);
    forwardCalls.push_back(planCall);forwardBlocks.push_back(SIM_BATCH);
    if(!V9_UBMEM_MERGE_OUTPUTS){forwardCalls.push_back(6);forwardBlocks.push_back(kOutputCores);}
    forwardCalls.push_back(7);forwardBlocks.push_back(SIM_BATCH*4);
    assert(calls==forwardCalls && blocks==forwardBlocks);
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
    calls.clear();blocks.clear();
    assert(kvip_ubmem_complete_plan_outputs(pointer,update,stream)==0);
    assert((calls==std::vector<unsigned>{9}) && (blocks==std::vector<unsigned>{SIM_BATCH}));
    // A failed plan must not launch its copy or publish Index state. Test
    // every launch boundary, using the same stream in both resolve kernels.
    for(auto failed:forwardCalls) {
        calls.clear();fail=failed;
        assert(kvip_ubmem_forward(pointer,update,stream)==71);
        auto stop=std::find(forwardCalls.begin(),forwardCalls.end(),failed)+1;
        assert(calls==std::vector<unsigned>(forwardCalls.begin(),stop));
    }
    fail=0;calls.clear();blocks.clear();
    assert(kvip_ubmem_resolve_plan(pointer,stream)==0);
    assert(kvip_ubmem_resolve_copy(pointer,stream)==0);
    assert((calls==std::vector<unsigned>{10,11}) && (blocks==std::vector<unsigned>{SIM_BATCH,kFinalCores}));
    calls.clear();assert(kvip_ubmem_forward(nullptr,pointer,pointer)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(nullptr,update,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(pointer,nullptr,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan(pointer,update,nullptr)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan_outputs(nullptr,update,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan_outputs(pointer,nullptr,stream)!=0 && calls.empty());
    assert(kvip_ubmem_complete_plan_outputs(pointer,update,nullptr)!=0 && calls.empty());
    assert(kvip_ubmem_resolve_plan(nullptr,stream)!=0 && calls.empty());
    assert(kvip_ubmem_resolve_plan(pointer,nullptr)!=0 && calls.empty());
    assert(kvip_ubmem_resolve_copy(nullptr,stream)!=0 && calls.empty());
    assert(kvip_ubmem_resolve_copy(pointer,nullptr)!=0 && calls.empty());
#if V9_UBMEM_FUSED_INIT
    assert(v9_init_li(nullptr,stream)!=0 && calls.empty());
    assert(v9_init_li(pointer,nullptr)!=0 && calls.empty());
    assert(v9_init_li(pointer,stream)==0);
    assert(calls==std::vector<unsigned>{12});
#endif
    for(uint32_t e=1;e<kRounds;++e)assert(Trigger(e)>Trigger(e-1));
    assert(Trigger(kRounds-1)==SIM_HISTORY/128);
}
