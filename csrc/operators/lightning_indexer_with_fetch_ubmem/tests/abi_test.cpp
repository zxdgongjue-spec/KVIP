#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstddef>
#include <vector>
#include "api.h"
#include "context.h"
namespace {std::vector<unsigned> calls;unsigned fail=0;}
#define LAUNCH(name,number) extern "C" uint32_t aclrtlaunch_kvip_ubmem_##name##_kernel(uint32_t,void*,void*) { \
 calls.push_back(number);return number==fail?71:0;}
LAUNCH(init,1)
LAUNCH(li,2)
LAUNCH(resolve,3)
LAUNCH(complete,4)
LAUNCH(direct_update_owned,5)
LAUNCH(direct_update_outputs,6)
LAUNCH(direct_apply_owned,7)
int main() {
    using namespace rotating;
    static_assert(sizeof(Context)==240 && sizeof(Row)==1024 && sizeof(UbmemRegion)==128);
    static_assert(offsetof(Row,error)==11*4 && offsetof(Row,pending)==211*4 && offsetof(Row,finalPending)==248*4);
    static_assert(offsetof(Context,ubmemRegion)==128 && offsetof(Context,ubmemStatus)==136 && offsetof(Context,lruSlots)==144);
    assert(kvip_ubmem_abi_version()==1 && kvip_ubmem_context_bytes()==240 && kvip_ubmem_region_bytes()==128);
    void* pointer=reinterpret_cast<void*>(1);
    assert(kvip_ubmem_forward(pointer,pointer,pointer)==0);
    assert((calls==std::vector<unsigned>{1,2,3,4,5,6,7}));
    calls.clear();fail=4;
    assert(kvip_ubmem_forward(pointer,pointer,pointer)==71);
    assert((calls==std::vector<unsigned>{1,2,3,4}));
    calls.clear();assert(kvip_ubmem_forward(nullptr,pointer,pointer)!=0 && calls.empty());
    for(uint32_t e=1;e<kRounds;++e)assert(Trigger(e)>Trigger(e-1));
    assert(Trigger(kRounds-1)==SIM_HISTORY/128);
}
