#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "prefetch.h"
#include "resolver.h"
#include "completion.h"
#include "init.h"
using namespace AscendC;
using namespace rotating;
using namespace rotating::init;
#endif
extern "C" __global__ __aicore__ void kvip_ubmem_init_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {InitializeFetchCore(GetContext(bindings),GetBlockIdx());}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_resolve_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {ReplicatedFinal(GetContext(bindings));}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_resolve_plan_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {ReplicatedFinal<true>(GetContext(bindings));}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_resolve_copy_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {CopyFinalPlan(GetContext(bindings));}
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_complete_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {uint32_t b=GetBlockIdx();if(b<SIM_BATCH)CompleteBatch(GetContext(bindings),b);}
#endif
}
