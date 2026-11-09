#pragma once
#include "kernel_operator.h"

// Cache visibility primitive used by the original rotating controller. There
// are no queue, network, or URMA entities in this operator.
namespace ubmem_device {
using namespace AscendC;
__aicore__ inline void CleanCacheLines(__gm__ uint8_t* address,uint64_t bytes) {
    if(!address || !bytes)return;
    uint64_t first=reinterpret_cast<uint64_t>(address)&~uint64_t(63);
    uint64_t last=(reinterpret_cast<uint64_t>(address)+bytes-1)&~uint64_t(63);
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(first));
    for(uint64_t offset=0;offset<=last-first;offset+=64)
        DataCacheCleanAndInvalid<uint8_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(global[offset]);
}
}
