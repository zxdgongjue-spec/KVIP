// SPDX-License-Identifier: Apache-2.0
#include "row_kernel.h"
// A real SQ producer + CQ consumer. Only the physical CQ entries are supplied by the host.
extern "C" __global__ __aicore__ void gate_camodel(GM_ADDR argbuf, GM_ADDR results, uint32_t enabled,uint32_t offset) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
    const uint32_t idx=offset+GetBlockIdx(), q=idx%4, own=idx==0?0:(q==1?2:31+q);
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(argbuf)+idx;
    auto out=reinterpret_cast<__gm__ uint64_t*>(results)+idx*64;
    auto post=reinterpret_cast<__gm__ rotating::FinalPost*>(out+32);
    const auto cuts=rotating::PlanTerminalBatches(own,idx<4 || idx>=8?320:52,0);
    const uint32_t group=enabled&&cuts.count==4?(idx/4+q)&3U:rotating::kGateDisabled;
    const uint64_t head=93;
    out[0]=GetSystemCycle();
    rotating::CommitTerminalPost(args,head,cuts.ends[0],post,group);
    const auto err=rotating::CompleteTerminalPost(args,post,cuts,head,group);
    out[1]=GetSystemCycle();out[2]=err;out[3]=cuts.count;out[4]=group;
    for(uint32_t i=0;i<cuts.count;++i)out[5+i]=cuts.ends[i];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(out),512);
 }
#endif
}
