// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef V9_COMPACT_RESERVED
#define V9_COMPACT_RESERVED 0
#endif
static_assert(V9_COMPACT_RESERVED==0 || V9_COMPACT_RESERVED==1,
              "Compact reserved-hot construction requires an explicit boolean");

#if V9_COMPACT_RESERVED && defined(__CCE_AICORE__)
namespace rotating {
using namespace AscendC;

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ScatterReservedHot(__gm__ uint32_t* output,
    const __gm__ uint32_t* candidates,uint32_t resident,uint32_t count) {
    for(uint32_t ordinal=threadIdx.x;ordinal<count;ordinal+=256) {
        uint32_t slot=asc_ldcg(candidates+ordinal);
        // Service LRU candidates are unique in-range slots. Keep malformed
        // input from writing outside the row; the decode oracle checks the map.
        if(slot<resident)asc_stcg(output+slot,ordinal);
    }
    asc_threadfence();
}

__aicore__ inline void RebuildReservedHot(uint64_t candidates,uint64_t reserved,
    uint64_t queryEnds,uint32_t batch,uint32_t resident,uint32_t count,
    LocalTensor<uint32_t> local) {
    if(!candidates || !reserved)return;
    if(queryEnds) {
        auto ends=reinterpret_cast<const __gm__ uint32_t*>(queryEnds)+batch*4;
        if(!(ends[0]|ends[1]|ends[2]|ends[3]))return;
    }
    // One init AIV owns the complete packed row. The clear must complete
    // before scatter; sharding these writes would require a global barrier.
    auto row=reinterpret_cast<__gm__ uint32_t*>(reserved)+uint64_t(batch)*resident;
    GlobalTensor<uint32_t> output;output.SetGlobalBuffer(row);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for(uint32_t offset=0;offset<resident;offset+=4096) {
        uint32_t words=resident-offset;if(words>4096)words=4096;
        DataCopy(output[offset],local,words);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Simt::VF_CALL<ScatterReservedHot>(Simt::Dim3{256,1,1},row,
        reinterpret_cast<const __gm__ uint32_t*>(candidates)+uint64_t(batch)*count,resident,count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
}
}
#endif
