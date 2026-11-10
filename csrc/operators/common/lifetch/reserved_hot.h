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

__simt_callee__ inline void ScatterReservedHot(__gm__ uint32_t* output,
    const __gm__ uint32_t* candidates,uint32_t resident,uint32_t count) {
    for(uint32_t ordinal=threadIdx.x;ordinal<count;ordinal+=256) {
        uint32_t slot=asc_ldcg(candidates+ordinal);
        // Service LRU candidates are unique in-range slots. Keep malformed
        // input from writing outside the row; the decode oracle checks the map.
        if(slot<resident)asc_stcg(output+slot,ordinal);
    }
    asc_threadfence();
}

__aicore__ inline uint64_t ClearReservedHot(uint64_t candidates,uint64_t reserved,
    uint64_t queryEnds,uint32_t batch,uint32_t resident,
    LocalTensor<uint32_t> local) {
    if(!candidates || !reserved)return 0;
    if(queryEnds) {
        auto ends=reinterpret_cast<const __gm__ uint32_t*>(queryEnds)+batch*4;
        if(!(ends[0]|ends[1]|ends[2]|ends[3]))return 0;
    }
    // Caller supplies the existing all-ones init buffer after V_MTE3, before
    // the LRU copy overwrites it. One AIV owns the complete packed row.
    // The caller must join this DMA batch with MTE3_S before reusing local or
    // launching the tail-init VF. Splitting this row across AIVs would require
    // a global barrier; returning its address does not imply DMA completion.
    auto row=reinterpret_cast<__gm__ uint32_t*>(reserved)+uint64_t(batch)*resident;
    GlobalTensor<uint32_t> output;output.SetGlobalBuffer(row);
    for(uint32_t offset=0;offset<resident;offset+=4096) {
        uint32_t words=resident-offset;if(words>4096)words=4096;
        DataCopy(output[offset],local,words);
    }
    return reinterpret_cast<uint64_t>(row);
}
}
#endif
