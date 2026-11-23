// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace lifetch_incremental {
using namespace AscendC;
// The generic TopK writes strict winners first and then equal-score winners
// in input order. If enough ties survive, new strict winners replace only
// the front of this suffix. Keep its earliest original IDs in the rest.
__simt_vf__ __aicore__ LAUNCH_BOUND(128)
inline void InsertIntoCutoffTies(__ubuf__ uint32_t* ids, __ubuf__ uint16_t* scores,
    __ubuf__ uint32_t* previous, __ubuf__ uint16_t* addedIds, __ubuf__ uint16_t* addedScores,
    uint32_t first, uint32_t aligned, uint32_t count, uint32_t base) {
    for (uint32_t i=threadIdx.x; i<2048-first; i+=128) {
        if (i<count) {
            ids[first+i]=base+addedIds[i];
            scores[first+i]=addedScores[i];
        } else {
            ids[first+i]=previous[first-aligned+i-count];
        }
    }
}

// Only the first merge since the old cutoff was loaded can use this path.
// The input packer retained only scores strictly above that cutoff.
__aicore__ inline bool MergeUnchangedCutoff(uint32_t count,uint32_t base,uint16_t cutoff,
    LocalTensor<uint32_t> ids,LocalTensor<uint16_t> scores,LocalTensor<uint32_t> previous,
    LocalTensor<uint16_t> addedIds,LocalTensor<uint16_t> addedScores) {
    if (!cutoff || count>=2048) return false;
    SetFlag<HardEvent::V_S>(EVENT_ID5); WaitFlag<HardEvent::V_S>(EVENT_ID5);
    if (scores.GetValue(2047-count)!=cutoff) return false;
    uint32_t low=0, high=2047-count;
    while(low<high) {
        const uint32_t mid=(low+high)/2;
        if (scores.GetValue(mid)==cutoff) high=mid;
        else low=mid+1;
    }
    // Stage the old ID suffix before overwriting any of its slots in parallel.
    const uint32_t aligned=low/8*8;
    DataCopy(previous,ids[aligned],2048-aligned);
    PipeBarrier<PIPE_V>();
    Simt::VF_CALL<lifetch_incremental::InsertIntoCutoffTies>(Simt::Dim3{128,1,1},
        (__ubuf__ uint32_t*)ids.GetPhyAddr(),(__ubuf__ uint16_t*)scores.GetPhyAddr(),
        (__ubuf__ uint32_t*)previous.GetPhyAddr(),(__ubuf__ uint16_t*)addedIds.GetPhyAddr(),
        (__ubuf__ uint16_t*)addedScores.GetPhyAddr(),low,aligned,count,base);
    SetFlag<HardEvent::V_S>(EVENT_ID5); WaitFlag<HardEvent::V_S>(EVENT_ID5);
    return true;
}
}
