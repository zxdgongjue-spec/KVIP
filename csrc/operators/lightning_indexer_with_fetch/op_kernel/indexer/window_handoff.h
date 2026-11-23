// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"

namespace kvip_window {
using namespace AscendC;
constexpr uint32_t kListWidth = 64;
constexpr uint32_t kRowWords = 4 * kListWidth;

// Each service lane owns all four Qs of its rows until posting completes.
// Metadata remains in the same UB bank that AdvanceWindow writes to GM.
struct Lists {
    LocalTensor<uint32_t> ids;
    LocalTensor<uint64_t> metadata;
};

__aicore__ inline void SaveLocalList(Lists& lists, uint32_t query, LocalTensor<uint32_t> compact, uint32_t count) {
    PipeBarrier<PIPE_V>();
    DataCopy(lists.ids[query*kListWidth], compact, (count+7)/8*8);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SaveGlobalList(GlobalTensor<uint32_t> jobs, LocalTensor<uint32_t> compact, uint32_t count) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID5); WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
    DataCopy(jobs, compact, (count+7)/8*8);
}

__aicore__ inline uint32_t ReadCounts(Lists* local, uint32_t row, GlobalTensor<uint32_t> metadata,
    LocalTensor<uint32_t> cfg, uint32_t (&counts)[4]) {
    uint32_t maximum = 0;
    if (local == nullptr) {
        DataCopy(cfg[128], metadata, 128);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    for (uint32_t q=0; q<4; ++q) {
        counts[q] = local ? uint32_t(local->metadata.GetValue((row*4+q)*16+3)) : cfg.GetValue(128+q*32+6);
        if (counts[q]>maximum) maximum=counts[q];
    }
    return maximum;
}

// width is the logical SIMT traversal stride. A local bank has a fixed
// physical stride, preserving exactly which tokens share each ballot warp.
__aicore__ inline LocalTensor<uint32_t> ReadInput(Lists* local, uint32_t row, uint32_t start, uint32_t width,
    GlobalTensor<uint32_t> compact, LocalTensor<uint32_t> scratch, uint32_t& physicalStride) {
    if (local != nullptr) {
        physicalStride=kListWidth;
        return local->ids[row*kRowWords+start];
    }
    physicalStride=width;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5); WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    DataCopyExtParams copy;
    copy.blockCount=4; copy.blockLen=width*4; copy.srcStride=(2048-width)*4; copy.dstStride=0;
    DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
    DataCopyPad(scratch,compact[start],copy,pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5); WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    return scratch;
}
}
