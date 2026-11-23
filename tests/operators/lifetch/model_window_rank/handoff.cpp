// SPDX-License-Identifier: Apache-2.0
// Measure the handoff from four completed per-Q window lists to the first union read.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "window_handoff.h"
#endif
using namespace AscendC;

extern "C" __global__ __aicore__ void handoff_probe(GM_ADDR input, GM_ADDR exchange, GM_ADDR outputs,
                                                    uint32_t count, uint32_t variant) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__CCE_AICORE__)
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 16384);
        auto words = storage.Get<uint32_t>();
        auto source = words[0], lists = words[512];
        auto metadata = words[1024].ReinterpretCast<uint64_t>();
        auto cfg = words[1280], validated = words[1536], stamps = words[2048], staged = words[2080];
        GlobalTensor<uint32_t> in, workspace, out;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(input));
        workspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(exchange));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputs));
        for (uint32_t k = 0; k < count; ++k) {
            auto row = in[k * 528];
            auto counts = row[8];
            auto jobs = workspace[256];
            auto globalMeta = workspace;
            auto dest = out[k * 528];
            uint32_t numRows = row.GetValue(0), inputCounts[8];
            for (uint32_t q = 0; q < 4 * numRows; ++q) {
                inputCounts[q] = counts.GetValue(q);
                metadata.SetValue(q*16+3, uint64_t(inputCounts[q]));
            }
            DataCopy(source, row[16], 512);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            kvip_window::Lists local{lists, metadata};
            auto option = variant ? &local : nullptr;
            uint32_t countsOut[2][4], strides[2], widths[2];
            bool pending = false;
            auto start = GetSystemCycle();
            for (uint32_t q = 0; q < 4 * numRows; ++q) {
                if (pending) { WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); pending = false; }
                if (inputCounts[q] == 0) continue;
                if (option) {
                    kvip_window::SaveLocalList(local, q, source[q*64], inputCounts[q]);
                } else {
                    kvip_window::SaveGlobalList(jobs[q*2048], source[q*64], inputCounts[q]);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID1); pending = true;
                }
            }
            if (pending) WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID5); WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
            for (uint32_t b = 0; b < numRows; ++b)
                DataCopy(globalMeta[b*128], metadata[b*64].ReinterpretCast<uint32_t>(), 128);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID5); WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
            LocalTensor<uint32_t> observed[2];
            for (uint32_t b = 0; b < numRows; ++b) {
                const uint32_t n = kvip_window::ReadCounts(option,b,globalMeta[b*128],cfg,countsOut[b]);
                widths[b] = (n+7)/8*8;
                strides[b] = 0;
                if (n) observed[b] = kvip_window::ReadInput(option,b,0,widths[b],jobs[b*8192],
                                                            staged[b*256],strides[b]);
            }
            PipeBarrier<PIPE_ALL>();
            const auto ticks = GetSystemCycle() - start;
            stamps.SetValue(0, uint32_t(ticks)); stamps.SetValue(1, uint32_t(ticks >> 32));
            for (uint32_t b = 0; b < numRows; ++b) {
                for (uint32_t q = 0; q < 4; ++q) {
                    const uint32_t n = countsOut[b][q];
                    stamps.SetValue(8+b*4+q,n);
                    if (n && n <= 64 && strides[b] <= 64)
                        DataCopy(validated[b*256+q*64],observed[b][q*strides[b]],(n+7)/8*8);
                }
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(dest,stamps,16);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(dest[16], validated, 512);
            PipeBarrier<PIPE_ALL>();
        }
    }
#endif
}
