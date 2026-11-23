// Device cycles of the unmodified production helper; inputs/outputs DMA outside the timed span.
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "ranked_candidates.h"
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void window_rank_probe(GM_ADDR records, GM_ADDR outputs,
                                                        uint32_t count, uint32_t variant) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__CCE_AICORE__)
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, 65536);
        auto ids = buffer.Get<uint32_t>();
        auto scores = ids[2048].ReinterpretCast<uint16_t>();
        auto output = ids[3072];
        auto scratch = ids[7168];
        GlobalTensor<uint32_t> in, out;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(records));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputs));
        for (uint32_t k = 0; k < count; ++k) {
            // 8 control words, 2048 IDs, 2048 uint16 scores, 4104 bitmap words.
            auto row = in[k * 7184];
            const uint32_t valid = row.GetValue(0), rank = row.GetValue(1);
            auto map = reinterpret_cast<__gm__ uint32_t*>(records) + k * 7184 + 3080;
            const bool empty = row.GetValue(2);
            DataCopy(ids, row[8], 2048);
            DataCopy(scores, row[2056].ReinterpretCast<uint16_t>(), 2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            uint64_t begin = GetSystemCycle();
            uint32_t n;
#ifndef BASELINE_ONLY
            if (variant) {
                n = kvip_prefetch_rank::SelectWindowBitmap(ids, scores, output, scratch, map,
                    131328, valid, rank, 64, nullptr, 32, empty);
            } else
#endif
            {
                n = kvip_prefetch_rank::SelectBitmap(ids, scores, output, scratch, map,
                    131328, valid, rank, 64, nullptr, 32, empty);
            }
            PipeBarrier<PIPE_ALL>();
            uint64_t elapsed = GetSystemCycle() - begin;
            auto result = out[k * 72];
            auto stats = ids[16000];
            stats.SetValue(0, n);
            stats.SetValue(1, uint32_t(elapsed));
            stats.SetValue(2, uint32_t(elapsed >> 32));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(result, stats, 8);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(result[8], output, 64);
            PipeBarrier<PIPE_ALL>();
        }
    }
#endif
}
