// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
#if defined(__CCE_AICORE__)
#include "context.h"
#include "indexer/resident_topk_layout.h"
#include "indexer/topk_high_histogram.h"
#endif

extern "C" __global__ __aicore__ void v9_li_kernel_resident_topk_high(GM_ADDR bindings)
{
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        using namespace AscendC;
        using namespace kvip_resident_topk;
        auto args = reinterpret_cast<__gm__ const uint64_t*>(bindings);
        auto table = HighTable(reinterpret_cast<GM_ADDR>(args[15]));
        const auto shape = rotating::LoadRuntimeShape(
            reinterpret_cast<__gm__ rotating::Context*>(args[14]));
        // Only the opt-in dense benchmark attaches a histogram table.
        if (!table || (shape.batch != 1 && shape.batch != 8) || shape.history != kHistory ||
            shape.resident != kResident || shape.indexHeads != 32 ||
            GetBlockNum() != shape.batch * kQueries * kShards) return;

        const uint32_t task = GetBlockIdx();
        const uint32_t query = task / kShards;
        const uint32_t shard = task % kShards;
        TPipe pipe;
        TBuf<TPosition::VECCALC> inputBuffer, histogramBuffer;
        pipe.InitBuffer(inputBuffer, kShardTokens * sizeof(uint16_t));
        pipe.InitBuffer(histogramBuffer, kBins * sizeof(uint32_t));
        auto input = inputBuffer.Get<uint16_t>();
        auto histogram = histogramBuffer.Get<uint32_t>();
        GlobalTensor<uint16_t> scores;
        GlobalTensor<uint32_t> shardHistograms;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(args[12]));
        shardHistograms.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table));
        DataCopy(input, scores[uint64_t(query) * kHistory + shard * kShardTokens], kShardTokens);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        topkb16gather::HistogramsHighVFImpl<uint16_t>(
            reinterpret_cast<__ubuf__ uint32_t*>(histogram.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(input.GetPhyAddr()), kShardTokens / 256, true);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(shardHistograms[task * kBins], histogram, kBins);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        pipe.Destroy();
    }
#endif
}
#endif
