// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "topk.h"

namespace kvip_resident_topk {
using namespace AscendC;

constexpr uint32_t kTableOffset = 216;
constexpr uint32_t kQueries = 4;
constexpr uint32_t kShards = 2;
constexpr uint32_t kBins = 256;
constexpr uint32_t kResident = 16384;
constexpr uint32_t kHistory = 131072;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kShardTokens = kResident / kShards;

__aicore__ inline GM_ADDR HighTable(GM_ADDR workspace)
{
    if (!workspace) return nullptr;
    return reinterpret_cast<GM_ADDR>(
        *reinterpret_cast<__gm__ const uint64_t*>(workspace + kTableOffset));
}

__simd_vf__ inline void MergeHigh(__ubuf__ uint32_t* counts)
{
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> first, second;
    for (uint16_t i = 0; i < kBins; i += 64) {
        LoadAlign(first, counts + i);
        LoadAlign(second, counts + kBins + i);
        Add(first, first, second, all);
        StoreAlign(counts + i, first, all);
    }
}

__aicore__ inline void Finish(LocalTensor<uint16_t> input, LocalTensor<uint32_t> indices,
    LocalTensor<uint16_t> scores, LocalTensor<uint32_t> scratch, GM_ADDR table, uint32_t query)
{
    GlobalTensor<uint32_t> shardHistograms;
    shardHistograms.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table));
    // Preserve the native radix pass and its input traversal order. Only the
    // high-byte histogram is produced on two independent AIVs.
    DataCopy(scratch, shardHistograms[query * kShards * kBins], kShards * kBins);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    auto hist = reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    auto high = hist + kShards * kBins;
    auto low = high + kBins;
    auto cutoff = low + kBins;
    auto temp = scratch[(kShards + 2) * kBins + 64].ReinterpretCast<uint16_t>();
    auto tempIds = reinterpret_cast<__ubuf__ uint16_t*>(temp.GetPhyAddr());
    auto values = reinterpret_cast<__ubuf__ uint16_t*>(input.GetPhyAddr());
    MergeHigh(hist);
    PipeBarrier<PIPE_V>();
    topkb16gather::FindHighTargetBinVFImpl(high, cutoff, hist, kResident - kTopK + 1);
    topkb16gather::HistogramsLowVFImpl<uint16_t>(hist, values, high, kResident / 256, true);
    topkb16gather::FindKthVFImpl(cutoff, hist, high, low);
    topkb16gather::FindIdxGTOutputVFImpl(tempIds, values, 0, cutoff, kResident / 128);
    topkb16gather::FindIdxEQOutputVFImpl(tempIds, values, 0, cutoff, kResident / 128);
    topkb16gather::FindValueOutputVFImpl(
        reinterpret_cast<__ubuf__ uint16_t*>(scores.GetPhyAddr()),
        values, tempIds, kTopK / 128);
    PipeBarrier<PIPE_V>();
    Cast(indices, temp, RoundMode::CAST_NONE, kTopK);
}
}
