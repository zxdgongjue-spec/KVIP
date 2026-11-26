#include "stage_trace.h"
#include "prefetch.h"
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
#include "window_frontier.h"
#endif
#if KVIP_STREAMING_LI
#include "pipeline_types.h"
#include "staged_topk_classify.h"
#endif
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "indexer_types.h"
#include "indexer_score.h"
#include "topk.h"
#if !V9_UBMEM
#include "recent_seed.h"
#include "incremental_vector_helpers.h"
#include "window_fast_vector.h"
#if KVIP_URMA_WINDOW_LI
#include "window_rank_pair.h"
#endif
#if KVIP_HOT_SCORE_PARTITION && KVIP_STREAMING_LI
#include "resident_topk_high.h"
#endif
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
#include "../../../common/lifetch/dedicated_cold_schedule.h"
#include "../../../common/lifetch/cold_score_partition.h"
#endif

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_6K = 6144;
template <typename QLIT>
class QLIVector {
public:
    rotating::RuntimeShape runtimeShape_;
#if !V9_UBMEM
#include "incremental_methods.h"
#endif
    __aicore__ inline void SetRotationContext(GM_ADDR p) {
        rotationContext_ = p;
        runtimeShape_ = rotating::LoadRuntimeShape(reinterpret_cast<__gm__ rotating::Context*>(p));
        rotationProgressGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            reinterpret_cast<__gm__ rotating::Context*>(p)->progress));
    }
    __aicore__ inline void PrepareRotationProgress() {
        // Prefix scores have been published; this 4KB output is dead until final TopK.
        PipeBarrier<PIPE_ALL>();
        rotationProgress_ = scoreOutLocal_.template ReinterpretCast<uint32_t>();
        rotationProgressBase_ = (runtimeShape_.resident / 128 / 64) * 64;
        rotating::PrepareProgress(reinterpret_cast<__ubuf__ uint32_t*>(rotationProgress_.GetPhyAddr()), rotationProgressBase_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    __aicore__ inline void BindPartitionedCold(GM_ADDR compute, uint32_t parts, bool separateService = false) {
        coldParts_ = parts;
        separateColdService_ = separateService;
        const uint32_t physical = blockId_ / 2;
        const uint32_t part = physical % parts;
        rotationCore_ = (physical / parts) * 2 + part / (parts / 2);
        coldProgress_ = compute + kvip_cold_score::ScoreBytes(runtimeShape_.batch, runtimeShape_.history);
        coldProgressGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(coldProgress_));
    }

    __aicore__ inline void BindDedicatedCold(GM_ADDR compute) {
        dedicatedProgress_ = compute + kvip_cold_score::ScoreBytes(runtimeShape_.batch, runtimeShape_.history);
        dedicatedProgressGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(dedicatedProgress_));
    }

    __aicore__ inline void SetDedicatedColdEpoch(uint32_t epoch) {
        dedicatedEpoch_ = epoch;
    }

    __aicore__ inline void RunDynamicService(uint32_t row, uint32_t epoch, uint32_t half) {
        dedicatedLane_ = row;
        AllocEventID();
        RotateFetch(epoch, half);
        FreeEventID();
    }

    __aicore__ inline void RunPartitionedWindowService(uint32_t slot, uint32_t half) {
        const uint32_t epoch = slot / runtimeShape_.serviceLanes;
        const uint32_t lane = slot % runtimeShape_.serviceLanes;
        dedicatedLane_ = lane;
        rotationCore_ = rotating::ServiceCore(runtimeShape_.batch, runtimeShape_.splitQ,
            runtimeShape_.serviceAics, epoch, lane / 2);
        AllocEventID();
        if (epoch) {
            KVIP_TRACE_BEGIN(driverWait, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
                Cold, DedicatedDriverWait, epoch, lane);
            const uint32_t part = (rotationCore_ % 2) * (coldParts_ / 2);
            const auto tiles = kvip_cold_score::Split(runtimeShape_.history, runtimeShape_.resident,
                half, epoch, part, coldParts_);
            const uint32_t end = tiles.end * kvip_cold_score::kTileTokens;
            const uint32_t firstQuery = (rotationCore_ / 2) * SIM_Q + (lane % 2) * 2;
            // Match the original rotating driver's two local Q rows.
            for (uint32_t q = firstQuery; q < firstQuery + 2; ++q) {
                while (ReadColdCursor(q, part) < end) {}
            }
        }
        RotateFetch(epoch, half);
        FreeEventID();
    }

    template <bool OwnWindow = false>
    __aicore__ inline void RunDedicatedService(uint32_t lane, uint32_t half) {
        dedicatedLane_ = lane;
        KVIP_TRACE_BEGIN(eventInit, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, LiEventsAcquire, 0, lane);
        AllocEventID();
        KVIP_TRACE_END(eventInit);
        for (uint32_t epoch = 0; epoch < runtimeShape_.rounds; ++epoch) {
            if (epoch) {
                if constexpr (OwnWindow) WaitOwnWindow(lane, epoch, half);
                else WaitDedicatedDriver(lane, epoch, half);
            }
            RotateFetch(epoch, half);
        }
        KVIP_TRACE_BEGIN(eventDrain, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, LiEventsDrain, 0, lane);
        FreeEventID();
    }

    __aicore__ inline void DrainColdPublication() {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WaitPartitionedHalf(uint32_t batch, uint32_t part, uint32_t epoch, uint32_t half) {
        KVIP_TRACE_BEGIN(partitionWaitScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, ColdPartitionWait, epoch, kvip_trace::PipeDetail(batch, (blockId_ % 2) * 2, part));
        DrainColdPublication();
        const uint32_t beginQ = batch * 4 + (blockId_ % 2) * 2;
        const uint32_t endPart = part + coldParts_ / 2;
        for (uint32_t p = part + 1; p < endPart; ++p) {
            const auto tile = kvip_cold_score::Split(runtimeShape_.history, runtimeShape_.resident, half,
                epoch, p, coldParts_);
            const uint32_t end = tile.end * kvip_cold_score::kTileTokens;
            for (uint32_t query = beginQ; query < beginQ + 2; ++query) {
                while (ReadColdCursor(query, p) < end) {}
            }
        }
    }
#endif
#if KVIP_PARALLEL_TERMINAL
    __aicore__ inline LocalTensor<uint32_t> TerminalCompact() { return incrementalNew_.template ReinterpretCast<uint32_t>(); }
    __aicore__ inline LocalTensor<uint32_t> TerminalScratch() { return topkSharedTmpLocal_; }
    __aicore__ inline LocalTensor<uint32_t> TerminalIds() { return indicesOutLocal_; }
    __aicore__ inline LocalTensor<uint32_t> TerminalWork() { return mrgValueLocal_.template ReinterpretCast<uint32_t>(); }
    __aicore__ inline void TerminalQuery(uint32_t query) {
        exportWindowCold_=true;rankTrace_=false;rankedEpoch_=runtimeShape_.rounds;
        incrementalMetadataActive_=false;incrementalHasNext_=false;
        windowCoveredEmpty_=false;
        uint32_t half=rotating::SequenceHalfWindow(runtimeShape_);
        rankHalf_=half;rankReady_=uint64_t(runtimeShape_.history)|(uint64_t(runtimeShape_.history)<<32);
        AdvanceIncremental(query,runtimeShape_.history,half?runtimeShape_.history:0,half);
        JoinColdOutput();
        if(incrementalOutputPending_){WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);incrementalOutputPending_=false;}
        PipeBarrier<PIPE_ALL>();
        exportWindowCold_=false;
    }
// Included inside the private terminal QLIVector class only.
// The production hot/cold/window QLIVector is not changed.
__aicore__ inline uint32_t TerminalQueryRuntime(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint32_t valid=ctx->queryEnds?
        reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]:runtimeShape_.history;
    auto meta=IncrementalMeta(query);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    const bool missing=valid && !meta[0];
    if(!valid || missing) {
        // An inactive graph row must still participate in the request-local
        // publication barriers, but must not consume last replay's TopK UB.
        Duplicate(indicesOutLocal_,UINT32_MAX,2048);
        meta[3]=0;
        meta[6]=0; // No compact miss payload is valid for an inactive Q.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
        PipeBarrier<PIPE_ALL>();
        return missing?1U:0U;
    }
    TerminalQuery(query);
    return 0;
}

#endif
    __aicore__ inline void RotateFetch(uint32_t epoch, uint32_t sequenceHalfWindow = 0) {
#if !V9_UBMEM
        const uint32_t core = RotationCore();
        const uint32_t serviceLane = ResolveServiceLane(core);
#if KVIP_STAGE_TRACE
        const uint64_t stageTrace = KVIP_TRACE_WORKSPACE(streamingWorkspace_);
        KVIP_TRACE_BEGIN(serviceScope, stageTrace, Cold, ServiceBody, epoch, serviceLane);
#endif
#if KVIP_SERVICE_LOCAL_COUNTS
        const bool stagedMetadata = AdvanceWindow(epoch, sequenceHalfWindow, serviceLane);
        // The same lane owns this UB until Service returns. Keep the GM
        // publication for later epochs and all existing ordering edges.
        const __ubuf__ uint64_t* windowMetadata = stagedMetadata && sequenceHalfWindow
            ? reinterpret_cast<const __ubuf__ uint64_t*>(incrementalMetadata_.GetPhyAddr()) : nullptr;
#else
        AdvanceWindow(epoch,sequenceHalfWindow,serviceLane);
#endif
#endif
        // Join this producer's hit publication. Other producers remain free.
        PipeBarrier<PIPE_ALL>();
#if defined(V9_UBMEM) && V9_UBMEM
        rotating::Service(rotationContext_, epoch, blockId_ / 2, blockId_ % 2,
#else
        rotating::Service(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,runtimeShape_.serviceLanes,runtimeShape_.tokenBytes,runtimeShape_.postThreads,runtimeShape_.postBank,runtimeShape_.batchedRows,runtimeShape_.scanTokens,rotationContext_, epoch, core, serviceLane,
#endif
#if !KVIP_STREAMING_LI
            topkSharedTmpLocal_, mrgValueLocal_, indicesOutLocal_, sequenceHalfWindow);
#else
            topkSharedTmpLocal_, mrgValueLocal_, indicesOutLocal_, sequenceHalfWindow,
            streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U)
#if !V9_UBMEM
#if KVIP_SERVICE_LOCAL_COUNTS
            , windowMetadata
#else
            , nullptr
#endif
#if KVIP_SERVICE_UB_CANDIDATES
            , windowCandidatesActive_
                ? reinterpret_cast<const __ubuf__ uint32_t*>(windowCandidates_.GetPhyAddr()) : nullptr
#endif
#endif
#if KVIP_STAGE_TRACE && !V9_UBMEM
            , stageTrace
#endif
        );
#endif
    }
    __aicore__ inline void LoadSequenceSplitThresholds(uint32_t batch, [[maybe_unused]] uint32_t epoch = 0) {
        KVIP_TRACE_BEGIN(loadScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, ColdThresholdRead, epoch, batch);
        constexpr uint32_t count = SIM_Q * 2048;
        constexpr uint32_t rowBytes = count * 6 + SIM_Q * 256;
        GlobalTensor<uint16_t> thresholds;
        thresholds.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(
            prefixOutput_ + batch * rowBytes + count * 6));
        DataCopy(prefixThreshold_, thresholds[(blockId_ % 2) * 256], 256);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;
    using QK_T = typename QLIT::qkType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info, uint32_t progressShard = 0,
#if !V9_UBMEM
                                     bool publishShardEnd = false, bool scoreOnly = false KVIP_PIPE_PARAM_DEFAULT);
#else
                                     bool publishShardEnd = false KVIP_PIPE_PARAM_DEFAULT);
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    __aicore__ inline void ProcessPrecomputedVec1(const QLICommon::RunInfo &info,
        uint32_t progressShard, bool publishShardEnd KVIP_PIPE_PARAM_DEFAULT);
#endif
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info, uint32_t queryRow = SIM_Q);
    __aicore__ inline void SetPrefixStorage(GM_ADDR output) { prefixOutput_ = output; }
#if KVIP_STREAMING_LI
    __aicore__ inline void SetStreamingWorkspace(GM_ADDR workspace) { streamingWorkspace_ = workspace; }
    __aicore__ inline void EmitFinalClassification(uint32_t query);
    __aicore__ inline void BeginFinalMetadata(uint32_t query);
#endif
    __aicore__ inline void ProcessResidentTopK(QLICommon::RunInfo info, uint32_t queryRow = SIM_Q) {
        info.actS2Size = runtimeShape_.resident;
        prefixActive_ = true;
        ProcessTopK(info, queryRow);
        prefixActive_ = false;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe, bool topkOnly);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const QLITilingData *__restrict tilingData, bool topkOnly);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID(
#if KVIP_STAGE_TRACE
        uint64_t trace = 0, uint32_t traceKernel = 0
#endif
    );

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;
    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void PublishColdTile(const QLICommon::RunInfo &info, uint32_t query,
        uint32_t queryCount, uint32_t progressShard, bool publishShardEnd);
    __aicore__ inline uint32_t RotationCore() const {
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if (coldParts_) return rotationCore_;
#endif
        return blockId_ / 2;
    }
#if !V9_UBMEM
    __aicore__ inline uint32_t ResolveServiceLane(uint32_t core) const {
#if KVIP_HOT_SCORE_PARTITION
        if (dedicatedLane_ != UINT32_MAX) return dedicatedLane_;
#endif
        return rotating::ServiceLane(runtimeShape_.batch, runtimeShape_.splitQ,
            runtimeShape_.serviceAics, core, blockId_ % 2);
    }
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    __aicore__ inline uint32_t ReadDedicatedCursor(uint32_t query, uint32_t epoch, uint32_t fragment) const {
        const auto offset = kvip_dedicated_cold::ProgressOffset(query, epoch, fragment);
        auto cursor = reinterpret_cast<__gm__ volatile uint32_t*>(dedicatedProgress_) + offset;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ uint32_t*>(cursor)),
            kvip_dedicated_cold::kProgressWords * sizeof(uint32_t));
        return *cursor;
    }

    __aicore__ inline uint64_t DedicatedColdReady(uint32_t query, uint32_t epoch, uint32_t half) const {
        KVIP_TRACE_BEGIN(readyScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, DedicatedReadySnapshot, epoch, query);
        uint32_t ends[(kvip_dedicated_cold::kRounds - 1) * kvip_dedicated_cold::kFragments] = {};
        for (uint32_t w = 1; w <= epoch; ++w) {
            for (uint32_t fragment = 0; fragment < kvip_dedicated_cold::kFragments; ++fragment) {
                ends[(w - 1) * kvip_dedicated_cold::kFragments + fragment] = ReadDedicatedCursor(query, w, fragment);
            }
        }
        const uint32_t end0 = kvip_dedicated_cold::Prefix(runtimeShape_.history, runtimeShape_.resident, half, epoch, 0, ends);
        const uint32_t end1 = kvip_dedicated_cold::Prefix(runtimeShape_.history, runtimeShape_.resident, half, epoch, 1, ends);
        return uint64_t(end0) | (uint64_t(end1) << 32);
    }

    __aicore__ inline void WaitOwnWindow(uint32_t row, uint32_t epoch, uint32_t half) const {
        KVIP_TRACE_BEGIN(scope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, DedicatedDriverWait, epoch, row);
        for (uint32_t fragment = 0; fragment < kvip_dedicated_cold::kFragments; ++fragment) {
            const auto tiles = kvip_dedicated_cold::Fragment(runtimeShape_.history, runtimeShape_.resident,
                half, epoch, fragment);
            if (tiles.begin == tiles.end) continue;
            const uint32_t end = tiles.end * kvip_dedicated_cold::kTileTokens;
            for (uint32_t q = 0; q < SIM_Q; ++q) {
                while (ReadDedicatedCursor(row * SIM_Q + q, epoch, fragment) < end) {}
            }
        }
    }

    __aicore__ inline void WaitDedicatedDriver(uint32_t lane, uint32_t epoch, uint32_t half) const {
        // Preserve the native rotating driver's request, half and Q pair,
        // even though the stable Service lane can act on a different request.
        const uint32_t driver = rotating::ServiceCore(runtimeShape_.batch, runtimeShape_.splitQ,
            runtimeShape_.serviceAics, epoch, lane / 2);
        const uint32_t shard = driver % 2;
        const uint32_t firstQuery = driver / 2 * 4 + (lane % 2) * 2;
        const uint32_t endFragment = shard ? kvip_dedicated_cold::kFragments : 1;
        KVIP_TRACE_BEGIN(driverScope, KVIP_TRACE_WORKSPACE(streamingWorkspace_),
            Cold, DedicatedDriverWait, epoch, lane);
        // The fragment ends are independent monotonic publications. The
        // driver gate only needs this half, not the rest of the batch.
        for (uint32_t fragment = shard; fragment < endFragment; ++fragment) {
            const auto range = kvip_dedicated_cold::Fragment(runtimeShape_.history, runtimeShape_.resident,
                half, epoch, fragment);
            if (range.begin == range.end) continue;
            const uint32_t limit = range.end * kvip_dedicated_cold::kTileTokens;
            for (uint32_t query = firstQuery; query < firstQuery + 2; ++query) {
                while (ReadDedicatedCursor(query, epoch, fragment) < limit) {}
            }
        }
    }

    __aicore__ inline uint32_t ReadColdCursor(uint32_t query, uint32_t part) const {
        const uint64_t offset = kvip_cold_score::ProgressOffset(query, part, coldParts_);
        auto cursor = reinterpret_cast<__gm__ volatile uint32_t*>(coldProgress_) + offset;
        kvaf_device::CleanCacheLines(
            reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ uint32_t*>(cursor)),
            kvip_cold_score::kProgressBytes);
        return *cursor;
    }

    __aicore__ inline uint64_t PartitionedColdReady(uint32_t query, uint32_t half) const {
        uint32_t cursors[kvip_cold_score::kMaxParts];
        for (uint32_t part = 0; part < coldParts_; ++part) cursors[part] = ReadColdCursor(query, part);
        const uint32_t lastEpoch = runtimeShape_.rounds - 1;
        const uint32_t end0 = kvip_cold_score::Prefix(cursors, runtimeShape_.history, runtimeShape_.resident,
            half, lastEpoch, 0, coldParts_);
        const uint32_t end1 = kvip_cold_score::Prefix(cursors, runtimeShape_.history, runtimeShape_.resident,
            half, lastEpoch, 1, coldParts_);
        return uint64_t(end0) | (uint64_t(end1) << 32);
    }
    GM_ADDR coldProgress_ = nullptr;
    uint32_t coldParts_ = 0;
    bool separateColdService_ = false;
    uint32_t rotationCore_ = 0;
    GlobalTensor<uint32_t> coldProgressGm_;
    GM_ADDR dedicatedProgress_ = nullptr;
    uint32_t dedicatedLane_ = UINT32_MAX;
    uint32_t dedicatedEpoch_ = 0;
    GlobalTensor<uint32_t> dedicatedProgressGm_;
#endif
    GM_ADDR rotationContext_ = nullptr;
#if KVIP_STREAMING_LI
    GM_ADDR streamingWorkspace_ = nullptr;
    bool streamingTopkOnly_ = false;
    bool streamingPrefetched_ = false;
    TBuf<TPosition::VECCALC> streamingValidBuf_, streamingReservedBuf_, streamingOutputBuf_, streamingBitmapBuf_;
    LocalTensor<uint32_t> streamingValid_, streamingReserved_, streamingOutput_, streamingBitmap_;
#endif
    GlobalTensor<uint32_t> rotationProgressGm_;
    LocalTensor<uint32_t> rotationProgress_;
    uint32_t rotationProgressBase_ = 0;

    GM_ADDR prefixOutput_ = nullptr;
    bool prefixActive_ = false;
    TBuf<TPosition::VECCALC> classifyBuf_;
    LocalTensor<uint16_t> classifyUB_;
    TBuf<TPosition::VECCALC> prefixThresholdBuf_;
    LocalTensor<uint16_t> prefixThreshold_;
#if !V9_UBMEM
    __aicore__ inline void MergeRecentScores(uint32_t query, uint32_t scoreRow, uint32_t availableEnd = UINT32_MAX) {
        auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        if (!ctx->tailBounds) return;
        uint32_t end = ctx->queryEnds ?
            reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query] : runtimeShape_.history;
        end = Min(end, availableEnd);
        // Fewer than K valid resident entries retain the original conservative
        // zero threshold. Padding must never create an artificial K-th score.
        if (end < topkCount_) return;
        auto range = kvip_recent_index::RecentRange(ctx, query/4, end);
        for (uint32_t begin = range.begin; begin < range.end;) {
            uint32_t count = Min(range.end - begin, trunkLen_);
            // Keep exact resident/previous IDs while the single-input TopK
            // compacts scores. Its histogram scratch starts after this bank.
            auto previous = topkSharedTmpLocal_;
            DataCopy(previous, indicesOutLocal_, topkCount_);
            DataCopy(mrgValueLocal_, scoreOutLocal_, topkCount_);
            uint32_t padded = QLICommon::Align(count, uint32_t(256));
            if (padded != count) Duplicate(mrgValueLocal_[topkCount_ + count / 256 * 256],
                                          uint16_t(0), padded - count / 256 * 256);
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
            DataCopyExtParams copy{1, count * uint32_t(sizeof(SCORE_T)), 0, 0, 0};
            DataCopyPadExtParams<SCORE_T> pad{true, 0, 0, 0};
            DataCopyPad(mrgValueLocal_[topkCount_],
                scoreGm[scoreRow * runtimeShape_.history + begin], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                    topkCount_ + padded, 0, 1, true);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Simt::VF_CALL<kvip_recent_index::Remap>(Simt::Dim3{512,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(previous.GetPhyAddr()), begin, count);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            begin += count;
        }
    }

    __aicore__ inline void SaveResident(uint32_t query, uint32_t scoreRow, bool publishProgress, uint32_t availableEnd, bool recentMerged = false) {
        if (!recentMerged) MergeRecentScores(query,scoreRow,availableEnd);
        TagResident(query);
#else
    __aicore__ inline void SaveResident(uint32_t query, uint32_t scoreRow, bool publishProgress) {
#endif
#if KVIP_STAGE_TRACE
        const uint64_t residentTrace = KVIP_TRACE_WORKSPACE(streamingWorkspace_);
        const uint32_t residentDetail = kvip_trace::PipeDetail(query / SIM_Q, query % SIM_Q);
#endif
        auto src = (__local_mem__ uint16_t *)scoreOutLocal_.GetPhyAddr();
        auto dst = (__local_mem__ uint16_t *)prefixThreshold_.GetPhyAddr() + (query % 2) * 128;
        KVIP_TRACE_BEGIN(reduceScope, residentTrace, Threshold, ResidentMinVf, 0, residentDetail);
        __VEC_SCOPE__ {
            MicroAPI::RegTensor<uint16_t> minimum, values, reduced;
            auto mask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(minimum, uint16_t(0xffff));
            for (uint16_t part = 0; part < 16; ++part) {
                MicroAPI::LoadAlign(values, src + part * 128);
                MicroAPI::Min(minimum, minimum, values, mask);
            }
            MicroAPI::ReduceMin(reduced, minimum, mask);
            MicroAPI::StoreAlign(dst, reduced, mask);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        KVIP_TRACE_END(reduceScope);
        KVIP_TRACE_BEGIN(prefixScope, residentTrace, Threshold, ThresholdOutputMte3, 0, residentDetail);
        constexpr uint32_t count = SIM_Q * 2048;
        constexpr uint32_t rowBytes = count * 6 + SIM_Q * 256;
        uint32_t batch = query / SIM_Q, q = query % SIM_Q;
        GlobalTensor<uint16_t> scores, thresholds;
        GlobalTensor<int32_t> indices;
        scores.SetGlobalBuffer((__gm__ uint16_t *)(prefixOutput_ + batch * rowBytes));
        indices.SetGlobalBuffer((__gm__ int32_t *)(prefixOutput_ + batch * rowBytes + count * 2));
        thresholds.SetGlobalBuffer((__gm__ uint16_t *)(prefixOutput_ + batch * rowBytes + count * 6));
        DataCopy(scores[q * 2048], scoreOutLocal_, 2048);
        DataCopy(indices[q * 2048], indicesOutLocal_.template ReinterpretCast<int32_t>(), 2048);
        DataCopy(thresholds[q * 128], prefixThreshold_[(query % 2) * 128], 128);
        PipeBarrier<PIPE_ALL>();
        KVIP_TRACE_END(prefixScope);
        auto context = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        if (context->lruSlots) {
            // TopK scratch is dead after the resident result is saved. Reuse
            // 16KiB: 4KiB scores, 8KiB victim IDs, 4KiB predicate scratch.
            auto scoreChunk = topkSharedTmpLocal_.template ReinterpretCast<uint16_t>();
            auto reservedChunk = scoreChunk[2048];
            auto hitChunk = scoreChunk[6144];
            GlobalTensor<uint32_t> victims;
            victims.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(context->lruSlots) + batch * 2048);
            GlobalTensor<uint8_t> bitmap;
            bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(context->hits) + query * (runtimeShape_.history / 4));
            uint32_t validEnd = context->queryEnds ?
                reinterpret_cast<__gm__ uint32_t*>(context->queryEnds)[query] : runtimeShape_.history;
            if (validEnd > runtimeShape_.resident) validEnd = runtimeShape_.resident;
            KVIP_TRACE_BEGIN(victimScope, residentTrace, Threshold, ResidentVictimIdsMte2, 0, residentDetail);
            DataCopy(reservedChunk.template ReinterpretCast<uint32_t>(), victims, 2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            KVIP_TRACE_END(victimScope);
            KVIP_TRACE_BEGIN(classifyScope, residentTrace, Threshold, ResidentVictimClassifyVf, 0, residentDetail);
#if !V9_UBMEM
            // One SIMT invocation covers both paths; zero selects GM only.
            Simt::VF_CALL<rotating::GatherVictimScoresUb>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(reservedChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scoreChunk.GetPhyAddr()),
                reinterpret_cast<const __ubuf__ uint16_t*>(mrgValueLocal_.GetPhyAddr()),
                reinterpret_cast<const __gm__ uint32_t*>(scoreGm.GetPhyAddr()) + scoreRow *
                    QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) / 2,
                recentMerged && streamingWorkspace_ ? runtimeShape_.resident : 0);
#else
            Simt::VF_CALL<rotating::GatherVictimScores>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(reservedChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scoreChunk.GetPhyAddr()),
                reinterpret_cast<const __gm__ uint32_t*>(scoreGm.GetPhyAddr()) + scoreRow *
                    QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) / 2);
#endif
            PipeBarrier<PIPE_V>();
            vector1::ClassifyResidentScores(
                reinterpret_cast<__ubuf__ uint16_t*>(scoreChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(reservedChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_[(query % 2) * 128].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(hitChunk.GetPhyAddr()), validEnd);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            KVIP_TRACE_END(classifyScope);
            KVIP_TRACE_BEGIN(hitScope, residentTrace, Threshold, ResidentHitBitmapMte3, 0, residentDetail);
            DataCopyExtParams copy;
            copy.blockCount = 16; copy.blockLen = 32;
            copy.srcStride = 7; copy.dstStride = 0;
            DataCopyPad(bitmap, hitChunk.template ReinterpretCast<uint8_t>(), copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
#if !V9_UBMEM
        // Private incremental TopK state, shared by all prefetch windows.
        KVIP_TRACE_BEGIN(windowScope, residentTrace, Threshold, ResidentWindowSeed, 0, residentDetail);
        auto windowMeta=reinterpret_cast<__gm__ uint64_t*>(rotating::TopKHeader(context))+query*16;
        windowMeta[0]=reinterpret_cast<uint64_t>(scoreGm.GetPhyAddr())+uint64_t(scoreRow)*runtimeShape_.history*2;
        windowMeta[1]=uint64_t(runtimeShape_.resident)|(uint64_t(runtimeShape_.resident)<<32);windowMeta[2]=0;windowMeta[3]=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(windowMeta),128);
        auto privatePrefix=reinterpret_cast<GM_ADDR>(context->incremental)+runtimeShape_.batch*512+batch*rowBytes;
        GlobalTensor<uint16_t> windowScores;GlobalTensor<uint32_t> windowIds;
        windowScores.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(privatePrefix)+q*2048);
        windowIds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(privatePrefix+count*2)+q*2048);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopy(windowScores,scoreOutLocal_,2048);DataCopy(windowIds,indicesOutLocal_,2048);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
        KVIP_TRACE_END(windowScope);
#endif
        if (publishProgress) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto ready = reinterpret_cast<__gm__ uint32_t*>(context->progress)+query*32;
            ready[0]=runtimeShape_.resident;
#if defined(V9_UBMEM) && V9_UBMEM
            ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),128);
#else
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ready),128);
#endif
        }
    }

    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    struct QLICommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe, bool topkOnly)
{
#if KVIP_STREAMING_LI
    streamingTopkOnly_ = topkOnly;
#endif
    if (!topkOnly) {
        // Fixpipe writes to the start of UB on its paired AIVs.
        pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));
        resMm1UB_ = resMm1Buf_.Get<QK_T>();
    }
    pipe->InitBuffer(prefixThresholdBuf_, 512);
    prefixThreshold_ = prefixThresholdBuf_.Get<uint16_t>();
    if (!topkOnly) {
        pipe->InitBuffer(classifyBuf_, 1024);
        classifyUB_ = classifyBuf_.Get<uint16_t>();
        pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
        weightUB_ = weightBuf_.Get<W_T>();
        pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
        weightFloatUB_ = weightFloatBuf_.Get<float>();
        pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * QLICommon::KScaleTiles * sizeof(SCALE_T));
        kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
        pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
        qScaleUB_ = qScaleBuf_.Get<SCALE_T>();
        pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
        vec1OutUB_ = outBuf_.Get<SCORE_T>();
    }

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);

#if KVIP_STREAMING_LI
    // Keep every original TopK buffer at its original UB offset. The extra
    // classification buffers occupy only the remaining space after TopK.
    if (topkOnly && streamingWorkspace_) {
        pipe->InitBuffer(streamingValidBuf_, 8192 * sizeof(uint32_t));
        pipe->InitBuffer(streamingReservedBuf_, 8192 * sizeof(uint32_t));
        pipe->InitBuffer(streamingOutputBuf_, 2048 * sizeof(uint32_t));
        pipe->InitBuffer(streamingBitmapBuf_, (rotating::kBitmapShards * rotating::kBitmapWords + 8) * sizeof(uint32_t));
        streamingValid_ = streamingValidBuf_.Get<uint32_t>();
        streamingReserved_ = streamingReservedBuf_.Get<uint32_t>();
        streamingOutput_ = streamingOutputBuf_.Get<uint32_t>();
        streamingBitmap_ = streamingBitmapBuf_.Get<uint32_t>();
    }
#endif

#if !V9_UBMEM
    pipe->InitBuffer(incrementalIdsBuf_, trunkLen_*2 + ((topkOnly && streamingWorkspace_) ? 0 : 2048*4) + 1024);
    incrementalNew_ = incrementalIdsBuf_.Get<uint16_t>();
    incrementalOld_ = (topkOnly && streamingWorkspace_) ? streamingOutput_ : incrementalNew_[trunkLen_].template ReinterpretCast<uint32_t>();
    incrementalMetadata_ = incrementalNew_[trunkLen_ + ((topkOnly && streamingWorkspace_) ? 0 : 4096)].template ReinterpretCast<uint64_t>();
    if (!topkOnly) {
        uint32_t used=2*CeilDiv(constInfo_.mBaseSize,2)*s2BaseSize_*sizeof(QK_T)+1536+6*CeilDiv(s1BaseSize_,2)*UB_BANK_DEPTH_STRIDE+2*s2BaseSize_*QLICommon::KScaleTiles*sizeof(SCALE_T)+2*CeilDiv(s1BaseSize_,2)*s2BaseSize_*sizeof(SCORE_T)+(topkCountAlign256_+trunkLen_)*sizeof(SCORE_T)+(topkCountAlign256_+64)*4+topkCountAlign256_*sizeof(SCORE_T)+topkSharedTmpSize+trunkLen_*2+8192+1024;
        incrementalPipeline_=used+kIncrementalInputBytes<=223232;
        if(incrementalPipeline_){pipe->InitBuffer(incrementalInputBuf_,kIncrementalInputBytes);incrementalInput_=incrementalInputBuf_.Get<uint16_t>();}
#if KVIP_SERVICE_UB_CANDIDATES
        constexpr uint32_t candidateBytes = 4 * rotating::kWindowCandidatePitch * sizeof(uint32_t);
        if (used + (incrementalPipeline_ ? kIncrementalInputBytes : 0) +
                candidateBytes <= 223232) {
            pipe->InitBuffer(windowCandidatesBuf_, candidateBytes);
            windowCandidates_ = windowCandidatesBuf_.Get<uint32_t>();
            windowCandidatesAllocated_ = true;
        }
#endif
    }
#endif
    if (!topkOnly) Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * QLICommon::KScaleTiles);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const QLITilingData *__restrict tilingData, bool topkOnly)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    if constexpr (std::is_same_v<SCORE_T, uint32_t>) {
        trunkLen_ = TRUNK_LEN_6K;
    } else {
        trunkLen_ = (s1BaseSize_ == 2 || topkOnly) ? TRUNK_LEN_16K : 8192;
    }
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                                           GlobalTensor<SCALE_T> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID(
#if KVIP_STAGE_TRACE
    uint64_t trace, uint32_t traceKernel
#endif
)
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    KVIP_TRACE_RECORD(scoreDrain, trace, traceKernel, kvip_trace::LiScoreOutputDrain, 0, 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    KVIP_TRACE_END(scoreDrain);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<SCALE_T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(SCALE_T);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[QLICommon::KScaleTiles * (runInfo.kScaleLoop % 2) * s2BaseSize_], kScaleGm[blockId *
                                constInfo_.keyDequantScaleStride0 + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(SCALE_T);
        for (int32_t i = 0; i < getLoopNum;) {
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            int32_t end = i + 1;
            // The scale stride can contain padding; only join dense pages.
            if (constInfo_.keyDequantScaleStride0 == kCacheBlockSize_) {
                while (end < getLoopNum &&
                    blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + end) == blockId + end - i) ++end;
            }
            int32_t stop = end == getLoopNum ? getLen : end * kCacheBlockSize_;
            copyInParams.blockLen = (stop - i * kCacheBlockSize_) * sizeof(SCALE_T);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[QLICommon::KScaleTiles * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
                                               resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
            i = end;
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(SCALE_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[QLICommon::KScaleTiles * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                            kScaleGm[runInfo.tensorKeyScaleOffset],
                                            copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info, uint32_t progressShard,
#if !V9_UBMEM
                                                   bool publishShardEnd, bool scoreOnly KVIP_PIPE_PARAM)
#else
                                                   bool publishShardEnd KVIP_PIPE_PARAM)
#endif
{
    auto pingpong = (info.loop % 2);
    auto qScalepingpong = (info.qScaleLoop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }

#if !V9_UBMEM
    if (info.LoadQuery()) {
#else
    if (info.isFirstS2InnerLoop) {
#endif
        KVIP_PIPE_SAMPLE(queryScope, pipeSample, AivQueryMte2);
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
        //weightsGm --> weightUB_
        int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
        DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockCount = curAivS1ProcNum;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
        wDataCopyExtParams.srcStride = 0;
        wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))],
                    weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);

        //qScaleGm  -->  qScaleUB_
        DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))],
                    qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
    }

    if ((info.s2Idx - info.s2Start) % QLICommon::KScaleTiles == 0) {
        KVIP_PIPE_SAMPLE(keyScope, pipeSample, AivKeyScaleMte2);
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = QLICommon::KScaleTiles * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : QLICommon::KScaleTiles * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    KVIP_PIPE_SAMPLE(bankScope, pipeSample, AivBankWait);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    KVIP_TRACE_END(bankScope);

    //CV同步
    KVIP_PIPE_SAMPLE(cubeScope, pipeSample, AivCubeWait);
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB
    KVIP_TRACE_END(cubeScope);
    LocalTensor<SCORE_T> outBase;
    if (std::is_same_v<SCORE_T, uint16_t>) {
        outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    } else {
        outBase = vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_];
    }
    auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * QLICommon::KScaleTiles * s2BaseSize_ + ((info.s2Idx - info.s2Start) % QLICommon::KScaleTiles) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    KVIP_PIPE_SAMPLE(vfScope, pipeSample, AivScoreVf);
#if !V9_UBMEM
    if (!scoreOnly && curS2Idx >= runtimeShape_.resident) {
#else
    if (curS2Idx >= runtimeShape_.resident) {
#endif
    vector1::BatchClassifyScore(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase,
                                        kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T),
                                        gSize_, curAivS1ProcNum,
        (__ubuf__ uint16_t *)prefixThreshold_.GetPhyAddr() + (curAivS1Idx % 2) * 128,
        (__ubuf__ uint16_t *)classifyUB_.GetPhyAddr() + pingpong * 256);
    } else {
    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase,
                                        kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T),
                                        gSize_, curAivS1ProcNum);
    }
    KVIP_TRACE_END(vfScope);
#if !V9_UBMEM
    if (info.LoadQuery()) {
#else
    if (info.isFirstS2InnerLoop) {
#endif
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }
    // Bank lifetime is independent of the number of prefetched scale tiles.
    // Return the bank only after its last consumer, including a partial tail.
    // A sequence shard can end before the whole cold phase. Return its
    // partial bank too, before the next aligned shard reuses that bank.
    if ((info.s2Idx - info.s2Start + 1) % QLICommon::KScaleTiles == 0 ||
        (info.s2Idx + 1) * s2BaseSize_ >= info.validS2Len || info.isLastS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    KVIP_PIPE_SAMPLE(storeScope, pipeSample, AivScoreMte3);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
#if !V9_UBMEM
    if (!scoreOnly && curS2Idx >= runtimeShape_.resident) {
#else
    if (curS2Idx >= runtimeShape_.resident) {
#endif
        PublishColdTile(info, curAivS1Idx, curAivS1ProcNum, progressShard, publishShardEnd);
    }
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    KVIP_TRACE_END(storeScope);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::PublishColdTile(const QLICommon::RunInfo &info, uint32_t query,
    uint32_t queryCount, uint32_t progressShard, bool publishShardEnd) {
    const uint32_t queryRow = info.bIdx * SIM_Q + query;
#if SIM_CAPTURE_HITS
    constexpr uint32_t prefixBytes = SIM_Q * 2048 * 6 + SIM_Q * 256;
    GlobalTensor<uint8_t> outputHits;
    outputHits.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(prefixOutput_ + runtimeShape_.batch * prefixBytes));
    DataCopyExtParams hitCopy;
    hitCopy.blockCount = queryCount;
    hitCopy.blockLen = 32;
    hitCopy.srcStride = 7; // classifyUB query stride is 256 bytes.
    hitCopy.dstStride = runtimeShape_.history / 4 - 32;
    DataCopyPad(outputHits[queryRow * (runtimeShape_.history / 4) + info.s2Idx * s2BaseSize_ / 4],
        classifyUB_[(info.loop % 2) * 256].template ReinterpretCast<uint8_t>(), hitCopy);
#endif
    if (publishShardEnd || (info.s2Idx + 1) % ROTATION_PUBLISH_TILES == 0 || info.isLastS2InnerLoop ||
        rotating::IsTrigger(runtimeShape_.history, runtimeShape_.resident, runtimeShape_.rounds, info.s2Idx + 1)) {
        const uint32_t tableBase = (info.s2Idx / 64) * 64;
        if (tableBase != rotationProgressBase_) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            rotating::PrepareProgress(reinterpret_cast<__ubuf__ uint32_t*>(rotationProgress_.GetPhyAddr()), tableBase);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            rotationProgressBase_ = tableBase;
        }
        DataCopyExtParams publish;
        publish.blockCount = queryCount;
        publish.blockLen = 32;
        publish.srcStride = 0;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if (dedicatedProgress_) {
            constexpr uint32_t queryStride = (kvip_dedicated_cold::kRounds - 1) *
                kvip_dedicated_cold::kFragments * kvip_dedicated_cold::kProgressWords * sizeof(uint32_t);
            publish.dstStride = queryStride - publish.blockLen;
            const uint32_t offset = kvip_dedicated_cold::ProgressOffset(queryRow, dedicatedEpoch_, progressShard);
            // The two UB banks can post to the same GM cursor. Order those
            // writes and make this tile's scores and predicates visible
            // before a separate Service lane can observe its cursor.
            PipeBarrier<PIPE_MTE3>();
            DataCopyPad(dedicatedProgressGm_[offset], rotationProgress_[(info.s2Idx % 64) * 16], publish);
        } else if (coldParts_) {
            // MTE3 has already queued this tile's predicates.
            // Each physical part has an exclusive cache line per query.
            // A separate Service lane may observe the final cursor as soon
            // as it is visible, so release the score and predicates first.
            if (separateColdService_ && publishShardEnd) PipeBarrier<PIPE_MTE3>();
            publish.dstStride = coldParts_ * kvip_cold_score::kProgressBytes - publish.blockLen;
            DataCopyPad(coldProgressGm_[kvip_cold_score::ProgressOffset(queryRow, progressShard, coldParts_)],
                rotationProgress_[(info.s2Idx % 64) * 16], publish);
        } else
#endif
        {
            publish.dstStride = 96;
            DataCopyPad(rotationProgressGm_[queryRow * 32 + progressShard * 16],
                rotationProgress_[(info.s2Idx % 64) * 16], publish);
        }
    }
}

#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessPrecomputedVec1(const QLICommon::RunInfo &info,
    uint32_t progressShard, bool publishShardEnd KVIP_PIPE_PARAM) {
    const uint32_t pingpong = info.loop % 2;
    const uint32_t query = (blockId_ % 2) * 2;
    auto scoresLocal = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    KVIP_PIPE_SAMPLE(bankScope, pipeSample, ColdT2BankWait);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    // MTE3_V orders the vector pipe. The V_MTE2 handoff makes that
    // wait and the preceding vector reads visible before MTE2 reuses UB.
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
    KVIP_TRACE_END(bankScope);
    const uint32_t scoreBytes = s2BaseSize_ * sizeof(SCORE_T);
    DataCopyExtParams copy{2, scoreBytes, (runtimeShape_.history - s2BaseSize_) * uint32_t(sizeof(SCORE_T)),
        (UB_BANK_DEPTH_STRIDE - scoreBytes) / 32, 0};
    DataCopyPadExtParams<SCORE_T> pad{false, 0, 0, 0};
    KVIP_PIPE_SAMPLE(loadScope, pipeSample, ColdT2ScoreMte2);
    DataCopyPad(scoresLocal, scoreGm[query * runtimeShape_.history + info.s2Idx * s2BaseSize_], copy, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    KVIP_TRACE_END(loadScope);
    auto scores = reinterpret_cast<__ubuf__ uint16_t*>(scoresLocal.GetPhyAddr());
    auto threshold = reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_.GetPhyAddr());
    auto hits = reinterpret_cast<__ubuf__ uint16_t*>(classifyUB_.GetPhyAddr()) + pingpong * 256;
    KVIP_PIPE_SAMPLE(classifyScope, pipeSample, ColdT2PredicateVf);
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint16_t> values;
        auto all = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t q = 0; q < 2; ++q) {
            MicroAPI::LoadAlign(values, scores + q * (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t)));
            vector1::ClassifyScore(values, threshold + q * 128, hits + q * 128, all);
        }
    }
    KVIP_TRACE_END(classifyScope);
    KVIP_PIPE_SAMPLE(publishScope, pipeSample, ColdT2PublishMte3);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    PublishColdTile(info, query, 2, progressShard, publishShardEnd);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    KVIP_TRACE_END(publishScope);
}
#endif

#if KVIP_STREAMING_LI
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::BeginFinalMetadata(uint32_t query)
{
#if !V9_UBMEM
    if (!streamingWorkspace_ || !streamingTopkOnly_ || runtimeShape_.resident!=16384) return;
#else
    if (!streamingWorkspace_ || !streamingTopkOnly_ || prefixActive_ || runtimeShape_.resident!=16384) return;
#endif
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);uint32_t b=query/SIM_Q;
    if(b>=runtimeShape_.batch)return;
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    GlobalTensor<uint32_t> src;
    if(ctx->hotValid){src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident);DataCopy(streamingValid_,src,8192);}
    if(ctx->reservedHot){src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident);DataCopy(streamingReserved_,src,8192);}
    streamingPrefetched_=true;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::EmitFinalClassification(uint32_t query)
{
    if (!streamingWorkspace_) return;
    auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors);
    uint32_t b=query/SIM_Q;
    if (b>=runtimeShape_.batch || runtimeShape_.resident>131072 ||
        !a->queryBitmaps || !a->queryErrors || !ctx->mapping || !ctx->lruSlots) {
        if (errors && b<runtimeShape_.batch) { errors[query*32]=1; kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+query*32),128); }
        return;
    }
    bool profile=(a->prefetchCount & 0x80000000U)!=0;
    auto liTicks=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(runtimeShape_.batch)*64+uint64_t(query)*16;
    if(profile)liTicks[0]=GetSystemCycle();
    // General LI combines score, window service and final TopK in one launch.
    // Wait for mapping publication, not for URMA completion. In sequence-split
    // mode the earlier LI kernels already form the publication boundary.
    if (!streamingTopkOnly_) {
        auto events=reinterpret_cast<__gm__ rotating::Event*>(ctx->events);
        uint64_t started=GetSystemCycle();
        for(uint32_t i=0;i<runtimeShape_.rounds*runtimeShape_.serviceLanes;++i) {
            while(true) {
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+i),64);
                if(events[i].complete)break;
                if(GetSystemCycle()-started>5000000000ULL) {
                    errors[query*32]=2;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+query*32),128);return;
                }
            }
        }
    }
    if(profile)liTicks[1]=GetSystemCycle();
    // A final-TopK-only phase has no outstanding cube/score work; mirror the
    // validated TopK flow and join the vector pipe before reusing its scratch.
    // The combined LI phase retains its original full-pipeline boundary.
    if(streamingTopkOnly_)PipeBarrier<PIPE_V>();
    else PipeBarrier<PIPE_ALL>();
    if(profile)liTicks[2]=GetSystemCycle();
    LocalTensor<uint32_t> valid,reserved,result,bitmap;
    uint32_t chunkSize=8192;
    if(streamingTopkOnly_) {
        valid=streamingValid_;reserved=streamingReserved_;result=streamingOutput_;bitmap=streamingBitmap_;
    } else {
        uint32_t bytes=2*CeilDiv(constInfo_.mBaseSize,2)*s2BaseSize_*sizeof(QK_T);
        uint32_t mergeBytes=(topkCountAlign256_+trunkLen_)*sizeof(SCORE_T);
        if(bytes<8192*sizeof(uint32_t) || mergeBytes<8192*sizeof(uint32_t)) chunkSize=4096;
        if(bytes<chunkSize*sizeof(uint32_t) || mergeBytes<chunkSize*sizeof(uint32_t)) {errors[query*32]=4;kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+query*32),128);return;}
        valid=resMm1UB_.template ReinterpretCast<uint32_t>();
        reserved=mrgValueLocal_.template ReinterpretCast<uint32_t>();
        result=topkSharedTmpLocal_;bitmap=topkSharedTmpLocal_[2048];
    }
    uint32_t bitmapWords=rotating::kBitmapShards*rotating::kBitmapWords;
#if !V9_UBMEM
    bool wideBitmap=streamingTopkOnly_ && taggedLocations_ && runtimeShape_.resident==16384 &&
        (topkCountAlign256_+trunkLen_)*sizeof(SCORE_T)>=8200*sizeof(uint32_t);
    if(wideBitmap) { bitmap=mrgValueLocal_.template ReinterpretCast<uint32_t>(); bitmapWords=8192; }
#else
    constexpr bool wideBitmap=false;
#endif
    Duplicate(bitmap,uint32_t(0),bitmapWords+8);PipeBarrier<PIPE_V>();
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
    uint32_t used=row->slots,tailBegin=0,tailEnd=0;
    if(ctx->tailBounds){auto tail=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;tailBegin=tail[0];tailEnd=tail[1];}
#if !V9_UBMEM
    bool overlapSecond=false;
    if(taggedLocations_) {
        if(profile)liTicks[3]=GetSystemCycle();
        const uint32_t words=(runtimeShape_.resident+31)/32;
        const uint32_t shardLimit=wideBitmap ? 16 : 8;
        const uint32_t shards=bitmapWords/words<shardLimit?bitmapWords/words:shardLimit;
        // All selected tokens for this Q share one ABC ring and its bounds.
        const uint32_t stride=uint32_t(ctx->outputStride);
        const uint32_t capacity=stride>runtimeShape_.resident?stride-runtimeShape_.resident:0;
        const uint32_t tailSize=tailEnd>tailBegin?Min(tailEnd-tailBegin,capacity):0;
        const uint32_t ringStart=tailSize?tailBegin%capacity:0;
        Simt::VF_CALL<rotating::ClassifyPreparedLocationsFlex>(Simt::Dim3{512,1,1},runtimeShape_.history,runtimeShape_.resident,b,ctx,
            tailBegin,tailSize,ringStart,capacity,
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bitmap[bitmapWords].GetPhyAddr()),shards);
        if(profile){SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);liTicks[4]=liTicks[5]=liTicks[6]=GetSystemCycle();}
        taggedLocations_=false;
    } else {
#endif
    GlobalTensor<uint32_t> validGm,reservedGm;
    if(ctx->hotValid)validGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident);
    if(ctx->reservedHot)reservedGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident);
    // Match the validated TopK experiment: once TopK has finished, its merge
    // and temporary buffers are dead. Load the second metadata half there
    // while the vector pipe classifies the first half.
#if !V9_UBMEM
    overlapSecond = streamingTopkOnly_ && streamingPrefetched_ &&
#else
    const bool overlapSecond = streamingTopkOnly_ && streamingPrefetched_ &&
#endif
        (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T) >= 8192 * sizeof(uint32_t) &&
        topkOp_.GetSharedTmpBufferSize() >= 8192 * sizeof(uint32_t);
    LocalTensor<uint32_t> secondValid,secondReserved;
    if(overlapSecond) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // Both destinations alias TopK vector scratch. PIPE_V orders vector
        // instructions only; establish the cross-pipe dependency before MTE2
        // overwrites the scratch used by the final TopK operation.
        // EVENT_ID0 belongs to VEC1_V_MTE2_EVENT_KSCALE and is already set
        // by AllocEventID. EVENT_ID3 is balanced in the TopK loop and free
        // here, so it cannot consume the older, unrelated signal.
        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
        secondValid=mrgValueLocal_.template ReinterpretCast<uint32_t>();
        secondReserved=topkSharedTmpLocal_;
        if(ctx->hotValid)DataCopy(secondValid,validGm[8192],8192);
        if(ctx->reservedHot)DataCopy(secondReserved,reservedGm[8192],8192);
    }
    for(uint32_t half=0;half<(runtimeShape_.resident+chunkSize-1)/chunkSize;++half){
        uint32_t count=runtimeShape_.resident-half*chunkSize; if(count>chunkSize)count=chunkSize;
        if(half && !overlapSecond)PipeBarrier<PIPE_ALL>();
        bool preloaded=half==0 && streamingTopkOnly_ && streamingPrefetched_;
        if(half && overlapSecond){valid=secondValid;reserved=secondReserved;}
        if(ctx->hotValid && !preloaded && !(half && overlapSecond))DataCopy(valid,validGm[half*chunkSize],count);
        if(ctx->reservedHot && !preloaded && !(half && overlapSecond))DataCopy(reserved,reservedGm[half*chunkSize],count);
        if(half && overlapSecond)PipeBarrier<PIPE_V>();
        if(half || !overlapSecond) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if(profile && half<2)liTicks[3+half*2]=GetSystemCycle();
        Simt::VF_CALL<rotating::ClassifyTopkHalf>(Simt::Dim3{512,1,1},runtimeShape_.history,runtimeShape_.resident,b,used,half*chunkSize,count,tailBegin,tailEnd,
          ctx->hotValid!=0,ctx->reservedHot!=0,ctx->tailBounds!=0,ctx,
          reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),
          reinterpret_cast<__ubuf__ uint32_t*>(valid.GetPhyAddr()),reinterpret_cast<__ubuf__ int32_t*>(reserved.GetPhyAddr()),
          reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bitmap[bitmapWords].GetPhyAddr()));
        if(profile && (half==0 || half+1==(runtimeShape_.resident+chunkSize-1)/chunkSize)){
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            liTicks[half==0?4:6]=GetSystemCycle();
        }
#if !V9_UBMEM
    }
#endif
    }
    if(profile)liTicks[10]=GetSystemCycle();
#if !V9_UBMEM
    const uint32_t wordsForMerge=(runtimeShape_.resident+31)/32;
    const uint32_t shardLimitForMerge=wideBitmap ? 16 : 8;
    const uint32_t shardsForMerge=bitmapWords/wordsForMerge<shardLimitForMerge?bitmapWords/wordsForMerge:shardLimitForMerge;
    Simt::VF_CALL<rotating::MergeClassifyBitmapFlex>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),runtimeShape_.resident,shardsForMerge,
        reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),
        ctx->sourceBlocks && ctx->reservedHot && wordsForMerge+16<=bitmapWords?
            reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+uint64_t(query)*2048:nullptr,
        reinterpret_cast<__ubuf__ uint32_t*>(bitmap[bitmapWords+1].GetPhyAddr()));
#else
    Simt::VF_CALL<rotating::MergeClassifyBitmap>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),runtimeShape_.resident);
#endif
    if(profile){SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);liTicks[11]=GetSystemCycle();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<uint32_t> output,bits;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+uint64_t(query)*2048);
    bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(query)*((runtimeShape_.resident+31)/32));
    DataCopy(output,result,2048);DataCopy(bits,bitmap,(runtimeShape_.resident+31)/32);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    errors[query*32+1]=bitmap.GetValue(bitmapWords+1);errors[query*32]=bitmap.GetValue(bitmapWords);kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+query*32),128);
    if(profile){liTicks[7]=GetSystemCycle();liTicks[8]=overlapSecond?1:0;liTicks[9]=streamingTopkOnly_?1:0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(liTicks),128);}
}
#endif

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo &info, uint32_t queryRow)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (queryRow < SIM_Q) {
        curAivS1Idx = curS1Idx + queryRow;
        curAivS1ProcNum = 1;
    }

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = queryRow < SIM_Q ? queryRow : blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = queryRow < SIM_Q ? queryRow : blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
#if KVIP_STREAMING_LI
        uint32_t streamingQuery = info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx;
        streamingPrefetched_ = false;
#endif

#if KVIP_STAGE_TRACE
        const uint64_t residentTrace = prefixActive_ ? KVIP_TRACE_WORKSPACE(streamingWorkspace_) : 0;
        const uint32_t residentDetail = kvip_trace::PipeDetail(info.bIdx, rowIdx);
#endif
        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        // Recompute each query bound: sparseMode=0 does not reset it in the original loop.
        validS2Len = constInfo_.attenMaskFlag ? static_cast<int32_t>(i) + cuRealAcSeq : cuRealAcSeq;
#if !V9_UBMEM
        const int32_t recentCausalEnd=int32_t(info.actS2SizeOrig)-int32_t(info.actS1Size)+int32_t(curAivS1Idx+i)+1;
        const uint32_t recentEnd=constInfo_.attenMaskFlag ? uint32_t(recentCausalEnd>0?recentCausalEnd:0) : uint32_t(info.actS2SizeOrig);
        bool recentMerged = false;
#endif
        // Causal ProcessTopK derives its bound from actS2SizeOrig, overriding
        // ProcessResidentTopK's actS2Size. Only resident scores exist here.
        if (prefixActive_) validS2Len = Min(validS2Len, static_cast<int32_t>(runtimeShape_.resident));
        auto rotationContext = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
        if (rotationContext->queryEnds) {
            uint32_t query = info.bIdx * SIM_Q + curS1Idx + rowIdx;
            uint32_t bound = reinterpret_cast<__gm__ uint32_t*>(rotationContext->queryEnds)[query];
            validS2Len = Min(validS2Len, static_cast<int32_t>(bound));
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            if (prefixActive_) {
                Duplicate(scoreOutLocal_, uint16_t(0), topkCount_);
#if !V9_UBMEM
                SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q, recentEnd);
#else
                SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q);
#endif
            }
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
#if KVIP_STREAMING_LI
    #if KVIP_STREAMING_LI
        if (!prefixActive_) EmitFinalClassification(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);
#endif
#endif
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
#if !V9_UBMEM
        if (!prefixActive_ && validS2Len >= runtimeShape_.resident &&
            (uint64_t(validS2Len)+127U)/128U*128U <= runtimeShape_.history) {
            FinalIncremental(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, validS2Len);
        } else if (validS2Len >= topkCount_) {
#else
        if (validS2Len >= topkCount_) {
#endif
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                KVIP_TRACE_BEGIN(hotInputScope, residentTrace, Threshold, ThresholdInputMte2, 0, residentDetail);
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                KVIP_TRACE_END(hotInputScope);
#if !V9_UBMEM && KVIP_STREAMING_LI
                kvip_recent_index::Range joinedRecent{0, 0};
                // The TopK input and scratch already reserve resident + K
                // entries. Combining the recent scores avoids a second TopK.
                // Keep the old path for shorter resident sequences and tails
                // larger than that existing capacity.
                if (prefixActive_ && streamingTopkOnly_ && runtimeShape_.resident == 16384 &&
                    validS2Len == 16384 && trunkLen_ == 16384) {
                    uint32_t query = info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx;
                    uint32_t end = rotationContext->queryEnds ?
                        reinterpret_cast<__gm__ uint32_t*>(rotationContext->queryEnds)[query] : runtimeShape_.history;
                    joinedRecent = kvip_recent_index::RecentRange(rotationContext, query / SIM_Q, Min(end, recentEnd));
                    uint32_t count = joinedRecent.end - joinedRecent.begin;
                    if (count && count <= topkCount_) {
                        uint32_t padded = QLICommon::Align(count, uint32_t(256));
                        if (padded != count) {
                            Duplicate(mrgValueLocal_[validS2LenAlign + count / 256 * 256], zero,
                                      padded - count / 256 * 256);
                            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        }
                        DataCopyExtParams recentCopy{1, count * uint32_t(sizeof(SCORE_T)), 0, 0, 0};
                        DataCopyPad(mrgValueLocal_[validS2LenAlign],
                            scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize,
                                                               (uint64_t)s2BaseSize_) + joinedRecent.begin],
                            recentCopy, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        validS2LenAlign += padded;
                        recentMerged = true;
                    }
                }
#endif
#if KVIP_STREAMING_LI
                BeginFinalMetadata(streamingQuery);
#endif
                KVIP_TRACE_BEGIN(hotTopkScope, residentTrace, Threshold, ThresholdTopKVf, 0, residentDetail);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM && KVIP_STREAMING_LI
                bool usedShardedHistogram = false;
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    if (prefixActive_ && streamingTopkOnly_ && !recentMerged &&
                        (runtimeShape_.batch == 1 || runtimeShape_.batch == 8) &&
                        runtimeShape_.history == kvip_resident_topk::kHistory &&
                        runtimeShape_.indexHeads == 32 && runtimeShape_.resident == kvip_resident_topk::kResident &&
                        validS2Len == kvip_resident_topk::kResident && topkCount_ == kvip_resident_topk::kTopK) {
                        const auto table = kvip_resident_topk::HighTable(streamingWorkspace_);
                        if (table) {
                            kvip_resident_topk::Finish(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                                topkSharedTmpLocal_, table, streamingQuery);
                            usedShardedHistogram = true;
                        }
                    }
                }
                if (!usedShardedHistogram)
#endif
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, prefixActive_);
                KVIP_TRACE_END(hotTopkScope);
#if !V9_UBMEM && KVIP_STREAMING_LI
                if (recentMerged) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    Simt::VF_CALL<kvip_recent_index::RemapJoined>(Simt::Dim3{512,1,1},
                        reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
                        runtimeShape_.resident, joinedRecent.begin, joinedRecent.end - joinedRecent.begin);
                    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                }
#endif
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero, QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
#if KVIP_STREAMING_LI
                    if (loopIdx+1 == s2LoopNum) BeginFinalMetadata(streamingQuery);
#endif
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        if (prefixActive_) {
            if (validS2Len < topkCount_) Duplicate(scoreOutLocal_, uint16_t(0), topkCount_);
#if !V9_UBMEM
            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q, recentEnd, recentMerged);
#else
            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q);
#endif
        } else {
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
#if KVIP_STREAMING_LI
        if (!prefixActive_) EmitFinalClassification(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);
#endif
    }
}
}  // namespace QLIKernel
#endif
