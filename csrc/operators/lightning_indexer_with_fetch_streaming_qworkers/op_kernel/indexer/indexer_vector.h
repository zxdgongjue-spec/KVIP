#include "prefetch.h"
#include "pipeline_types.h"
#include "staged_topk_classify.h"
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

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_6K = 6144;
template <typename QLIT>
class QLIVector {
public:
    rotating::RuntimeShape runtimeShape_;
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
    __aicore__ inline void RotateFetch(uint32_t epoch, uint32_t sequenceHalfWindow = 0) {
        // Join this producer's hit publication. Other producers remain free.
        PipeBarrier<PIPE_ALL>();
#if defined(V9_UBMEM) && V9_UBMEM
        rotating::Service(rotationContext_, epoch, blockId_ / 2, blockId_ % 2,
#else
        rotating::Service(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,runtimeShape_.serviceLanes,runtimeShape_.tokenBytes,runtimeShape_.postThreads,runtimeShape_.postBank,runtimeShape_.batchedRows,runtimeShape_.scanTokens,rotationContext_, epoch, blockId_ / 2, blockId_ % 2,
#endif
#if defined(V9_UBMEM) && V9_UBMEM
            topkSharedTmpLocal_, mrgValueLocal_, indicesOutLocal_);
#else
            topkSharedTmpLocal_, mrgValueLocal_, indicesOutLocal_, sequenceHalfWindow,
            streamingWorkspace_ && (reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_)->prefetchCount & 0x80000000U));
#endif
    }
    __aicore__ inline void LoadSequenceSplitThresholds(uint32_t batch) {
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
                                     bool publishShardEnd = false);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info, uint32_t queryRow = SIM_Q);
    __aicore__ inline void SetPrefixStorage(GM_ADDR output) { prefixOutput_ = output; }
    __aicore__ inline void SetStreamingWorkspace(GM_ADDR workspace) { streamingWorkspace_ = workspace; }
    __aicore__ inline void EmitFinalClassification(uint32_t query);
    __aicore__ inline void BeginFinalMetadata(uint32_t query);
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
    __aicore__ inline void FreeEventID();

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
    GM_ADDR rotationContext_ = nullptr;
    GM_ADDR streamingWorkspace_ = nullptr;
    bool streamingTopkOnly_ = false;
    bool streamingPrefetched_ = false;
    TBuf<TPosition::VECCALC> streamingValidBuf_, streamingReservedBuf_, streamingOutputBuf_, streamingBitmapBuf_;
    LocalTensor<uint32_t> streamingValid_, streamingReserved_, streamingOutput_, streamingBitmap_;
    GlobalTensor<uint32_t> rotationProgressGm_;
    LocalTensor<uint32_t> rotationProgress_;
    uint32_t rotationProgressBase_ = 0;

    GM_ADDR prefixOutput_ = nullptr;
    bool prefixActive_ = false;
    TBuf<TPosition::VECCALC> classifyBuf_;
    LocalTensor<uint16_t> classifyUB_;
    TBuf<TPosition::VECCALC> prefixThresholdBuf_;
    LocalTensor<uint16_t> prefixThreshold_;
    __aicore__ inline void SaveResident(uint32_t query, uint32_t scoreRow, bool publishProgress) {
        auto src = (__local_mem__ uint16_t *)scoreOutLocal_.GetPhyAddr();
        auto dst = (__local_mem__ uint16_t *)prefixThreshold_.GetPhyAddr() + (query % 2) * 128;
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
            DataCopy(reservedChunk.template ReinterpretCast<uint32_t>(), victims, 2048);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Simt::VF_CALL<rotating::GatherVictimScores>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(reservedChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scoreChunk.GetPhyAddr()),
                reinterpret_cast<const __gm__ uint32_t*>(scoreGm.GetPhyAddr()) + scoreRow *
                    QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) / 2);
            PipeBarrier<PIPE_V>();
            vector1::ClassifyResidentScores(
                reinterpret_cast<__ubuf__ uint16_t*>(scoreChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(reservedChunk.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_[(query % 2) * 128].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(hitChunk.GetPhyAddr()), validEnd);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams copy;
            copy.blockCount = 16; copy.blockLen = 32;
            copy.srcStride = 7; copy.dstStride = 0;
            DataCopyPad(bitmap, hitChunk.template ReinterpretCast<uint8_t>(), copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
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
    streamingTopkOnly_ = topkOnly;
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
        pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
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

    if (!topkOnly) Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
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
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
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
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_], kScaleGm[blockId *
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
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
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
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                            kScaleGm[runInfo.tensorKeyScaleOffset],
                                            copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info, uint32_t progressShard,
                                                   bool publishShardEnd)
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

    if (info.isFirstS2InnerLoop) {
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

    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : 16 * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB
    LocalTensor<SCORE_T> outBase;
    if (std::is_same_v<SCORE_T, uint16_t>) {
        outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    } else {
        outBase = vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_];
    }
    auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    if (curS2Idx >= runtimeShape_.resident) {
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
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
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
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
#if SIM_CAPTURE_HITS
    if (curS2Idx >= runtimeShape_.resident) {
        constexpr uint32_t prefixBytes = SIM_Q * 2048 * 6 + SIM_Q * 256;
        GlobalTensor<uint8_t> outputHits;
        outputHits.SetGlobalBuffer((__gm__ uint8_t *)(prefixOutput_ + runtimeShape_.batch * prefixBytes));
        DataCopyExtParams hitCopy;
        hitCopy.blockCount = curAivS1ProcNum;
        hitCopy.blockLen = 32;
        hitCopy.srcStride = 7; // classifyUB query stride is 256 bytes.
        hitCopy.dstStride = runtimeShape_.history / 4 - 32;
        DataCopyPad(outputHits[(info.bIdx * SIM_Q + curAivS1Idx) * (runtimeShape_.history / 4) + curS2Idx / 4],
                    classifyUB_[pingpong * 256].template ReinterpretCast<uint8_t>(), hitCopy);
    }
#endif

    if (curS2Idx >= runtimeShape_.resident && (publishShardEnd || (info.s2Idx + 1) % ROTATION_PUBLISH_TILES == 0 || info.isLastS2InnerLoop ||
        rotating::IsTrigger(runtimeShape_.history,runtimeShape_.resident,runtimeShape_.rounds,info.s2Idx + 1))) {
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
        publish.blockCount = curAivS1ProcNum; // Q2: one query/AIV; Q4: two
        publish.blockLen = 32;
        publish.srcStride = 0;
        publish.dstStride = 96;
        DataCopyPad(rotationProgressGm_[(info.bIdx * SIM_Q + curAivS1Idx) * 32 + progressShard * 16],
            rotationProgress_[(info.s2Idx % 64) * 16], publish);
    }
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::BeginFinalMetadata(uint32_t query)
{
    if (!streamingWorkspace_ || !streamingTopkOnly_ || prefixActive_ || runtimeShape_.resident!=16384) return;
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
    if (b>=runtimeShape_.batch || runtimeShape_.resident!=16384 || runtimeShape_.history!=131072 ||
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
    constexpr uint32_t bitmapWords=rotating::kBitmapShards*rotating::kBitmapWords;
    Duplicate(bitmap,uint32_t(0),bitmapWords+8);PipeBarrier<PIPE_V>();
    uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
    uint32_t used=row->slots,tailBegin=0,tailEnd=0;
    if(ctx->tailBounds){auto tail=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;tailBegin=tail[0];tailEnd=tail[1];}
    GlobalTensor<uint32_t> validGm,reservedGm;
    if(ctx->hotValid)validGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*runtimeShape_.resident);
    if(ctx->reservedHot)reservedGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->reservedHot)+uint64_t(b)*runtimeShape_.resident);
    // Match the validated TopK experiment: once TopK has finished, its merge
    // and temporary buffers are dead. Load the second metadata half there
    // while the vector pipe classifies the first half.
    const bool overlapSecond = streamingTopkOnly_ && streamingPrefetched_ &&
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
    for(uint32_t half=0;half<runtimeShape_.resident/chunkSize;++half){
        if(half && !overlapSecond)PipeBarrier<PIPE_ALL>();
        bool preloaded=half==0 && streamingTopkOnly_ && streamingPrefetched_;
        if(half && overlapSecond){valid=secondValid;reserved=secondReserved;}
        if(ctx->hotValid && !preloaded && !(half && overlapSecond))DataCopy(valid,validGm[half*chunkSize],chunkSize);
        if(ctx->reservedHot && !preloaded && !(half && overlapSecond))DataCopy(reserved,reservedGm[half*chunkSize],chunkSize);
        if(half && overlapSecond)PipeBarrier<PIPE_V>();
        if(half || !overlapSecond) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if(profile && half<2)liTicks[3+half*2]=GetSystemCycle();
        Simt::VF_CALL<rotating::ClassifyTopkHalf>(Simt::Dim3{512,1,1},runtimeShape_.history,runtimeShape_.resident,b,used,half*chunkSize,chunkSize,tailBegin,tailEnd,
          ctx->hotValid!=0,ctx->reservedHot!=0,ctx->tailBounds!=0,ctx,
          reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()),
          reinterpret_cast<__ubuf__ uint32_t*>(valid.GetPhyAddr()),reinterpret_cast<__ubuf__ int32_t*>(reserved.GetPhyAddr()),
          reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(bitmap[bitmapWords].GetPhyAddr()));
        if(profile && (half==0 || half+1==runtimeShape_.resident/chunkSize)){
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            liTicks[half==0?4:6]=GetSystemCycle();
        }
    }
    if(profile)liTicks[10]=GetSystemCycle();
    Simt::VF_CALL<rotating::MergeClassifyBitmap>(Simt::Dim3{512,1,1},reinterpret_cast<__ubuf__ uint32_t*>(bitmap.GetPhyAddr()));
    if(profile){SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);liTicks[11]=GetSystemCycle();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<uint32_t> output,bits;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+uint64_t(query)*2048);
    bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a->queryBitmaps)+uint64_t(query)*512);
    DataCopy(output,result,2048);DataCopy(bits,bitmap,512);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    errors[query*32]=bitmap.GetValue(bitmapWords);kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(errors+query*32),128);
    if(profile){liTicks[7]=GetSystemCycle();liTicks[8]=overlapSecond?1:0;liTicks[9]=streamingTopkOnly_?1:0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(liTicks),128);}
}

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
        uint32_t streamingQuery = info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx;
        streamingPrefetched_ = false;

        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        // Recompute each query bound: sparseMode=0 does not reset it in the original loop.
        validS2Len = constInfo_.attenMaskFlag ? static_cast<int32_t>(i) + cuRealAcSeq : cuRealAcSeq;
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
                SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q);
            }
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (!prefixActive_) EmitFinalClassification(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                BeginFinalMetadata(streamingQuery);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, prefixActive_);
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
                    if (loopIdx+1 == s2LoopNum) BeginFinalMetadata(streamingQuery);
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
            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx, vecOffset, queryRow == SIM_Q);
        } else {
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        if (!prefixActive_) EmitFinalClassification(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);
    }
}
}  // namespace QLIKernel
#endif
