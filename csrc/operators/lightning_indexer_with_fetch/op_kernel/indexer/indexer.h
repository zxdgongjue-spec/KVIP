#include "context.h"
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
 * \file quant_lightning_indexer_kernel.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_KERNEL_H
#define QUANT_LIGHTNING_INDEXER_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "indexer_types.h"
#include "stage_trace.h"
#include <indexer_vector.h>
#include "indexer_cube.h"
#include "../../../common/lifetch/hot_score_partition.h"

namespace QLIKernel {
using namespace QLICommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

enum class SequenceLiPhase { Hot, Threshold, Cold, Final, UnifiedScore, PrecomputedCold, DedicatedCold };

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    uint32_t actS2Size = 0U;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
};

template <typename QLIT>
class QuantLightningIndexerKernel {
public:
    __aicore__ inline QuantLightningIndexerKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengthsK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *workspace, const QLITilingData *__restrict tiling, TPipe *tPipe,
                                uint32_t seqParts = 2);
    __aicore__ inline void Process();
    template <SequenceLiPhase Phase>
    __aicore__ inline void ProcessSequenceSplit(uint32_t parts = 2);
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    __aicore__ inline void ProcessDedicatedCold(GM_ADDR compute);
#endif

    // =================================类型定义区=================================
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using OUT_T = typename QLIT::outputType;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;
    using SCORE_T = typename QLIT::scoreType;

    QLIMatmul<QLIT> matmulService;
    QLIVector<QLIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;
    uint32_t resident_=0;
public:
    rotating::RuntimeShape runtimeShape_;
#if KVIP_STAGE_TRACE
    uint64_t stageTrace_ = 0;
    uint32_t traceWindow_ = 0;
#endif
    bool sequenceSplit_ = false;
    bool sequenceTopKOnly_ = false;
#if !V9_UBMEM
    GM_ADDR rotationContext_ = nullptr;
#endif
private:
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t M_BASE_SIZE = 128;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S2_BASE_SIZE = 128;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    bool rotationServed_ = false;
    uint32_t usedCoreNum = 0U;

    QLICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    QLICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitTilingData(const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, QLICommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    template <bool Precomputed = false>
    __aicore__ inline void ScanSequenceHalf(uint32_t begin, uint32_t end, uint32_t shard,
#if !V9_UBMEM
                                           uint32_t& loop, uint32_t& scaleLoop, bool last, bool newQuery = false,
                                           bool scoreOnly = false, bool reloadQuery = false);
    __aicore__ inline bool ScoreRecent(uint32_t& loop, uint32_t qScaleLoop,
                                      uint32_t shard = 0, uint32_t shards = 1, bool reuseRetained = false);
    __aicore__ inline void FinishBaseBlock(const QLICommon::RunInfo& runInfo);
#else
                                           uint32_t& loop, uint32_t& scaleLoop, bool last);
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
    __aicore__ inline void BindDedicatedQuery(GM_ADDR compute, uint32_t batch);
    __aicore__ inline void ScanDedicatedFragment(uint32_t batch, uint32_t half, uint32_t epoch, uint32_t fragment,
        uint32_t& loop, uint32_t& scaleLoop, bool last, bool reloadQuery = false);
#endif
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
#if !V9_UBMEM
                                            QLICommon::RunInfo runInfo, uint32_t qScaleLoop, uint32_t kScaleLoop,
                                            bool reuseQuery = false);
#else
                                            QLICommon::RunInfo runInfo, uint32_t qScaleLoop, uint32_t kScaleLoop);
#endif
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size, uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLICommon::RunInfo &runInfo,
                                               uint32_t qScaleLoop, uint32_t kScaleLoop);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::InitTilingData(const QLITilingData *__restrict tilingData)
{
    resident_ = tilingData->resident;
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.tSize = tilingData->tSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.outputLayout = Q_LAYOUT_T;  // 输出和输入形状一致

    if (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }
    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = S1_BASE_SIZE;

    if (tilingData->splitQ && constInfo.tSize <= 2 * tilingData->aicCount) {
        constInfo.mBaseSize /= 2;
        constInfo.s1BaseSize /= 2;
    }

    constInfo.setL2DisableFlag = !sequenceSplit_ && constInfo.tSize / constInfo.batchSize <= constInfo.s1BaseSize;
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe, sequenceTopKOnly_);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
}

template <typename QLIT>
__aicore__ inline uint32_t QuantLightningIndexerKernel<QLIT>::GetActualSeqLen(uint32_t bIdx,
                                                             uint32_t actualLenDims, bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                             uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                        uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
    actS2Size = actS2SizeOrig;
}

template <typename QLIT>
__aicore__ inline uint32_t QuantLightningIndexerKernel<QLIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx,
                                                                     uint32_t actS1Size,
                                                                     uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) +
        validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize));
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig));
    validS2Len = Max(validS2Len, 1);
    tempLoopInfo.validS2Len = validS2Len;
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIT>
__aicore__ inline uint32_t QuantLightningIndexerKernel<QLIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isLDOpen ? CeilDiv(actS2Size, constInfo.s2BaseSize) : (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isLDOpen ?
                GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) :
                (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

#if !V9_UBMEM
#else

#endif
// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename QLIT>
__aicore__ void inline QuantLightningIndexerKernel<QLIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                QLICommon::SplitCoreInfo &info)
{
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreIdx = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        splitCoreInfo.isCoreEnable = true;
    } else {
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    bool findLastCoreEnd = true;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.s2Start = 0;
                info.gS1Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx,
                    actS1Size, actS2SizeOrig);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.gS1Start = gS1Idx;
                info.bN2Start = bN2Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isLDOpen ? s2BaseNum : (actS2Size > 0 ? 1 : 0);
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.s2Start = s2Idx;
                    info.gS1Start = gS1Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isLDOpen ?
                        s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1 : s2BaseNum - 1;
#if !V9_UBMEM
#else

#endif

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.s2End = 0;
                            info.gS1End = 0;
                        }
                        return;
                    }
                    findLastCoreEnd = true;
                    coreIdx++;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ?
                        minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::DealActSeqLenIsZero(uint32_t bIdx,
                                                                    uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
            uint32_t tBase = bIdx == 0 ?
                0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +  // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                  // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize *
                                           constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount +  // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                        // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::Init(__gm__ uint8_t *query,
                                              __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                              __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                              __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                              __gm__ uint8_t *blockTable,
                                              __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                              const QLITilingData *__restrict tiling, TPipe *tPipe,
                                              uint32_t seqParts)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // 获取分核信息
    if (sequenceSplit_) {
        splitCoreInfo.bN2Start = aiCoreIdx / seqParts;
        splitCoreInfo.bN2End = splitCoreInfo.bN2Start;
        splitCoreInfo.s2End = constInfo.kSeqSize / constInfo.s2BaseSize - 1;
    } else {
        SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);
    }

    pipe = tPipe;

    uint64_t offset = 0;
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize = constInfo.s1BaseSize * QLICommon::Align((uint64_t)constInfo.kSeqSize,
                                    (uint64_t)constInfo.s2BaseSize)  * sizeof(SCORE_T);
    uint32_t scoreRow = sequenceSplit_ ? splitCoreInfo.bN2Start : aiCoreIdx;
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + scoreRow * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling, sequenceTopKOnly_);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)keyScale);
        if (constInfo.setL2DisableFlag) {
            kScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecWorkspaceTensor(scoreGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        if (constInfo.setL2DisableFlag) {
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx,
            tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        tempLoopInfo.validS2Len = tempLoopInfo.actS2Size;
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail = (tempLoopInfo.s2BasicSizeTail == 0) ?
        constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                     QLICommon::RunInfo &runInfo, uint32_t qScaleLoop,
                                                     uint32_t kScaleLoop)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;
    runInfo.validS2Len = tempLoopInfo.validS2Len;
    runInfo.qScaleLoop = qScaleLoop;
    runInfo.kScaleLoop = kScaleLoop;
    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QLICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QLICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ?
                0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum *
            constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize *
        constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
            totalOutputSize = constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        } else {
            uint32_t tBase = constInfo.batchSize == 0 ? 0 : constInfo.tSize;
            totalOutputSize = tBase * constInfo.kHeadNum * constInfo.sparseCount;
        }
        uint64_t singleCoreSize =
            QLICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIT>
template <bool Precomputed>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ScanSequenceHalf(
#if !V9_UBMEM
    uint32_t begin, uint32_t end, uint32_t shard, uint32_t& loop, uint32_t& scaleLoop, bool last, bool newQuery,
    bool scoreOnly, bool reloadQuery)
#else
    uint32_t begin, uint32_t end, uint32_t shard, uint32_t& loop, uint32_t& scaleLoop, bool last)
#endif
{
    splitCoreInfo.s2Start = begin;
    tempLoopInfo.s2LoopEnd = end - 1;
    tempLoopInfo.validS2Len = end * S2_BASE_SIZE;
    for (uint32_t tile = begin; tile < end; ++tile, ++loop) {
        if ((tile - begin) % QLICommon::KScaleTiles == 0) ++scaleLoop;
        QLICommon::RunInfo info;
        CalcRunInfo(loop, tile, info, 0, scaleLoop);
#if !V9_UBMEM
        info.isFirstS2InnerLoop = loop == 0 || ((newQuery || reloadQuery) && tile == begin);
#else
        info.isFirstS2InnerLoop = loop == 0;
#endif
        info.isLastS2InnerLoop = last && tile + 1 == end;
#if !V9_UBMEM
        info.reuseQuery = newQuery && tile == begin;
#endif
#if KVIP_STAGE_TRACE
        // First and last tile preserve pipeline transitions without turning
        // the full history scan into a trace of every individual submission.
        kvip_trace::ScanSample pipeSample;
        if (stageTrace_ && (tile == begin || tile + 1 == end)) {
            uint32_t firstQ = kvip_trace::kAllQueries;
            if ASCEND_IS_AIV { firstQ = (tmpBlockIdx % 2) * 2; }
            pipeSample = {stageTrace_, traceWindow_ ? kvip_trace::Cold : kvip_trace::Hot,
                traceWindow_, kvip_trace::PipeDetail(info.bIdx, firstQ, tile)};
        }
#endif
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if constexpr (Precomputed) {
            vectorService.ProcessPrecomputedVec1(info, shard, tile + 1 == end KVIP_PIPE_ARG);
        } else
#endif
        {
            if ASCEND_IS_AIC {
                matmulService.ComputeMm1(info KVIP_PIPE_ARG);
            } else {
#if !V9_UBMEM
                vectorService.ProcessVec1(info, shard, tile + 1 == end, scoreOnly KVIP_PIPE_ARG);
#else
                vectorService.ProcessVec1(info, shard, tile + 1 == end KVIP_PIPE_ARG);
#endif
            }
        }
    }
#if !V9_UBMEM
}

// Score the recent range without publishing prefetch progress. The key/scale
// bank is drained at the range boundary; Q may be retained or reused.
template <typename QLIT>
__aicore__ inline bool QuantLightningIndexerKernel<QLIT>::ScoreRecent(
    uint32_t& loop, uint32_t qScaleLoop, uint32_t shard, uint32_t shards, bool reuseRetained)
{
    auto ctx = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    if (!ctx->tailBounds) return false;
    uint32_t end = tempLoopInfo.actS2Size;
    if (ctx->queryEnds) {
        auto ends = reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds) + tempLoopInfo.bIdx * SIM_Q;
        uint32_t maximum = 0;
        for (uint32_t q = 0; q < SIM_Q; ++q) if (ends[q] > maximum) maximum = ends[q];
        if (maximum < end) end = maximum;
    }
    auto range = kvip_recent_index::RecentRange(ctx, tempLoopInfo.bIdx, end);
    auto tiles = kvip_recent_index::Tiles(range, shard, shards);
    if (tiles.begin == tiles.end) return false;
    const uint32_t oldStart = splitCoreInfo.s2Start, oldEnd = tempLoopInfo.s2LoopEnd;
    const uint32_t oldValid = tempLoopInfo.validS2Len;
    splitCoreInfo.s2Start = tiles.begin;
    tempLoopInfo.s2LoopEnd = tiles.end - 1;
    tempLoopInfo.validS2Len = tiles.end * S2_BASE_SIZE;
    uint32_t scaleLoop = 0;
    for (uint32_t tile = tiles.begin; tile < tiles.end; ++tile, ++loop) {
        if ((tile - tiles.begin) % QLICommon::KScaleTiles == 0) ++scaleLoop;
        QLICommon::RunInfo info;
        CalcRunInfo(loop, tile, info, qScaleLoop, scaleLoop);
        if (reuseRetained) {
            info.isFirstS2InnerLoop = false;
            info.reuseQuery = true;
        }
        info.retainQuery = !reuseRetained;
        if ASCEND_IS_AIC {
            matmulService.ComputeMm1(info);
        } else {
            vectorService.ProcessVec1(info, 0, false, true);
        }
    }
    splitCoreInfo.s2Start = oldStart;
    tempLoopInfo.s2LoopEnd = oldEnd;
    tempLoopInfo.validS2Len = oldValid;
    return true;
#endif
}

template <typename QLIT>
template <SequenceLiPhase Phase>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessSequenceSplit(uint32_t parts)
{
    uint32_t batch = splitCoreInfo.bN2Start;
    uint32_t shard = aiCoreIdx % parts;
    CalcGS1LoopParams(batch);
    CalcS2LoopParams(batch, 0);

    if constexpr (Phase == SequenceLiPhase::Threshold || Phase == SequenceLiPhase::Final) {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            QLICommon::RunInfo info;
            CalcRunInfo(0, 0, info, 0, 0);
            if constexpr (Phase == SequenceLiPhase::Threshold) {
                KVIP_TRACE_BEGIN(thresholdScope, stageTrace_, Threshold, LiHotTopK, 0, batch * SIM_Q + tmpBlockIdx % SIM_Q);
                vectorService.ProcessResidentTopK(info, tmpBlockIdx % SIM_Q);
            } else {
                KVIP_TRACE_BEGIN(finalScope, stageTrace_, Final, LiFinalTopK, 0, batch * SIM_Q + tmpBlockIdx % SIM_Q);
                vectorService.ProcessTopK(info, tmpBlockIdx % SIM_Q);
            }
            vectorService.FreeEventID();
        }
    } else {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            if constexpr (Phase != SequenceLiPhase::PrecomputedCold) {
                CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT);
                CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
            }
        } else {
            matmulService.AllocEventID();
        }

        uint32_t loop = 0, scaleLoop = 0;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
        if constexpr (Phase == SequenceLiPhase::UnifiedScore) {
            KVIP_TRACE_RECORD(unifiedScanScope, stageTrace_, kvip_trace::Hot,
                kvip_trace::ScanPhase(), 0, batch);
            const auto tiles = kvip_hot_score::Split(runtimeShape_.history, shard, parts);
            ScanSequenceHalf(tiles.begin, tiles.end, shard, loop, scaleLoop, true, false, true);
        } else
#endif
        if constexpr (Phase == SequenceLiPhase::Hot) {
            KVIP_TRACE_RECORD(hotScanScope, stageTrace_, kvip_trace::Hot, kvip_trace::ScanPhase(), 0, batch);
#if !V9_UBMEM
            auto ctxRecent = reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
            uint32_t recentEnd = tempLoopInfo.actS2Size;
            auto boundsRecent = reinterpret_cast<__gm__ uint32_t*>(ctxRecent->tailBounds);
            bool potentiallyRecent = boundsRecent &&
                boundsRecent[tempLoopInfo.bIdx * 2] < boundsRecent[tempLoopInfo.bIdx * 2 + 1] &&
                ctxRecent->outputStride > runtimeShape_.resident;
            if (potentiallyRecent && ctxRecent->queryEnds) {
                auto ends = reinterpret_cast<__gm__ uint32_t*>(ctxRecent->queryEnds) + tempLoopInfo.bIdx * SIM_Q;
                uint32_t maximum = 0;
                for (uint32_t q = 0; q < SIM_Q; ++q) if (ends[q] > maximum) maximum = ends[q];
                if (maximum < recentEnd) recentEnd = maximum;
            }
            bool hasRecent = false;
            if (potentiallyRecent) {
                auto recentRange = kvip_recent_index::RecentRange(ctxRecent, tempLoopInfo.bIdx, recentEnd);
                auto recentTiles = kvip_recent_index::Tiles(recentRange, shard, parts);
                hasRecent = recentTiles.begin != recentTiles.end;
            }
#endif
            const auto tiles = kvip_hot_score::Split(resident_, shard, parts);
#if !V9_UBMEM
            // Keep Q and its scale until the shard's recent tail finishes. Key
            // and key-scale banks still drain at each discontinuous range.
            ScanSequenceHalf(tiles.begin, tiles.end, shard, loop, scaleLoop, !hasRecent, false);
            if (hasRecent) ScoreRecent(loop, 0, shard, parts, true);
#else
            ScanSequenceHalf(tiles.begin, tiles.end, shard, loop, scaleLoop, true);
#endif
        } else {
            uint32_t halfTokens = rotating::SequenceHalfWindow(runtimeShape_);
            uint32_t half = halfTokens / S2_BASE_SIZE;
            uint32_t serviceCore = aiCoreIdx;
            bool serviceLeader = true;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
            if (parts > 2) {
                const uint32_t shardParts = parts / 2;
                serviceCore = batch * 2 + shard / shardParts;
                serviceLeader = shard % shardParts == 0;
            }
#endif
            const uint32_t serviceEpoch = rotating::ServiceRank(runtimeShape_.batch, runtimeShape_.splitQ,
                serviceCore) / runtimeShape_.serviceAics;
            if ASCEND_IS_AIV {
                vectorService.LoadSequenceSplitThresholds(batch);
                vectorService.PrepareRotationProgress();
                if (serviceLeader && serviceEpoch == 0) vectorService.RotateFetch(0, halfTokens);
            }
            uint32_t begin = resident_ / S2_BASE_SIZE + shard * half;
            for (uint32_t epoch = 1; epoch < runtimeShape_.rounds; ++epoch, begin += 2 * half) {
                auto tiles = kvip_hot_score::TileRange{begin, begin + half};
                if (tiles.end > runtimeShape_.history / S2_BASE_SIZE) tiles.end = runtimeShape_.history / S2_BASE_SIZE;
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
                if (parts > 2) tiles = kvip_cold_score::Split(runtimeShape_.history, runtimeShape_.resident,
                    halfTokens, epoch, shard, parts);
#endif
                {
#if KVIP_STAGE_TRACE
                    traceWindow_ = epoch;
#endif
                    KVIP_TRACE_RECORD(coldScanScope, stageTrace_, kvip_trace::Cold,
                        kvip_trace::ScanPhase(), epoch, batch);
                    ScanSequenceHalf<Phase == SequenceLiPhase::PrecomputedCold>(
                        tiles.begin, tiles.end, shard, loop, scaleLoop, epoch + 1 == runtimeShape_.rounds);
                }
                if ASCEND_IS_AIV {
                    if (serviceLeader && serviceEpoch == epoch) {
#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
                        if (parts > 2) vectorService.WaitPartitionedHalf(batch, shard, epoch, halfTokens);
#endif
                        vectorService.RotateFetch(epoch, halfTokens);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
        } else {
            matmulService.FreeEventID();
            KVIP_TRACE_RECORD(cubeTailScope, stageTrace_,
                (Phase == SequenceLiPhase::Hot || Phase == SequenceLiPhase::UnifiedScore)
                    ? kvip_trace::Hot : kvip_trace::Cold,
                kvip_trace::LiCubeWait, runtimeShape_.rounds, batch);
            CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
        }
    }
}

#if KVIP_HOT_SCORE_PARTITION && !V9_UBMEM
template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::BindDedicatedQuery(GM_ADDR compute, uint32_t batch)
{
    splitCoreInfo.bN2Start = batch;
    splitCoreInfo.bN2End = batch;
    CalcGS1LoopParams(batch);
    CalcS2LoopParams(batch, 0);
    if ASCEND_IS_AIV {
        GlobalTensor<SCORE_T> row;
        row.SetGlobalBuffer(reinterpret_cast<__gm__ SCORE_T*>(compute) +
            uint64_t(batch) * runtimeShape_.history * kvip_dedicated_cold::kQueries);
        vectorService.InitVecWorkspaceTensor(row);
        vectorService.LoadSequenceSplitThresholds(batch);
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ScanDedicatedFragment(uint32_t batch,
    uint32_t half, uint32_t epoch, uint32_t fragment, uint32_t& loop, uint32_t& scaleLoop, bool last, bool reloadQuery)
{
    const auto tiles = kvip_dedicated_cold::Fragment(runtimeShape_.history, resident_, half, epoch, fragment);
#if KVIP_STAGE_TRACE
    traceWindow_ = epoch;
#endif
    KVIP_TRACE_RECORD(scanScope, stageTrace_, kvip_trace::Cold, kvip_trace::ScanPhase(), epoch, batch);
    ScanSequenceHalf(tiles.begin, tiles.end, fragment, loop, scaleLoop, last, false, false, reloadQuery);
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessDedicatedCold(GM_ADDR compute)
{
    const uint32_t half = rotating::SequenceHalfWindow(runtimeShape_);
    if (aiCoreIdx >= kvip_dedicated_cold::kProducerGroups) {
        if ASCEND_IS_AIV {
            vectorService.RunDedicatedService(tmpBlockIdx - 2 * kvip_dedicated_cold::kProducerGroups, half);
        }
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT);
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
        vectorService.PrepareRotationProgress();
    } else {
        matmulService.AllocEventID();
    }

    constexpr uint32_t groupsPerPair = 3;
    const uint32_t firstBatch = aiCoreIdx / groupsPerPair * 2;
    const uint32_t slot = aiCoreIdx % groupsPerPair;
    if (slot < 2) BindDedicatedQuery(compute, firstBatch + slot);
    uint32_t loop = 0, scaleLoop = 0;
    for (uint32_t epoch = 1; epoch < runtimeShape_.rounds; ++epoch) {
        if ASCEND_IS_AIV { vectorService.SetDedicatedColdEpoch(epoch); }
        if (slot < 2) {
            const uint32_t batch = firstBatch + slot;
            ScanDedicatedFragment(batch, half, epoch, 0, loop, scaleLoop, false);
            ScanDedicatedFragment(batch, half, epoch, 1, loop, scaleLoop, epoch + 1 == runtimeShape_.rounds);
        } else {
            for (uint32_t batch = firstBatch; batch < firstBatch + 2; ++batch) {
                // Finish work that used the old query and its tensor bindings.
                // Keep the FIXP ping-pong counter aligned across both consumers.
                PipeBarrier<PIPE_ALL>();
                BindDedicatedQuery(compute, batch);
                ScanDedicatedFragment(batch, half, epoch, 2, loop, scaleLoop, true, true);
            }
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        KVIP_TRACE_BEGIN(cubeTailScope, stageTrace_, Cold, LiCubeWait, runtimeShape_.rounds, tempLoopInfo.bIdx);
        // Each bank receives one final credit from each of the two AIVs.
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT);
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
            QLICommon::ConstInfo::CROSS_VC_EVENT + QLICommon::ConstInfo::AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
            QLICommon::ConstInfo::CROSS_VC_EVENT + QLICommon::ConstInfo::AIV0_AIV1_OFFSET + 1);
    }
}
#endif

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    QLICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    uint32_t qScaleLoop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
#if !V9_UBMEM
            bool hadRecent = ScoreRecent(gloop, qScaleLoop);
#endif
            uint32_t kScaleLoop = 0;
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                if ((s2LoopIdx - splitCoreInfo.s2Start) % QLICommon::KScaleTiles == 0) {
                    ++kScaleLoop;
                }
#if !V9_UBMEM
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop,
                                 hadRecent && s2LoopIdx == splitCoreInfo.s2Start);
#else
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
#endif
                ++gloop;
            }
            ++qScaleLoop;
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + 1);
    }
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          QLICommon::RunInfo runInfo, uint32_t qScaleLoop,
#if !V9_UBMEM
                                                          uint32_t kScaleLoop, bool reuseQuery)
#else
                                                          uint32_t kScaleLoop)
#endif
{
    CalcRunInfo(loop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
#if !V9_UBMEM
    runInfo.reuseQuery = reuseQuery;
#endif
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
#if !V9_UBMEM
    } else {
        vectorService.ProcessVec1(runInfo);
    }
    FinishBaseBlock(runInfo);
}

template <typename QLIT>
__aicore__ inline void QuantLightningIndexerKernel<QLIT>::FinishBaseBlock(const QLICommon::RunInfo& runInfo)
{
    uint32_t s2LoopIdx = runInfo.s2Idx;
    if ASCEND_IS_AIC {
#endif
        if (!rotationServed_ && rotating::EpochAt(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,s2LoopIdx + 1, aiCoreIdx) >= 0) {
            CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(8);
            rotationServed_ = true;
        }
    } else {
#if !V9_UBMEM
#else
        vectorService.ProcessVec1(runInfo);
#endif
        if ((runInfo.s2Idx + 1) * 128 == resident_) {
            vectorService.ProcessResidentTopK(runInfo);
            vectorService.PrepareRotationProgress();
        }
        if (!rotationServed_ && rotating::EpochAt(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,s2LoopIdx + 1, aiCoreIdx) >= 0) {
            vectorService.RotateFetch(rotating::EpochAt(runtimeShape_.batch,runtimeShape_.history,runtimeShape_.resident,runtimeShape_.splitQ,runtimeShape_.rounds,runtimeShape_.serviceAics,s2LoopIdx + 1, aiCoreIdx));
            CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(8);
            rotationServed_ = true;
        }
        if (runInfo.isLastS2InnerLoop) {   // 本核s2last
            vectorService.ProcessTopK(runInfo);
        }
    }
}

}  // namespace QLIKernel
#endif  // QUANT_LIGHTNING_INDEXER_KERNEL_H
