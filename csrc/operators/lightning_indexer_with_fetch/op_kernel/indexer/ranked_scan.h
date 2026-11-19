/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// LI scan from KVIP 1d2486a76151ae37662c47aab835d06df50be866.
// Included inside QLIKernel for ranked URMA only. Window scheduling, rank
// ownership and Fetch admission remain defined by this integration.
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info, uint32_t progressShard,
                                                   bool publishShardEnd, bool scoreOnly)
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

    if (info.LoadQuery()) {
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
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = QLICommon::KScaleTiles * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : QLICommon::KScaleTiles * s2BaseSize_;
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
    auto kScaleBase = kScaleUB_[kScalepingpong * QLICommon::KScaleTiles * s2BaseSize_ + ((info.s2Idx - info.s2Start) % QLICommon::KScaleTiles) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    if (!scoreOnly && curS2Idx >= runtimeShape_.resident) {
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
    if (info.LoadQuery()) {
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
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
#if SIM_CAPTURE_HITS
    if (!scoreOnly && curS2Idx >= runtimeShape_.resident) {
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

    if (!scoreOnly && curS2Idx >= runtimeShape_.resident && (publishShardEnd || (info.s2Idx + 1) % ROTATION_PUBLISH_TILES == 0 || info.isLastS2InnerLoop ||
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
