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
 * \file quant_lightning_indexer_topk.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_TOPK_H
#define QUANT_LIGHTNING_INDEXER_TOPK_H

#include "kernel_operator.h"
#include "li_types.h"
#include "topk_common.h"
#include "topk_common.h"
#include "pruned_topk.h"

namespace topk {
template<typename T>
class LITopk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                    LocalTensor<T>& inputLocal,
                                    uint32_t s2SeqLen)
    {
    }
};
template<>
class LITopk<uint32_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * QLICommon::Align(topK, (uint32_t)256): 两块hisIndexLocal
        // 5 * 256：histogramsLocal + idxLocal[0-3]
        // 64：nkValueLocal
        uint64_t bufferSize1 = (2 * QLICommon::Align(topK, (uint32_t)256) + 5 * 256 + 64)  * sizeof(uint32_t);
        // QLICommon::Align(topK, (uint32_t)256) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (QLICommon::Align(topK, (uint32_t)256) + trunkLen)  * sizeof(uint32_t);
        uint64_t reuseBufferSize = QLICommon::Align(topK, (uint32_t)256) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2 - reuseBufferSize;
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen = trunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer, LocalTensor<uint32_t>& indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndex1 = indicesOutLocal;
        LocalTensor<uint32_t> hisIndex2 = sharedTmpBuffer[0];
        hisIndexLocal[0] = hisIndex1;
        hisIndexLocal[1] = hisIndex2;
        histogramsLocal = hisIndex2[QLICommon::Align(topK, (uint32_t)256)]; // 256:topk对齐256
        idxLocal0 = histogramsLocal[256]; // 256: 本地内存对齐基线
        idxLocal1 = idxLocal0[256]; // 256: 同上
        idxLocal2 = idxLocal1[256]; // 256: 同上
        idxLocal3 = idxLocal2[256]; // 256: 同上
        nkValueLocal = idxLocal3[256]; // 256: 同上
        tmpIndexLocal = nkValueLocal[64]; // 64: 单核/单线程输出元素容量（G维度切分阈值）
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t>& mrgValueLocal, LocalTensor<uint32_t>& indicesOutLocal,
                                      LocalTensor<uint32_t>& hisValueLocal, uint32_t s2SeqLen,
                                      uint32_t loopIdx, uint32_t s2LoopNum)
    {
        if (s2LoopNum == 1) {
            topkb32gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal,
                                            mrgValueLocal, histogramsLocal, idxLocal0, idxLocal1,
                                            idxLocal2, idxLocal3, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, tmpIndexLocal, QLICommon::Align(topK, (uint32_t)256)); // 256:topk对齐256
        } else {
            if (loopIdx == 0) {
                topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal,
                                              histogramsLocal, idxLocal0, idxLocal1, idxLocal2,
                                              idxLocal3, nkValueLocal, topK, s2SeqLen);
                PipeBarrier<PIPE_V>();
                AscendC::DataCopy(hisIndexLocal[(loopIdx + 1) % 2], // 2:pingpong
                                  tmpIndexLocal, QLICommon::Align(topK, (uint32_t)256)); // 256:topk对齐256
            } else {
                topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal,
                                              histogramsLocal, idxLocal0, idxLocal1, idxLocal2,
                                              idxLocal3, nkValueLocal, topK, s2SeqLen);
                PipeBarrier<PIPE_V>();
                topkb32gather::LiTopKGatherVF(
                    hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal, // 2:pingpong
                    tmpIndexLocal, hisIndexLocal[loopIdx % 2], // 2:pingpong
                    topK,
                    loopIdx * trunkLen - QLICommon::Align(topK, (uint32_t)256), // 256:topk对齐256
                    s2SeqLen);
                if (loopIdx == s2LoopNum - 1) {
                    PipeBarrier<PIPE_V>();
                    if ((loopIdx + 1) % 2 == 1) { // 2:pingpong
                        AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2], // 2:pingpong
                            QLICommon::Align(topK, (uint32_t)256)); // 256:topk对齐256
                    }
                }
            }
        }
    }
private:
    LocalTensor<uint32_t> hisIndexLocal[2]; // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;  // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal0;        // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal1;        // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal2;        // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal3;        // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;     // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;    // 每trunkLen + topK的临时index
    uint32_t topK = 512;
    uint32_t trunkLen = 8192;
};

template<>
class LITopk<uint16_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * QLICommon::Align(topK, (uint32_t)256):两块hisIndexLocal；3 * 256：histogramsLocal idxHighLocal idxLowLocal；64：nkValueLocal
        uint64_t bufferSize1 = (2 * QLICommon::Align(topK, (uint32_t)256) + 3 * 256 + 64)  * sizeof(uint32_t);
        // QLICommon::Align(topK, (uint32_t)256) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (QLICommon::Align(topK, (uint32_t)256) + trunkLen)  * sizeof(uint16_t);
        uint64_t reUseBufferSize = QLICommon::Align(topK, (uint32_t)256) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2 - reUseBufferSize;
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen =  trunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer, LocalTensor<uint32_t>& indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndexLocal1 = indicesOutLocal;
        // 256: 将 topK 实际分配容量向上取整至 256 的倍数
        LocalTensor<uint32_t> hisIndexLocal2 = sharedTmpBuffer[0];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[QLICommon::Align(topK, (uint32_t)256)]; // 256: 同上
        idxHighLocal = histogramsLocal[256]; // 256: 本地内存对齐基线步长。
        idxLowLocal = idxHighLocal[256]; // 256: 延续对齐基线步长
        nkValueLocal = idxLowLocal[256]; // 256: 固定偏移步长
        LocalTensor<uint32_t> tmpIndexLocalTmp = nkValueLocal[64]; // 64: 单核/单线程输出容量
        tmpIndexLocal = tmpIndexLocalTmp.template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void operator()(LocalTensor<uint16_t>& mrgValueLocal, LocalTensor<uint32_t>& indicesOutLocal,
                                      LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum, bool returnScores = false)
    {
        if (s2LoopNum == 1) {
            if (returnScores) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            } else {
            topkb16gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            }
            PipeBarrier<PIPE_V>();
            Cast(indicesOutLocal, tmpIndexLocal, RoundMode::CAST_NONE, topK);
            return;
        }

        if (loopIdx == 0) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE, topK); // 2: 双缓冲的“对侧”Bank
        } else {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, // 2: 双缓冲的Bank
                mrgValueLocal, tmpIndexLocal, hisIndexLocal[loopIdx % 2], // 2: 双缓冲的Bank
                topK, loopIdx * trunkLen - QLICommon::Align(topK, (uint32_t)256), s2SeqLen); // 256: 硬件对齐粒度
            if (loopIdx == s2LoopNum - 1) {
                PipeBarrier<PIPE_V>();
                if ((loopIdx + 1) % 2 == 1) { // 1: 直接访问 Bank 1; 2: 双缓冲总数
                    AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2], // 1: 直接访问 Bank 1; 2: 双缓冲总数
                        QLICommon::Align(topK, (uint32_t)256)); // 256: 硬件对齐粒度; 2: 双缓冲的Bank
                }
            }
        }
    }
    // Resume the exact normal multi-trunk merge after Service has borrowed
    // UB: caller restored previous ordered indices into sharedTmp bank 1.
    __aicore__ inline void MergeNext(LocalTensor<uint16_t>& values,LocalTensor<uint32_t>& indices,
        LocalTensor<uint16_t>& scores,uint32_t padded,uint32_t absoluteStart) {
        topkb16gather::LiTopKVF<true>(tmpIndexLocal,scores,values,histogramsLocal,
            idxHighLocal,idxLowLocal,nkValueLocal,topK,padded);
        PipeBarrier<PIPE_V>();
        topkb16gather::LiTopKGatherVF(indices,scores,values,tmpIndexLocal,
            hisIndexLocal[1],topK,absoluteStart-QLICommon::Align(topK,uint32_t(256)),padded);
        PipeBarrier<PIPE_V>();
    }
#if V9_PRUNED_TOPK
    __aicore__ inline void MergePruned(LocalTensor<uint16_t>& values,
        LocalTensor<uint16_t>& scores,LocalTensor<uint16_t>& positions,
        uint32_t count,uint32_t absoluteStart,uint32_t previousBank,uint32_t outputBank) {
        kvip_pruned::CompactPositions(
            reinterpret_cast<__ubuf__ uint16_t*>(positions.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(values.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(nkValueLocal.GetPhyAddr()),topK,count);
        // Squeeze's AR byte count is a vector result, not a scalar dependency
        // of the call. Join V->S before reading it or sizing the next TopK.
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        const uint32_t kept=uint32_t(GetSpr<SpecialPurposeReg::AR>()/sizeof(uint16_t));
        const uint32_t padded=QLICommon::Align(topK+kept,uint32_t(256));
        kvip_pruned::GatherScores(
            reinterpret_cast<__ubuf__ uint16_t*>(values.GetPhyAddr())+topK,
            reinterpret_cast<__ubuf__ uint16_t*>(positions.GetPhyAddr()),kept,padded-topK);
        PipeBarrier<PIPE_V>();
        topkb16gather::LiTopKVF<true>(tmpIndexLocal,scores,values,histogramsLocal,
            idxHighLocal,idxLowLocal,nkValueLocal,topK,padded);
        PipeBarrier<PIPE_V>();
        kvip_pruned::RestorePositions(
            reinterpret_cast<__ubuf__ uint16_t*>(tmpIndexLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(positions.GetPhyAddr()),topK);
        PipeBarrier<PIPE_V>();
        // LiTopKVF already produced the selected scores. This gather maps only
        // indices, using the original cold positions and absolute block start.
        topkb16gather::LiTopKGatherVF(hisIndexLocal[outputBank],scores,values,tmpIndexLocal,
            hisIndexLocal[previousBank],topK,absoluteStart-topK,padded);
        PipeBarrier<PIPE_V>();
    }
#endif
private:
    LocalTensor<uint32_t> hisIndexLocal[2];      // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;       // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idxHighLocal;          // 输入数据高8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLowLocal;           // 输入数据低8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;          // next_k 暂存Buf 64 * 4B
    LocalTensor<uint16_t> tmpIndexLocal;         // 每trunkLen + topK的临时index
    uint32_t topK = 512;
    uint32_t trunkLen = 16384;
};
}
#endif
