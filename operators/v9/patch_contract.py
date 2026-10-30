# ruff: noqa: E501
# Long string anchors intentionally match the checked-in native C++ verbatim.
"""Adapt V9's generated native LI to real mapped hot/cold tokens.

Anchors fail closed when the checked-in native implementation changes. The
native Cube score path and TopK primitives remain the numerical implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "base"))
from patch_installed_prefix import replace_once


def replace_body(text, signature, body):
    start = text.index(signature)
    brace = text.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:brace] + "{\n" + body + "\n}" + text[end:]


def patch(root):
    kernel = root / "arch35/quant_lightning_indexer_kernel.h"
    text = '#include "contract_device.h"\n' + kernel.read_text(encoding="utf-8")
    # Match compile-time service geometry exactly, including B > AIC_COUNT.
    for anchor in ("if (constInfo.tSize <= 2 * V9_AIC_COUNT) {", "if (false) { // rotating Q4 tiling"):
        if anchor in text:
            text = replace_once(text, anchor, "if (V9_SPLIT_Q) {")
            break
    text = replace_once(
        text,
        "public:",
        """public:
    __aicore__ inline void SetMappedContext(GM_ADDR p) {
        mappedContext_ = reinterpret_cast<__gm__ rotating::Context*>(p);
        if ASCEND_IS_AIC { matmulService.SetMappedContext(p); }
    }
""",
    )
    text = replace_once(
        text,
        "    bool rotationServed_ = false;",
        """    bool rotationServed_ = false;
    bool mappedHot_ = false;
    __gm__ rotating::Context* mappedContext_ = nullptr;""",
    )
    old = """            uint32_t kScaleLoop = 0;
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                if ((s2LoopIdx - splitCoreInfo.s2Start) % 16 == 0) {
                    ++kScaleLoop;
                }
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);
                ++gloop;
            }
            ++qScaleLoop;"""
    new = """            auto c = rotating::ContractOf(mappedContext_);
            uint32_t length = rotating::ContractLength(c, tempLoopInfo.bIdx);
            uint32_t hotCount = rotating::ContractPtr<int32_t>(c->hot_counts)[tempLoopInfo.bIdx];
            // Each query group finishes its mapped hot pass before the cold
            // page scan. H0 is frozen by init and never changed by prefetch.
            for (uint32_t phase = 0; phase < 2; ++phase) {
                mappedHot_ = phase == 0;
                uint32_t size = mappedHot_ ? hotCount : rotating::ColdExtent(c);
                if ASCEND_IS_AIC { matmulService.SetMappedPhase(mappedHot_); }
                else {
                    vectorService.SetMappedPhase(mappedHot_, tempLoopInfo.bIdx, tempLoopInfo.gS1Idx);
                    if (!mappedHot_) vectorService.PrepareRotationProgress();
                }
                if (!size) continue;
                tempLoopInfo.actS2Size = size;
                tempLoopInfo.validS2Len = size;
                tempLoopInfo.s2BasicSizeTail = size % 128 ? size % 128 : 128;
                tempLoopInfo.s2LoopEnd = (size + 127) / 128 - 1;
                splitCoreInfo.s2Start = 0;
                uint32_t kScaleLoop = 0;
                for (uint32_t tile = 0; tile <= tempLoopInfo.s2LoopEnd; ++tile) {
                    if (tile % 16 == 0) ++kScaleLoop;
                    ProcessBaseBlock(gloop++, tile, runInfo, qScaleLoop, kScaleLoop);
                }
                ++qScaleLoop;
            }
            tempLoopInfo.actS2Size = length;"""
    text = replace_once(text, old, new)
    text = text.replace(
        "!rotationServed_ && rotating::EpochAt(s2LoopIdx + 1, aiCoreIdx) >= 0",
        "!mappedHot_ && !rotationServed_ && "
        "rotating::ContractEpoch(rotating::ContractOf(mappedContext_), s2LoopIdx + 1, aiCoreIdx) >= 0",
    )
    text = text.replace(
        "vectorService.RotateFetch(rotating::EpochAt(s2LoopIdx + 1, aiCoreIdx));",
        "vectorService.RotateFetch(rotating::ContractEpoch("
        "rotating::ContractOf(mappedContext_), s2LoopIdx + 1, aiCoreIdx));",
    )
    text = replace_once(
        text,
        """        if ((runInfo.s2Idx + 1) * 128 == SIM_RESIDENT) {
            vectorService.ProcessResidentTopK(runInfo);
            vectorService.PrepareRotationProgress();
        }
""",
        "",
    )
    text = replace_once(
        text,
        "            vectorService.ProcessTopK(runInfo);",
        """            if (mappedHot_) vectorService.ProcessResidentTopK(runInfo);
            else vectorService.ProcessMappedTopK(runInfo);""",
    )
    kernel.write_text(text, encoding="utf-8")

    cube = root / "arch35/quant_lightning_indexer_service_cube.h"
    text = '#include "contract_device.h"\n' + cube.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "public:",
        """public:
    __aicore__ inline void SetMappedContext(GM_ADDR p) {
        mappedContext_ = reinterpret_cast<__gm__ rotating::Context*>(p);
    }
    __aicore__ inline void SetMappedPhase(bool hot) { mappedHot_ = hot; }
""",
    )
    text = replace_once(
        text,
        "private:",
        """private:
    __gm__ rotating::Context* mappedContext_ = nullptr;
    bool mappedHot_ = false;
""",
    )
    anchor = "    uint64_t s2L1Offset = 0;"
    text = replace_once(
        text,
        anchor,
        """    {
        auto c = rotating::ContractOf(mappedContext_);
        if (mappedHot_ || !rotating::IndexOwned(c,runInfo.bIdx)) {
        GlobalTensor<K_T> bank;
        bank.SetGlobalBuffer(rotating::ContractPtr<K_T>(rotating::IndexKey(c,runInfo.bIdx,mappedHot_)));
        Nd2NzParams p;
        p.ndNum=1; p.nValue=s2L1RealSize; p.dValue=constInfo_.headDim;
        p.srcDValue=constInfo_.headDim;
        p.dstNzC0Stride=CeilAlign(s2L1RealSize,uint64_t(BLOCK_CUBE));
        p.dstNzNStride=1; p.srcNdMatrixStride=0; p.dstNzMatrixStride=0;
        DataCopy(keyL1_[(keyL1BufIdx_%KEY_BUF_NUM)*keyBufferOffset_],bank[s2GmOffset*constInfo_.headDim],p);
        return;
        }
    }
"""
        + anchor,
    )
    cube.write_text(text, encoding="utf-8")

    vector = root / "arch35/quant_lightning_indexer_service_vector.h"
    text = '#include "contract_scores.h"\n' + vector.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    GM_ADDR rotationContext_ = nullptr;",
        """    GM_ADDR rotationContext_ = nullptr;
    bool mappedHot_ = false;
    uint32_t mappedCount_ = 0, mappedRow_ = 0, mappedColdBase_ = 0;
    uint32_t mappedExtent_ = 0, mappedMaskTiles_[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    __aicore__ inline __gm__ rotating::Context* MappedContext() {
        return reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    }""",
    )
    text = replace_once(
        text,
        "public:",
        """public:
    __aicore__ inline void SetMappedPhase(bool hot, uint32_t b, uint32_t group) {
        mappedHot_ = hot;
        if (hot) Duplicate(prefixThreshold_, uint16_t(0), 256);
        auto c=rotating::ContractOf(MappedContext());
        // Separate aligned hot/cold tiles share one persistent score row.
        constInfo_.kSeqSize=c->score_stride;
        scoreGm.SetGlobalBuffer(rotating::ContractPtr<SCORE_T>(c->scores)+
            (b*4+group*s1BaseSize_)*c->score_stride);
        mappedColdBase_=(rotating::ContractPtr<int32_t>(c->hot_counts)[b]+127)/128*128;
        mappedExtent_=hot?rotating::ContractPtr<int32_t>(c->hot_counts)[b]:rotating::ColdCount(c,b);
        uint32_t length=rotating::ContractLength(c,b);
        for(uint32_t i=0;i<3;++i) {
            mappedMaskTiles_[i]=UINT32_MAX;
            if(length<=i)continue;
            int32_t at=rotating::IndexPositions(c,b)[length-1-i];
            if(hot && at<=-2)mappedMaskTiles_[i]=uint32_t(-at-2)/128;
            if(!hot && at>=0)mappedMaskTiles_[i]=uint32_t(at)/128;
        }
    }
    __aicore__ inline void ProcessMappedTopK(const QLICommon::RunInfo &info) {
        auto c = rotating::ContractOf(MappedContext());
        auto full=info;
        full.actS2Size=mappedColdBase_+rotating::ColdCount(c,info.bIdx);
        full.validS2Len=full.actS2Size;
        // Causal masking already follows logical IDs at tile production.
        bool mask=constInfo_.attenMaskFlag;
        constInfo_.attenMaskFlag=false;
        ProcessTopK(full);
        constInfo_.attenMaskFlag=mask;
    }
""",
    )
    text = replace_once(
        text,
        "        rotationContext_ = p;",
        """        rotationContext_ = p;
        Duplicate(prefixThreshold_, uint16_t(0), 256);""",
    )
    text = replace_once(text, "rotationProgressBase_ = (SIM_RESIDENT / 128 / 64) * 64;", "rotationProgressBase_ = 0;")
    text = replace_body(
        text,
        "__aicore__ inline void ProcessResidentTopK(",
        """        mappedCount_ = info.actS2Size;
        prefixActive_ = true;
        bool mask = constInfo_.attenMaskFlag;
        constInfo_.attenMaskFlag = false;
        ProcessTopK(info);
        constInfo_.attenMaskFlag = mask;
        prefixActive_ = false;""",
    )
    text = replace_body(
        text,
        "__aicore__ inline void SaveResident(",
        """        auto c = rotating::ContractOf(MappedContext());
        uint32_t b = query / c->queries, q = query % c->queries;
        auto src=(__local_mem__ uint16_t*)scoreOutLocal_.GetPhyAddr();
        auto dst=(__local_mem__ uint16_t*)prefixThreshold_.GetPhyAddr()+(q%2)*128;
        __VEC_SCOPE__ {
            MicroAPI::RegTensor<uint16_t> minimum,values,reduced;
            auto mask=MicroAPI::CreateMask<uint16_t,MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(minimum,uint16_t(0xffff));
            for(uint16_t part=0;part<16;++part) {
                MicroAPI::LoadAlign(values,src+part*128);
                MicroAPI::Min(minimum,minimum,values,mask);
            }
            MicroAPI::ReduceMin(reduced,minimum,mask);
            MicroAPI::StoreAlign(dst,reduced,mask);
        }
        PipeBarrier<PIPE_V>();
        Simt::VF_CALL<rotating::ContractThreshold>(Simt::Dim3{256, 1, 1},
            MappedContext(), b, q, reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(prefixThreshold_.GetPhyAddr()) + (q % 2) * 128,
            scoreGm.GetPhyAddr() + mappedRow_ * c->score_stride, mappedCount_);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);""",
    )
    text = replace_once(
        text,
        "    // startS2一定能整除kCacheBlockSize_",
        """    {
        auto c=rotating::ContractOf(MappedContext());
        if (mappedHot_ || !rotating::IndexOwned(c,batchId)) {
        GlobalTensor<float> bank;
        bank.SetGlobalBuffer(rotating::ContractPtr<float>(rotating::IndexScale(c,batchId,mappedHot_)));
        DataCopyExtParams copy{1,uint32_t(getLen*sizeof(float)),0,0,0};
        DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(kScaleUB[16*(runInfo.kScaleLoop%2)*s2BaseSize_],bank[startS2],copy,pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        return;
        }
    }
    // startS2一定能整除kCacheBlockSize_""",
    )
    text = text.replace("if (curS2Idx >= SIM_RESIDENT) {", "if (!mappedHot_) {")
    # Preserve native MTE3 score/hit/progress ordering, not PIPE_ALL per tile.
    anchor = "    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);"
    text = replace_once(
        text,
        anchor,
        """    if (!mappedHot_) vec1OutGmOffset += mappedColdBase_;
    // Only the three causal-tail tokens and physical padding need masking.
    // All other tiles retain the original SIMD -> MTE3 path without a SIMT VF.
    if (uint32_t(curS2Idx)+128>mappedExtent_ ||
        info.s2Idx==mappedMaskTiles_[0] || info.s2Idx==mappedMaskTiles_[1] || info.s2Idx==mappedMaskTiles_[2]) {
    Simt::VF_CALL<rotating::MaskContractScores>(Simt::Dim3{256, 1, 1},
        MappedContext(), info.bIdx, curAivS1Idx, curAivS1ProcNum, curS2Idx,
        info.actualSingleProcessSInnerSize, mappedHot_,
        reinterpret_cast<__ubuf__ uint16_t*>(outBase.GetPhyAddr()), UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
        reinterpret_cast<__ubuf__ uint32_t*>(classifyUB_[pingpong * 256].GetPhyAddr()));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }
"""
        + "    if (mappedHot_ || uint32_t(curS2Idx) < mappedExtent_) {\n" + anchor + "\n    }",
    )
    text = text.replace("if (curS2Idx >= SIM_RESIDENT &&", "if (!mappedHot_ &&")
    text = text.replace(
        "rotating::IsTrigger(info.s2Idx + 1)",
        "rotating::ContractEpoch(rotating::ContractOf(MappedContext()),info.s2Idx+1,blockId_/2)>=0",
    )
    text = replace_once(
        text,
        "            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);",
        "            mappedRow_ = vecOffset;\n            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);",
    )
    anchor = """        } else {
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);"""
    text = replace_once(
        text,
        anchor,
        """        } else {
        Simt::VF_CALL<rotating::FinalizeContractTopK>(Simt::Dim3{256, 1, 1},
            MappedContext(), info.bIdx, uint32_t(curS1Idx+rowIdx),
            reinterpret_cast<__ubuf__ uint32_t*>(indicesOutLocal_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(topkSharedTmpLocal_.GetPhyAddr()),
            scoreGm.GetPhyAddr() + vecOffset * constInfo_.kSeqSize, uint32_t(validS2Len));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);""",
    )
    vector.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
