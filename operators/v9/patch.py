"""Rotation extends Stage4 only; no Stage5 score-path WQE patch is applied."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'base'))
from patch_installed_prefix import replace_once

def patch(root, enabled=True, q4=True):
    kernel = root / 'arch35/quant_lightning_indexer_kernel.h'
    text = '#include "rotation_abi.h"\n' + kernel.read_text()
    # Q4 per AIC makes one cold traversal cover all queries. Apply identically
    # to the controlled native binary; also measure unchanged native Q2.
    if q4:
        text = replace_once(text, 'if (constInfo.tSize <= 64) {', 'if (false) { // rotating Q4 tiling')
    else:
        text = replace_once(text, 'if (constInfo.tSize <= 64) {', 'if (constInfo.tSize <= 2 * V9_AIC_COUNT) {')
    if enabled:
        text = replace_once(text, '    uint32_t aiCoreIdx = 0U;', '    uint32_t aiCoreIdx = 0U;\n    bool rotationServed_ = false;')
        text = replace_once(text, '        matmulService.ComputeMm1(runInfo);', '''        matmulService.ComputeMm1(runInfo);
        if (!rotationServed_ && rotating::EpochAt(s2LoopIdx + 1, aiCoreIdx) >= 0) {
            CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(8);
            rotationServed_ = true;
        }''')
        anchor = '        if (runInfo.isLastS2InnerLoop) {   // 本核s2last'
        text = replace_once(text, anchor, '''        if (!rotationServed_ && rotating::EpochAt(s2LoopIdx + 1, aiCoreIdx) >= 0) {
            vectorService.RotateFetch(rotating::EpochAt(s2LoopIdx + 1, aiCoreIdx));
            CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(8);
            rotationServed_ = true;
        }
''' + anchor)
    if enabled:
        text = replace_once(text, '            vectorService.ProcessResidentTopK(runInfo);',
            '            vectorService.ProcessResidentTopK(runInfo);\n            vectorService.PrepareRotationProgress();')
    kernel.write_text(text)
    if enabled:
        vector_path = root / 'arch35/quant_lightning_indexer_service_vector.h'
        bounded = replace_once(vector_path.read_text(), 'trunkLen_ = TRUNK_LEN_16K;',
            'trunkLen_ = s1BaseSize_ == 2 ? TRUNK_LEN_16K : 8192; // Q2 retains native TopK; Q4 must fit the UB budget')
        vector_path.write_text(bounded)
    if not enabled:
        return
    vector = root / 'arch35/quant_lightning_indexer_service_vector.h'
    text = '#include "' + str(Path(__file__).resolve().parent / 'rotation_device.h') + '"\n' + vector.read_text()
    text = replace_once(text, 'public:', '''public:
    __aicore__ inline void SetRotationContext(GM_ADDR p) {
        rotationContext_ = p;
        rotationProgressGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            reinterpret_cast<__gm__ rotating::Context*>(p)->progress));
    }
    __aicore__ inline void PrepareRotationProgress() {
        // Prefix scores have been published; this 4KB output is dead until final TopK.
        PipeBarrier<PIPE_ALL>();
        rotationProgress_ = scoreOutLocal_.template ReinterpretCast<uint32_t>();
        rotationProgressBase_ = (SIM_RESIDENT / 128 / 64) * 64;
        rotating::PrepareProgress(reinterpret_cast<__ubuf__ uint32_t*>(rotationProgress_.GetPhyAddr()), rotationProgressBase_);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }
    __aicore__ inline void RotateFetch(uint32_t epoch) {
        // Join this producer's hit publication. Other producers remain free.
        PipeBarrier<PIPE_ALL>();
        rotating::Service(rotationContext_, epoch, blockId_ / 2, blockId_ % 2,
            topkSharedTmpLocal_, mrgValueLocal_, indicesOutLocal_);
    }
''')
    text = replace_once(text, 'private:', '''private:
    GM_ADDR rotationContext_ = nullptr;
    GlobalTensor<uint32_t> rotationProgressGm_;
    LocalTensor<uint32_t> rotationProgress_;
    uint32_t rotationProgressBase_ = 0;
''')
    anchor = '    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);\n    CrossCoreSetFlag'
    text = replace_once(text, anchor, '''    if (curS2Idx >= SIM_RESIDENT && ((info.s2Idx + 1) % ROTATION_PUBLISH_TILES == 0 || info.isLastS2InnerLoop ||
        rotating::IsTrigger(info.s2Idx + 1))) {
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
        DataCopyPad(rotationProgressGm_[(info.bIdx * SIM_Q + curAivS1Idx) * 32],
            rotationProgress_[(info.s2Idx % 64) * 16], publish);
    }
''' + anchor)
    # Coalesce physically adjacent scale pages. Preserve arbitrary page tables
    # by ending each DMA run at a discontinuity (and preserve the tail length).
    old = """        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(SCALE_T);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
                                               resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }"""
    new = """        for (int32_t i = 0; i < getLoopNum;) {
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
        }"""
    text = replace_once(text, old, new)
    vf = root / 'arch35/vf/quant_lightning_indexer_vector1.h'
    vftext = vf.read_text()
    # Masked Duplicate defaults to ZEROING: produce the exact 0/1 flags in
    # one instruction instead of two duplicates followed by Select.
    vftext = replace_once(vftext, """    AscendC::MicroAPI::Duplicate(one, uint16_t(1));
    AscendC::MicroAPI::Duplicate(zero, uint16_t(0));
    AscendC::MicroAPI::Select(output, one, zero, selected);""", """    AscendC::MicroAPI::Duplicate<uint16_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(output, uint16_t(1), selected);""")
    # Publish the predicate itself (256 mask bits / 128 BF16 lanes), not
    # 128 uint16 flags. The consumer reloads the same native predicate layout.
    vftext = replace_once(vftext,
        "AscendC::MicroAPI::Duplicate<uint16_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(output, uint16_t(1), selected);\n    AscendC::MicroAPI::StoreAlign(hits, output, all);",
        "AscendC::MicroAPI::StoreAlign(hits, selected);")
    begin = text.index("        GlobalTensor<uint16_t> outputHits;")
    end = text.index("    }\n#endif", begin)
    text = text[:begin] + """        GlobalTensor<uint8_t> outputHits;
        outputHits.SetGlobalBuffer((__gm__ uint8_t *)(prefixOutput_ + SIM_BATCH * prefixBytes));
        DataCopyExtParams hitCopy;
        hitCopy.blockCount = curAivS1ProcNum;
        hitCopy.blockLen = 32;
        hitCopy.srcStride = 7; // classifyUB query stride is 256 bytes.
        hitCopy.dstStride = SIM_HISTORY / 4 - 32;
        DataCopyPad(outputHits[(info.bIdx * SIM_Q + curAivS1Idx) * (SIM_HISTORY / 4) + curS2Idx / 4],
                    classifyUB_[pingpong * 256].template ReinterpretCast<uint8_t>(), hitCopy);
""" + text[end:]
    vf.write_text(vftext)
    vector.write_text(text)

if __name__ == '__main__':
    patch(Path(sys.argv[1]), bool(int(sys.argv[2])), bool(int(sys.argv[3])) if len(sys.argv)>3 else True)
