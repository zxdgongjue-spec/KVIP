"""Copy installed LI; add resident TopK without changing its score pipeline."""
from pathlib import Path
import shutil
import sys


def replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f"unsupported installed LI source anchor: {anchor[:100]}")
    return text.replace(anchor, replacement, 1)


def patch(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    # The quantized kernel includes its sibling LI vector primitives.
    sibling = destination.parent / "lightning_indexer"
    if not sibling.exists():
        try:
            sibling.symlink_to((source.parent / "lightning_indexer").resolve(), target_is_directory=True)
        except OSError:
            # Windows source-generation checks need no administrator privilege.
            shutil.copytree(source.parent / "lightning_indexer", sibling)
    vector = destination / "arch35/quant_lightning_indexer_service_vector.h"
    text = vector.read_text()
    anchor = "    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info);"
    text = replace_once(text, anchor, anchor + r'''
    __aicore__ inline void SetPrefixStorage(GM_ADDR output) { prefixOutput_ = output; }
    __aicore__ inline void ProcessResidentTopK(QLICommon::RunInfo info) {
        info.actS2Size = SIM_RESIDENT;
        prefixActive_ = true;
        ProcessTopK(info);
        prefixActive_ = false;
    }
''')
    text = replace_once(text, "private:", r'''private:
    GM_ADDR prefixOutput_ = nullptr;
    bool prefixActive_ = false;
    TBuf<TPosition::VECCALC> prefixThresholdBuf_;
    LocalTensor<uint16_t> prefixThreshold_;
    __aicore__ inline void SaveResident(uint32_t query) {
        auto src = (__local_mem__ uint16_t *)scoreOutLocal_.GetPhyAddr();
        auto dst = (__local_mem__ uint16_t *)prefixThreshold_.GetPhyAddr();
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
        DataCopy(thresholds[q * 128], prefixThreshold_, 128);
    }
''')
    anchor = "    resMm1UB_ = resMm1Buf_.Get<QK_T>();"
    text = replace_once(text, anchor, anchor + "\n    pipe->InitBuffer(prefixThresholdBuf_, 256);\n    prefixThreshold_ = prefixThresholdBuf_.Get<uint16_t>();")
    anchor = "        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);\n        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);"
    text = replace_once(text, anchor, "        if (prefixActive_) {\n            SaveResident(info.indiceOutOffset / topkCount_ + curS1Idx + rowIdx);\n        } else {\n" + anchor.split("\n        SetFlag")[0] + "\n        }\n        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);")
    text = replace_once(text,
        "topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1);",
        "topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, prefixActive_);")
    vector.write_text(text)
    topk = destination / "arch35/vf/quant_lightning_indexer_topk.h"
    text = topk.read_text()
    text = replace_once(text,
        "LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum)",
        "LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum, bool returnScores = false)")
    anchor = "            topkb16gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);"
    text = replace_once(text, anchor, "            if (returnScores) {\n" + anchor.replace("<false>", "<true>") + "\n            } else {\n" + anchor + "\n            }")
    topk.write_text(text)
    kernel = destination / "arch35/quant_lightning_indexer_kernel.h"
    text = kernel.read_text()
    anchor = "        vectorService.ProcessVec1(runInfo);"
    text = replace_once(text, anchor, anchor + "\n        if ((runInfo.s2Idx + 1) * 128 == SIM_RESIDENT) {\n            vectorService.ProcessResidentTopK(runInfo);\n        }")
    kernel.write_text(text)


if __name__ == "__main__":
    patch(Path(sys.argv[1]), Path(sys.argv[2]))
    if len(sys.argv) > 3 and int(sys.argv[3]) >= 4:
        from patch_installed_classify import patch as classify
        classify(Path(sys.argv[2]))
    if len(sys.argv) > 3 and int(sys.argv[3]) >= 5:
        from patch_installed_async_post import patch as post
        post(Path(sys.argv[2]))
