"""Fuse cold threshold comparison into the BF16-weight score VF, no new VF."""
from pathlib import Path
import re

from patch_installed_prefix import replace_once


def functions(text, qualifier):
    for match in re.finditer(qualifier + r" inline void \w+\(", text):
        start = match.start()
        brace = text.index("{", match.end())
        depth, end = 1, brace + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        yield text[start:end]


def patch(destination):
    vf = destination / "arch35/vf/quant_lightning_indexer_vector1.h"
    text = vf.read_text()
    extra = r'''
__simd_callee__ inline void ClassifyScore(AscendC::MicroAPI::RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits,
    AscendC::MicroAPI::MaskReg& all) {
    AscendC::MicroAPI::RegTensor<uint16_t> limit, one, zero, output;
    AscendC::MicroAPI::MaskReg selected;
    AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(limit, threshold);
    AscendC::MicroAPI::Compare<uint16_t, CMPMODE::GE>(selected, score, limit, all);
    AscendC::MicroAPI::Duplicate(one, uint16_t(1));
    AscendC::MicroAPI::Duplicate(zero, uint16_t(0));
    AscendC::MicroAPI::Select(output, one, zero, selected);
    AscendC::MicroAPI::StoreAlign(hits, output, all);
}
'''
    for function in functions(text, "__simd_callee__"):
        signature = function.split("{", 1)[0]
        if "Finalize(" not in signature or "__ubuf__ uint16_t*" not in signature:
            continue
        body = function.replace("Finalize(", "FinalizeClassify(", 1)
        head, tail = body.split("{", 1)
        head = head.rstrip()[:-1] + ", __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits)\n"
        if "out1_" in signature:
            tail = tail.rsplit("}", 1)[0] + "ClassifyScore(regOut[0], threshold, hits, maskAllB16);\nClassifyScore(regOut[1], threshold + 128, hits + 128, maskAllB16);\n}"
        else:
            tail = tail.rsplit("}", 1)[0] + "ClassifyScore(regOut, threshold, hits, maskAllB16);\n}"
        extra += head + "{" + tail + "\n"
    found = 0
    for function in functions(text, "__simd_vf__"):
        signature = function.split("{", 1)[0]
        if "__ubuf__ bfloat16_t* weight" not in signature:
            continue
        found += 1
        body = function.replace("MulWeightAndReduceSum", "ClassifyMulWeightAndReduceSum", 1)
        head, tail = body.split("{", 1)
        head = head.rstrip()[:-1] + ", __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits)\n"
        tail = tail.replace("Finalize(", "FinalizeClassify(")
        tail = tail.replace("out0_, out1_);", "out0_, out1_, threshold, hits);")
        tail = tail.replace("maskAllB16, out_);", "maskAllB16, out_, threshold, hits);")
        extra += "template <typename QK_T>\n" + head + "{" + tail + "\n"
    if found != 2:
        raise ValueError("expected two BF16 score VF overloads")
    wrapper = next(functions(text, "__aicore__"))
    if "BatchMulWeightAndReduceSum" not in wrapper:
        raise ValueError("unexpected score wrapper")
    wrapper = wrapper.replace("BatchMulWeightAndReduceSum", "BatchClassifyScore", 1)
    head, tail = wrapper.split("{", 1)
    head = head.rstrip()[:-1] + ", __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits)\n"
    tail = tail.replace("MulWeightAndReduceSum", "ClassifyMulWeightAndReduceSum")
    tail = tail.replace("gSize);", "gSize, threshold, hits);")
    extra += "template <typename QK_T, typename W_T, typename SCALE_T, typename SCORE_T>\n" + head + "{" + tail
    # Append within vector1 namespace; keep original functions unchanged.
    text = replace_once(text, "\n}\n\n#endif", "\n" + extra + "\n}\n\n#endif")
    vf.write_text(text)

    vector = destination / "arch35/quant_lightning_indexer_service_vector.h"
    text = vector.read_text()
    text = replace_once(text, "pipe->InitBuffer(prefixThresholdBuf_, 256);", "pipe->InitBuffer(prefixThresholdBuf_, 512);\n    pipe->InitBuffer(classifyBuf_, 1024);\n    classifyUB_ = classifyBuf_.Get<uint16_t>();")
    text = replace_once(text, "    bool prefixActive_ = false;", "    bool prefixActive_ = false;\n    TBuf<TPosition::VECCALC> classifyBuf_;\n    LocalTensor<uint16_t> classifyUB_;")
    text = replace_once(text, "prefixThreshold_.GetPhyAddr();", "prefixThreshold_.GetPhyAddr() + (query % 2) * 128;")
    text = replace_once(text, "DataCopy(thresholds[q * 128], prefixThreshold_, 128);", "DataCopy(thresholds[q * 128], prefixThreshold_[(query % 2) * 128], 128);")
    start = text.index("    vector1::BatchMulWeightAndReduceSum(")
    end = text.index(";", start) + 1
    original = text[start:end]
    classified = original.replace("BatchMulWeightAndReduceSum", "BatchClassifyScore")
    classified = classified[:-2] + ",\n        (__ubuf__ uint16_t *)prefixThreshold_.GetPhyAddr() + (curAivS1Idx % 2) * 128,\n        (__ubuf__ uint16_t *)classifyUB_.GetPhyAddr() + pingpong * 256);"
    text = text[:start] + "    if (curS2Idx >= SIM_RESIDENT) {\n" + classified + "\n    } else {\n" + original + "\n    }" + text[end:]
    anchor = "    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);"
    export = r'''
#if SIM_CAPTURE_HITS
    if (curS2Idx >= SIM_RESIDENT) {
        constexpr uint32_t prefixBytes = SIM_Q * 2048 * 6 + SIM_Q * 256;
        GlobalTensor<uint16_t> outputHits;
        outputHits.SetGlobalBuffer((__gm__ uint16_t *)(prefixOutput_ + SIM_BATCH * prefixBytes));
        DataCopyExtParams hitCopy = copyOutParams;
        hitCopy.srcStride = 0;
        DataCopyPad(outputHits[(info.bIdx * SIM_Q + curAivS1Idx) * SIM_HISTORY + curS2Idx],
                    classifyUB_[pingpong * 256], hitCopy);
    }
#endif
'''
    text = replace_once(text, anchor, anchor + export)
    vector.write_text(text)
