"""Copy current production LI into the build directory; patch only candidate output."""
from pathlib import Path
import shutil
import sys
repo = Path(__file__).resolve().parents[3]
dst = Path(sys.argv[1]) / 'csrc/operators'
shutil.copytree(repo / 'csrc/operators', dst, dirs_exist_ok=True)
p = dst / 'lightning_indexer_with_fetch/op_kernel/indexer/indexer_score.h'
s = p.read_text()
helper = '''
__simd_callee__ inline void ClassifyColdOutput(AscendC::MicroAPI::RegTensor<uint16_t>& score,
    __ubuf__ uint16_t* threshold, __ubuf__ uint16_t* hits, AscendC::MicroAPI::MaskReg& all) {
#if EXPERIMENT_COMPACT
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> limit,packed,ones;RegTensor<uint32_t> total;
    RegTensor<int16_t> positions;MaskReg selected;
    LoadAlign<uint16_t,LoadDist::DIST_BRC_B16>(limit,threshold);
    Compare<uint16_t,CMPMODE::GT>(selected,score,limit,all);
    ClearSpr<SpecialPurposeReg::AR>();
    Squeeze<uint16_t,AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,score,selected);
    UnalignRegForStore scoreStore;auto scoreDst=hits;
    StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(scoreDst,packed,scoreStore);StoreUnAlignPost(scoreDst,scoreStore);
    Arange(positions,int16_t(0));ClearSpr<SpecialPurposeReg::AR>();
    Squeeze<uint16_t,AscendC::MicroAPI::GatherMaskMode::STORE_REG>(packed,reinterpret_cast<RegTensor<uint16_t>&>(positions),selected);
    UnalignRegForStore idStore;auto idDst=hits+128;
    StoreUnAlign<uint16_t,PostLiteral::POST_MODE_UPDATE>(idDst,packed,idStore);StoreUnAlignPost(idDst,idStore);
    Duplicate<uint16_t,MaskMergeMode::ZEROING>(ones,uint16_t(1),selected);
    ReduceSum(total,ones,all);
    auto one=CreateMask<uint32_t,MaskPattern::VL1>();StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(hits+256),total,one);
#else
    ClassifyScore(score,threshold,hits,all);
#endif
}
'''
needle='__simd_callee__ inline void ReduceSumFinalizeClassify('
assert s.count(needle)==1
s=s.replace(needle,helper+'\n'+needle)
s=s.replace('ClassifyScore(regOut, threshold, hits, maskAllB16);','ClassifyColdOutput(regOut, threshold, hits, maskAllB16);')
s=s.replace('ClassifyScore(regOut[0], threshold, hits, maskAllB16);','ClassifyColdOutput(regOut[0], threshold, hits, maskAllB16);')
s=s.replace('ClassifyScore(regOut[1], threshold + 128, hits + 128, maskAllB16);','ClassifyColdOutput(regOut[1], threshold + 128, hits + (EXPERIMENT_COMPACT ? 512 : 128), maskAllB16);')
p.write_text(s)
p = dst / 'lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h'
s = p.read_text().replace('pipe->InitBuffer(classifyBuf_, 1024);','pipe->InitBuffer(classifyBuf_, EXPERIMENT_COMPACT ? 4096 : 1024);')
s = s.replace('classifyUB_.GetPhyAddr() + pingpong * 256','classifyUB_.GetPhyAddr() + pingpong * (EXPERIMENT_COMPACT ? 1024 : 256)')
a=s.index('        DataCopyExtParams hitCopy;');b=s.index('\n    }\n#endif',a)
old=s[a:b]
new='''        DataCopyExtParams hitCopy;
        hitCopy.blockCount=curAivS1ProcNum;hitCopy.blockLen=544;
        hitCopy.srcStride=15; // UB query stride 1024 bytes, payload 544 bytes.
        hitCopy.dstStride=(runtimeShape_.history/128-1)*544;
        DataCopyPad(outputHits[((info.bIdx*SIM_Q+curAivS1Idx)*(runtimeShape_.history/128)+curS2Idx/128)*544],
            classifyUB_[pingpong*1024].template ReinterpretCast<uint8_t>(),hitCopy);'''
s=s[:a]+'#if EXPERIMENT_COMPACT\n'+new+'\n#else\n'+old+'\n#endif'+s[b:]
p.write_text(s)
