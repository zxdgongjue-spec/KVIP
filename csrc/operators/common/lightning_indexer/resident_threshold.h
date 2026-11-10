// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "topk_common.h"

#ifndef V9_THRESHOLD_ONLY
#define V9_THRESHOLD_ONLY 0
#endif
static_assert(V9_THRESHOLD_ONLY==0 || V9_THRESHOLD_ONLY==1,"Threshold-only prefix must be boolean");

namespace kvip_prefix {
// The one-trunk path has exactly the same zero padding as stock LiTopKVF.
// Short/invalid queries and multi-trunk prefixes retain the original path.
__aicore__ inline bool UseThresholdOnly(bool prefix,int32_t valid,uint32_t topk,uint32_t trunk) {
    return V9_THRESHOLD_ONLY && prefix && topk==2048 && valid>=int32_t(topk) &&
        valid<=int32_t(trunk) && valid<=16384;
}

__simd_vf__ inline void BroadcastThreshold(__ubuf__ uint16_t* output,__ubuf__ uint32_t* kth) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> value;
    auto all=CreateMask<uint16_t,MaskPattern::ALL>();
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    LoadAlign<uint16_t,LoadDist::DIST_BRC_B16>(value,reinterpret_cast<__ubuf__ uint16_t*>(kth));
    StoreAlign(output,value,all);
}

// Reuse the production BF16-sortable-key histogram stages, including their
// exact tie cutoff. No float conversion, approximate rank or new sort is used.
// The input and work buffers are the caller's existing TopK scratch. All
// returned values occupy the existing 128-halfword per-query threshold row.
__aicore__ inline void ResidentThresholdOnly(AscendC::LocalTensor<uint16_t> input,
    AscendC::LocalTensor<uint32_t> work,AscendC::LocalTensor<uint16_t> output,
    uint32_t padded,uint32_t topk) {
    auto scores=reinterpret_cast<__ubuf__ uint16_t*>(input.GetPhyAddr());
    auto hist=reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr());
    auto high=hist+256,low=high+256,kth=low+256;
    topkb16gather::HistogramsHighVFImpl<uint16_t>(hist,scores,padded/256,true);
    topkb16gather::FindHighTargetBinVFImpl(high,kth,hist,padded-topk+1);
    topkb16gather::HistogramsLowVFImpl<uint16_t>(hist,scores,high,padded/256,true);
    topkb16gather::FindKthVFImpl(kth,hist,high,low);
    AscendC::PipeBarrier<PIPE_V>();
    BroadcastThreshold(reinterpret_cast<__ubuf__ uint16_t*>(output.GetPhyAddr()),kth);
}
}  // namespace kvip_prefix
