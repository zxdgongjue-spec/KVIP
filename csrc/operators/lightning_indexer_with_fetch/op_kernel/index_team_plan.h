// SPDX-License-Identifier: Apache-2.0
#pragma once

// The device consumer writes one compact, ordered stream of physical moves.
constexpr uint32_t kIndexPhases = 5;
constexpr uint32_t kIndexMaxChunks = (kPrefetch + kMaxFinal) / CHUNK + kIndexPhases;
constexpr uint32_t kIndexEdgesOffset = 32;
static_assert(kIndexMaxChunks < kIndexEdgesOffset, "chunk directory overlaps edges");
__simt_callee__ inline void RecordIndexChunk(__gm__ uint32_t* plan, __ubuf__ uint32_t* src,
        __ubuf__ uint32_t* dst, __ubuf__ uint32_t* ctl, uint32_t n, uint32_t moves,
        uint32_t resident) {
    uint32_t offset=ctl[5];
    for (uint32_t i=threadIdx.x;i<moves;i+=THREADS) {
        uint32_t j=i<n?i:CHUNK+i-n;
        uint32_t to=dst[j] | (i>=n && dst[j]<resident ? 0x80000000U : 0U);
        asc_stcg(plan+kIndexEdgesOffset+2*(offset+i),src[j]);
        asc_stcg(plan+kIndexEdgesOffset+2*(offset+i)+1,to);
    }
    asc_syncthreads();
    if (!threadIdx.x) {
        ctl[5]=offset+moves;
        asc_stcg(plan+ctl[4],ctl[5]);
    }
    asc_threadfence(); asc_syncthreads();
}
