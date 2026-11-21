// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace rotating {
constexpr uint32_t kCompactMissLimit=64;
constexpr uint32_t kPositionBits=11;
constexpr uint32_t kPositionMask=(1U<<kPositionBits)-1;
// Run inside the final bitmap merge. Its secondary shards are dead at this point.
// Order entries by the original TopK position, including duplicate tokens.
__simt_callee__ inline void ExportCompactMisses(__ubuf__ uint32_t* indices,
    __ubuf__ uint32_t* loc,__ubuf__ uint32_t* scan,__gm__ uint32_t* entries,
    __ubuf__ uint32_t* count, uint32_t capacity=kCompactMissLimit){
    uint32_t tid=threadIdx.x,warp=tid/32,lane=tid%32,bits=0;
    for(uint32_t j=0;j<4;++j){
        uint32_t i=tid*4+j;
        if(loc[i]==UINT32_MAX && indices[i]!=UINT32_MAX)bits|=1U<<j;
    }
    uint32_t own=__popc(bits),n=own;
    for(uint32_t d=1;d<32;d*=2){auto x=asc_shfl_up(n,d);if(lane>=d)n+=x;}
    if(lane==31)scan[warp]=n;
    asc_syncthreads();
    if(!warp){
        uint32_t v=lane<16?scan[lane]:0;
        for(uint32_t d=1;d<16;d*=2){auto x=asc_shfl_up(v,d);if(lane>=d)v+=x;}
        if(lane<16)scan[lane]=v;
    }
    asc_syncthreads();
    if(scan[15]<=capacity){
        uint32_t out=n-own+(warp?scan[warp-1]:0);
        for(uint32_t j=0;j<4;++j)if(bits&(1U<<j)){
            uint32_t i=tid*4+j;asc_stcg(entries+out++,(indices[i]<<kPositionBits)|i);
        }
    }
    if(!tid)*count=scan[15]+1; // zero identifies classifiers without this output
    asc_threadfence();asc_syncthreads();
}

// LI has completed before the tail launch. Packets stay immutable even when
// a peer turns its sourceBlocks and locations into fetch descriptors.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void LoadOwnerMissPackets(
    __gm__ uint32_t *packets, __ubuf__ uint32_t *ids, __ubuf__ uint32_t *positions,
    uint32_t q, uint32_t last, uint32_t n0, uint32_t n1, uint32_t n2, uint32_t n3) {
    uint32_t sizes[4] = {n0, n1, n2, n3}, cursor = 0;
    constexpr uint32_t stride = 1U << kPositionBits;
    for (uint32_t j = 0; j <= last; ++j) {
        for (uint32_t i = threadIdx.x; i < sizes[j]; i += 512) {
            uint32_t entry = asc_ldcg(packets + j * stride + i);
            ids[cursor + i] = entry >> kPositionBits;
            if (j == q)
                positions[i] = entry & kPositionMask;
        }
        cursor += sizes[j];
    }
}
}
