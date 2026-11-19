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
    __ubuf__ uint32_t* count){
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
    if(scan[15]<=kCompactMissLimit){
        uint32_t out=n-own+(warp?scan[warp-1]:0);
        for(uint32_t j=0;j<4;++j)if(bits&(1U<<j)){
            uint32_t i=tid*4+j;asc_stcg(entries+out++,(indices[i]<<kPositionBits)|i);
        }
    }
    if(!tid)*count=scan[15]+1; // zero identifies classifiers without this output
    asc_threadfence();asc_syncthreads();
}
}
