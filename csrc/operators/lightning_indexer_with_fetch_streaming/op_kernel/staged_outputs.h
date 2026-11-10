// SPDX-License-Identifier: Apache-2.0
// Prepare one query's service outputs while the original logical IDs remain
// immutable. The caller copies the resulting contiguous rows with MTE3.
#pragma once
namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PrepareStagedQueryOutputs(__gm__ uint64_t* a,uint32_t batchCount,
    uint32_t batch,uint32_t query,__ubuf__ uint32_t* selected,
    __ubuf__ uint32_t* rowMasks,__ubuf__ uint32_t* logicalValues) {
    auto owners=reinterpret_cast<__gm__ uint32_t*>(a[9]);
    auto logical=reinterpret_cast<__gm__ int64_t*>(a[3]);
    auto unpack=reinterpret_cast<__gm__ int64_t*>(a[20]);
    auto valid=reinterpret_cast<__gm__ int64_t*>(a[21]);
    uint32_t sourceRow=batch*4+query,owner=asc_ldcg(owners+batch);
    for(uint32_t row=threadIdx.x;row<batchCount*4;row+=512){
        bool active=asc_ldcg(valid+row)!=0;
        uint64_t source=active?uint64_t(asc_ldcg(unpack+row)):0;
        if(active && source==sourceRow)asc_atomic_or(rowMasks+row/32,1U<<(row%32));
        else if(!active && query==0 && row/4==batch)asc_atomic_or(rowMasks+2+row/32,1U<<(row%32));
    }
    for(uint32_t k=threadIdx.x;k<kTop;k+=512){
        uint32_t token=selected[query*kTop+k];
        logicalValues[k]=token==UINT32_MAX?UINT32_MAX:uint32_t(asc_ldcg(logical+uint64_t(owner)*uint32_t(a[24])+token));
    }
    asc_syncthreads();
}
}
