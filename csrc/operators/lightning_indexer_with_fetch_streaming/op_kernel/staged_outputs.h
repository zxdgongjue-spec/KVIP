// SPDX-License-Identifier: Apache-2.0
// Publish the existing service output ABI while original logical IDs are
// still immutable. Different Resolver workers own disjoint source queries.
#pragma once
namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void PublishStagedQueryOutputs(__gm__ uint64_t* a,uint32_t batchCount,
    uint32_t batch,uint32_t query,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* loc) {
    auto owners=reinterpret_cast<__gm__ uint32_t*>(a[9]);
    auto logical=reinterpret_cast<__gm__ int64_t*>(a[3]);
    auto unpack=reinterpret_cast<__gm__ int64_t*>(a[20]);
    auto valid=reinterpret_cast<__gm__ int64_t*>(a[21]);
    auto logicalOut=reinterpret_cast<__gm__ int32_t*>(a[22]);
    auto slotsOut=reinterpret_cast<__gm__ int32_t*>(a[23]);
    if(!a[20]||!a[21]||!a[22]||!a[23])return;
    uint32_t sourceRow=batch*4+query,owner=asc_ldcg(owners+batch);
    for(uint32_t row=0;row<batchCount*4;++row){
        bool active=asc_ldcg(valid+row)!=0;
        uint64_t source=active?uint64_t(asc_ldcg(unpack+row)):0;
        if(active ? source!=sourceRow : (query!=0 || row/4!=batch))continue;
        for(uint32_t k=threadIdx.x;k<kTop;k+=512){
            uint32_t token=active?selected[query*kTop+k]:UINT32_MAX;
            int32_t id=token==UINT32_MAX?-1:int32_t(asc_ldcg(logical+uint64_t(owner)*uint32_t(a[24])+token));
            int32_t slot=active?int32_t(loc[query*kTop+k]):-1;
            asc_stcg(logicalOut+uint64_t(row)*kTop+k,id);
            asc_stcg(slotsOut+uint64_t(row)*kTop+k,slot);
        }
    }
    asc_threadfence();asc_syncthreads();
}
}
