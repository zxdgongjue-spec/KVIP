// SPDX-License-Identifier: Apache-2.0
#pragma once

// Only thread zero elects one worker for this row. Preserve the owner until
// the next invocation's planner opens a new claim.
__simt_callee__ inline void TryFourthBody(__gm__ uint32_t* claim,uint32_t block,__ubuf__ uint32_t* won) {
    if(threadIdx.x==0)*won=asc_atomic_cas(claim,0U,block+1)==0U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ClaimFourth(
        __gm__ uint32_t* claim,uint32_t block,__ubuf__ uint32_t* won) {
    TryFourthBody(claim,block,won);
}

// Lane 0 also owns DDR addresses, lane 1 logical IDs and lane 2 scales.
// Invalidation applies to freed hot positions, exactly as in ApplyBatch.
// A whole chunk fits in private UB even when its swaps include long cycles.
template<bool Claim=false>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void CopyFields(
    __gm__ uint32_t* edges,uint32_t chunks,uint32_t lane,
    __gm__ uint64_t* keys,__gm__ uint32_t* scales,__gm__ uint64_t* addr,
    __gm__ uint64_t* logical,__gm__ uint32_t* valid,__ubuf__ uint64_t* saved,
    __gm__ uint32_t* claim=nullptr,uint32_t block=0) {
    uint32_t tid=threadIdx.x,first=0;
    auto src=reinterpret_cast<__ubuf__ uint32_t*>(saved+5*kMaxMoves);
    auto dst=src+kMaxMoves;
    for(uint32_t chunk=0;chunk<chunks;++chunk){
        uint32_t end=asc_ldcg(edges+chunk), n=end-first;
        for(uint32_t i=tid;i<n;i+=256){
            src[i]=asc_ldcg(edges+kIndexEdgesOffset+2*(first+i));
            dst[i]=asc_ldcg(edges+kIndexEdgesOffset+2*(first+i)+1);
        }
        asc_syncthreads();
        for(uint32_t x=tid;x<4*n;x+=256){
            uint32_t i=x/4,j=x%4;
            saved[x]=asc_ldcg(keys+uint64_t(src[i])*16+4*lane+j);
        }
        for(uint32_t i=tid;i<n;i+=256){
            if(lane<2)saved[4*kMaxMoves+i]=asc_ldcg((lane?logical:addr)+src[i]);
            else if(lane==2)saved[4*kMaxMoves+i]=asc_ldcg(scales+src[i]);
        }
        asc_syncthreads();
        for(uint32_t x=tid;x<4*n;x+=256){
            uint32_t i=x/4,j=x%4;
            asc_stcg(keys+uint64_t(dst[i]&0x7fffffffU)*16+4*lane+j,saved[x]);
        }
        for(uint32_t i=tid;i<n;i+=256){
            uint32_t at=dst[i]&0x7fffffffU;
            if(lane<2)asc_stcg((lane?logical:addr)+at,saved[4*kMaxMoves+i]);
            else if(lane==2){
                asc_stcg(scales+at,uint32_t(saved[4*kMaxMoves+i]));
                if(dst[i]>>31)asc_stcg(valid+at,0U);
            }
        }
        // A later chunk may use a position written by this one.
        asc_threadfence(); asc_syncthreads(); first=end;
    }
    // All threads finished the last scatter and no longer use the saved keys.
    // Return the CAS result through that same UB allocation and V_S event.
    if constexpr(Claim)TryFourthBody(claim,block,reinterpret_cast<__ubuf__ uint32_t*>(saved));
}
