// SPDX-License-Identifier: Apache-2.0
#pragma once
// Test fixture, identical on both arms. Keep the native final misses and the row counter.
// Sort membership is checked independently before replacing just their order.
constexpr uint32_t kPrivateFinalWords = 32 + 2 * 2048;
__simt_callee__ inline bool MatchFinalOrder(__gm__ uint32_t* plan, __ubuf__ uint32_t* ids,
                                             uint32_t count, uint32_t lane, __ubuf__ uint32_t* seen) {
    if (count != asc_ldcg(plan)) return false;
    for(uint32_t i=lane;i<(count+31)/32;i+=32) seen[i]=0;
    asc_threadfence_block();
    bool matched = true;
    auto sorted = plan + 32 + 2048;
    for (uint32_t i=lane; i<count; i+=32) {
        const uint32_t token=ids[i];
        uint32_t lo=0, hi=count;
        while (lo<hi) {
            const uint32_t mid=(hi+lo)/2;
            if (asc_ldcg(sorted+mid)<token) lo=mid+1; else hi=mid;
        }
        if(lo>=count || asc_ldcg(sorted+lo)!=token) { matched=false; continue; }
        const uint32_t bit=1U<<(lo&31U);
        if(asc_atomic_or(seen+(lo>>5),bit)&bit) matched=false;
    }
    if (asc_ballot(matched)==UINT32_MAX) {
        for (uint32_t i=lane;i<count;i+=32) ids[i]=asc_ldcg(plan+32+i);
        asc_threadfence_block();
        return true;
    }
    return false;
}
__simt_callee__ inline uint32_t ReserveCapturedOrder(
    __gm__ uint32_t* plan, __gm__ uint32_t* slots, uint32_t count,
    __gm__ uint32_t* flags, uint32_t generation, uint32_t q, uint32_t lane, bool sameSet, uint32_t maxPolls=65536) {
    uint32_t base = 0;
    auto ready = flags + 24*32 + 4;
    const uint32_t previous=asc_ldcg(plan+1);
    uint32_t status=sameSet ? 1U : 2U;
    if (previous < 4) {
        uint32_t polls=0;
        for (;;) {
            const bool done = lane != 0 || asc_ldcg(ready+previous)==generation;
            if (asc_ballot(done)==UINT32_MAX) break;
            if (++polls==maxPolls) { status|=4U; break; }
        }
    }
    if (!lane) {
        asc_stcg(plan+2,status);
        if (count) base=asc_atomic_add(slots,count);
        asc_threadfence();
        asc_stcg(ready+q,generation);
        asc_threadfence();
    }
    return asc_shfl(base,0);
}
