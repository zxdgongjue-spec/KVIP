#pragma once
#include "fusion3_contract.h"
#include "index_cache_update.h"

namespace fusion3 {
// Existing consumers enqueue attention without reading the final status. On
// any failure, retract every output belonging to this source before returning.
// A malformed source has no source owner and is poisoned by its destination.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PoisonFailedOutputs(
    __gm__ uint64_t* a, uint32_t b, uint32_t batch, bool allDestinations) {
    auto logical = reinterpret_cast<__gm__ int32_t*>(a[22]);
    auto slots = reinterpret_cast<__gm__ int32_t*>(a[23]);
    auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
    auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
    for (uint32_t row = 0; row < batch * 4; ++row) {
        uint32_t owner = row / 4;
        if (!allDestinations && asc_ldcg(valid + row)) {
            int64_t source = asc_ldcg(unpack + row);
            if (source >= 0 && uint64_t(source) < uint64_t(batch) * 4) owner = uint32_t(source) / 4;
        }
        if (owner != b) continue;
        for (uint32_t k = threadIdx.x; k < 2048; k += 512) {
            asc_stcg(logical + uint64_t(row) * 2048 + k, int32_t(-1));
            asc_stcg(slots + uint64_t(row) * 2048 + k, int32_t(-1));
        }
    }
    asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void StoreUpdateResult(
    __gm__ uint64_t* a, uint32_t b, uint32_t error, bool emptyPlan) {
    if (!threadIdx.x) {
        if (a[6]) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a[6]) + b, error);
        if (emptyPlan && a[17]) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a[17]) +
                               uint64_t(b) * kExtendedTransferWords, uint32_t(0));
    }
    asc_threadfence();
}
// Hashes are dead on entry; records and prefix/status remain live. The output
// occupies only [0,32800), disjoint from records [65536,98304).
template<uint32_t Threads>
__simt_callee__ inline void CompactSmall(__ubuf__ uint32_t* work,
    __ubuf__ uint32_t* status, uint32_t resident) {
    uint32_t tid = threadIdx.x;
    auto dst = work + 2 * kPlanHash, src = dst + kPlanRecords;
    constexpr uint32_t per = kPlanRecords / Threads;
    uint32_t count = 0;
    for (uint32_t i = tid * per; i < (tid + 1) * per; ++i)
        count += src[i] != UINT32_MAX && src[i] != dst[i];
    uint32_t rank = PlanPrefix<Threads>(count, status + 32);
    for (uint32_t i = tid; i < kTransferHeaderWords; i += Threads) work[i] = 0;
    asc_syncthreads();
    if (tid == Threads - 1) work[0] = rank + count;
    for (uint32_t i = tid * per; i < (tid + 1) * per; ++i) {
        uint32_t s = src[i], d = dst[i];
        if (s == UINT32_MAX || s == d) continue;
        if (i >= kPlanCandidates && d < resident) d |= kInvalidateHot;
        work[kTransferHeaderWords + 2 * rank] = d;
        work[kTransferHeaderWords + 2 * rank + 1] = s;
        ++rank;
    }
    asc_syncthreads();
}

// The producer supplies a high-water prefix, not 10240 live records. Keep
// closure semantics/order identical to BuildExtendedPlan, but initialize only
// that prefix. Both extra columns are completely written before publication.
template<uint32_t Threads>
__simt_callee__ inline void BuildBoundedExtended(__gm__ uint32_t* inputSlots,
    __gm__ uint32_t* changes, uint32_t count, uint32_t history, uint32_t resident,
    __ubuf__ uint32_t* hashes, __ubuf__ uint32_t* status, __gm__ uint32_t* records) {
    uint32_t tid = threadIdx.x;
    if (count > kExtendedCandidates) { if (!tid) status[0] = 1; return; }
    auto dstSet = hashes, srcSet = hashes + kExtendedHash;
    auto dst = records, src = records + kExtendedRecords;
    for (uint32_t i = tid; i < 2 * kExtendedHash; i += Threads) hashes[i] = UINT32_MAX;
    for (uint32_t i = tid; i < count; i += Threads) {
        asc_stcg(dst + i, UINT32_MAX); asc_stcg(src + i, UINT32_MAX);
    }
    asc_threadfence(); asc_syncthreads();
    for (uint32_t i = tid; i < count; i += Threads) {
        uint32_t s = asc_ldcg(changes + i), d = asc_ldcg(inputSlots + i);
        if (s == UINT32_MAX) continue;
        if (s >= history || d >= resident) { asc_atomic_or(status, 1U); continue; }
        asc_stcg(src + i, s); asc_stcg(dst + i, d);
        if (!ExtendedInsert(dstSet, d) || !ExtendedInsert(srcSet, s)) asc_atomic_or(status, 1U);
    }
    asc_threadfence(); asc_syncthreads();
    if (status[0]) return;
    uint32_t per = (count + Threads - 1) / Threads, out = 0, free = 0;
    for (uint32_t i = tid * per; i < (tid + 1) * per && i < count; ++i) {
        uint32_t s = asc_ldcg(src + i), d = asc_ldcg(dst + i);
        if (s != UINT32_MAX) { out += !ExtendedContains(srcSet, d); free += !ExtendedContains(dstSet, s); }
    }
    uint32_t outRank = kExtendedCandidates + PlanPrefix<Threads>(out, status + 32);
    uint32_t freeRank = kExtendedCandidates + PlanPrefix<Threads>(free, status + 64);
    if (tid == Threads - 1) status[1] = outRank + out - kExtendedCandidates;
    for (uint32_t i = tid * per; i < (tid + 1) * per && i < count; ++i) {
        uint32_t s = asc_ldcg(src + i), d = asc_ldcg(dst + i);
        if (s != UINT32_MAX) {
            if (!ExtendedContains(srcSet, d)) asc_stcg(src + outRank++, d);
            if (!ExtendedContains(dstSet, s)) asc_stcg(dst + freeRank++, s);
        }
    }
    asc_threadfence(); asc_syncthreads();
}

template<uint32_t Threads>
__simt_callee__ inline void PublishBoundedExtended(__gm__ uint32_t* records,
    __ubuf__ uint32_t* status, __gm__ uint32_t* plan, uint32_t resident, uint32_t primary) {
    uint32_t tid = threadIdx.x, extras = status[1];
    if (primary > kExtendedCandidates || extras > primary) { if (!tid) status[0] = 1; return; }
    uint32_t total = primary + extras, per = (total + Threads - 1) / Threads, count = 0;
    auto dst = records, src = records + kExtendedRecords;
    for (uint32_t i = tid * per; i < (tid + 1) * per && i < total; ++i) {
        uint32_t record = i < primary ? i : kExtendedCandidates + i - primary;
        uint32_t s = asc_ldcg(src + record), d = asc_ldcg(dst + record);
        count += s != UINT32_MAX && s != d;
    }
    uint32_t rank = PlanPrefix<Threads>(count, status + 32);
    if (tid == Threads - 1) asc_stcg(plan, rank + count);
    for (uint32_t i = tid * per; i < (tid + 1) * per && i < total; ++i) {
        uint32_t record = i < primary ? i : kExtendedCandidates + i - primary;
        uint32_t s = asc_ldcg(src + record), d = asc_ldcg(dst + record);
        if (s == UINT32_MAX || s == d) continue;
        if (record >= kExtendedCandidates && d < resident) d |= kInvalidateHot;
        asc_stcg(plan + kTransferHeaderWords + 2 * rank, d);
        asc_stcg(plan + kTransferHeaderWords + 2 * rank + 1, s); ++rank;
    }
}

template<uint32_t Threads, bool Extended>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void MakePlan(
    __gm__ uint64_t* a, uint32_t b, __ubuf__ uint32_t* work) {
    uint32_t tid = threadIdx.x;
    auto status = work + (Extended ? 2 * kExtendedHash : kPlanWords);
    if (tid < 128) status[tid] = 0;
    asc_syncthreads();
    auto slots = reinterpret_cast<__gm__ uint32_t*>(a[4]) + b * kExtendedCandidates;
    auto changes = reinterpret_cast<__gm__ uint32_t*>(a[5]) + b * kExtendedCandidates;
    auto plan = reinterpret_cast<__gm__ uint32_t*>(a[17]) + uint64_t(b) * kExtendedTransferWords;
    if constexpr (!Extended) {
        BuildUpdatePlan<Threads>(slots, changes, uint32_t(a[24]), uint32_t(a[25]), work, status);
        asc_syncthreads();
        if (!status[0]) CompactSmall<Threads>(work, status, uint32_t(a[25]));
    } else {
        uint32_t count = asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[26]) + b);
        auto records = reinterpret_cast<__gm__ uint32_t*>(a[27]) + uint64_t(b) * kExtendedWorkWords;
        BuildBoundedExtended<Threads>(slots, changes, count, uint32_t(a[24]), uint32_t(a[25]),
                                     work, status, records);
        asc_syncthreads();
        if (!status[0]) PublishBoundedExtended<Threads>(records, status, plan, uint32_t(a[25]), count);
    }
    asc_threadfence();
    asc_syncthreads();
    // Keep the existing observable GM plan ABI, while Apply-small reads UB.
    if constexpr (!Extended) {
        uint32_t words = status[0] ? 0 : kTransferHeaderWords + 2 * work[0];
        for (uint32_t i = tid; i < words; i += Threads) asc_stcg(plan + i, work[i]);
    }
    if (!tid) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a[6]) + b, status[0] ? kPlanError : 0U);
        if (status[0]) { asc_stcg(plan, uint32_t(0)); work[0] = 0; }
    }
    asc_threadfence();
    asc_syncthreads();
}
} // namespace fusion3
