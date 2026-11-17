// SPDX-License-Identifier: Apache-2.0
// Shared preparation for the exact low path. Only errors and capacity
// fallback publish atomics; the common path uses the existing Pack join.
#pragma once

namespace post_index_prepare {
constexpr uint32_t kErrorWord = 24;
constexpr uint32_t kBlockingWord = 26;
constexpr uint32_t kPostFailureWord = 31;

// The caller has completed Producer/OutputWorker's existing SyncAll. Only
// block0 invokes this helper, either inside its existing Pack VF or through
// PrepareOnly if its slice is empty/preflight failed. No entity is accessed.
// All stores bypass the scalar cache: error/mask words in the same 64B line
// can receive peer atomics concurrently without a dirty-line writeback race.
__simt_callee__ inline void PrefixScan(
    __gm__ uint32_t* rowStatus, uint32_t statusStride, uint32_t batch,
    __gm__ uint32_t* prefix) {
    if (threadIdx.x < 32) {
        const uint32_t lane = threadIdx.x;
        const bool geometry = batch && batch <= 16;
        const uint32_t value = geometry && lane < batch ?
            asc_ldcg(rowStatus + lane * statusStride + 1) : 0;
        const uint32_t invalid = asc_ballot(value > 8192);
        uint32_t sum = value <= 8192 ? value : 0;
        for (uint32_t step = 1; step < 32; step <<= 1) {
            const uint32_t previous = asc_shfl(sum, (lane + 32 - step) % 32, 32);
            if (lane >= step) sum += previous;
        }
        if (!lane) {
            asc_stcg(prefix, 0U);
            if (invalid || !geometry)
                asc_atomic_or(reinterpret_cast<__gm__ uint64_t*>(prefix + kErrorWord), uint64_t(1));
        }
        if (geometry && lane < batch) asc_stcg(prefix + lane + 1, sum);
        // PrefixPack may return immediately after its initial validation.
        // Publish these stores even when the ordinary Pack tail is skipped.
        asc_threadfence();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void PrepareOnly(__gm__ Context* ctx, __gm__ uint32_t* prefix) {
    PrefixScan(reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus),
               kRowStatusWords, ctx->shape.batch, prefix);
    asc_threadfence();
}

// The normal path never invokes this VF. It runs after any native preflight
// or Pack error, or when queue capacity requires the existing blocking Get.
// uint64 union deliberately preserves error bits above bit31.
__simt_vf__ __aicore__ LAUNCH_BOUND(32)
inline void PublishPreflight(__gm__ uint32_t* prefix, uint64_t error, uint32_t blocking) {
    if (!threadIdx.x) {
        if (error)
            asc_atomic_or(reinterpret_cast<__gm__ uint64_t*>(prefix + kErrorWord), error);
        if (blocking) asc_atomic_or(prefix + kBlockingWord, 1U);
    }
    asc_threadfence();
}
}  // namespace post_index_prepare
