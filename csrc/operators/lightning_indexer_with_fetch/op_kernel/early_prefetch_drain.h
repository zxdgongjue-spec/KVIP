// SPDX-License-Identifier: Apache-2.0
// Give the Index-update worker sole ownership of prefetch completion. The
// resolver may then publish conflicting Get ranges without waiting for its
// own Q4/output join before someone observes an already completed transfer.
#pragma once

#if KVIP_EARLY_PREFETCH_DRAIN && !V9_UBMEM
__aicore__ inline void CompletePrefetchEarly(__gm__ Args* a, uint32_t b) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto done = reinterpret_cast<__gm__ BatchFlag*>(a->flags) + b * kFlagCount + 7;
    bool profile = (a->prefetchCount & 0x80000000U) != 0;
    uint64_t polls = 0, drain = 0, arrival = 0;
    uint64_t begin = profile ? GetSystemCycle() : 0;
    CompletePrefetch(ctx, b, true, profile ? &polls : nullptr,
                     profile ? &drain : nullptr, profile ? &arrival : nullptr);
    if (profile) {
        // Consumer owns this full cache line. Writing FinalizeStreaming's
        // ft[4:8] here would race its ft[0] store on another core.
        auto saved = reinterpret_cast<__gm__ uint64_t*>(a->times) + b * 64 + 32;
        saved[0] = begin;
        saved[1] = GetSystemCycle();
        saved[2] = polls;
        saved[3] = drain;
        saved[4] = arrival;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(saved), 40);
    }
    // CompletePrefetch has flushed its queue, status and Row updates. Both
    // the transport worker and finalizer acquire this isolated flag line.
    done->error = row->error;
    done->ready = 1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done), sizeof(*done));
}

__aicore__ inline uint64_t JoinEarlyPrefetch(
    __gm__ Args* a, uint32_t b, __gm__ uint64_t* ft, bool profile) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    auto row = reinterpret_cast<__gm__ Row*>(ctx->rows) + b;
    auto done = reinterpret_cast<__gm__ BatchFlag*>(a->flags) + b * kFlagCount + 7;
    uint64_t start = GetSystemCycle();
    while (true) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(done), sizeof(*done));
        if (done->ready) break;
        if (GetSystemCycle() - start > 5000000000ULL) return kvaf_device::kCqTimeout;
    }
    // This core may have read row->slots earlier. Reacquire the two lines
    // updated by the other core instead of reading a cached pre-drain error
    // or pending count. Producer's final-state line at +960 is disjoint.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row), 64);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending), sizeof(row->pending));
    if (profile) {
        auto saved = reinterpret_cast<__gm__ uint64_t*>(a->times) + b * 64 + 32;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(saved), 40);
        // ft[0:4] retain the finalizer timeline. The earlier Consumer interval
        // remains in saved[0:2], otherwise ft[4] could precede ft[0].
        ft[4] = GetSystemCycle();
        ft[5] = saved[2];
        ft[6] = saved[3];
        ft[7] = saved[4];
    }
    return done->error;
}
#endif
