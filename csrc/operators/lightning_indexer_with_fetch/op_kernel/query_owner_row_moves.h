// SPDX-License-Identifier: Apache-2.0
constexpr uint32_t kIndexRowMask = 0x7fffffffU;
constexpr uint32_t kDependencyBits = 8, kDependencyMask = (1U << kDependencyBits) - 1;
constexpr uint32_t kRowCopyCapacity = 2 * CHUNK;
// Each staged row has 16 key words and 3 metadata words (64-bit), followed
// by three 32-bit arrays for its destination, dependencies and source row.
constexpr uint32_t kRowCopyWords = (2 * (16 + 3) + 3) * kRowCopyCapacity;

// RecordIndexChunk emits a closed permutation, including displaced victims.
// Track both ends: a destination is also a source whose old value must survive.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void StampDependencies(
    __gm__ uint32_t *plan, __gm__ uint32_t *deps, __gm__ uint32_t *last, uint32_t first, uint32_t end) {
    auto pairs = plan + kIndexEdgesOffset;
    for (uint32_t c = first; c < end; ++c) {
        uint32_t lo = c ? asc_ldcg(plan + c - 1) : 0, hi = asc_ldcg(plan + c);
        for (uint32_t i = lo + threadIdx.x; i < hi; i += 512) {
            uint32_t src = asc_ldcg(pairs + 2 * i), dst = asc_ldcg(pairs + 2 * i + 1) & kIndexRowMask;
            uint32_t readAfter = asc_ldcg(last + src), writeAfter = asc_ldcg(last + dst);
            // Chunks of the same query are already issued in order. Zero
            // represents an input that is ready without a cross-query wait.
            asc_stcg(deps + i, (readAfter <= first ? readAfter : 0) |
                               ((writeAfter <= first ? writeAfter : 0) << kDependencyBits));
        }
        asc_threadfence();
        asc_syncthreads();
        for (uint32_t i = lo + threadIdx.x; i < hi; i += 512) {
            asc_stcg(last + (asc_ldcg(pairs + 2 * i + 1) & kIndexRowMask), c + 1);
        }
        asc_threadfence();
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void CopyReadyRows(
    __gm__ Args *a, uint32_t owner, uint32_t lo, uint32_t hi, __gm__ uint32_t *pairs, __gm__ uint32_t *dependencies,
    uint32_t previous, uint32_t ready, __ubuf__ uint64_t *buf, __ubuf__ uint32_t *pending) {
    constexpr uint32_t cap = kRowCopyCapacity;
    auto key = reinterpret_cast<__gm__ uint64_t *>(a->keys) + uint64_t(owner) * a->history * 16;
    auto scale = reinterpret_cast<__gm__ uint32_t *>(a->scales) + uint64_t(owner) * a->history;
    auto bind = reinterpret_cast<__gm__ uint64_t *>(a->updateBindings);
    auto addr = reinterpret_cast<__gm__ uint64_t *>(a->addresses) + uint64_t(owner) * bind[7];
    auto id = reinterpret_cast<__gm__ uint64_t *>(a->logical) + uint64_t(owner) * a->history;
    auto valid = reinterpret_cast<__gm__ uint32_t *>(a->valid) + uint64_t(owner) * a->resident;
    auto dest = reinterpret_cast<__ubuf__ uint32_t *>(buf + 19 * cap);
    auto masks = dest + cap, srcRows = masks + cap;
    uint32_t tid = threadIdx.x, n = hi - lo;
    if (!tid) {
        *pending = 0;
    }
    asc_syncthreads();
    for (uint32_t i = tid; i < n; i += 512) {
        if (!previous) {
            masks[i] = asc_ldcg(dependencies + lo + i);
            srcRows[i] = asc_ldcg(pairs + 2 * (lo + i));
            dest[i] = asc_ldcg(pairs + 2 * (lo + i) + 1);
        }
        uint32_t dep = masks[i], src = srcRows[i];
        uint32_t readAfter = 1U << (dep & kDependencyMask), writeAfter = 1U << (dep >> kDependencyBits);
        uint32_t missing = (readAfter | writeAfter) & ~ready;
        if (missing) {
            asc_atomic_or(pending, missing);
        }
        if ((ready & readAfter) && !(previous & readAfter)) {
            buf[16 * cap + i] = asc_ldcg(scale + src);
            buf[17 * cap + i] = asc_ldcg(addr + src);
            buf[18 * cap + i] = asc_ldcg(id + src);
        }
    }
    asc_syncthreads();
    for (uint32_t x = tid; x < n * 16; x += 512) {
        uint32_t i = x / 16, readAfter = 1U << (masks[i] & kDependencyMask);
        if ((ready & readAfter) && !(previous & readAfter)) {
            buf[x] = asc_ldcg(key + uint64_t(srcRows[i]) * 16 + x % 16);
        }
    }
    asc_threadfence();
    asc_syncthreads();
    for (uint32_t x = tid; x < n * 16; x += 512) {
        uint32_t i = x / 16, dep = masks[i];
        uint32_t required = (1U << (dep & kDependencyMask)) | (1U << (dep >> kDependencyBits));
        if ((required & ready) == required && (required & previous) != required) {
            uint32_t target = dest[i], d = target & kIndexRowMask;
            asc_stcg(key + uint64_t(d) * 16 + x % 16, buf[x]);
            if (!(x % 16)) {
                asc_stcg(scale + d, uint32_t(buf[16 * cap + i]));
                asc_stcg(addr + d, buf[17 * cap + i]);
                asc_stcg(id + d, buf[18 * cap + i]);
                if (target >> 31) {
                    asc_stcg(valid + d, 0U);
                }
            }
        }
    }
    asc_threadfence();
    asc_syncthreads();
}
