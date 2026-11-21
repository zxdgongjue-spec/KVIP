// SPDX-License-Identifier: Apache-2.0
#include "query_owner_row_moves.h"

static_assert(kIndexMaxChunks < kOwnerChunkLimit && kIndexMaxChunks < 32,
              "chunk publications and dependencies must fit");

__aicore__ inline __gm__ uint32_t *LastWriters(__gm__ Args *a, uint32_t batch, uint32_t b) {
    // The first six history-sized planes per request hold shadow maps and moves.
    // Append the stamps after all requests so the existing plan offsets still apply.
    return reinterpret_cast<__gm__ uint32_t *>(a->denseShadow) + (uint64_t(batch) * 6 + b) * a->history;
}

__aicore__ __attribute__((noinline)) inline uint32_t CopyChunks(
    __gm__ Args *a, uint32_t b, uint32_t first, uint32_t end, __gm__ uint32_t *plan, __gm__ uint32_t *deps,
    __gm__ BatchFlag *flags) {
    constexpr uint32_t words = kRowCopyWords + 32;
    static_assert(words * sizeof(uint32_t) <= kFinalScratchBytes);
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, words * sizeof(uint32_t));
    auto mem = storage.Get<uint32_t>();
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t *>(ctx->rowOwners)[b] : b;
    uint32_t error = 0, ready = 1, chunk = first;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(plan), kIndexEdgesOffset * sizeof(uint32_t));
    for (; chunk < end && !error; ++chunk) {
        uint32_t lo = chunk ? plan[chunk - 1] : 0, hi = plan[chunk], previous = 0;
        while (!error) {
            Simt::VF_CALL<CopyReadyRows>(Simt::Dim3{512, 1, 1}, a, owner, lo, hi, plan + kIndexEdgesOffset, deps,
                previous, ready, reinterpret_cast<__ubuf__ uint64_t *>(FinalPtr(mem)), FinalPtr(mem[kRowCopyWords]));
            VectorDone();
            uint32_t need = mem.GetValue(kRowCopyWords);
            if (!need) {
                break;
            }
            previous = ready;
            // Consume any available predecessor; never poll an unrelated chunk.
            uint64_t start = GetSystemCycle();
            do {
                for (uint32_t bits = need; bits; bits &= bits - 1) {
                    uint32_t bit = __builtin_ctz(bits);
                    auto f = flags + bit - 1;
                    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t *>(f), sizeof(*f));
                    if (f->ready) {
                        ready |= 1U << bit;
                        error |= f->error;
                    }
                }
                if (GetSystemCycle() - start > 5000000000ULL) {
                    error |= 2U;
                }
            } while (previous == ready && !error);
        }
        Signal(flags + chunk, 0, error);
    }
    // Wake consumers of unexecuted chunks. Already-published receipts stay stable.
    for (; chunk < end; ++chunk) {
        Signal(flags + chunk, 0, error);
    }
    pipe.Destroy();
    return error;
}

// The existing all-AIV exit join guarantees that the next invocation has no
// live readers. Only the last writer clears each row's replay-private stamp.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void RetireWriters(
    __gm__ Args *a, uint32_t b, uint32_t first, uint32_t end) {
    auto ctx = reinterpret_cast<__gm__ Context *>(a->context);
    auto dense = reinterpret_cast<__gm__ uint32_t *>(a->denseShadow);
    auto last = dense + (uint64_t(ctx->shape.batch) * 6 + b) * a->history;
    auto plan = dense + (uint64_t(b) * 6 + 2) * a->history;
    for (uint32_t c = first; c < end; ++c) {
        uint32_t lo = c ? asc_ldcg(plan + c - 1) : 0, hi = asc_ldcg(plan + c);
        for (uint32_t i = lo + threadIdx.x; i < hi; i += 512) {
            uint32_t row = asc_ldcg(plan + kIndexEdgesOffset + 2 * i + 1) & kIndexRowMask;
            asc_atomic_cas(last + row, c + 1, 0U);
        }
    }
    asc_threadfence();
}
