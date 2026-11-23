// SPDX-License-Identifier: Apache-2.0
#pragma once
// One optional 768-byte record per AIV. Times use the local system counter.
// Keep the Python field table in StreamingLIFetchForward in the same order.
namespace joined_trace {
enum Field {
    Entry, TopkEnd, ClassPublished, PrefetchReady, PrefetchEnd, ResolveBegin, RawWait, RawReady,
    CompactBegin, CompactEnd, DenseBegin, BitmapWait, BitmapReady, DenseEnd, SaveBegin, SaveEnd,
    ResolveEnd, RawRelease, ConflictWait, ConflictReady, GetBegin, CallbackBegin,
    ClassificationWait, ClassificationReady, AdaptBegin, AdaptEnd, IndexWait, IndexReady,
    IndexBegin, ApplyBegin, ApplyEnd, Q1ReadWait, Q1ReadReady, CommitBegin, CommitEnd, IndexRelease,
    CallbackEnd, GetEnd, DoneRelease, FinalizeBegin, FinalizeEnd, BarrierEnter, BarrierExit,
    RetireIssued, MissCount, SafeCount, BucketWords, PrefixCount, Protection, CompactSuccess,
    FirstDoorbell, LastPost, CompletionDone, PostActiveCycles, CqDrainCycles, FinalPosts,
    FinalCqEvents, Prefetched, Q4PrefetchPolls, Q4PrefetchWaitCycles, DescriptorSyncBegin,
    DescriptorSyncEnd, DenseIndex, ShortLru,
    ResolverRestoreBegin, ResolverRestoreEnd, IndexRestoreBegin, IndexRestoreEnd, IndexSaveBegin,
    IndexSaveEnd, ResolverRestoreBytes, ResolverSaveBytes, IndexRestoreBytes, IndexSaveBytes,
    LocationsSaveBytes, PeerMapsReady, ClassMissCount, LruInputBytes, PrefetchPublish,
    FinalFlag6Wait, FinalFlag6Ready, FinalFlag7Wait, FinalFlag7Ready, FinalFlag8Wait,
    FinalFlag8Ready, FinalFlag9Wait, FinalFlag9Ready, FinalFlag10Wait, FinalFlag10Ready,
    FinalFlag11Wait, FinalFlag11Ready, FinalOutputWait, FinalOutputReady, DenseInitBegin,
    DenseInitEnd, QueryPrefixUpper, FieldCount
};
static_assert(FieldCount == 96, "joined owner trace record");
using Record = __gm__ uint64_t *;
__aicore__ inline Record Row(__gm__ pipeline_ab::Args *a, uint32_t core) {
    return a->queryOwnerTrace ? reinterpret_cast<Record>(a->queryOwnerTrace) + core * FieldCount : nullptr;
}
__aicore__ inline void Mark(Record r, Field f) {
    if (r) r[f] = AscendC::GetSystemCycle();
}
__aicore__ inline uint32_t DmaBytes(uint32_t words) {
    return ((words + 7) & ~7U) * sizeof(uint32_t);
}
// Both hash tables have a power-of-two capacity; the optional protection bitmap
// and preceding slot prefix use the same padding as FromGm/ToGm.
__aicore__ inline uint32_t ResolverBytes(uint32_t buckets, uint32_t prefix, uint32_t bitmapWords) {
    return 2 * DmaBytes(buckets) + DmaBytes(prefix) + DmaBytes(32) + DmaBytes(bitmapWords);
}
} // namespace joined_trace
