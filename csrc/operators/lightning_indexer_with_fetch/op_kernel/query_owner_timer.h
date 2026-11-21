// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "pipeline_types.h"
namespace owner_timer {
// The Q4 finalizer owns [0,12), one ordered wait interval per flag 6..11.
enum Field { FinalFlagWaits=0, MetadataBegin=12, MetadataEnd, CompactBegin, CompactEnd,
    TailBegin=16, TailBlock, PrefetchWait, PrefetchReady, PrefetchEnd,
    ResolveBegin, RawWait, RawReady, ResolveEnd, RawRelease, ConflictWait,
    ConflictReady, GetBegin, GetEnd, AdaptBegin, AdaptEnd, IndexWait, IndexReady,
    IndexBegin, IndexRelease, IndexEnd, DoneRelease, FinalizeBegin, FinalizeEnd,
    CleanupEnd, BarrierBegin, BarrierEnd, RetireIssued, Count, Safe, Processed, Queue,
    PrefetchCqObserved, PrefetchPolls, PrefetchDrainCycles, PrefetchDrainReturn,
    PrefetchPending, PrefetchCompleted, DescriptorSyncBegin, DescriptorSyncEnd,
    DenseResolveBegin, DenseResolveEnd, ApplyBegin, ApplyEnd, SaveStateBegin, SaveStateEnd };
constexpr uint32_t stride=848;
__aicore__ inline __gm__ uint64_t *Row(__gm__ pipeline_ab::Args *a, uint32_t query) {
    return a && a->queryOwnerTrace ? reinterpret_cast<__gm__ uint64_t *>(a->queryOwnerTrace) + query*stride : nullptr;
}
__aicore__ inline void Mark(__gm__ uint64_t *r, Field f) {
    if (r) r[f] = AscendC::GetSystemCycle();
}
} // namespace owner_timer
