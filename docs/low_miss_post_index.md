# Zero-window URMA: Post, Index, CQ

`low_miss_post_index=True` is an opt-in implementation of the automatic low
route. It keeps normal hot-KV promotion and the persistent hot-first Index.
It does not implement read-through or periodic cache refresh.

The captured graph remains `init → LI/TopK → streaming tail`. Init publishes
the existing batch-wide device decision once. The high route retains the
windowed pipeline. On a low round the same tail kernel runs:

1. Build the exact Q4 union, protect selected hot slots, allocate destinations
   and produce TND slot outputs using the original logical mapping.
2. Partition exact Get descriptors across `2 * aic_count` AIVs and post them
   through exclusively leased native KVAF queues.
3. After **all** Post calls finish, reuse those AIVs for Index maintenance.
   URMA reads continue in hardware while keys, scales, DDR addresses and
   logical IDs are exchanged. These arrays do not alias the hot-KV payload.
4. After **all** Index work finishes, the same AIVs drain their CQ completions.
   Only successful completion publishes hot validity and route feedback.

Missing KV whose Index is already resident refills its own hot slot; it still
counts as a real Get. Cold positions swap with unique, unselected hot slots.
Consequently different Index events have disjoint endpoints and can execute
on all participating AIVs without constructing a permutation/shadow map.
The finalizer omits empty prefetch completion, obsolete role-flag polling and
queue diagnostics belonging to the old worker layout. It retains completion
counts, output validation, failure invalidation and success-only feedback.

## Safety and resource contract

- The feature is separate from read-through, compact-plan and SIMT routing
  experiments; their combinations are rejected. Defaults stay disabled.
- Each AIV has an exclusive single-Jetty view. The existing KVAF fused-context
  patch already exposes `PostDirectRequests` and `WaitForCompletion`. The
  tail calls those helpers directly, retaining scalar state across Index work.
  No additional KVAF patch, runtime ABI, WQE/CQE implementation or native Get
  rebuild is needed. The original blocking Get remains the capacity fallback.
- Before any Post, all cores validate plans and queue capacity. If a whole
  slice cannot fit in one SQ window, the whole invocation uses bounded native
  Get before Index. This is an explicitly counted fallback, not an async
  timing result. A real queue/descriptor error is not silently retried.
- Every successfully posted queue reaches its CQ wait even when a peer
  reports an error. A CQ timeout poisons the queue and forbids reuse; it does
  not establish that outstanding DMA has stopped. The blocking fallback
  retains KVAF's original failure handling. A failed request invalidates its
  owner's hot validity and does not commit successful predictor feedback.
- Hardware `SyncAll` uses vector-only mixed scheduling
  (`KERNEL_TYPE_MIX_AIV_1_0`) and batch scheduling. Every launched AIV follows
  the same phase barriers; no CPU synchronization or extra graph node is
  introduced. The batch-scheduling attribute also applies on high rounds,
  so high-route performance must be measured, not inferred from source.

The workspace adds one 128-byte record per participating AIV, allocated before
capture. It adds no KV scratch arena and retains the 256-byte streaming Args
ABI. Headline samples do not read clocks in this record. Optional separate
diagnostic replays record plan/pack/Post/Index/CQ boundaries; overlapping
cross-core intervals cannot be summed into graph latency.

## Validation and expected effect

CPU differential tests cover Q4 deduplication, invalid-resident self-refills,
all-miss limits, disjoint full-tuple exchanges and descriptor partitions. The
native KVAF CPU device model tests deferred CQ completion, released/reused UB,
capacity fallback, errors and queue poisoning with address/undefined-behavior
sanitizers. These checks do not establish CANN compilation, device cache
coherence or a speedup.

The intended reduction is exposed `Get wait + Index` approaching their
overlap, plus removal of separate worker-role handoffs. Exact planning,
posting, phase barriers, completion and actual Index moves remain necessary.
Use same-run strong serial and old automatic-route controls before and after
the candidate. Compare both mean and median, preserve first/high samples,
report steady low samples separately, and reject correctness failures before
interpreting timings. No 35% gain is asserted without those NPU results.
