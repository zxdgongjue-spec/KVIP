# Bounded URMA prefetch admission

This opt-in experiment extends the post-threshold pipeline after merging KVIP
main `532df2c8dfcb27cf8ab203f70d5bdf3510d77b6c`. The pre-experiment merge is
`a99fe7c1e906ae96e3041280617297333b861e14`. None of the new options is a
measured NPU speedup until the complete graph passes target validation.

## Separate selection correctness from speculation

The resident plus recent-Index Kth is an exact lower bound for final TopK. It
continues to protect hot victims and constrain exact selection. An independent
prefetch gate may be raised by a nonnegative number of *ordered uint16 score-key
steps*. This is not a percentage of a floating-point score. It only suppresses
cold speculative admission. Final repair must retrieve every selected missing
token, including tokens suppressed by the raised gate. Resident victim
classification continues to use the original lower bound.

BF16 ties matter: a single key step can remove many candidates. Raising a gate
can remove useful prefetch as well as waste. No miss-rate-to-gate formula is
assumed. Compare complete-graph time, useful early reads and final reads, rather
than choosing the policy with the smallest unused-prefetch count.

## Per-request controls

`LIFetchBinding.set_admission_controls(cpu_int32_Bx4)` binds four values per
packed request: `(total_limit, window_limit, key_steps, carry)`.

* Total is 0..2048; window is 1..480; key steps are 0..65535; carry is 0 or 1.
* Controls are immutable during execution. The setter rejects capture-time
  calls and device tensors. Update on the submission stream after prior users
  complete, before replay. After first allocation the table address is retained.
* With carry, window `e` may admit at most
  `min(480, (e+1)*window_limit - posted_so_far)`; negative values become zero.
  Existing total, victim-slot and transport limits further bound this value.
* Carry reuses earlier **unused credits**. It does not revisit candidates that
  an earlier window rejected; final repair handles those candidates.
* `None` preserves the original fixed policy. There are new pointer/null
  checks, but no new allocation, upload, barrier or kernel in that path.
  Zero key steps reuse the original threshold tensor; positive steps require
  a separate 512-byte local gate buffer and setup which are inside graph time.

The original 512 setting is the request's *total* prefetch budget, with a
64-token window limit. It is not 512 per window. The existing data structures
already support 2048 speculative slots plus 8192 final Q4 slots. Runtime limits
remain enforced. The appended URMA Context pointer changes its bridge ABI to
13 (336 bytes). The separately built UBmem Context is unchanged.

## Host planning without a new device synchronization

`kvip.runtime.admission_policy` contains three pure-CPU components:

1. `DemandHistory` consumes completed selected-read demand, keyed by request
   generation and layer group. It predicts using an exponentially weighted
   mean and absolute error. Current-step TopK is never a predictor input.
2. `allocate_controls` distributes a bounded batch budget across request
   owners, respecting each owner's capacity. This is host planning, not a
   device pool that can steal another owner's occupied slots.
3. `PathSelector` compares calibrated **full-chain** costs, including state
   conversion, with a margin and consecutive confirmations. Cold starts,
   out-of-range predictions and mismatched fingerprints retain the current
   path. The lower/mean/upper prediction band must all favour a switch.

Calibration must match the layer/workload class, shape, packing, history,
payload width, NUMA placement, kernel/Get hashes and policy. Five groups with
different score and request distributions do not form a valid one-dimensional
calibration curve merely because their total read counts are close.

These are planning interfaces, **not automatic vLLM graph dispatch**. The
caller must supply compatible persistent Index/cache layouts, completed
feedback and calibrated transitions. Existing CPU miss-slot lists are victim
choices, not this step's final cold misses. Prefetch plus final transfers also
include speculation waste. Do not add a synchronous `.cpu()` to obtain missing
feedback or treat offline snapshot restoration as a free serving transition.

## Direct descriptors for ready repair

The opt-in build flag `ready_direct_ranges=True` requires bounded ready-batch
fetch. A single contiguous ready range can be passed directly from its three
canonical descriptor arrays to the existing synchronous Get. This removes
packing and intermediate GM writes for that case. Multiple ranges keep the
original pack path. Range validation, descriptor visibility, completion checks,
CQ ownership and buffer lifetime remain unchanged.

This does not convert Get to asynchronous Submit/Poll or eliminate its CQ wait.
Such a change needs a matching KVAF ABI and per-batch completion ownership;
publishing row completion early on the existing shared CQ would be unsafe.

## Validation boundary

CPU tests execute the admission arithmetic with sanitizers, gate preparation,
binding ABI, bounded allocation and causal policy. The CPU score survey uses
captured score bits and exact TopK IDs but approximates producer arrival order;
it is a traffic counterfactual, not a hardware simulator.

NPU replay must check bit-exact scores, unchanged ordered TopK/final thresholds,
KV/Index state, zero outstanding transfers and each budget. Compare G2/G7/G10/
G17/G4 using strong serial (32 workers, post 128, inflight 2048, two jettys/core).
Record full-graph mean and median, real transfer counts and control drift.
Fixed controls are uploaded before capture; device gate/quota work is timed,
but per-step CPU planning/H2D and offline snapshot restoration are not.
