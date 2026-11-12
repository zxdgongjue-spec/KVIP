# P1 — Variable-length Rotation Epoch closure

Baseline: `0324c8090ffcf8c86444e015ff7c3c9f6a18391c`, with Native LI and
V9 LI mixed-Q/K, zero-prefetch, two-round A5 device tests passing.

## Scope and failure mechanism

`rotating::EpochAt` assigns one service epoch to a fixed LI core
(`context.h`). `QuantLightningIndexerKernel::SplitCore` assigns actual LI
work according to the per-request `actualSeqLengthsQ/K` (`indexer.h`).
Thus a designated core can have no work or finish before the fixed epoch's
`Trigger`. The next epoch's `FastService` waits on
`previous->complete`, which could previously remain zero forever.

This patch **does not rewrite LI TopK or reschedule speculative fetches**.
It closes a *missing* epoch when the assigned AIV exits without triggering
its service. The completion marker carries `skipped=1`, and **does not**
record any fetched data or advance row/queue counts.

## Device control flow

1. `v9_li_kernel` uses the already-established
   `prefetchLimit != 0` switch.
2. Each AIV tracks whether `RotateFetch` served its assigned epoch.
3. After normal `ProcessMain`, or after the no-work `ProcessInvalid`
   path, an unserved AIV calls `CompleteUnreachedRotation`.
4. `CompleteUnreachedEpoch` checks static epoch/lane ownership, waits for
   **the same lane of the previous epoch** when present, and publishes
   `Event{core,epoch,skipped=1,complete=1}` with cache-line maintenance.
5. A reached `FastService` still publishes its original `complete=1`
   with `skipped=0`.

**Safety invariant**: for every allocated `(epoch,lane)` there must be
exactly one publisher, and a later epoch may only become complete after the
previous epoch on that lane is complete.

This P1 implementation targets the direct URMA V9 path. UBmem service
integration and true rebalancing of service work are out of scope.

## Device test admission

Use the dedicated `--li-only-varlen` mode with
`batch=4,history=8192,resident=2048,index-heads=32,miss-percent=0`.
It permits `--prefetch-tokens 0` (regression) or `64` (minimal active
Rotation Service). With `--intermediate-fetches 8`, positive budget means
8 tokens/window. No Resolver/Final Fetch/Index update is launched in this
mode; the KVAF URMA session and queues are initialized.

Run the existing zero-prefetch P0 test first. After it passes, run the
same command with `--prefetch-tokens 64`, a distinct output folder, and
`--li-only-varlen`.

Validation checks:
- Both rounds' Q/K arrays and persistent buffer addresses match expectations.
- Native/V9 full TopK ID sets agree for all 13 valid Query rows; 19 padded
  Query rows contain only `-1`.
- `v9_init`, `v9_li`, and `v9_wait` complete.
- For positive budget, every `events[epoch,lane]` has `complete=1`, and
  `skipped` is 0 or 1. Record the number of skipped lanes, **without**
  asserting an exact count (core assignment can change with tiling).
- Prefetch CQ status, row error, and pending WQE are clean.
- Resource lifetime completes normally with no `fetch_quarantine`.

The CPU-only topology tests check epoch owner uniqueness and source wiring;
they **are not evidence of an A5 device PASS**.

## Follow-up boundaries

Positive-budget, zero-miss success confirms the Epoch dependency chain and
no-hang behavior only. It does **not** demonstrate real miss processing,
prefetch effectiveness, DMA payload correctness, final Fetch, Index update,
four-layer Index Share, or service throughput.

Any device timeout/fault means FAIL and should preserve quarantine. Do not
bypass a pending DMA protection check or free the corresponding buffers just
to rerun. Preserve stage JSONL, native logs, and the report.

Once P1 is validated on A5, test nonzero misses with stage-by-stage
Fetcher/Resolver/Update checks before returning to the GLM-5.3 service.
