# LIFetch performance ledger

Numbers measured on device 5 (`Ascend950PR`), batch 16, history 131072,
resident 16384, index heads 32, Q4, TopK 2048, graph replay, warmup 4,
iterations 20, trace rounds 3. Server-side KV: 656 B per token.

## 1. Device baseline (main)

| miss | core | full | update exposed | plan | move | fetch | prefetch | final | unused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 460.5 | 503.9 | 43.4 | 78.0 | 47.3 | 54.0 | 7110-7320 | 461-570 | 1066-1298 |
| 30% | 729.8 | 817.6 | 87.8 | 88.0 | 69.6 | 313.2 | 7744-8000 | 7744-8000 | 1088-1599 |
| 50% | 898.0 | 991.2 | 93.2 | 93.0 | 77.6 | 452.3 | 7488-7872 | 11776-12032 | 1152-1534 |

All values in microseconds; `core` is the fused path without Index update,
`full` adds it. Stage rows come from the same-replay core trace.

Every kernel is on one stream, so the Index update is fully serialised between
`v9_resolve` and `v9_fetch_update`: it delays the final repair by its whole
duration. Nothing overlaps it except the tail of the prefetch DMA.

## 2. Committed optimizations (`perf/update-hot-path`)

| commit | change | simulator evidence |
|---|---|---|
| `4e87083` | plan caches both membership results in a 2-bit mask per record; scale/address/logical moves fan out to three workers instead of one | B16/n=2048 220400 -> 153059 ticks (-30.6%), B1/n=2048 190235 -> 134976 (-29.1%) |
| `1effb23` | service output mapping partitioned into 512-element tiles across up to eight AIVs per request; the first share still plans | B16/n=410 170320 -> 95995 (-43.6%), B8 168810 -> 88689 (-47.5%) |

`KVIP_LIFETCH_PUBLISH_SPLIT=1` restores the old publish shape for A/B.

### Simulator calibration

`update_sim` on Ascend950PR is deterministic (three repeats within 0.02%) but
it has no L2 model. The `logical` table is 1 MB per owner row and the publish
path gathers it randomly; the model charges every gather a full memory access,
so publish looks like two thirds of the update. On device `aiv_cache_update`
is 95-142 us while `plan` alone is 75-93 us, which rules that share out.

Consequence: treat simulator deltas on memory-heavy paths as a *ranking*, not a
percentage. The trustworthy expectation from these two commits is the plan+move
share (-26..30% of those kernels), i.e. roughly 13-28 us end to end.

## 3. Structural findings

**Service work grows linearly with the epoch count.** `serviceLanes =
2 * (liCores // rounds)` and every epoch rescans all batch rows over the fresh
`history / rounds` slice. Total scanned rows are `rounds^2 / 4`, each covering
`history / rounds`, so total service work is `rounds * history / 4` while a
single interruption stays constant. Doubling epochs doubles the time LI spends
serving prefetch. The 30/50% runs pin `epochs=[1024 ...]` not because of the
budget but because of this structure.

**Fetch is the high-miss bottleneck.** 19904 requests x 656 B in 452 us is about
29 GB/s, plausible for scattered 656 B reads. The lever is therefore not
bandwidth but *where* a read happens: prefetch overlaps LI, the final repair
does not.

**Unused prefetch is 12-17% of reads.** These are speculative tokens that never
enter the final TopK; they cost slots and DDR but return to no consumer.

## 4. Open experiments (need the device)

1. Epoch sweep 4/6/8/12 at miss 50. Judge on `total_us` only; watch `li_us`
   rise against `fetch_us` and `final` falling. If 4 wins, the current default
   of 8 is already too many.
2. Prefetch budget 256/512/1024 at the winning epoch count.
3. Service event cross-check: `span_sum` from `events` against the measured
   `li_us` for the same round. The same table now carries `budget` and
   `demanded` per epoch: `demanded` far below `budget` means prefetch is
   limited by how few new misses a window produces, not by the budget, and the
   prefetch-token sweep would then be a no-op.
4. Init journal: `InitializeRound` rewrites `mapping` for the whole history
   (12-13 us per replay). A journal of written tokens bounds that to the
   prefetched set; the sentinel semantics must be kept, so a stale entry must
   still read as "no mapping".
5. Score cutoff: `CompactMisses` truncates in token order, so at high miss the
   budget goes to the earliest positions rather than the highest scores.

## 5. Reproduction

```bash
bash scripts/build.sh --kvaf-root <path> && source scripts/env.sh
bash scripts/bench_lifetch_batch.sh <tag> 5 127.0.0.1:28885
python3 scripts/compare_batch.py tests/results/lifetch-batch/<base> tests/results/lifetch-batch/<cand>
```

`tests/operators/lifetch/simulator` builds the Index update kernel for the
Ascend950PR model: `./update_sim <changes> <batch> [owned|outputs]`.
