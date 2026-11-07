# Proposal: overlap one replay's tail with the next replay's LI

Status: proposal only. Nothing here is implemented or measured beyond the
device numbers in `docs/lifetch-perf.md`.

## Why

Per replay (batch 16, history 131072, resident 16384):

| miss | LI | update | final repair | tail total | whole replay |
|---|---:|---:|---:|---:|---:|
| 10% | 285 | 43 | 54 | 97 (19%) | 504 |
| 30% | 286 | 88 | 313 | 401 (49%) | 818 |
| 50% | 287 | 93 | 452 | 545 (55%) | 991 |

The tail is everything after `v9_resolve`: the Index update and the final
repair. All six kernels sit on one stream, so the tail of replay *n* cannot
start before replay *n* ends, and replay *n+1* cannot start before that.
The tail is already 55% of the replay at 50% miss, and it grows with miss
while LI stays flat.

## Idea

Double-buffer the rotation workspace and run two streams:

```
stream A: LI(n) -> resolve(n) -> LI(n+1) -> resolve(n+1) ...
stream B:            update(n) -> fetch(n) -> update(n+1) ...
```

The final repair of one replay then overlaps the LI of the next. The ceiling is
`min(tail, LI)`, i.e. about 285 us at any miss rate - roughly 29% at 50% miss
and 19% at 10%.

## What blocks it today

1. **Shared rotation buffers.** `allocate_rotation` comments that "groups
   execute serially on the same stream" and every group's bindings point at the
   same `mapping`, `changes`, `locations`, `rows`, `destinations`. Overlap needs
   two of each: `mapping` alone is `batch * history * 4` = 8 MB for these shapes.
2. **One slot pool.** The resolver of replay *n+1* may allocate a slot that the
   final repair of replay *n* is still writing. `reservedHot` prevents double
   booking *within* a replay only; cross-replay reservation is not implemented.
3. **`hot_valid` visibility.** `PublishDestinationResidency` runs at the end of
   the final repair; the next resolver reads `hotValid` to classify hits.
   Overlapping makes that read racy.
4. **KVAF queue ownership.** State and CQ consumer indices are per
   `(row, worker)` queue. Two replays in flight would share a queue and its
   cursor; the queue state would need to be double buffered too, or the two
   replays would need disjoint queues (`final_workers * 2` queues per row).
5. **Graph capture.** The captured graph is a linear chain on the caller's
   stream. Two streams need events inside the graph, and the harness currently
   captures on one stream only (`capture_scope`).

## Phased plan

1. **Measure the ceiling first.** Time `resolve(n)` end to `LI(n+1)` start; if
   the serving loop already serialises on something outside the replay, the
   whole proposal is moot.
2. **Double buffer `mapping` only** and prove a stale mapping cannot be read
   across replays. This is the same invariant the init journal work needs, so
   the two share a test.
3. **Cross-replay slot reservation**: extend `reservedHot` to carry the owning
   replay index instead of an ordinal, so a slot reserved by replay *n* is
   never handed to replay *n+1*.
4. **Queue partitioning** on the KVAF side, then two-stream capture.

## Risks

- Correctness is silent when it fails: a slot read too early returns another
  token's KV. The acceptance test has to compare bytes, not just counts.
- Memory: doubling rotation buffers costs about 8 MB per layer group at these
  shapes; a serving deployment with many groups may not afford it.
- The benefit is capped by LI, so it only pays at miss rates above roughly 30%.
  Below that the tail is 97 us and the added complexity is not worth it.
