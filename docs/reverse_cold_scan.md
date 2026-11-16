# URMA contiguous-block reverse scan experiment

Base: upstream `main` at `532df2c8dfcb27cf8ab203f70d5bdf3510d77b6c`,
plus PR15 (`4385ac75435c3e872b5e77a4b66e98f8fa5838a8`). The main test-entry
refactor is retained when applying PR15. No adaptive graph routing or older
post-threshold optimizations are included.

## Mechanism and cost

`build_libraries(..., reverse_cold=True)` reverses the order of complete
2048-position cold Index blocks. Within each block, all 128-position tiles
remain in ascending order. Resident positions, partial boundary blocks and the
causal tail retain their original positions. The permutation only covers blocks
valid for all four packed queries; each query's validity boundary is unchanged.

The LI producer keeps a monotonic **scan position** for bitmap storage, progress,
scale-bank rotation and Fetch window scheduling. It converts that position to a
**physical Index position** for key/scale reads and score writes. Fetch's existing
bitmap-expansion SIMT code converts admitted IDs back to physical positions.
Its resume cursor remains in scan coordinates. Final TopK therefore sees the
same physical score array and keeps the same tie-breaking order.

There is no Index/KV repacking, Top4096 cache, additional workspace tensor,
kernel launch, synchronization event, sorting pass or transfer of a schedule
from the CPU. Added work is four query-bound reads per producer row and service
row, integer block-address conversion per tile, and integer ID conversion in
the existing candidate-expansion function. This is not zero-cost: instruction
count and non-forward traversal can affect the native pipeline and memory
system. Only complete graph timing can establish a benefit.

The experiment keeps resident+recent scoring and the existing fixed threshold.
It does not claim the simulated prefix-TopK update or retry queue is present.
Those mechanisms require their own implementation and native cost accounting.

## Boundaries

- Default is OFF; UBmem is unchanged. No Context or tensor ABI change.
- Sequence-split LI keeps forward order. The experiment harness rejects that
  geometry instead of presenting an unchanged fallback as a reverse result.
- This is an isolated frozen-input replay, not a measurement of service TPOT or
  closed-loop cache evolution.
- Full graph timing includes all required device work; restoration, input
  import, verification and JIT compilation remain outside timing.
- Local host tests cover the permutation, causal boundaries, scale-DMA
  contiguity and actual consumer ID decoding. Ascend compilation and device
  correctness/performance still require the remote run.

## Comparison

Use one arena and identical pre-state for five variants:

| Variant | Scan | Window / total prefetch limit |
|---|---|---|
| forward | Forward | 64 / 512 |
| reverse | Reverse complete blocks | 64 / 512 |
| forward-wide | Forward | 128 / 1024 |
| reverse-wide | Reverse complete blocks | 128 / 1024 |
| forward-end | Forward, repeated drift reference | 64 / 512 |

The 2×2 comparison separates ordering from budget changes. The wide limits use
existing capacity and admission code, not a new queue. Compare full graph mean
and median, mis-prefetch bytes, final repair bytes and total bytes. A reduction
in final repair alone is insufficient. Successful correctness checks require
identical ordered final selection and exact initial/final threshold keys.
Aggregate useful prefetch is not a measure of overlap or completion before the
last window; do not label it `coldEarly` or `lead`.
