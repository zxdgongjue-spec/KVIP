# P0-C5 — S2 multi-AIC admission (source audit and CPU oracle)

Baseline: Ascend950DT_9582, B4 H131072 R8192, Prefetch=0, KVIP HEAD 5791543.

## What the P0-C4 A5 evidence actually establishes

- CPU 74/74 PASS and A5 LI-only 10/10 in both Q2 and Q4 modes.
- Q1, Q2-mode K=32/64/96/128K: 68.142/125.437/181.886/238.683 us.
- Each extra 32K costs 57.295/56.449/56.797 us; mean 56.85 us.
- Q1, K8K is nonstationary (-43.79% half drift); do not extrapolate from it.
- These are whole-v9_li graph-amortized costs, NOT isolated MatMul timing.
- Real S2 multi-AIC acceleration has NOT been demonstrated.

## Source-specific hazards

- indexer_types.h: constInfo.isLDOpen defaults false.
- indexer.h: SplitCore can represent S2 tile shards but isLDOpen=true alone is unsafe.
- indexer.h: scoreGm currently lives in a core-private workspace slice.
- indexer.h: ProcessTopK runs at every core's last S2 block.
- indexer_vector.h: ProcessTopK expects a full logical score row and writes final indices.
- indexer_cube.h: AIC ComputeMm1 already works in 128-token S2 tiles; 1 AIC maps to 2 AIV.
- runtime_shape.h: liCores and Epoch geometry depend on Batch/splitQ, not S2 shards.

## C5a — First A5 experiment: Score-only, fixed 1/2/4 AIC groups

1. Leave production v9_li_kernel untouched; implement a separate opt-in diagnostic kernel.
2. Start with B4, Q=[1,0,0,0], actual K=128K, Prefetch=0; static 1/2/4 shards.
3. Shard boundaries must be complete 128-token tiles; first shard owns resident 0..8192.
4. Preserve MM and Vec1 pairing. Each group writes disjoint global Token ranges
   in a logical per-query Score GM array; do not use the existing per-core-private
   Score buffer without an explicit stitch step.
5. Compare EVERY Score tile, including cold 96K..128K, with an independently
   validated single-core Score result. The current Resident-only TopK PASS is inadequate.
6. Measure T_score_only(1/2/4) with an ABBA paired NPU Graph after correctness.

## C5b — Separate single-writer TopK consumer

- Run the already-known Streaming TopK over the complete, stitched scores in
  a subsequent kernel to guarantee producer completion and unique output ownership.
- Compare complete TopK against Native with deliberate high-scoring cold winners
  near 32K/64K/96K/128K. Verify exact global IDs and tie behavior separately.
- Report end-to-end T_score + T_topk + synchronization vs the original whole LI.

## C5c — Only then try parallel Partial TopK and Global Merge

- Four shards give up to 4*2048=8192 partial candidates per Query.
- Score:uint16 + ID:uint32 uses roughly 48KiB per Query before metadata.
- CPU TopK-union theorem is valid under the explicit total-order comparator.
  Native CANN tie policy must be empirically verified, not assumed.
- Do not alter Rotation, MIS, BatchGet or Owner/Follower until C5b net LI gain is proven.

CPU admission:
python3 -m pytest -q tests/operators/lifetch/test_s2_split_oracle.py
python3 tests/operators/lifetch/support/s2_split_oracle.py

These CPU tests are a mathematical planning reference only; no A5 Score kernel was implemented.