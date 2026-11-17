# Incremental TopK: offline cost audit

`python -m kvip.diagnostics.window_audit` reads completed scan-order result
JSONs, their recorded profiler exports, and the original input snapshots.
It does not import torch, allocate device memory, compile kernels, launch
KVAF, or rewrite an existing result. Use a new output directory.

Required arguments:

```text
--results DIR           gN.json and gN-diagnostic.json from a completed study
--inputs DIR            gN-input/manifest.json and its referenced tensor files
--expected-native SHA   exact native revision recorded by the study
--output NEW_DIR        new analysis output
```

Defaults: groups 2, 7, 10, 17, 4; profiler groups 7, 2. The audited study uses
the register/light-ties/inplace variants, a resident phase and eight cold
windows. Source and snapshot hashes and stored correctness checks must match.

## Timing interpretation

`HEADLINE` uses the existing uninstrumented graph measurements. Mean and
median controls are chosen independently from the faster first/end repeats.
`CONTROL` reports the drift between those repeats. A negative latency delta
means faster; a positive percentage gain means faster.

`PROFILE` reads one independent single-graph profiler replay. If no recognized
Chrome device trace exists, it reads one canonical `kernel_details.csv`.
Epoch timestamps are subtracted as Decimal before conversion to float.
Duplicate or ambiguous exports are rejected rather than added. Overlapping
kernel duration sums and interval spans are preserved separately.

The streaming kernel combines resolve, final fetch and index update. Its
duration does not identify the cost of those individual functions. Neither
these profiler times nor logical CPU work counts are substituted for the
headline graph timing. Dividing a LI delta by eight is not a measurement of
each window: seed preparation, final TopK reuse, service and pipeline changes
also affect that delta.

## Real CPU work and model inputs

`gN/windows.csv` contains one row for each of 64 queries × 8 cold windows:

- Physical and newly scored positions, causal and seed exclusions included.
- Candidates strictly above the preceding exact Kth score.
- Old Kth tie group size and the empty/dense/sparse/tie/inplace branch.
- Old TopK entries retained, eligible IDs published, and IDs repeated from
  the preceding publication. Repeated publication is **not** repeated DDR I/O:
  the device service still checks its submitted-ID map.
- Source-level selection/packing loop extents. These are logical item counts,
  not microseconds, physical memory bytes or utilization.

The CPU check compares dense and sparse/inplace selection for exact ordered
scores and IDs after every window. It uses captured score keys and physical
Index positions, preserving the native seed, reverse block order and tie
rules. It does not predict asynchronous Q4 visibility, URMA completion, or
device wall time.

`gN/probe_inputs.npz` exports **all** queries, with no real KV or Index vectors:

| Array | Shape / meaning |
|---|---|
| `scores` | `[B,4,H]`, captured uint16 score keys |
| `query_ends` | `[B,4]`, causal valid bounds |
| `threshold_bounds`, `tail_bounds` | `[B,2]`, recent seed and ABC bounds |
| `eligible` | `[B,H]`, frozen physical Index fetch eligibility |
| `expected_ids`, `expected_scores` | `[B,4,9,2048]`, ordered seed and eight window outputs |
| `expected_thresholds` | `[B,4,9]`, exact Kth keys |

Use `recent_positions(resident, bounds, query_end)` for each seed and
`scan_windows(history, resident, query_ends[row], True)` for its eight physical
window ranges. Mask positions outside the query or already in the seed. The
manifest records geometry, the snapshot hash, and four candidate-count
quantile queries as convenient component-probe starting points. Every query
is exported; quantiles must not replace full parallel work in a timing model.

A CANN probe consuming these inputs must first reproduce every expected
ordered output. It must then include seed selection, all eight windows,
state copies, list publication and synchronization, and document which
producer/consumer and Cube/vector dependencies it models. Passing this CPU
check alone provides no guarantee of matching full-device timing.
