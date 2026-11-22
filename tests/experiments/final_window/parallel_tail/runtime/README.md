# Experimental four-Q terminal integration

`compare.py` runs the current checkout's original LIFetch chain (A) and the
private four-AIV-per-request terminal chain (B) against the same snapshot using
real URMA. It builds isolated shared libraries and does not install or replace
production libraries. This is an experimental validation entry, not a qualified
production implementation or a claim that the 10/28-us targets are met.

## Run

From the KVIP repository root:

```bash
python tests/experiments/final_window/parallel_tail/runtime/compare.py \
  /vllm-workspace/vllm-ascend/tmp/lifetch-failure/pid-91271-generation-48 \
  --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF \
  --kvaf-server 127.0.0.1:28885 --device 4 \
  --windows 4 --prefetch-tokens 1024 --prefetch-rank-start 1024 \
  --warmup 2 --iterations 3 --timeout-seconds 1200 \
  --output "/tmp/lifetch-parallel-tail-$(date +%Y%m%d_%H%M%S)"
```

The source snapshot runner is vendored from commit `c239791` and adapted with
checked replacement sites. The older `final_window/compare.py` tests a different
implementation; do not label its B results as this four-core terminal.

## What the comparison checks

- Exact effective geometry and prefetch configuration equality between A/B.
- Recognized expected kernel chains and measured sample counts. B must contain
  the new terminal kernel and must not contain the old final-TopK/resolver pair.
- Input equality before each replay, completion status and selected-token bounds.
- Every measured replay's selected logical IDs and KV equality between A/B.
- Per-kernel means, full tail, total kernel sum and prefix-kernel A/B deltas.

Kernel sums exclude graph gaps. These results measure the snapshot operator
chain, not vLLM throughput. The script does not establish complete Index/state
correctness on hardware merely from matching selected KV. Detailed diagnostic
mapping/header snapshots and general capacity handling still require work.

## Integration

- `prepare_prefix.py` copies the producer sources byte-for-byte and adds a host
  entry launching original LI kernels 0/1/2. Non-sequence-split geometry is
  rejected before launch because that path already performs final TopK.
- `prepare_native.py` generates the terminal kernel with live row and owner
  offsets, routed outputs, Index updates, CQ completion and retirement.
- `live_binding.py` captures immutable WQE templates before graph capture.
  Queue heads, row owners and query ends remain live. Each request uses one
  prefetch queue and four independent terminal queue views.
- `transport.h` uses KVAF posting and CQ routines. Prefetch completion does not
  clear the odd queue's status because that queue is now Q1-owned.
- `native_forward.py` joins the original prefix and new tail without host tensor
  work in the forward path.

## Validation and current limits

The retained Index-split/status-warp implementation measured **33.49 us** in
its software-CQ model with detailed stamps off, and **34.41 us** with stamps on.
Posting/output synchronization in the instrumented run was **11.634445 us**.
These are simulator results, not real URMA timings; the 10/28-us targets have
not been met. Two repeated full state audits passed, including final status.
Native compilation passed. Host tests passed (10 tests, 19 subtests).

The model assumes CQ readiness after 5 us and does not model NIC KV traffic.
The direct LRU range is limited to 2048 slots; larger requirements report an
error. Detailed mapping/header snapshots and broader hardware qualification
remain outstanding. Source fixtures under `evidence/v537` are required by the
checked generator and are not performance results for the retained integration.

## Extract a compact result

```bash
python tests/experiments/final_window/parallel_tail/runtime/brief.py "$run"
```

This prints configuration, per-kernel means, tail and total differences, and
per-replay logical/KV equality. Failed runs print only the relevant log tail.
