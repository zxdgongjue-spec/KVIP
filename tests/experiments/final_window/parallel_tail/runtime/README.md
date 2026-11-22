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
correctness on hardware merely from matching selected KV. Broader hardware qualification still requires work.

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

The retained capacity/exchange implementation and its source hashes are recorded
in `capacity_model/validation.json`. The ordinary 230-change fixture measured
**37.12 us**, compared with **34.92 us** for the unchanged terminal: **+2.20 us
(6.3%)** remains. The initial capacity candidate took 41.47 us; do not describe
the retained version as having zero regression. Host checks passed (23 tests)
and the real NPU-mode shared library compiled and linked under CANN 9.1.
Hardware acceptance of this revision remains outstanding.

The model assumes CQ readiness after 5 us and does not model NIC KV traffic.
The terminal retains the original 2048-slot LRU prefix. When more slots are
needed, it extends that list using resident slots absent from all four final
hot-hit sets. The allocation and protection workspace belongs to the capture
binding; it is allocated before capture. Ordinary requests perform their
capacity probe inside the existing pack task. Exact cross-Q counting and
extension run only at the capacity boundary.

Overlapping Index exchanges snapshot their payload before any worker writes,
then distribute the updates across four workers. The scratch-capacity check
remains: very large overlapping permutations that do not fit its bounded UB
workspace fail rather than silently corrupting Index. The 4000-record test is
an independent-exchange workload, not proof that every 4000-record overlapping
permutation is supported. Synthetic/real operator bench integration is separate
from this snapshot AB entry. Source fixtures under `evidence/v537` are required by the
checked generator and are not performance results for the retained integration.

## Extract a compact result

```bash
python tests/experiments/final_window/parallel_tail/runtime/brief.py "$run"
```

This prints configuration, per-kernel means, tail and total differences, and
per-replay logical/KV equality. Failed runs print only the relevant log tail.


## Reproduce the capacity regression model

Generate a candidate with the same capacity and exchange adapters used by the
native AB build (run from the repository root):

```bash
python tests/experiments/final_window/parallel_tail/runtime/capacity_model/prepare.py   /tmp/lifetch-capacity-model --changes 230 --candidate
```

Supported fixture sizes are 230, 2048 through 2100, and 4000. Omit `--candidate`
for the unchanged terminal baseline; that baseline cannot handle overflow.
The generated CMake project is for the CANN simulator. Run its
`native_tail_sim 9` application with `msprof op simulator` and the matching SoC.
Mode 9 performs two launches, restores the same input before each, and preserves
publication generations. It does not represent two evolving decode rounds.
The oracle checks WQEs, TopK, counts, output slots and logical IDs, all Index
payload/metadata, validity, retirement and transport status.

```bash
python tests/experiments/final_window/parallel_tail/runtime/capacity_model/report.py run9.log
```

The report rejects partial runs or failed audits. Its timings are simulator
wall-time estimates, not real URMA measurements. Use the snapshot AB command
above for hardware acceptance.
