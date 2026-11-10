# KVIP

Ascend A5 cache operators extracted from vLLM-DSA's KVOP. Python 3.12,
a matching CANN and `torch`/`torch_npu` runtime are required for NPU validation.
URMA additionally requires a reachable KVAF CPU service; UBmem owns local Host DDR
in each NPU worker and does not connect to a CPU service. Installation is editable: keep this checkout beside
the installed Python package because startup builds use its C++ sources.

## Layout and API

```text
csrc/operators/
  lidu/                             # op_host / op_kernel / torch_extension
  common/lightning_indexer/          # shared LI/score/TopK device functions
  common/lifetch/                    # resident selection, budgets and Index-update planner
  lightning_indexer_with_fetch/
    include/                        # public ABI, Context and runtime shape
    op_host/                        # launch entry points and Index-update host binding
    op_kernel/                      # LI/Fetch lifecycle, prefetch, resolver, Index update
      indexer/                      # URMA runtime-shape LI scheduling
  lightning_indexer_with_fetch_ubmem/
    include/                        # UBmem launch, region and completion ABIs
    op_host/                        # graph-capturable launch sequence
    op_kernel/                      # static LI, controller, pipelined copies, resolver/update
python/kvip/
  operators/lidu.py
  operators/lifetch/                 # URMA binding, forward, workspace, tiling, Index update
  operators/lifetch_ubmem/            # startup specialization, binding, workspace and update
  cache/                            # LRU, ABC and persistent Index storage
  adapters/kvaf.py                   # URMA Runtime lease and shared transport queues
  adapters/ubmem.py                  # local mapping/output/graph lifetime
  adapters/vllm/                     # inputs, selector, startup binding and paged Index import
  runtime/                          # build cache, library loader, hardware query, integrity
  _compat.py                        # centralized lazy aliases for legacy kvip imports
scripts/                            # existing build/environment entry points and common KVAF patches
patches/kvaf-background-put.patch    # shared background Put binding compatibility
patches/kvaf-ubmem.patch             # optional local-DDR runtime/Get/Put/fused-lease increment
tests/
  operators/lidu/
  operators/lifetch/                 # single CLI, update tests, support and resolver_reference
  cache/                            # LRU / ABC
  adapters/                         # vLLM inputs / KVAF transport
  runtime/                          # package boundaries and relocation checks
```

The graph test's Index-update fixture subclasses the production binding; it does
not build another ABI. Workspace reservation and launch validation use the same
scratch specification. KVAF and LIFetch use one library loader. Index ingestion
from vLLM pages belongs to the adapter subclass, not generic persistent storage.
LIFetch headers containing templates remain headers to preserve compilation.

```python
import kvip

lru = kvip.SlotLRU()
# Existing out-parameter LIDU ABI, on the current Torch NPU stream:
# kvip.lidu(topk, actual_query_lengths, ..., block_size)
# Existing ABC descriptor ABI:
# kvip.update_abc(token_to_slot, slot_to_token, descriptors, requests, history, tail)

# Bind once, before graph capture, using the existing prepared launch objects:
forward = kvip.LIFetch(rotating, update)
forward()  # round init -> LI/prefetch -> resolver -> outputs/Index update -> Fetch completion
```

`LIFetch` accepts `LIFetchBinding` and `DirectIndexUpdate` objects;
it is not a new tensor ABI. It retains both objects and uses their preallocated
bindings. No tensor allocation, synchronization, fixture reset or correctness
readback is added to `forward()`. Transport/workspace ownership must outlive graph
replay. Device errors remain available through the existing status/feedback path.
Individual launch methods remain internal diagnostic interfaces.

The vLLM adapters build both transports with `compact_reserved=True`. The existing
init launch reconstructs each active packed row's reserved-hot inverse map from
its 2048 LRU candidates; serving no longer constructs or uploads the dense
`[group, packed row, hot slot]` map. Buffer addresses and graph bindings stay
stable, and newly active rows are cleared before rebuilding. Standalone builds
default to the captured dense-input path for reproducible comparisons. Diagnostic
captures explicitly label the derived inverse map and verify its device value
after init; offline replay does not measure the eliminated service H2D transfer.

`kvip.QueryInputs`, `kvip.LIFetchWorkspace`, `kvip.IndexCacheStorage`,
`kvip.build_libraries` and `kvip.bind_selectors` expose the existing service setup.
`bind_selectors` requires the current vLLM KVAF policy/writer contract; this is not
an arbitrary vLLM-version plugin. The `kvop` package has been removed. External integrations must import `kvip`.
Legacy `kvip.fused_*` imports resolve lazily to the canonical modules through
`_compat.py`; there are no duplicate implementation or forwarding files. Migrating the service installation to
this repository is separate from deleting its existing KVOP checkout.

## Build

Run inside the Ascend container, with the matching Torch NPU installation already
available. The following downloads the pinned KVAF revision if necessary and
builds KVAF, LIDU and LIFetch including Index update:

```bash
git clone --branch main https://gitcode.com/CacheOS/KVIP.git /vllm-workspace/KVIP
cd /vllm-workspace/KVIP
source /usr/local/Ascend/cann/set_env.sh
bash scripts/build.sh
source scripts/env.sh
```

If KVIP is already checked out, build from that checkout; no second clone or
branch switch is required. The build uses the current KVIP source and pins only
the external KVAF dependency.

To use the existing KVAF source instead, replace the build command with:

```bash
bash scripts/build.sh --kvaf-root /vllm-workspace/vllm-ascend/third_party/KVAF
source scripts/env.sh
```

The default CANN target is `Ascend950PR_9599`, matching the original v9 build.
For another A5 SKU, pass its concrete CANN target with `--soc`; `Ascend950` alone
is not a valid target for this CANN build API. The NPU test discovers the device
SoC through ACL unless `--soc` is specified.

The supported KVAF base is `1c1310387ebfb6776a07316874e924265b209ada`.
Preparation retains the existing three common patches in `scripts/patches` and
applies `patches/kvaf-background-put.patch` for background Put's GIL release and
capability flag. `--transport ubmem` additionally
applies `patches/kvaf-ubmem.patch` to that checkout. Preparation recognizes an
already patched tree and preserves unrelated edits; conflicting changes fail
without a forced reset. A separate KVAF checkout/build is recommended per transport.
`--prepare-only` prepares the dependency without replacing the active build selection;
`--skip-kvaf-build` requires matching existing build products.

```bash
bash scripts/build.sh --transport ubmem --kvaf-root /vllm-workspace/KVAF-ubmem \
    --cann /usr/local/Ascend/ascend-toolkit/latest --soc Ascend950DT_9582 --jobs 2
source scripts/env.sh
```

UBmem builds the worker-local Host DDR allocator and Get/Put kernels without
UMDK or CPU services. `thp` (default) allocates host memory, requests transparent
huge pages and registers the device-visible mapping; actual huge-page coverage
is reported separately and depends on the host. `base` uses the regular-page VMM
path. Both paths keep the allocation alive until outstanding work and retained
fused contexts are released. The existing key/slot metadata and background Put
ownership remain in KVAF. URMA remains the default build.

The serial vLLM path uses native LI/LIDU followed by KVAF UBmem BatchGet;
`decode_backend=li_fetch` selects the independent UBmem LI/TopK/Fetch/Index-update
pipeline. Both use UBmem BatchPut and the same local DDR arena. Shared Indexer
layers fetch the complete grouped record (656 or 2624 bytes), so the following
layers reuse the fetched payload. UBmem retains its own copy scheduling and
completion checks while reusing the transport-independent algorithms above.

The original LIDU build uses its own `csrc/operators/lidu/build` directory.

## Verify

CPU contract and source checks:

```bash
cd /vllm-workspace/KVIP
python -m pip install pytest ruff
bash scripts/check.sh
```

Graph-mode native/fused comparison using real DDR and the current service LRU:

```bash
cd /vllm-workspace/KVIP
source scripts/env.sh
python tests/operators/lifetch/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --batches 16 --history 131072 --resident 16384 --index-heads 32 --miss-percent 10 --intermediate-fetches 8 --prefetch-tokens 512 --warmup 4 --iterations 20 --trace-rounds 3
```

This uses Q4, TopK 2048, and reports the actual quantized miss fraction. Full-path
timing calls `kvip.LIFetch`, including mandatory initialization and Index update.
Fixture restoration, byte/SFA verification and CPU LRU are outside timing. A
separate profiled pass reports init/LI/resolve/Fetch device durations. Actual DDR
reads and unused prefetches are printed. Summaries, status and errors are saved
under the printed `tests/results/lifetch/<run>/summary.log`; detailed logs stay
beside it. Snapshot and stateful correctness checks are retained.

The default total prefetch budget is 512, divided evenly over the windows (64
for eight windows). Final misses first consume unprotected LRU candidates, then
reuse LIDU's hot bitmap selector from slot zero. There is no temporary KV region:
all final repairs persist in hot storage, and ABC follows hot storage directly.
The resolver exports destination/source records for every final repair to Index
update. Final descriptors are split into nonconflicting and conflicting ranges. Native
KVAF BatchGet submits the nonconflicting ranges before joining prefetch CQ, then
fetches the conflicting ranges. Its inflight control and completion handling are
reused without a separate final WQE submitter. The test also checks an empty hot cache, victim collisions,
full Index/scale/address ownership, and multiple graph replays, outside timing.

For the original/adapted/fixed resolver comparison, use the same entry:

```bash
python tests/operators/lifetch/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --batches 8,16 --history 131072 --resident 16384 --index-heads 32 --miss-percent 10 --intermediate-fetches 8 --prefetch-tokens 512 --warmup 4 --iterations 20 --resolver-compare
```

The CPU suite executes the production resolver and shared LIDU selector with
empty/populated hot storage, duplicate queries, padding and ABC mappings. A CANN
simulator fixture additionally executes the compiled planner/mover and resolver
without network transport:

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S tests/operators/lifetch/simulator -B build/hot-fallback-sim \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann \
  -DKVAF_INCLUDE="$PWD/third_party/KVAF/access_operator/demo/aiv_batch_get/include"
cmake --build build/hot-fallback-sim -j2
msprof op simulator --application="$PWD/build/hot-fallback-sim/update_sim 410 8" \
  --soc-version=Ascend950PR_9599 --output=tests/results/update-sim-410-b8 --launch-count=2 --timeout=90
msprof op simulator --application="$PWD/build/hot-fallback-sim/update_sim 3000 1" \
  --soc-version=Ascend950PR_9599 --output=tests/results/update-sim-3000 --launch-count=2 --timeout=90
msprof op simulator --application="$PWD/build/hot-fallback-sim/resolver_sim 3000" \
  --soc-version=Ascend950PR_9599 --output=tests/results/resolver-sim-3000 --launch-count=1 --timeout=90
```

The fixtures require their own `PASS` line after all tensor checks. A profiler
success message or a partial trace following timeout is not a correctness pass.
Simulator times are device-model times, not hardware graph or TPOT measurements.

Independent LIDU and ABC device checks:

```bash
python tests/operators/lidu/test_lidu.py --device 5
python tests/cache/test_abc.py --device 5
```

## Migration evidence and limits

Source baseline: vLLM-DSA commit `dad47855b20cbe8b74bf412f05757043d22ce0a0`.
LIDU originally migrated byte-identically; its bitmap count/enumeration now lives
in a shared header used by LIFetch fallback as well. At the initial migration,
retained LI/Fetch/resolver/update function bodies
were compared after removing include paths, comments and whitespace. Both Q2 and
Q4 materialized LI sources match the prior generation pipeline; the build no
longer rewrites source files. LI/Fetch and Index update keep their separate
compiler options. Unused non-quantized LI implementations and legacy simulator
fixture entry points were omitted. Original resolver comparison sources are test
only, disabled in normal builds.

`scripts/migration_sources.json` records source provenance and hashes of the
migrated operator files. Hash checks detect accidental edits, not device correctness.
CPU tests include host execution of the Index-update planner/mover and the unified
launch ordering/error path. The hot-fallback change was compiled with CANN 9.1
and checked with its operator simulator as described above. Hardware graph
correctness and performance still require A5 validation; simulator results are
not hardware measurements.

The imported CANN LI headers retain their original license notices; their license
is reproduced in `LICENSE-CANN`. Apache-2.0 applies to files so marked.

## Runtime shapes

Installation does not take device, batch, history, resident, heads or window count.
Host tiling evaluates the original partition/window formulas when a launch object
is bound, then stores its immutable parameters in the device Context before graph
capture. No per-shape compilation is required. Index update reads history/resident
from its bindings. ABI versions are now LIFetch 11 and Index update 3; rebuild both.

Supported runtime dimensions: batch 1–32 (also limited by available AIC count),
index heads 32/64, 1–32 requested windows, 256-aligned history/resident with
2048 <= resident <= 131072 and resident < history <= 210944. Existing UB/queue constraints are checked
before launch. Q4, TopK 2048 and the existing element types remain fixed. These
limits preserve the original kernel's storage budget; this is not unrestricted
shape support. Each captured graph still owns its shape-specific buffers and
launch object. Changing shape requires another binding/capture, not another binary.

To check different batch/history/resident/head/window combinations with one
compiled library (the runner prints and checks its SHA256):

```bash
cd /vllm-workspace/KVIP
source scripts/env.sh
python tests/operators/lifetch/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --runtime-shapes --miss-percent 10 --warmup 4 --iterations 20 --trace-rounds 3
```

The matrix includes B1/B8/B16, history 65536/131072, resident 8192/16384,
32/64 heads and 4/8 requested windows, using real 656-byte KV records. Each case
retains native/fused byte checks, SFA checks, stateful checks and graph timings.
The runtime conversion changes scalar plumbing and generated code; preserved
formulas alone do not prove unchanged device performance. A5 validation is required.

## Known vLLM integration limitation

The service has observed `InvalidQueue` when an ordinary leased KVAF Get follows
LIFetch on shared persistent queues. The root cause is unresolved; successful
standalone operator tests do not establish that this mixed call sequence is safe.
This does not identify all KVAF Get calls, or the native submission inside
LIFetch, as failing paths.

The current vLLM workaround copies retiring ABC tokens directly into hot HBM
before the next forward overwrites their ABC positions. It avoids the ordinary
Get previously used for this promotion, including across the ABC ring boundary.
The user verified this service path runs. Initial history restoration and fused
final fetch still retain their existing Get paths; this is not a general queue
handoff fix. Do not reintroduce an ordinary Get for ABC promotion without resolving
and validating the shared-queue handoff. Include this limitation in the merge PR.

## UBmem hot-only final repair

UBmem fusion ABI 2 uses the same final TopK protection and hot-slot selection
as URMA, in `csrc/operators/common/lifetch/resident_selection.h`. Missing rows
replace unprotected persistent hot slots; the payload layout is hot + ABC, with
no separate 8192-token payload region. UBmem retains its eight-stage copy pipeline
and native completion checks. The shared Index planner handles up to 10240 change
records (2048 prefetch candidates plus the Q4 union); plan/work buffers are shared
across sequential layer groups and allocated before memory profiling.

Rebuild UBmem fusion and use the matching vLLM integration before serving. Source
hashes include the shared headers and old ABI 1 libraries are rejected. Resident
capacity must be at least 8192 for the worst-case Q4 union. This hot-slot repair changes the fusion ABI, while retaining the KVAF UBmem
runtime ABI and the URMA transport implementation.

Both bindings default to 512 speculative fetches per request/round, accept a
total budget of 0..2048, and cap each window at its configured allowance (at
most 480). Device admission shares the same saturating budget calculation.
The final TopK repair is independent of this speculative budget: setting it to
zero disables prefetch but still retrieves every required cold row. The UBmem
smoke check accepts `--prefetch-tokens 0 512` to check both cases, including eager
execution, graph replay, changed owner packing, and Index/payload consistency.

UBmem fusion currently specializes batch 1–16, 32/64 Indexer heads and
656/2624-byte records before profiling/graph capture; history and hot capacity
are 256-aligned, with `8192 <= resident < history <= 210944` and
`resident <= 131072`. The binary cache includes the shape and source hashes.
The forward path uses preallocated storage without per-step compilation or a
new host synchronization. URMA's runtime-shape limits described above are separate.

## Compare a captured service group without model weights

`python -m kvip.diagnostics.replay` accepts `--transport ubmem|urma` and
`--mode serial|fused`. `python -m kvip.diagnostics.compare` runs all four cases
in separate, sequential processes and prints a comparison table. Both reuse the
existing KVAF installation and operator build paths; no new build script or
KVAF patch is required. Keep UBmem and URMA in separate KVAF checkouts/builds.

Given a verified service snapshot and built runtimes:

```bash
python -B -m kvip.diagnostics.compare \
  --snapshot "$snapshot" \
  --ubmem-build-dir "$ubmem_build" \
  --urma-build-dir "$urma_build" \
  --kvaf-server 127.0.0.1:28999 \
  --device 0 --soc Ascend950DT_9582 \
  --ubmem-serial-workers 32 --urma-serial-workers 32 \
  --urma-serial-max-inflight 0 \
  --warmup 2 --iterations 10 --stage-iterations 3 \
  --result-dir "$new_result_directory"
```

Start one dedicated local `kvaf_cpu_service` for URMA, using the snapshot's
NUMA node and page policy. For device 0 on the validated A5 layout this is
`numactl --cpunodebind=0 --membind=2 .../kvaf_cpu_service --listen 127.0.0.1
--port 28999 --ub-device udmac1d1e2 --memory-page-mode thp`.
Stop it after the comparison. This CPU peer is not a vLLM/model service.
URMA restores the original keys and fixed-slot offsets through its existing
Put/access-table APIs. Both transports allocate the full captured arena even
when exported payload is sparse; the BS16 capture needs approximately 109 GiB
of DDR at a time. THP coverage and physical pages are not replicated by the
snapshot, and the peer's effective page policy must not be inferred from the
source manifest alone.

The measured boundary is query/Index ready through LI, TopK, Fetch, cache
update and TND slot outputs ready. A 2624-byte record includes all four shared
layers; this is a group latency, not a measured 656-byte layer latency or TPOT.
Model work, initial Put, representation conversion, state restoration and
correctness readback are outside the device events. Each replay restores the
original hot/ABC state and checks the logical TopK set and selected KV bytes.
Native LI is executed; the captured TopK is only the correctness reference.

Native LI sees logical token order, while fusion sees the captured hot-first
permutation and ranks BF16 scores. Those orders can break a cutoff tie
differently. Serial replay first checks unique, causal IDs; unequal sets pass
only when every substituted ID is at the captured set's minimum BF16 score in
an untimed selected-union reference calculation using fusion's final
`CAST_ROUND` BF16 rounding rule. The first accepted native ID set must remain
unchanged in eager, full-graph and stage replays. It reports `exact` or
`bf16_boundary_ties`, replacement counts and a selection hash. The CPU FP64
reference rounded to BF16 is an equivalence check against the captured set,
not a bit-exact CANN implementation or a full-history numerical proof. Other
score differences remain failures with token/score diagnostics. All KV byte,
miss-count and resident-map checks still apply to the actual native selection.
Read volumes therefore describe each path's selection, not forced identical
miss IDs. No saved TopK is injected into the measured chain.

Serial reuses CANN's TND/PA_BSND causal LI and production LIDU/Get. Its logical
paged Index and resident maps are reconstructed from the captured hot-first
permutation outside timing. The archive does not record the original serial
map allocation padding, so this is a semantic input reconstruction rather
than a byte-for-byte reconstruction of that backend's former allocations.
Serial Get defaults match serving: UBmem 32 workers, URMA 2, explicitly
overridable. Fused bindings use their original geometry and worker contracts.
These defaults are compatibility settings, not tuned performance baselines.
The candidate `--urma-serial-workers 32 --urma-serial-max-inflight 0`
removes the runtime's default total
inflight cap of 2048 and uses its existing whole-slice submission path. It is
limited to URMA serial replay; fusion queue geometry and UBmem are unchanged.
Runtime workers and the effective inflight policy are recorded in each result.
On node24's captured B16/Q4/2624-byte workload this candidate measured
979.080 us, worse than the earlier 2-worker result of 895.811 us. Increasing
workers alone is not a validated baseline improvement.
The older V9 serial implementation used two jettys per worker, no total inflight
cap and 32-WQE posts. Current KVAF retains two jettys but uses 128-WQE posts.
To reproduce the old post width without replacing the current Runtime, pass
`--urma-get-post32` to the existing installer/build entry. Its focused optional
KVAF patch builds `bin/kvaf_execute_batch_get_post32.o`; the default Get object
and fusion remain unchanged. Select it with `--urma-serial-post-size 32` in
`compare` or URMA-serial `replay`. Results include the selected object's hash.
This reuses the earlier post/concurrency policy, not the old binary.
Its optimum worker count depends on the real misses and four-layer width.
An optional `--urma-serial-workers-candidates 16 32` adds one serial run and
reports both results, using the lower validated device mean for the ratio.
With stage diagnostics enabled, URMA serial also runs one native Get profile
after headline timing. It reports active requests, posts, completion counts,
post-active and drain-wait cycles per core. These cycles are not converted to
microseconds or added to device latency. The extra pass restores and verifies
the same pre-state; profiling is disabled afterward, including on failure.

Backend-specific `--ubmem-fused-tuning` and `--urma-fused-tuning` accept JSON
objects. For the captured four-layer B16 case, an explicit candidate is:

```bash
--ubmem-fused-tuning '{"parallel_init":false,"balanced_index":true,"skip_inactive":true,"compact_reserved":true,"grouped_copy":true,"prefetch_limit":256,"window_limit":32}' \
--urma-fused-tuning '{"compact_reserved":true,"prefetch_limit":512,"window_limit":64}'
```

UBmem's opt-in `grouped_copy` pairs independently validated 2624-byte rows in
four pipeline stages. It keeps the original eight-row scratch footprint and
load lookahead. Positive representable address gaps use two-block DMA;
reversed/repeated addresses use ordered single-row operations, and an odd tail
is drained normally. The 656-byte pipeline is unchanged. Index TND publication
assigns complete query rows to available AIVs (64 at B16), retaining
plan-before-output-before-apply ordering. Both backends skip speculative bitmap
scans when the admission budget is exhausted, preserving event completion and
final repair. Suppression counts cover candidates actually examined, not bits
skipped by this gate. Init still constructs all required mapping/protection
state inside timing; the previously regressing parallel-init option remains
disabled in this candidate.

UBmem completion bookkeeping and row-local Index planning now share one AIV
launch after resolve has joined all Fetch workers. TND output publication stays
in a separate launch so every row's plan is ready before cross-row reads;
Index apply remains last. Completion/error checks and hot-valid publication
are preserved. The device chain has six launches instead of seven. This
removes a boundary, not the work inside completion; hardware measurements are
still needed to establish its benefit.

The summary separately reports correctness and `performance_target.target_met`.
The performance target requires both at least 35% lower device-graph mean and
at least 1.5x speedup against the measured serial path (the former is stricter:
approximately 1.5385x). Candidate-call wall time and instrumented stages do not
enter that ratio. Host models or a successful compile do not establish target
attainment; the same-snapshot NPU result is required.

Node24's later NPU4 run measured UBmem serial/fused at 684.875/676.665 us
and URMA serial/fused at 584.858/583.266 us. Neither met that target. Both
URMA paths improved substantially on the different NPU, although fusion does
not use the standalone post32 object; this does not isolate a post32 benefit.
Native serial diagnostics show completion waiting dominates posting cycles.
The current URMA resolver records deferred final Get descriptors; the final
Fetch kernel submits and waits for those reads after early Index update.
`index_update_before_final_get` names that boundary explicitly. It may overlap
prefetch DMA, but it is not evidence that final-demand reads overlap Index.
The adjacent diagnostic stages are `resolve_final_plan` and
`final_get_and_completion`; the latter includes submitting the final Get,
not just waiting for previously submitted work.

Two further UBmem switches are opt-in and preserve all queries and KV bytes:

- `tagged_map=true`: encode the per-row epoch alongside mapping slots. Normal
  rounds advance the epoch instead of clearing all history entries. First use,
  invalid epoch state and 14-bit epoch wrap still clear inside Init. It requires
  serial Init and a bounded output stride; old epochs decode as invalid. No
  preparation is moved outside the graph. Warm replay measures steady-state
  epochs, not first-use or wrap latency.
- `final_copy_stages=8`, with `grouped_copy=true` and 2624-byte rows: increase
  final-copy capacity from 8 to 16 in-flight rows using retired resolver scratch.
  Window-copy depth remains unchanged. Default final depth remains 4 stages.

`replay --fused-variants` accepts two to four objects with unique `name` and
`tuning` fields for either UBmem or URMA. It uses the existing ablation runner
to import DDR once and retain one Runtime, access table and transport connection,
then builds, restores, verifies and times each case separately. Comparing
current, tagged-map-only, and tagged-map-plus-deeper-final isolates the two
incremental changes without a new model service or a worker sweep. The
snapshot's original geometry is retained; mapping diagnostics decode epoch
tags before counting useful/unused prefetch. New tagged captures also retain
the raw mapping tensor as `post.mapping_raw`.

The following experiments retain all completion, output and Index work inside
the graph. They are opt-in and require real-snapshot correctness/performance
validation before use in serving:

- `rolling_threshold=true` reuses the resident TopK as the first 16K chunk,
  then performs each existing BF16 TopK merge when that score chunk completes.
  The updated kth score filters later score tiles. Published bitmaps are never
  rewritten. Final TopK copies the saved indices instead of repeating the
  chunk merges. State uses the existing prefix GM buffers, because Fetch
  borrows the TopK UB scratch. Extra state transfers and synchronization can
  offset the saved sorting/prefetch work. Both backends require resident=16384,
  split-Q2 and `threshold_only=false`; unsupported shapes are rejected.
- URMA `single_planner=true` computes the immutable final slot plan once per
  request and publishes all original final-worker ranges. Final Get concurrency
  is unchanged. `skip_final_partition=true` independently skips the old
  safe/conflicting descriptor partition. Both require `parallel_final=true`
  and `async_final=false`: this path joins prefetch before any final Get, so the
  safe-prefix split is unnecessary. Neither option adds a kernel launch.
- `reserved_ub=true` builds the reserved-hot inverse map in a 64 KiB UB
  scratch area, then publishes the entire row with one DMA. It replaces
  2048 scattered GM stores per active request, retaining the same overwrite
  protection and the existing tail-init VF. It requires resident=16384,
  `compact_reserved=true`, `simt_init=false`, and UBmem `parallel_init=false`.
  Init scratch grows by 64 KiB; init remains inside the measured graph.
- UBmem `cached_buckets=true` retains the hash bucket of each selected miss
  in currently unused slot scratch, then reuses it in the subsequent rank and
  location passes. The slots become destination-slot storage only after all
  bucket consumers finish. Outputs, ordering and the selected hot slots are
  unchanged; this reduces repeated hash probes, not payload copy bytes.
- URMA `post_final_before_index=true` joins each row's prefetch and submits
  its whole final-read ranges before Index. The same final workers drain their
  own queues after Index, and the final completion/error checks remain. It
  requires `parallel_final=true`, `async_final=false`; it is compatible with
  single-planner/skip-partition. Final payload reads can overlap Index metadata
  work without destination conflicts with prefetch. The row worker submits
  four queues sequentially, so increased submit cost may offset hidden NIC
  wait. No kernel launch is added. Stage diagnostics identify the combined
  prefetch join/submission explicitly.

Three follow-up experiments retain the measured window/selection policies:

- UBmem `fused_init=true` runs the **complete existing initialization** inside
  the mixed LI kernel. All launched AIVs finish a hardware collective before
  LI can publish its cube/vector credits. The independent init and LI entries
  remain available. This removes one launch boundary and can overlap AIC setup;
  it does not remove initialization work from timing. It requires
  `parallel_init=false`. Diagnostic timing uses `init_li_with_prefetch` as one
  inseparable stage.
- UBmem `final_copy_rows=4, final_copy_stages=6` increases the final-copy ring
  from 16 rows (8 pairs) to 24 rows. Its 62,976-byte scratch fits the same
  retired 64 KiB selection/location banks; live address vectors are untouched.
  Exact forward arithmetic runs share a DMA command; reversed, repeated or
  incompatible addresses keep their original order. No gap bytes are read or
  written. This requires grouped 2624-byte copy and `ordered_pair=false`.
  Window copy and the default 4/8-stage pair paths are unchanged.
- URMA `merge_final_completion=true` drains all of a row's posted final queues
  in the existing completion kernel, then performs the original completion
  checks. It requires `post_final_before_index=true`. Each CQ still has one
  consumer; an error does not skip sibling drains. This removes the separate
  final-wait launch, but serial polling of a row's four queues can offset that
  saving. Diagnostic timing is `final_wait_and_completion`.

Node24 NPU4 measured revision `9b6b696`: UBmem serial 682.726 us versus
fused `reserved_ub` 649.735 us (4.83% lower mean; 4.91% lower p50).
Adding cached buckets regressed to 657.909 us, so keep that option disabled.
URMA serial was 605.932 us versus fused reserved-map/final-post overlap
477.283 us (21.23% lower mean; 20.05% lower p50). The different mean/median
reductions expose serial timing outliers. Neither meets the 35% target;
the follow-up experiments above have no measured NPU performance claim yet.

Extended Index planning now clears and scans only its producer high-water
prefix and the corresponding path-closing prefix. Their ABI offsets and record
order are unchanged; old records outside those ranges cannot be consumed.
This only benefits rows that use the extended planner.

Replay also reports `prefetch_admission` from existing Row counters, after
timing: per-window posted/suppressed candidates, strictly window-limited epochs,
other/tied limits, zero-budget epochs and active-row unused budget. Suppression
counts examined per-window candidates, not globally unique or final cold misses;
windows skipped after budget exhaustion do not count their unseen candidates.
`prefetch_limit` is per request across all windows, while `window_limit` applies
to each individual service visit. Unused capacity in an early window is not a
separate bank of credits, and candidates skipped at that visit are not retried
automatically. Compare window changes at a fixed total budget before increasing
the total; report complete graph latency rather than summing diagnostic stages.

Node24 NPU4's `7f3a767` window sweep measured UBmem means of 646.998 us
(window32), 668.706 us (window64), 671.793 us (window128), and 651.503 us
(window64+rolling), with a fixed per-request total of 256. Wider windows moved
work from final Fetch into LI but increased the full graph. Retain window32
as that snapshot's measured baseline; rolling32 has not been measured.
URMA measured 512.010 us (window64), 493.294 us (window128), 488.878 us
(window128+single planner+skip partition), and 481.299 us (plus rolling),
with total512: 6.00% below its same-run fused baseline. These results do not
establish 35% improvement over serial, whose current-version timings must be
measured separately. The new init/hash/post experiments remain opt-in.

After running serial references and fused variants on one revision, reuse
their JSON files without another device run:

```bash
python3 -m kvip.diagnostics.compare --from-results \
  ubmem-serial.json urma-serial.json ubmem-variants.json urma-variants.json \
  --output summary.json --quiet
```

The command rejects failed or mixed-snapshot/revision/geometry/scope cases,
retains all candidate means, and selects the lowest validated fused mean in
each sweep. It reports target attainment against the saved serial references and shows
median reduction alongside the mean; the target still uses the full graph mean.
`--split-index-stages` also works for URMA: it separates the existing
early-output/plan launch from the owned Index apply launch. This localizes
Index cost without adding work or splitting the uninstrumented full graph.
This is selection from a finite sweep, not an independent remeasurement of
the winner; the summary records that limitation.

Three additional opt-in switches target the remaining stages; all default to
false and preserve the selected KV bytes and required state updates:

- UBmem `ordered_pair=true` requires `grouped_copy=true` and 2624-byte rows.
  Final workers own balanced, disjoint rank ranges instead of rescanning every
  descriptor. Reverse-address pairs may swap source and destination together
  to use one strided Host read; overlapping destinations or unrepresentable
  strides retain the exact-row fallback. 656-byte sector ownership is unchanged.
- UBmem `merge_outputs=true` publishes each packed batch's TND outputs inside
  the existing completion/planning VF, reducing the graph from six launches
  to five. Each writer uses its own plan status; invalid/padding rows still
  receive -1. Index apply remains a later kernel so output reads see old logical
  IDs. This trades output parallelism for one fewer launch and needs an NPU A/B.
- URMA `async_final=true` requires `parallel_final=true`. Resolver workers post
  final descriptors that cannot conflict with outstanding prefetch writes,
  without waiting for CQ completion. Those reads can overlap Index update.
  After prefetch completion, workers post conflicting tails and drain their
  private queues. Queue bounds, completion errors and poisoned-queue rejection
  remain active. Stage names distinguish submission, Index overlap and final
  drain; no necessary preparation is moved outside timing.

Same-runtime variants control DDR placement and connection setup within each
transport. They do not eliminate load/thermal variability, and their printed
speedup compares fused variants, not the separate serial performance target.
Node24 NPU4 measured the UBmem ordered-pair/output-merge combination at
864.554 us versus 677.495 us before, a 27.61% regression. The instrumented
completion/Index chain rose from 87.746 to 267.312 us while final Fetch stayed
near 196 us; one fewer launch did not compensate for reduced output parallelism.
This paired run does not isolate ordered-pair copying, so both switches remain
off in the next baseline. URMA async-final measured 672.136 versus 668.285 us:
the shorter final drain was offset by longer prefetch completion. It also
remains off; neither experiment establishes the serial-relative target.

Three shared switches, also disabled by default, reduce work inside the timed
graph without changing the captured inputs or the final TopK algorithm:

- `simt_init=true` combines control-record zeroing, initial LRU copying and Row
  publication into the existing init VF. Bulk sentinel initialization, epoch
  reset, hot-slot protection and URMA queue validation stay in init. It does
  not add a launch or move preparation outside timing. UBmem cannot combine
  it with `parallel_init=true`.
- `threshold_only=true` computes the exact resident-prefix cutoff with the
  existing BF16 sortable-key histogram stages, avoiding a full TopK whose
  selected resident IDs are unused. Short/invalid and multi-trunk prefixes
  keep the original path; final full-history TopK and tie behavior are unchanged.
- `cold_only_prefetch=true` scans speculative candidates from the resident
  boundary onward and omits unused hot-candidate bitmap production. Selected
  candidate-hot tokens are still repaired by the final resolver. It may shift
  useful work from prefetch to final Fetch, so lower read traffic alone is not
  a latency win.

Node24 NPU4's paired run enabling these three switches together measured
UBmem at 694.429 versus 698.832 us (p50 694.940 versus 692.621 us), and URMA
at 711.141 versus 681.243 us. This is no stable UBmem improvement and a 4.39%
URMA regression. Candidate-hot prefetch was about 91% useful before the change;
removing it increased final reads while unused cold prefetch barely changed.
Keep `cold_only_prefetch=false` for the next experiment. Init/LI stage gains
do not establish a full-graph gain or isolate the three switches individually.

Two further switches remain opt-in, with all new work included in graph timing:

- UBmem `single_plan=true` uses one final planner per request instead of four
  redundant planners, then launches the original four copy workers on the same
  stream. Existing descriptors, slot ownership, byte copies and error checks
  are preserved. The added launch and descriptor reload must earn their cost;
  diagnostic stages are `resolve_plan` and `final_fetch`.
- Shared `staged_outputs=true` stages the resident logical-ID table, packed Q4
  TopK, slot locations and TND map in UB. Hot lookup reuses this table and whole
  output rows use DMA; cold logical IDs retain their original GM lookup. Apply
  remains a later kernel. Resident lengths above 16384 or batches above 32
  retain the original path; invalid/padding output behavior is unchanged.

Node24 NPU4's `51efb06` four-path run measured device means of 679.101 us
(UBmem serial), 586.706 us (URMA serial, 32 workers/post32/inflight0),
670.706 us (UBmem fused, single plan plus staged outputs), and 515.975 us
(URMA fused, staged outputs). Fusion reduced latency by 1.24% and 12.06%,
respectively; neither met the 35% target. UBmem output diagnostics fell to
16.427 us, but its split plan/fetch diagnostics were 47.976/172.170 us.
These are instrumented stages, not additive full-graph components. The large
URMA final-Get change across runs does not establish that output staging
overlapped that Get: this configuration still submits final Get after Index.

The next UBmem comparison retains staged outputs and separately tests removing
the extra single-plan launch, then enabling the existing `ordered_pair` path.
The latter assigns each copy worker a contiguous rank range and can coalesce
reverse-ordered source pairs after swapping both sources and destinations.
It preserves payload bytes and destination ownership; window copies use the
same pair logic, so prefetch timing and waste may change. This needs device
measurement; prior ordered-pair results also enabled the regressing merged
Index path and did not isolate copy performance.

Round initialization now fills only the 2048 candidate change records. Final
resolver writers publish all extended records that Index can read, including
sentinel holes for ranks whose actual changes live in the candidate prefix.
This removes 512 KiB of initialization writes for B16 (the original one-worker
init removes 32 fill commands), while adding bounded sentinel writes in the
resolver. It preserves replayed state, empty/error handling and the existing
Index high-water mark. It is a small reduction of redundant work, not evidence
that the whole init diagnostic interval can be eliminated.

Use `replay --quiet` or `compare --quiet` to print only progress, effective
tuning, correctness, timing, read counts and stage diagnostics. Successful
native build/link chatter is discarded; failures retain the log and show its
last 80 lines. Reprint a saved result without rebuilding or touching an NPU:
`python -m kvip.diagnostics.replay --report-only /absolute/path/result.json`
(the same option on `compare` reads its `summary.json`). Full result JSON still
contains the measurements; compact output does not change timing boundaries.
Compact reports also show actual, useful (excluding unused prefetch), and cold
payload MiB and decimal GB/s normalized by the same full-graph mean. These
are operator-chain throughput figures, not hardware bandwidth. A separate
instrumented serial Get rate is shown only when that stage has its own byte
samples and mean time; full-graph bytes and stage p50 are never mixed.

Replay readback additionally classifies actual prefetch and unused prefetch
against each request's frozen initial cold, candidate-hot, other-hot and ABC
residency. Mapping coordinates are converted through the pre-call logical-ID
permutation; epoch tags are decoded first. This bookkeeping is outside timing,
as are the existing output checks. `prefetch_unclassified` makes any gap against
the native transfer counter explicit rather than assigning it to cold demand.
Summary columns take separate medians; their sums need not equal the median
of the per-iteration sum. Use `reads_samples` for exact per-round accounting.

Serial graphs use the Torch NPU graph allocator for native LI/gather outputs;
fused graphs retain the existing KVAF graph API. The supported explicit stream
argument is documented in [Torch NPU graphs](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/graphs.py).

The four full-path means/medians come from uninstrumented production graphs.
`--stage-iterations` additionally captures each existing stage without changing
its internal launches, then measures the ordered stage graphs with device events.
These additional graph/event boundaries can change scheduling and overlap;
stage times locate large costs but are not an exact additive decomposition of
the full-path latency. URMA's original early Index update can overlap prefetch;
only the explicit async-final path above also posts final-demand reads before it.
UBmem's `li_with_prefetch` includes concurrent LI/TopK and speculative copies,
so it must not be interpreted as pure LI compute time.

Read counters distinguish unused speculative prefetch from serial LIDU's
candidate-hot refetch. The latter rereads selected hot slots reserved for
replacement and is separate from genuine cold demand. Equal source inputs do
not require equal transport bytes or identical speculative scheduling.
ABC mappings are not DDR prefetches. The comparison saves four case JSON files
and `summary.json`, without persistent native setup logs or profiler traces.
Host tests cover layout reconstruction, ABI arguments, restoration, validation
and reporting; actual four-path correctness/timing still requires NPU execution.

### Isolate UBmem init, Index update and speculative reads

For an existing snapshot, run the three-case experiment without a model or CPU
peer service:

```bash
python -B -m kvip.diagnostics.replay \
  --snapshot "$snapshot" --build-dir "$ubmem_build" \
  --transport ubmem --mode fused --device 0 --soc Ascend950DT_9582 \
  --ubmem-ablation --split-index-stages \
  --warmup 2 --iterations 10 --stage-iterations 3 \
  --output "$new_result_json"
```

This restores the captured DDR arena once and runs three sequential cases:

1. `baseline`: existing init/Index kernels and captured prefetch budgets.
2. `optimized`: shard init's map clears across available AIV cores, and spread
   Index scale/address/logical-ID movement across three of the four workers
   instead of assigning all metadata to one worker. Both small and extended
   Index plans preserve the snapshot-before-scatter rule for permutation cycles.
3. `optimized_half_prefetch`: the same optimized binary, with per-request and
   per-window prefetch budgets halved (512/64 becomes 256/32). LI, TopK,
   window count, history and four-layer record width stay unchanged.

Each iteration restores the original pre-state outside timing and runs the
existing correctness checks; the three cases share DDR but not mutable hot
cache/workspace state. Full graphs retain the production launch sequence.
`--split-index-stages` diagnoses completion plus plan generation, TND outputs
and field movement separately as `complete_index_plan`, `index_outputs` and
`index_apply`. Without that flag they are grouped as `complete_index_update`.
Do not compare that combined scope directly with old index-only timings. These diagnostic
graph boundaries are excluded from full-graph timing. The result JSON keeps
all timings, read counts, effective tuning and source/binary hashes.

The optimization switches are `Specialization` fields (`parallel_init`,
`balanced_index`). Parallel init remains off in serving; balanced Index movement
is enabled for UBmem serving after the real-snapshot validation. URMA Index
movement is unchanged. The existing
specialization cache builds one baseline and one optimized library, then reuses
the optimized library for the smaller budget. Reducing prefetch can reduce
wasted reads but increase final-fetch latency, so compare complete graph times
before choosing a budget. Local host-model checks do not establish NPU speedup.

### Skip padded LI work and construct protection maps from compact inputs

Both fusion backends keep graph capacity and addresses stable, but skip LI's
matrix/vector/history loop for a request whose four device query bounds are
zero. The inactive core still participates in its assigned prefetch window and
cube/vector synchronization. Window service waits for active producers to reach
the corresponding history boundary before sampling their candidate state.
Init, final resolution and other control work retain their existing capacity.

Serving also stops constructing/uploading the dense reserved-hot inverse map
on the CPU. It uploads the existing LRU candidate lists; the original init
kernel rebuilds the inverse map for active packed rows on the NPU. The mapping
uses packed request rows, not persistent owner IDs. For 21 groups, 17 allocated
rows and 16384 hot slots this removes 22.3125 MiB of extra H2D per worker/step;
the candidate upload already existed. No extra kernel or CPU readback is added.
Both repository revisions must be updated together. A compact-mode capture
labels the inverse map as derived from its saved LRU input and verifies the
device result after the call, rather than saving the preceding round's map.
The reconstruction reuses init's existing all-ones scratch and DMA completion
barrier, then scatters inside the existing ABC-tail VF call. It avoids a second
16 KiB scratch fill and standalone VF launch while retaining clear-before-scatter
ordering and a single writer per protection row.

To validate these changes from a verified B16 snapshot without a model service:

```bash
python -B -m kvip.diagnostics.replay \
  --snapshot "$snapshot" --build-dir "$ubmem_build" \
  --transport ubmem --mode fused --device 0 --soc Ascend950DT_9582 \
  --service-ablation --active-batches 1 16 --split-index-stages \
  --warmup 2 --iterations 10 --stage-iterations 3 \
  --output "$new_result_json"
```

Five cases share one restored DDR arena: B1 dense, B1 skip-only, B1 skip+compact,
B16 dense and B16 skip+compact. All disable the regressing parallel-init option,
keep balanced Index movement, and halve the captured prefetch budget (512/64
becomes 256/32). B1 is explicitly a derived padding view at B16 capacity: it
preserves one real request's packed inputs, owner, DDR and hot state. It is not
a new B1 service capture or a B1-specialized kernel. Valid TopK/KV/Index results
are checked against the corresponding captured requests, and padding outputs
must remain -1. Before each compact run the inverse map is poisoned outside
timing; correctness checks require the kernel to rebuild the active rows.

Compare B1 dense vs skip-only for padded-LI savings, and skip-only vs compact
for inverse-map construction overhead. The full B16 comparison checks the
unchanged active workload. The original `timing` boundary remains device-only.
The original archive is never rewritten, and JSON provenance records the derived
selection and original manifest hash.

To include the reproducible host preparation in the same comparison, add
`--measure-call-preparation`. For just the captured B16 workload, use:

```bash
python -B -m kvip.diagnostics.replay \
  --snapshot "$snapshot" --build-dir "$ubmem_build" \
  --transport ubmem --mode fused --device 0 --soc Ascend950DT_9582 \
  --service-ablation --active-batches 16 --measure-call-preparation \
  --split-index-stages --warmup 2 --iterations 10 --stage-iterations 3 \
  --output "$new_result_json"
```

The two cases share one arena import and the same real queries, DDR/hot state,
balanced Index movement and 256/32 prefetch budgets. They compare legacy dense
protection metadata against device reconstruction. Both retain full correctness
checks. The result separates these boundaries:

| Result | Measured interval |
| --- | --- |
| `timing` | Inputs already on device through full fusion graph completion, including required init reconstruction |
| `candidate_call.wall` | Captured candidate lists ready on CPU, staging copy and legacy inverse-map construction, necessary H2D, graph submission and synchronized completion |
| `candidate_call.cpu_prepare` | CPU staging and inverse-map construction within the wall-time pass |
| `candidate_call.h2d_diagnostic` | Separate event-instrumented upload interval; enabled by `--stage-iterations` |

The host-wall pass has no NPU timing-event submissions in its measured interval.
Pinned buffers, ordinals and array views are allocated beforehand; diagnostic
validation, archive reads, DDR import, state reset and correctness readbacks
remain outside every timed interval. Both input buffers are poisoned before
each call so missing candidate uploads or protection reconstruction cannot pass
by using the previous valid state. The CPU staging buffers remain owned until
stream completion. H2D diagnostics and device-only graph samples are separate
runs, not additive components of host wall time.

This is a **single-group candidate-ready call**, not the complete service Target
step. A one-group archive does not contain scheduler/LRU feedback, generation of
the candidates, owner lifecycle, new Index ingest, shared ABC maintenance, or
the combined preparation/upload for all groups. These exclusions are recorded
in JSON. One must not multiply the measured group preparation latency by 21.
The B16 archive has 16 packed rows: its dense table is 1 MiB and candidate list
128 KiB. The observed service allocation has 17 rows including the guard and
21 groups: the deleted dense upload is 22.3125 MiB **once per worker/Target step**;
the pre-existing 2.7890625 MiB candidate upload remains. The snapshot reproduces
that optimization's local cost, but cannot establish a full-service TPOT gain.
