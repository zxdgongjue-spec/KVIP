# KVIP

Ascend A5 cache operators extracted from vLLM-DSA's KVOP. Python 3.12,
a matching CANN and `torch`/`torch_npu` runtime are required for NPU validation.
URMA and UBmem require a reachable KVAF CPU service. UBmem imports the service's
Host DDR mapping into the NPU process. Installation is editable: keep this checkout beside
the installed Python package because startup builds use its C++ sources.

## Layout and API

```text
csrc/operators/
  lidu/                             # op_host / op_kernel / torch_extension
  common/lifetch/                    # resident selection, budgets and Index-update planner
  lightning_indexer_with_fetch/
    include/                        # public ABI, Context and runtime shape
    op_host/                        # launch entry points and Index-update host binding
    op_kernel/                      # LI/Fetch lifecycle, prefetch, resolver, Index update
      indexer/                      # shared LI/score/TopK device functions
  lightning_indexer_with_fetch_ubmem/
    include/                        # UBmem launch, region and completion ABIs
    op_host/                        # graph-capturable launch sequence
    op_kernel/                      # runtime-shaped LI, controller, pipelined copies, resolver/update
python/kvip/
  operators/lidu.py
  operators/lifetch/                 # shared forward and workspace layout; URMA binding and update
  operators/lifetch_ubmem/           # UBmem startup build, binding, workspace and update
  cache/                            # LRU, ABC and persistent Index storage
  adapters/kvaf.py                   # URMA queues and UBmem mapped-region lease lifetimes
  adapters/vllm/                     # inputs, selector, startup binding and paged Index import
  runtime/                          # build cache, library loader, hardware query, integrity
  _compat.py                        # centralized lazy aliases for legacy kvip imports
scripts/                            # existing build/environment entry points and common KVAF patches
tests/
  operators/lidu/
  operators/lifetch/                 # unified CLI, correctness/performance tests and diagnostics
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

`LIFetch` accepts either the URMA or UBmem rotating binding and its matching
Index update; it selects the transport launcher when the binding is created.
It is not a new tensor ABI. It retains both objects and uses their preallocated
bindings. No tensor allocation, synchronization, fixture reset or correctness
readback is added to `forward()`. Transport/workspace ownership must outlive graph
replay. Device errors remain available through the existing status/feedback path.
Individual launch methods remain internal diagnostic interfaces.

`kvip.QueryInputs`, `kvip.LIFetchWorkspace`, `kvip.IndexCacheStorage`,
`kvip.build_libraries` and `kvip.bind_selectors` expose the existing service setup.
`bind_selectors` dispatches on the policy configuration's transport and requires
the current vLLM KVAF policy/writer contract; this is not
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

The build pins KVAF `main` at `87e261ae4d07d4868cfdbb306d15f3f137edfc3d`.
KVIP applies its unified Runtime LIFetch patches and the UBmem device headers
when selected. Preparation can reuse an already patched checkout.
`--prepare-only` prepares the dependency without replacing the active build selection;
`--skip-kvaf-build` requires matching existing build products.

```bash
bash scripts/build.sh --transport ubmem --kvaf-root /vllm-workspace/KVAF-ubmem \
    --cann /usr/local/Ascend/ascend-toolkit/latest --soc Ascend950DT_9582 --jobs 2
source scripts/env.sh
```

UBmem uses KVAF's existing CPU service, Host DDR allocator and Get/Put kernels.
The fused lease retains the imported NPU mapping and its graph-owned buffers.
The CPU service manages memory placement; worker-local allocation, THP and NUMA
settings from the earlier local-DDR prototype are not part of this path. URMA
remains the default build.

The UBmem single-operator path uses the CPU-service DDR region and KVAF BatchPut.
Fused final fetch consumes the shared descriptors through KVAF's patched UBmem device Get.
vLLM integration with this CPU-service-backed interface has not yet been validated.

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
python tests/operators/lifetch/test_lifetch.py diagnose --transport urma --device 5 --kvaf-server 127.0.0.1:28885 --batches 16 --history 131072 --resident 16384 --index-heads 32 --miss-percent 10 --intermediate-fetches 8 --prefetch-tokens 512 --warmup 4 --iterations 20 --level 2
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

The host suite checks classification, staged Index exchanges, Fetch control flow,
and launch/error contracts. The NPU runner validates the actual full pipeline:
cold starts, hot-victim collisions, every Index field, selected KV, all valid hot
KV and repeated stateful graph replays. Historical replicated-resolver and
standalone Index planner/mover fixtures were removed with their implementation.

Independent LIDU and ABC device checks:

```bash
python tests/operators/lidu/test_lidu.py --device 5
python tests/cache/test_abc.py --device 5
```

## LIFetch tests

The single entry is `tests/operators/lifetch/test_lifetch.py`. Choose `check`,
`bench`, `diagnose`, or `all`; select URMA/UBmem with `--transport`.
SoC detection is automatic. Set an explicit KVAF root when it is not the
checkout's `third_party/KVAF`. No historical source checkout is created.

```bash
source /usr/local/Ascend/cann/set_env.sh
source scripts/env.sh
python tests/operators/lifetch/test_lifetch.py check --suite init \
  --transport urma --device 5 --kvaf-server 127.0.0.1:28905
python tests/operators/lifetch/test_lifetch.py bench --transport ubmem \
  --device 5 --kvaf-server 127.0.0.1:28905 --batches 1,2,4,8,16
python tests/operators/lifetch/test_lifetch.py diagnose --transport urma \
  --device 5 --kvaf-server 127.0.0.1:28905 --batches 16 --level 2
```

Initialization tests execute Put/Get and recreate resources. Bench records
unprofiled paired graph timings; diagnose performs a separate profiled pass.
Completed timing remains visible when numerical validation fails. Missing
cases, execution errors and incomplete traces are reported explicitly.
Host models and regression tests live under `support/`.
See [test design](docs/performance/lifetch-tests.md) for the target interfaces
and measurement definitions; real trace replay and diagnostic completeness
are still being integrated on this branch.

## Migration evidence and limits

Source baseline: vLLM-DSA commit `dad47855b20cbe8b74bf412f05757043d22ce0a0`.
LIDU originally migrated byte-identically; its bitmap count/enumeration now lives
in a shared header used by LIFetch fallback as well. At the initial migration,
retained LI/Fetch/resolver/update function bodies
were compared after removing include paths, comments and whitespace. Both Q2 and
Q4 materialized LI sources match the prior generation pipeline; the build no
longer rewrites source files. The current pipeline uses shared LI classification
and staged Resolver/Index update/Fetch. Old replicated-resolver and standalone
Index update launchers and their build targets have been removed.

`scripts/migration_sources.json` records the initial source provenance; its old
hashes are historical, not an assertion that optimized files are unchanged.
Hardware correctness and performance still require A5 validation.

The imported CANN LI headers retain their original license notices; their license
is reproduced in `LICENSE-CANN`. Apache-2.0 applies to files so marked.

## Runtime shapes

Installation does not take device, batch, history, resident, heads or window count.
Host tiling evaluates the original partition/window formulas when a launch object
is bound, then stores its immutable parameters in the device Context before graph
capture. No per-shape compilation is required. Index update reads history/resident
from its bindings. Rebuild LIFetch after updating; it no longer depends on a separate Index update library.

Supported runtime dimensions: batch 1–16 (also limited by the concurrently resident AIV budget),
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
python tests/operators/lifetch/test_lifetch.py bench --transport urma --device 5 --kvaf-server 127.0.0.1:28885 --batches 1,2,4,8,16 --miss-percent 10 --warmup 4 --iterations 20
```

The command covers BS1/2/4/8/16 with 656-byte KV records. History, resident,
heads and requested windows can be selected for separate benchmark runs. Each case
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

UBmem fusion ABI 3 uses the same final TopK protection and hot-slot selection
as URMA, in `csrc/operators/common/lifetch/resident_selection.h`. Missing rows
replace unprotected persistent hot slots; the payload layout is hot + ABC, with
no separate 8192-token payload region. UBmem retains its eight-stage copy pipeline
and native completion checks. The shared Index planner handles up to 10240 change
records (2048 prefetch candidates plus the Q4 union); plan/work buffers are shared
across sequential layer groups and allocated before memory profiling.

Rebuild UBmem fusion and use the matching vLLM integration before serving. Source
hashes include the shared headers and incompatible earlier ABI libraries are rejected. Resident
capacity must be at least 8192 for the worst-case Q4 union; smaller capacities
are admitted but a union exceeding available hot slots reports an error.
This hot-slot repair changes the fusion ABI, while retaining the KVAF UBmem
runtime ABI and the URMA transport implementation.

Both bindings default to 512 speculative fetches per request/round, accept a
total budget of 0..2048, and cap each window at its configured allowance (at
most 480). Device admission shares the same saturating budget calculation.
The final TopK repair is independent of this speculative budget: setting it to
zero disables prefetch but still retrieves every required cold row. The UBmem
smoke check accepts `--prefetch-tokens 0 512` to check both cases, including eager
execution, graph replay, changed owner packing, and Index/payload consistency.

UBmem uses one library per CANN/SoC build for batch 1–16, 32/64 Indexer
heads and both 656/2624-byte records; runtime geometry is prepared before
graph capture. History and hot capacity are 256-aligned, with
`2048 <= resident < history <= 210944` and `resident <= 131072`. The binary
cache key includes source and toolchain identity but not workload shape. The
controller processes more than eight request rows per service lane in bounded
chunks. BS > 16 is rejected at binding, without a legacy fallback. Generalized hot capacities require NPU validation. Compilation and
host models do not prove runtime correctness or performance.

### Unified streaming LIFetch entry

`kvip.LIFetch(rotating, update)` and `kvip.bind_selectors` use the same pipeline
for URMA and UBmem: init, shared LI/final TopK classification, and fused
per-query Resolver/Index update/Fetch. The old complete launch symbols and the
separate experimental operator directory have been removed. Old standalone
Resolve/Fetch/Index-update entry points are also removed. Native LI/LIDU/Get
remains the independent serial reference. Init and LI launchers serve the shared
full pipeline; Fetch-only diagnostics execute its actual Fetch worker.

BS1–16 is supported; a larger batch raises an explicit error during binding.
The default uses two Fetch AIVs per Q and three pipeline AIVs per request.
All tasks must fit concurrently: `3 * BS + 4 * q_workers <= 2 * AIC_count`.
The public binding allocates before graph capture. Service layer groups reuse
stream-ordered temporary storage while retaining distinct Index and launch
bindings. KVAF patches and transport-specific leases are unchanged.

The unified test entry uses this production pipeline. For a full stage trace:

```bash
python tests/operators/lifetch/test_lifetch.py diagnose \
  --transport urma --kvaf-root /path/to/KVAF --kvaf-server 127.0.0.1:28885 \
  --device 4 --batches 1,2,4,8,16 --miss-percent 10 \
  --warmup 4 --iterations 20 --level 2 --timeout-seconds 600
```

Use `--transport ubmem` with its matching runtime and service. Both transports
select the current formal implementation; there is no public optimization switch.
Host checks and compilation do not replace validation on NPU hardware.
