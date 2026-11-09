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
    op_kernel/                      # runtime-shaped LI, controller, pipelined copies, resolver/update
python/kvip/
  operators/lidu.py
  operators/lifetch/                 # URMA binding, forward, workspace, tiling, Index update
  operators/lifetch_ubmem/            # startup build, runtime binding, workspace and update
  cache/                            # LRU, ABC and persistent Index storage
  adapters/kvaf.py                   # URMA Runtime lease and shared transport queues
  adapters/ubmem.py                  # local mapping/output/graph lifetime
  adapters/vllm/                     # inputs, selector, startup binding and paged Index import
  runtime/                          # build cache, library loader, hardware query, integrity
  _compat.py                        # centralized lazy aliases for legacy kvip imports
scripts/                            # existing build/environment entry points and common KVAF patches
patches/kvaf-background-put.patch    # shared background Put binding compatibility
patches/kvaf-ubmem-fused-lease.patch # optional mapped-region fused lease for KVAF UBmem
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
applies `patches/kvaf-ubmem-fused-lease.patch` to that checkout. Preparation recognizes an
already patched tree and preserves unrelated edits; conflicting changes fail
without a forced reset. A separate KVAF checkout/build is recommended per transport.
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

The UBmem single-operator path uses the CPU-service DDR region and the existing
KVAF BatchPut and BatchGet transport. Fused LIFetch retains its own copy scheduling
and completion checks while reusing the transport-independent algorithms above.
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

## URMA / UBmem matched single-operator check

Build one KVAF checkout with both transports and start its CPU service. The
example uses NPU 5 and its URMA device `udmac0d1e3`; adjust the device and SoC
for the machine. Port 28905 must be free. Logs remain under this checkout.

```bash
source /usr/local/Ascend/cann/set_env.sh
bash scripts/build.sh --transport ubmem --kvaf-root "$PWD/third_party/KVAF-ubmem-review" --soc Ascend950PR_9599 --jobs 2
source scripts/env.sh
mkdir -p tests/results
nohup third_party/KVAF-ubmem-review/build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28905 --ub-device udmac0d1e3 \
  > tests/results/kvaf-ubmem-review-service.log 2>&1 < /dev/null &
```

```bash
python tests/operators/lifetch_ubmem/compare_transports.py \
  --device 5 --soc Ascend950PR_9599 \
  --kvaf-root "$PWD/third_party/KVAF-ubmem-review" \
  --urma-server 127.0.0.1:28905 --ubmem-server 127.0.0.1:28905 \
  --batch 16 --history 131072 --resident 16384 --index-heads 32 \
  --miss-percent 10 30 50 --warmup 4 --iterations 20
```

The runner first checks batch, width and history changes against one binary, then
uses the same controlled Q4 LI fixture, Index permutation, LRU
candidate order and 656-byte KV payload. UBmem checks native LI, KV bytes,
Index/slot ownership, real SFA, eager, graph replay and padded rows. Timed graph
replays restore the fixture before the event pair. UBmem and URMA use different
output strides and capture implementations, so their reported events are not a
same-process paired experiment. Detailed logs and one short summary are in
`tests/results/ubmem-compare`. NPU results are not yet established by host tests.

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

UBmem uses one library per CANN/SoC build for batch 1–32, 32/64 Indexer
heads and both 656/2624-byte records; runtime geometry is prepared before
graph capture. History and hot capacity are 256-aligned, with
`2048 <= resident < history <= 210944` and `resident <= 131072`. The binary
cache key includes source and toolchain identity but not workload shape. The
controller processes more than eight request rows per service lane in bounded
chunks. B17–32 and resident below 8192 require NPU validation. Compilation and
host models do not prove runtime correctness or performance.
