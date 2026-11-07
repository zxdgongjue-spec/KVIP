# KVIP

Ascend A5 cache operators extracted from vLLM-DSA's KVOP. Python 3.12,
CANN 9.1, matching `torch`/`torch_npu`, and a reachable KVAF CPU service are
required for NPU validation. Installation is editable: keep this checkout beside
the installed Python package because startup builds use its C++ sources.

## Layout and API

```text
csrc/operators/
  lidu/                             # op_host / op_kernel / torch_extension
  lightning_indexer_with_fetch/
    include/                        # public ABI, Context and runtime shape
    op_host/                        # launch entry points and Index-update host binding
    op_kernel/                      # LI/Fetch lifecycle, prefetch, resolver, Index update
      indexer/                      # original LI/score/TopK template implementation
python/kvip/
  operators/lidu.py
  operators/lifetch/                 # binding, forward, workspace, tiling, Index update
  cache/                            # LRU, ABC and persistent Index storage
  adapters/kvaf.py                   # Runtime lease and shared transport queues
  adapters/vllm/                     # inputs, selector, startup binding and paged Index import
  runtime/                          # build cache, library loader, hardware query, integrity
  _compat.py                        # centralized lazy aliases for legacy kvip imports
scripts/                            # build, environment and KVAF patches
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
git clone --branch codex/lifetch-hot-resolver https://gitcode.com/CacheOS/KVIP.git /vllm-workspace/KVIP
cd /vllm-workspace/KVIP
source /usr/local/Ascend/cann/set_env.sh
bash scripts/build.sh
source scripts/env.sh
```

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
Preparation applies the three patches in `scripts/patches`, recognizes an already
patched tree, and never resets existing edits. A different revision or conflicting
patch fails explicitly. `--prepare-only` prepares dependencies without compiling;
`--skip-kvaf-build` is only for an already rebuilt patched KVAF. One binary supports both 656-byte and
2624-byte KV records through runtime parameters. Build logs are under `build/lifetch/*/build.log`.
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
