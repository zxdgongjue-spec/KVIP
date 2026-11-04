# KVIP

Ascend A5 cache operators extracted from vLLM-DSA's KVOP. Python 3.12,
CANN 9.1, matching `torch`/`torch_npu`, and a reachable KVAF CPU service are
required for NPU validation. Installation is editable: keep this checkout beside
the installed Python package because startup builds use its C++ sources.

## Layout and API

- `csrc/operators/lidu/`: original LIDU host, kernel, Torch extension and build.
- `csrc/operators/lightning_indexer_with_fetch/`: LI/Fetch, resolver, and mandatory
  Index update. The update is compiled separately with its original flags and
  linked into the same launch library.
- `python/kvip/`: public operators, LRU, ABC, input/workspace and transport adapters.
- `tests/`: one entry per functional module; `support/` contains shared fixtures,
  real KVAF setup, correctness checks and timing helpers, not additional test CLIs.
- `scripts/`: dependency preparation, build, environment and KVAF patches.

```python
import kvip

lru = kvip.SlotLRU()
# Existing out-parameter LIDU ABI, on the current Torch NPU stream:
# kvip.lidu(topk, actual_query_lengths, ..., block_size)
# Existing ABC descriptor ABI:
# kvip.update_abc(token_to_slot, slot_to_token, descriptors, requests, history, tail)

# Bind once, before graph capture, using the existing prepared launch objects:
forward = kvip.LIFetch(rotating, update)
forward()  # round init -> LI/prefetch -> resolver -> Fetch completion -> Index update
```

`LIFetch` accepts the existing `RotatingLaunch` and `DirectIndexUpdate` objects;
it is not a new tensor ABI. It retains both objects and uses their preallocated
bindings. No tensor allocation, synchronization, fixture reset or correctness
readback is added to `forward()`. Transport/workspace ownership must outlive graph
replay. Device errors remain available through the existing status/feedback path.
Individual launch methods remain internal diagnostic interfaces.

`kvip.FusedQueryInputs`, `kvip.FusedWorkspace`, `kvip.IndexCacheStorage`,
`kvip.build_libraries` and `kvip.bind_selectors` expose the existing service setup.
`bind_selectors` requires the current vLLM KVAF policy/writer contract; this is not
an arbitrary vLLM-version plugin. The `kvop` import namespace aliases the same
package path for existing service imports. Migrating the service installation to
this repository is separate from deleting its existing KVOP checkout.

## Build

Run inside the Ascend container, with the matching Torch NPU installation already
available. The following downloads the pinned KVAF revision if necessary and
builds KVAF, LIDU and LIFetch including Index update:

```bash
git clone --branch codex/clean-slate https://gitcode.com/CacheOS/KVIP.git /vllm-workspace/KVIP
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

The supported KVAF base is `c4375e13d7a9519d907ab6a5d11227795f207286`.
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
python tests/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --batches 16 --history 131072 --resident 16384 --index-heads 32 --miss-percent 10 --intermediate-fetches 8 --warmup 4 --iterations 20 --trace-rounds 3
```

This uses Q4, TopK 2048, and reports the actual quantized miss fraction. Full-path
timing calls `kvip.LIFetch`, including mandatory initialization and Index update.
Fixture restoration, byte/SFA verification and CPU LRU are outside timing. A
separate profiled pass reports init/LI/resolve/Fetch device durations. Actual DDR
reads and unused prefetches are printed. Summaries, status and errors are saved
under the printed `tests/results/lifetch/<run>/summary.log`; detailed logs stay
beside it. Snapshot and stateful correctness checks are retained.

For the original/adapted/fixed resolver comparison, use the same entry:

```bash
python tests/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --batches 8,16 --history 131072 --resident 16384 --index-heads 32 --miss-percent 10 --intermediate-fetches 8 --warmup 4 --iterations 20 --resolver-compare
```

Independent LIDU and ABC device checks:

```bash
python tests/test_lidu.py --device 5
python tests/test_abc.py --device 5
```

## Migration evidence and limits

Source baseline: vLLM-DSA commit `dad47855b20cbe8b74bf412f05757043d22ce0a0`.
LIDU files are byte-identical. At the initial migration, retained LI/Fetch/resolver/update function bodies
were compared after removing include paths, comments and whitespace. Both Q2 and
Q4 materialized LI sources match the prior generation pipeline; the build no
longer rewrites source files. LI/Fetch and Index update keep their separate
compiler options. Unused non-quantized LI implementations and legacy simulator
fixture entry points were omitted. Original resolver comparison sources are test
only, disabled in normal builds.

`scripts/migration_sources.json` records source provenance and hashes of the
migrated operator files. Hash checks detect accidental edits, not device correctness.
CPU tests include host execution of the Index-update planner/mover and the unified
launch ordering/error path. NPU compilation, graph correctness and performance
must still be verified on A5: this migration was prepared on a machine without
CANN/NPU, so no new device performance result is claimed.

The imported CANN LI headers retain their original license notices; their license
is reproduced in `LICENSE-CANN`. Apache-2.0 applies to files so marked.

## Runtime shapes

Installation does not take device, batch, history, resident, heads or window count.
Host tiling evaluates the original partition/window formulas when a launch object
is bound, then stores its immutable parameters in the device Context before graph
capture. No per-shape compilation is required. Index update reads history/resident
from its bindings. ABI versions are now LIFetch 10 and Index update 2; rebuild both.

Supported runtime dimensions: batch 1–32 (also limited by available AIC count),
index heads 32/64, 1–32 requested windows, 256-aligned history/resident with
2048 <= resident < history <= 210944. Existing UB/queue constraints are checked
before launch. Q4, TopK 2048 and the existing element types remain fixed. These
limits preserve the original kernel's storage budget; this is not unrestricted
shape support. Each captured graph still owns its shape-specific buffers and
launch object. Changing shape requires another binding/capture, not another binary.

To check different batch/history/resident/head/window combinations with one
compiled library (the runner prints and checks its SHA256):

```bash
cd /vllm-workspace/KVIP
source scripts/env.sh
python tests/test_lifetch.py --device 5 --kvaf-server 127.0.0.1:28885 --runtime-shapes --miss-percent 10 --warmup 4 --iterations 20 --trace-rounds 3
```

The matrix includes B1/B8/B16, history 65536/131072, resident 8192/16384,
32/64 heads and 4/8 requested windows, using real 656-byte KV records. Each case
retains native/fused byte checks, SFA checks, stateful checks and graph timings.
The runtime conversion changes scalar plumbing and generated code; preserved
formulas alone do not prove unchanged device performance. A5 validation is required.
