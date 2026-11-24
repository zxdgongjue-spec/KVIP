# MR28 A5 prefetch comparison

This runner uses P2 Score and the existing four-AIV-per-request Terminal for all three runs.

| Variant | Service candidate storage | Post size |
| --- | --- | --- |
| `baseline` | Original GM path | Original submission |
| `ub` | UB deduplication and candidate handoff | Original submission |
| `batch` | Original GM path | At most 64 entries per post |

Select BS1, 2, 4, 8, or 16 with repeatable `--batch`. The default is BS16.
All runs use Q4, 32 Index heads, 128K history, 16K resident tokens,
10% synthetic miss, requested W4, pf512, rank threshold 1024, and seed 43.
The per-window cap is 128. Native P2 uses W2 for BS1, so its effective early
cap is 256 and its split variant retains grouped posting. BS2 through BS16
use W4. The split path needs more than 64 admitted tokens in a window.
The runner verifies matching fixture hashes, 32 effective AICs, and each
case's actual P2 and Terminal geometry.

```bash
bash experiments/lifetch_prefetch/run_a5_prefetch.sh --dry-run

# Run inside the A5 container, with the unified KVAF Runtime already listening.
KVIP_A5_DEVICE=0 \
KVIP_KVAF_SERVER=127.0.0.1:28986 \
bash experiments/lifetch_prefetch/run_a5_prefetch.sh --batch 16

# Run the five sizes sequentially in one matrix.
bash experiments/lifetch_prefetch/run_a5_prefetch.sh \
  --batch 1 --batch 2 --batch 4 --batch 8 --batch 16
```

The default KVAF checkout is `$KVIP_A5_ROOT/KVAF-unified-813f1750`. Set `KVIP_KVAF_ROOT` for a different path to that revision. If dependencies are missing, invoke this wrapper with `--prepare-deps` on the host and `--build-deps` in the container. If needed, start `$KVIP_KVAF_ROOT/build-cpu/runtime/kvaf_cpu_service --listen 127.0.0.1 --port 28986` in a separate container shell. Set `KVIP_LIDU_ROOT` if the compatible LIDU extension and custom OPP were built in a different checkout. Use one A5 benchmark at a time.

The matrix gives each size and variant its own worker and reuses the build
within each variant. It preserves checked results if a worker stops without
confirmed completion.

The run writes `manifest.json`, `summary.csv`, and `summary.json` under `$KVIP_AB_OUTPUT/mr28_prefetch.*` along with per-variant worker logs and build flags. The measured graph with detailed stamps disabled supplies `kv_ready_us` as an event mean and `profiler_p50_us` as the profiled fused invocation median. The separate checked diagnostic replay supplies the first software SQ doorbell, early-to-Terminal lead, early and final CQ scopes, and repeated source tokens within early posts, within Final Get, and across the two. The request-local token identities come from the transfer ordinals captured before Terminal retirement and the pre-replay Index. The reported doorbell-to-CQ scope includes overlapped work. CQ join time measures software wait separately. The report reconciles each request/window with actual doorbell counts and identifies how many posted tokens used the UB handoff. It reports which windows posted in chunks, which used grouped posting despite exceeding 64 entries, and whether splitting was exercised. Raw details remain in each variant's `fused/diagnose`. Normalized per-sample rows are in `fused/prefetch_protocol.json`.

`get_comparison` distinguishes identical early/late count vectors, redistribution with unchanged per-request totals, and different per-request totals. `matched_total_kv_ready_ratio` is populated for the first two conditions. `observed_kv_ready_ratio` reports the same-fixture latency including any change in transferred work.

## CANN Service smoke

[Short model results](results/short_service.md) and [CSV](results/short_service.csv).

The short model runs the actual Service code with seeded Q4 compact candidates.
It uses BS2, history 4096, resident 2048, W4, and pf512. Every row starts the SQ
32 entries before wrap. Window 0 must post 88 unique tokens despite duplicates
across Q. Window 1 repeats those tokens and must post zero. The model checks
the actual SQ descriptors, slot and token identity, owner bits, and the single
signaled WQE and pending CQE per nonempty window. Doorbells are modeled by GM
writes. This fixture measures Service from entry and does not run Score,
incremental TopK, real URMA, Final Get, or Terminal.

Run the following inside a disposable Ascend950 CANN 9.1 container with the
checkout and patched KVAF headers mounted. The script edits that container's
simulator model configuration.

```bash
KVAF_INCLUDE=/kvaf KVIP_MODEL_OUTPUT=/output \
bash experiments/lifetch_prefetch/model/run.sh 2 4096 2048 2
```

The CPU protocol and report checks run without Docker.

```bash
python3 -B experiments/lifetch_prefetch/test_prefetch.py
```
