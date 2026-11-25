# A5 LIFetch Service comparisons

## MR25 Terminal paired benchmark

Run `run_mr25_handoff.sh` from the MR !28 checkout in `lifetch-aligned-a5`.

```bash
bash /home/w00568585/li-aligned-a5/KVIP-mr28/experiments/lifetch_mr23_service/run_mr25_handoff.sh
```

The launcher builds two libraries from this checkout using the pinned KVAF and the same CANN toolchain. `main` builds with `KVIP_SERVICE_LOCAL_COUNTS=OFF`. `local_counts` builds with it `ON`. The option defaults to off and applies to the original LI kernels. Both workers require the MR25 Terminal route with P2 and the original Cold rotation. On the candidate, the same lane reads TopK's four compact counts from the existing UB metadata when the P2 lane owns at most two requests. The candidate IDs, cross-window GM publication, local readiness, predecessor ordering, prefetch policy and Terminal stay on the shared path.

Each worker runs BS1/8/16, Q4, H131072, R16384, G32, seed43 and miss10%. Both use the same resident threshold rule, rank start 1024, prefetch budget 512 per request and four requested windows. The runner records the actual window and Service lane counts. It uses two warmups and ten measured graph calls per case. Add `--comparison-dry-run` to print the 6-case plan without device or KVAF setup.

The final `Summary:` points to `summary.csv`. Compare `paired=True` rows. `kv_ready_us` is the graph Event mean through completed CQ and Index. The four same-graph kernel spans are `hot_us`, `threshold_us`, `cold_service_us` and `terminal_us`. Cold includes its rotating Service. The Get counters are token requests per complete batch invocation, split into `prefetch_requests_per_call` and `final_get_requests_per_call`. The runner reads them after the measured graph completes. `both_posted_prefetch` indicates that both variants issued early Get requests. The raw counters by call, request, window and Q are in `*/miss10/case-b*/fused/path_result.json`.

## Historical MR23 three-route benchmark

This experiment is delivered on the MR25 merge tree `609fa26f`. Its launcher stages a separate checkout of `7b4e3d52` and applies `runner.patch` there. The comparison runner exports the MR23 baseline `a56154c3` from Git history. All three measured routes use the MR23 Final and completed Resolve/Get/Index tail.

| Route | Score and Service |
| --- | --- |
| `mr23` | Original P2 history split and rotating Service inside Cold |
| `window_budget` | Uniform Score per window, then a separate Service launch |
| `full_budget` | Uniform full-history Score, then threshold and a Cold T2 kernel with rotating Service |

The last route moves full-history scoring before the threshold. Its measured difference from `window_budget` includes the changed Score order, the Cold score reread and the removal of separate Service launches.

### Run on A5

On the A5 host, fetch the MR and the pinned Score T2 history. Use a detached worktree in the directory shared with the container.

```bash
A5_ROOT=/home/w00568585/li-aligned-a5
REPO="$A5_ROOT/KVIP"
MR_DIR="$A5_ROOT/KVIP-mr28"
git -C "$REPO" fetch origin \
  refs/merge-requests/28/head:refs/remotes/origin/mr-28 \
  refs/heads/codex/lifetch-score-t2:refs/remotes/origin/codex/lifetch-score-t2 &&
if [ -e "$MR_DIR/.git" ]; then
  git -C "$MR_DIR" switch --detach origin/mr-28
else
  git -C "$REPO" worktree add --detach "$MR_DIR" origin/mr-28
fi
```

Enter the container with `docker exec -it lifetch-aligned-a5 bash` and use the A5 bootstrap script. It loads the CANN environment from the KVAF build cache and probes the local KVAF port with Python. It reuses a listening service and starts the pinned binary if the port is closed. Each invocation saves its console log under `/home/w00568585/li-aligned-a5/results/mr28.*`.

```bash
bash /home/w00568585/li-aligned-a5/KVIP-mr28/experiments/lifetch_mr23_service/run_mr28_container.sh
```

#### Direct launcher

Use the existing container's CANN and Python environment with `torch_npu`, the registered LIDU operator, and a running KVAF service. KVAF should use the built `1c1310387ebfb6776a07316874e924265b209ada` checkout.

If the pinned Score T2 object is missing from the local Git history, fetch `refs/heads/codex/lifetch-score-t2` from the KVIP main repository. Run the launcher from the review checkout.

```bash
KVIP_KVAF_ROOT=/path/to/KVAF-score-t2-1c131038 \
KVIP_KVAF_SERVER=127.0.0.1:28985 \
KVIP_A5_DEVICE=0 \
bash experiments/lifetch_mr23_service/run_a5.sh
```

Set `KVIP_PYTHON` to select the container interpreter. If the existing custom OPP location is separate, set `KVIP_LIDU_OPP` to its built vendor directory, for example the directory ending in `_custom_opp/vendors/customize`. Its `op_api/lib/libcust_opapi.so` must exist. The launcher prepends that path to the existing `ASCEND_CUSTOM_OPP_PATH`.

The defaults use three workers for nine cases at BS1/8/16, miss 10%, Q4, H128K, R16K and seed 43. The resident threshold rule, rank start 1024, prefetch budget 512 and four requested windows are shared. Candidate history parts are 16/4/2. Each case has two warmups and ten measured calls. The full cohort requires 32 effective AIC and the associated 64 AIV MIX execution budget.

Set `KVIP_AB_OUTPUT` to relocate the results and isolated checkout. The default is `results/mr23_service` under the invoking directory. Add `--comparison-dry-run` to prepare both source snapshots and enumerate the workers without opening a device or transport.

### Interpret the result

Open `summary.csv` at the printed `Summary:` path. Use accepted rows and `fused_event_us` for complete graph Event time per call through KV completion. `over_mr23` is each route's ratio to the original baseline. `over_window_budget` compares the two new routes directly.

When `pipeline_count_status=PASS`, `prefetch_requests_per_call` and `final_get_requests_per_call` give actual early and final token Get requests per batch invocation. Check for a positive prefetch count in each route to confirm that all three posted early requests. Per-request counts are in `case-b*/fused/path_result.json` under `pipeline_counts.records`.

The full route requires four ordered LI kernels in its measured trace, Score, threshold, Cold T2 and Final. The runner rejects an independent Service launch in that route. `cold_t2_us` covers the combined reread, classification and Service work.
## Hot and Cold Score comparisons on MR28

Run this experiment in the same A5 container as `run_mr25_handoff.sh`. MR28 checkouts that include MR26 use the unified KVAF setup below.

```bash
bash experiments/lifetch_mr23_service/run_mr25_score.sh --batch 1 --batch 2 --batch 4 --batch 8 --batch 16
```

Pass a single `--batch N` to run one case. Pass `--comparison-dry-run` to print and save the plan without an NPU. Pass `--reverse-order` to run the partitioned arm first. Each batch has an independent output and two separate workers. All workers in one command share the same compiled bridge, with `KVIP_SERVICE_LOCAL_COUNTS=ON`.

Both arms use synthetic dense Q4/G32, H128K/R16K, miss 10%, seed 43, requested W4, prefetch budget 512, initial rank 1024, 2 warmups and 10 graph invocations. The partitioned entry uses all available Hot Score groups up to the resident page budget. Scores land in the original canonical GM rows. Threshold, Cold, rotating Service and Terminal use the same launches in both arms.

At effective C32, the five partition counts per request are 32/16/8/4/2. Each MIX task computes all four Q over its assigned resident pages. BS16 uses the original Hot kernel in both arms and serves as the equal-geometry control.

The output includes Hot, Threshold, Cold/Service, Terminal and KV-ready times, plus allocated MIX groups in `summary.csv`. Hot and Cold/Service are profiler kernel spans. KV ready is the per-invocation graph Event mean.

To evaluate Cold alone and then Hot plus Cold with one P2 baseline per batch, run

```bash
bash experiments/lifetch_mr23_service/run_mr25_score.sh \
  --batch 1 --batch 8 --batch 16 --stage cold --stage hot-cold
```

Use `--batch 1` for a short run. Add `--stage hot` to rerun the Hot-only control in the same command. Both Cold candidates keep the two logical half-windows and the original rotation of Service lanes. Each half uses multiple MIX producers with independent progress. Service consumes their contiguous progress and uses the existing thresholds, candidate selection and prefetch budgets. The threshold launch includes candidate progress initialization. The BS16 arms use the original P2 launches at C32.

`paired` requires exact prefetch and Final Get vectors for every measured request and invocation. `get_comparison=same_request_totals` identifies runs where the vectors differ and each request still completes the same total number of Gets. The `observed_*_over_p2` fields retain the observed timing ratios for those runs and the console flags the change in early/final placement. Correctness, fixture, library, route and sample checks apply to both. The default Hot-only invocation keeps the strict exit status. Explicit Cold comparisons return success for a valid same-total observation and keep the strict result in `paired`.

## Unified Score on the latest main

Run one synthetic Q4/G32, H128K/R16K, miss 10%, seed 43 case at each of BS1, 2, 4, 8 and 16. Each case compares three routes in a shared library with local counts OFF and uses the current main Terminal, requested W4, original resident threshold, prefetch budget 512 and rank start 1024.

| Route | Score producer | Window consumer |
| --- | --- | --- |
| `p2` | Native Hot P2 then Cold P2 | Native Cold with rotating Service |
| `full_p2` | Full-history Score P2, direct GM output | Cold T2 reads existing scores after Threshold and runs rotating Service |
| `full_uniform` | Same full-history kernel at P16/P16/P8/P4/P2 on C32 | The same Cold T2 |

### Shared A5 checkout setup

On the host, prepare network dependencies in the existing shared tree. This command selects KVAF `813f1750` and prepares the UMDK checkout used by KVAF.

```bash
bash experiments/lifetch_mr23_service/run_mr28_unified_score.sh --prepare-deps
```

In the A5 container, with the serving Python and CANN environment, build once if this unified KVAF checkout and the current KVIP LIDU are unbuilt.

```bash
bash experiments/lifetch_mr23_service/run_mr28_unified_score.sh --build-deps
```

In the same container, run all five batches.

```bash
bash experiments/lifetch_mr23_service/run_mr28_unified_score.sh
```

Set `KVIP_KVAF_ROOT` in both places to reuse a compatible dependency checkout. The default is `$KVIP_A5_ROOT/KVAF-unified-813f1750`, alongside the earlier SDK checkout. The launcher defaults to a separate CPU service port, `127.0.0.1:28986`. Use `KVIP_KVAF_SERVER` to select an existing compatible service. Use `KVIP_A5_DEVICE` for the container-visible ordinal.

Use `--batch 1` for the three-route smoke case, repeat `--batch` to select multiple sizes, or add `--comparison-dry-run` to check the plan offline. Optional `--prefetch-trace` records first-prefetch doorbell lead time relative to Terminal entry during a separate diagnostic replay.

`score_over_full_p2` compares the two equal-scope full-history Score stages. `observed_kv_ready_over_native_p2` compares complete KV-ready time with native P2. Check `get_vs_native_p2` and `get_vs_full_p2` for equality of Get vectors, equality of per-request totals only, or different totals. Prefetch, final and total Get counts remain visible even when workloads differ. BS16 uses the same unified P2 entry in both full-history arms.

## A5 msprof for the main fused baseline

In the existing A5 container checkout, run one checked replay of main `d6cfd392`'s P2 path.

```bash
bash experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh
```

The default case is BS16, Q4, H128K, R16K, 32 index heads, miss 10%, requested W4, and the existing prefetch threshold with budget 512 and rank start 1024. The launcher accepts `--batch 1|2|4|8|16`, `--miss-percent N`, and optional `--warmup N`. The stock P2 planner reduces BS1 to two effective windows. Use `--dry-run` to see the plan. It uses the same container-visible `KVIP_A5_DEVICE` and unified KVAF service as the unified Score runner above. If the pinned dependencies are already built, start with the capture command. Otherwise use `--prepare-deps` on the networked host and `--build-deps` in the container. Both paths use the existing shared directory.

The worker compiles an isolated diagnostic bridge and finishes KVAF BatchPut, counter calibration, and any warmups before starting CANN profiling. On the original lease stream, it wraps each of the four native launches with CANN device markers and stops profiling after stream completion. Export succeeds only after the kernel tasks and all eight device marker endpoints are verified. The original four launches are Hot Score, resident Threshold, Cold Score with rotating Service, then the current four-AIV-per-request Terminal. The instrumentation follows MR12's source-cycle scopes and leaves the default build flag off. Hot and Cold keep their original MIX partition. Threshold, candidate deduplication, prefetch, final Get, and Index publication retain the baseline ordering.

Under the printed `Run output` directory, open `lifetch_business_trace.json` first for the detailed fused stages. Open `lifetch_msprof_trace.json` for the CANN time domain and native task intervals.

| Output | Contents |
| --- | --- |
| `lifetch_business_trace.json` | Detailed kernel scopes grouped by logical AIC/AIV lane, with window and request/query tags |
| `lifetch_msprof_trace.json` | CANN trace with a `KVIP fused LIFetch` track for its four marker-verified device kernel phases |
| `msprof/PROF_*/mindstudio_profiler_output/msprof_*.json` | Unmodified native CANN export with the original `KVIP_LIFETCH_*_BEGIN/END` markers |
| `worker/miss*/case-b*/baseline_stage_report.json` | Coverage, stage timing summary, Top2048/KV/Index verification and prefetch/final Get counts |
| `worker/miss*/case-b*/baseline_stage_trace.bin` | Original source-cycle records used for the stage timeline |
| `manifest.json` | Revisions, fixture, command, exit status and paths |

The two views describe the same capture and retain their measured clocks. The source view uses a measured device-counter frequency. `manifest.json` records whether CANN reports each task by kernel name or whether it was identified as the unique compute task between that stage’s device markers. Scope spans include existing waits and marker overhead.

### Per-core MTE and VF stages

In `lifetch_business_trace.json`, expand the individual AIC and AIV lanes and
the `Observed source lane counts` counters. Device events carry the logical
lane, subblock and raw AI Core group ID. The stage report's `stages` list
also contains the distinct AIC/AIV lanes per stage and window. Group IDs may
be shared by AIC/AIV subcores.

The finer source scopes annotate AIC MTE2 loads, MTE1 staging, Cube and FIXP,
AIV MTE2 input loads, VF/vector scoring, MTE3 writeback, and waits. The
window and Terminal entries include TopK selection, candidate/descriptor
handling, final Resolve, KV transfer and persistent Index publication.
Hot/Cold tile details are sampled at the first and last tile per mapped
shard/window. Submit intervals and existing fence/wait intervals retain
their source semantics. Source lane counts measure overlap of those
observed scopes and include observer overhead.

### Optional on-board instruction pipeline

Run the same command with `--instructions` for a separate native capture.
It uses single-device application replay, with a fresh KVAF lease and one
checked uninstrumented forward on each application pass. The vendor
collector's operator filter targets the original four LIFetch kernels.

```bash
bash experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh --instructions
```

For a smaller instruction capture, use
`--instructions --instruction-stage cold-service --instruction-pipes 'mte2|mte3'`.
The default instruction pipes are MTE1, MTE2 and MTE3. An installation
that advertises both `InstrTimeline` and application replay is required.
Set `KVIP_MSOPPROF` to the path of a compatible standalone msopprof if it
is installed. The CANN 9.1 container used for compilation did not advertise
`InstrTimeline` in its bundled CLI.

Open the listed `instruction/raw/OPPROF_*/**/visualize_data.bin` files in
MindStudio Insight. `instruction/instruction_status.json` records the
requested filters, replay report paths and native artifacts found. Verify
the instruction tracks in Insight and correlate by kernel name and source
location. Hardware instruction sampling covers a subset of cores. The
generic InstrTimeline route gives no VF-body completeness guarantee. Use
the per-core source trace for VF launch and dependency boundaries.

### BS8 P2 versus Cold P4 phase comparison

Run on A5 with the existing unified KVAF dependency.

```bash
bash experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh --cold-compare
```

This selects BS8, Q4/G32, H128K/R16K, miss 10%, W4 and seed 43. Both arms
retain the resident threshold, rank start 1024, prefetch budget 512,
rotating Service, and four-AIV-per-request Terminal. The P2 arm uses the
stock launch schedule. The P4 arm subdivides only Cold. Both use one
diagnostic bridge, with the UB candidate and split-prefetch-post
experiments disabled. The default is `--local-counts on` for the same
experimental setting as the historical 186.97/205.55 us comparison.
Pass `--local-counts off` to use main's default in both arms.

Open the `p2/` and `cold_p4/` business and native timelines in the
printed output directory. The Cold P4 trace exposes `WaitPartitionedHalf`
before rotating Service. It brackets the leader's publication fence and
polls on helper cursors. The separate `Service predecessor wait` scope
waits for the previous window's Service publication.

`cold_trace_comparison.json` includes the native Cold+Service wall span,
the historical reference, and actual Get vectors compared by request.
`cold_trace_scopes.csv` contains both arms and signed deltas for each scope
and window. Use `max_lane_union_us` for the lane with the most time inside
one scope and `source_union_us` for that scope's interval union across
lanes. The report also includes score MTE3 writeback and output-bank wait
samples, per-Q incremental TopK, candidate deduplication, and WQE/SQ
submission. Score MTE3 samples cover first and last tiles and record
source command issue and existing fences.

The paired trace captures one instrumented eager replay per arm. Compare
Get placement and totals alongside the native wall delta. Per-scope
intervals can overlap, and their deltas cannot be summed into the
historical 18.58 us difference from the uninstrumented graph run.

### BS8 unified Score end-to-end profiling

In the A5 container, use the existing built unified KVAF dependencies.

```bash
bash experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh --unified-compare
```

This runs one checked, marked fused replay for each of native P2, full
Score P2, and full Score P4. All use BS8, Q4/G32, H128K/R16K, miss 10%,
seed 43, W4, rank 1024, and prefetch budget 512. Local counts defaults
to OFF. Pass `--dry-run` to inspect the plan without CANN or KVAF.

- `unified_trace_native.json` puts the four native msprof tasks from
  each route on parallel lanes using Score start as each lane's origin.
- `unified_trace_stages.csv` compares those task durations and gaps.
  `unified_trace_comparison.json` adds Score-to-Terminal device spans,
  per-window Get counts and comparisons, and first-prefetch doorbell
  timing from the source counter.
- `unified_trace_scopes.csv` reports per-window and per-stage source
  intervals, including union time across cores and maximum lane union.
  For each route, `lifetch_business_trace.json` contains the individual
  AIC/AIV tracks and `lifetch_msprof_trace.json` contains the CANN tracks.

Cold T2 records first and last tile samples for bank reuse, the GM score
read with its existing fence, threshold classification, and predicate/
progress publication. The shared Service records predecessor wait,
incremental TopK, deduplication, WQE submission and actual batched
doorbells. Terminal retains its Top2048, Resolve, final Get/CQ, Index,
prefetch CQ and completion scopes. Source scope unions can overlap, and
the relative source and native clocks are independent. Use
`run_mr28_unified_score.sh --batch 8` for an uninstrumented graph Event
mean with the same synthetic case.

### BS16 Cold Score and independent window Service

Run the native P2 and dedicated Service comparison in the A5 container.
Both use the same bridge and BS16, Q4/G32, H128K/R16K, W4, original
prefetch budget 512, and rank start 1024. The default miss rate is 10%.

```bash
bash experiments/lifetch_mr23_service/run_mr28_dedicated_service.sh
```

For a quick graph check of the candidate on its own, use
`--route dedicated --iterations 1 --warmup 0`. The worker reports
pre-capture eager, capture, warmup and measured replay separately. If
synchronization fails, see `graph_failure.log` in the case's fused directory.
For a failure during the first eager forward, add `--staged-eager` to log
completion of each of the original four launches on the same stream.
See `eager_failure.log` in the fused directory. If the stages and eager
correctness check pass, the runner continues to the usual graph measurement.
Use ordinary dispatch for performance comparison, since the extra host
fences alter scheduling.
If the first incomplete stage is Cold, repeat the same diagnostic with
`--route dedicated --legacy-credit-tail`. This build changes only the two
AIV1 credit waits at Cold exit. It keeps the W0 readiness fix and prefetch
configuration and is a diagnostic control.

Use `--dry-run` to inspect the plan. Run `--prepare-deps` on the host
if the existing unified KVAF sources are absent, and `--build-deps`
in the container if that checkout is unbuilt. All runs use the existing
shared checkout.

The candidate retains four same-stream launches. Hot and Terminal use the
native kernels. Threshold also clears the candidate's private readiness
ledger on every replay. A single Cold launch contains 24 Score MIX groups
and 16 dedicated Service AIVs. The remaining eight AICs in the launch do
not participate in the producer credit protocol.

Score groups come in sets of three for each pair of requests. A
request-local group scores the lower half and the beginning of the upper
half. The shared group scores the disjoint upper tails of both requests.
In W1 and W2 every group scans 200 Q4 tiles, and in W3 each pair scans
198, 198, and 196 tiles. The logical half-window and the original
canonical Score rows stay intact.

Cube results flow directly to the two paired Score AIVs in UB. Remote
Service AIVs consume the reduced Score and mask from GM after readiness
publication. Readiness has one independent cursor per query, window
and fragment. Service can advance only through contiguous ready ranges,
so a faster upper tail cannot make an unfinished prefix readable. Each
request keeps a single Service owner for incremental TopK, Q4 dedup,
prefetch budget, and SQ updates. W0 uses the native empty Cold cursor
while Cold scoring starts. In W1–W3, Service waits for the corresponding native
rotating driver's logical half window to be visible in GM, then consumes
the target request's current ready ranges. The gate preserves the logical
trigger, and publication timing can change the prefetch and final Get
split. Terminal joins early and final Get CQ.

The longest producer receives 598 tiles versus 450 in native P2. This
is a workload count. Any end-to-end improvement depends on removing
enough exposed credit wait and overlapping useful Service. Compare
Score, Threshold, Cold, Terminal, graph Event, and per-request prefetch
and final Get counts before assigning a latency change to the schedule.
The run writes `summary.csv` and `requests.csv`. The optional
`--prefetch-trace` adds a separate replay with prefetch submission
timestamps and their lead over Terminal.

### BS16 standalone msprof replay

In the A5 container, collect each route separately with the same fixture
and local counts disabled. Each invocation warms up twice and captures
one marked fused forward. The profiling scope starts after fixture
initialization and BatchPut.

```bash
runner=experiments/lifetch_mr23_service
for route in p2 dedicated; do
    bash "$runner/run_mr28_baseline_msprof.sh" \
        --route "$route" --batch 16 --miss-percent 10 \
        --warmup 2 --timeout-seconds 1800 || break
done
```

Each output directory contains `lifetch_msprof_trace.json` for the native
task spans and `lifetch_business_trace.json` for the per-core business
and MTE/VF source scopes. The Dedicated trace separates the driver wait
and ready cursor snapshot from the predecessor wait, per-Q TopK,
deduplication, and prefetch submission. Compare the same clock domain
across routes and use the separate graph run for KV-ready latency.
