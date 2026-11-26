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
| `lifetch_handoff_report.json` | Compact source handoff anchors, body segments and observed scopes, plus native same-stream gaps |
| `msprof/PROF_*/mindstudio_profiler_output/msprof_*.json` | Unmodified native CANN export with the original `KVIP_LIFETCH_*_BEGIN/END` markers |
| `worker/miss*/case-b*/baseline_stage_report.json` | Coverage, stage timing summary, Top2048/KV/Index verification and prefetch/final Get counts |
| `worker/miss*/case-b*/baseline_stage_trace.bin` | Original source-cycle records used for the stage timeline |
| `manifest.json` | Revisions, fixture, command, exit status and paths |

The two views describe the same capture and retain their measured clocks. The source view uses a measured device-counter frequency. `manifest.json` records whether CANN reports each task by kernel name or whether it was identified as the unique compute task between that stage’s device markers. Scope spans include existing waits and marker overhead.

For Hot to Resident and Resident to Cold, open the corresponding source
`breakdown` tracks. These split the business interval at the last recorded
previous kernel body end and the first following body start. Read the
Score output-bank drain, cutoff and window state scopes on the AIV lanes.
Use the native handoff track to measure the same-stream device task gap.
The small handoff report contains both views, so it can be shared without
exporting the full trace. Refresh the source and capture again to include
the new device scopes.

Open `Handoff activity and dependencies | source clock` to see which
recorded scope and logical lane cover each part of a handoff. Each slice
shows one representative scope and lists the observed lanes. The
dependency track names the device-visible state and the stage ordering
gate. The dynamic route uses request-local readiness between Resident
and Cold within the same device kernel.

On the native view, `Handoff device task activity` splits each same-stream
gap by timed profiling marks, other timed device tasks, overlap, and
time with no timed task record. A mark without duration remains a
boundary timestamp. Use the separate source and native clocks.

Open the logical AIC and AIV lanes to see `Handoff gap` spans during those
same intervals. Each span marks time without a recorded LIFetch source
scope on that lane, after excluding kernel bodies and other recorded work
such as cursor resets. Device-side utilization requires separate hardware
evidence. To get the lane view from an existing capture, re-render its
`baseline_stage_trace.bin` with
`tests/operators/lifetch/support/baseline_stage_trace.py`, passing the
recorded `counter_mhz` and route. That rendering runs offline.
Both offline decoders accept `--handoff-report` for sharing only the
handoff evidence. The native decoder can produce this report without
writing a second copy of the full msprof JSON.

To redraw an existing capture from the container checkout:

```bash
bash experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh --redraw-latest
```

The script selects the latest complete marked capture from the shared
results root and prints the new `handoff-redraw.*` directory. It replays
each arm of a completed comparison, and `--route p2` can select a specific
route. The source activity and compact device activity timelines are
written alongside the two small handoff reports. This operation uses
saved artifacts and runs without starting a worker, KVAF or a new capture.

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

### BS16 dynamic Score/Service experiment

Refresh the MR28 checkout on the host. The existing shared checkout and KVAF
build are then available inside the A5 container. Run the two commands in the
container with its CANN/KVAF environment. Set `KVIP_A5_DEVICE` to the device
ordinal visible inside that container. The runner defaults to ordinal 0.

```bash
cd /home/w00568585/li-aligned-a5/KVIP-mr28
runner=experiments/lifetch_mr23_service
bash "$runner/run_mr28_dynamic_service.sh" --warmup 2 --iterations 10
bash "$runner/run_mr28_baseline_msprof.sh" --dynamic-compare --warmup 2 --timeout-seconds 1800
```

Both use one fixture (BS16, Q4, H128K, resident 16K, miss 10%, W4,
32 AIC/64 AIV and prefetch budget 512 per request). Local candidate-count
optimization stays off in both arms. `static_ready` uses fixed Dedicated Score
and Service workers. `dynamic` executes each Resident Q once, then assigns
Cold Score fragments to free MIX groups and Service windows to free AIVs.
Both arms wait for the same request's Resident result, current scores and
prior window publication. The existing prefetch and Terminal paths run in
both arms.

The performance runner writes `summary.json`, `summary.csv` and
`requests.csv`. The graph Event covers Hot through Terminal, including
completion. The dynamic kernel overlaps Resident, Cold and Service, so its
profiler measurement covers that whole kernel. The fixed control reports
Resident and Cold/Service separately and also their sum. A matched latency
ratio requires identical per-request prefetch and final Get vectors. Raw
observations and vector differences remain in the report.

The profiling runner captures one checked forward per arm after fixture
setup and BatchPut. Open `dynamic_trace_native.json` to compare native
device intervals. Open each arm's `lifetch_business_trace.json` to inspect
logical AIC/AIV lanes, dynamically assigned fragments and Service windows,
and sampled MTE/VF scopes. `dynamic_trace_comparison.json` reports the first
W1 prefetch doorbell, native elapsed spans and per-window Get counts.
`dynamic_trace_scopes.csv` summarizes the instrumented scopes. Native task
timestamps and source-scope timestamps use separate clock domains.

## MR28 P2 and early W1 Score on A5

The opt-in route keeps the original rotating P2 Service and Parallel
Terminal. While the other AIVs compute resident thresholds, the last
`min(4, liCores / 2)` MIX groups score the first two tiles of their W1
halves. Their AIVs then compute their own resident thresholds. The
original Cold producer AIVs read back and classify the scored tiles
before continuing with the native suffix. BS16 still uses MIX 28–31
for B14 and B15.

Fetch the MR on the **host** and use the existing shared checkout.
Prepare KVAF sources here only if the pinned checkout is missing.

```bash
cd /home/w00568585/li-aligned-a5/KVIP-mr28
git status --short
git fetch origin refs/merge-requests/28/head && git switch --detach FETCH_HEAD
git rev-parse --short HEAD
# If the pinned KVAF checkout is missing
# bash experiments/lifetch_mr23_service/run_mr28_p2_w1.sh --prepare-deps
docker exec -it -w "$PWD" lifetch-aligned-a5 bash
```

In the **container**, use the logical NPU ID visible to the container.
Build dependencies only if the existing KVAF or LIDU artifacts are missing.
The paired runner builds the candidate bridge with the experiment enabled.

```bash
runner=experiments/lifetch_mr23_service/run_mr28_p2_w1.sh
export KVIP_A5_DEVICE=0
# bash "$runner" --build-deps
batches=(--batch 1 --batch 2 --batch 4 --batch 8 --batch 16)
bash "$runner" "${batches[@]}" --route p2_w1 --staged-eager --warmup 0 --iterations 1
bash "$runner" "${batches[@]}" --route both --warmup 2 --iterations 10
```

Omitting `--batch` runs BS16. The comparison uses Q4, history 128K,
resident 16K, miss 10%, seed 43, and a prefetch budget of 512 per request.
Both arms request four windows and retain the 128-token per-window
budget. Native P2 uses two effective windows on BS1 and four at all
other supported batch sizes.

Each measured replay verifies KV and Index and checks the actual device
kernels. Read `summary.csv` for Hot, Threshold, Cold+Service, Terminal
and KV-ready latency. A sweep compiles the bridge once, writes a combined
`summary.csv`, and keeps the per-route logs and `requests.csv` in each
`bs<N>` directory. The BS1 request file leaves W2 and W3 blank. Each
request records prefetch by window and final Get by query. The runner
reports whether request-level counts match between routes. A worker error
stops the sweep and preserves completed cases. A separate
`--prefetch-trace` run collects prefetch diagnostics.
When total Get counts differ, the ratio is labeled as a raw observation
and the CSV sets `timing_eligible` to false.

In the H7168/resident 4096 CANN model, the candidate shortened Hot through
Cold+Service from 238,125 to 229,244 device ticks. Prefetch declined from
2,612 to 2,368 WQEs and the first W1 doorbell moved 16,266 ticks later.
This A5 runner measures the remaining Terminal and KV-ready effects.

## MR28 Cold Score via GM

Run on A5 from the shared checkout with the existing KVAF build. The
comparison uses the same synthetic Q4, H128K, resident 16K, miss 10%,
seed 43 and original prefetch budget. Start with BS16.

```bash
export KVIP_A5_DEVICE=0
runner=experiments/lifetch_mr23_service/run_mr28_cold_gm.sh
bash "$runner" --batch 16 --route cold_gm_pipeline --staged-eager --warmup 0 --iterations 1
bash "$runner" --batch 16 --route all --warmup 2 --iterations 10
bash "$runner" --batch 16 --route all --warmup 1 --iterations 3 --prefetch-trace
```

- `p2` uses the native route with rotating Service.
- `cold_gm_fenced` starts the Cold raw Score and resident Top2048 after
  the Hot Score launch. In one MIX kernel the first cohort scores first,
  and the other cohort computes resident Top2048 first. The existing
  Cold T2 consumer starts after the full-batch producer kernel finishes.
- `cold_gm_pipeline` uses the same two producer cohorts. After the full
  raw Score and resident thresholds for a request are visible, its AIVs
  can load GM Score tiles and classify them while other requests are
  still producing scores. The native rotating Service uses the
  classified progress and window order. W0 only needs the resident
  thresholds for the requests assigned to that service lane. Terminal
  still follows the producer and consumer kernel on the transport stream.

Both candidates reuse the canonical Score arena and append one
128-byte readiness line per query. Each new replay begins with the
Hot kernel clearing those lines. The same kernel and stream order
provide the readiness reset and the final Terminal boundary. At BS16,
128K and Q4, the producer covers the complete Cold range
`[16384, 131072)`, including W1, W2 and W3. The two MIX groups for each
request split that range into equal halves of 448 tiles. The batch
writes 14,336 Cold tiles. The consumer performs an additional 14 MiB
of logical raw Score reads for classification before the original
window ranking uses the published Score and predicates.

The paired report includes KV/Index validation, each stage's profiler
mean, KV-ready graph timing and per-request Get counts. The separate
diagnostic replay writes AIV phase stamps, envelope overlaps,
prefetch doorbells and Terminal CQ timing to `detail_samples.json`.
The overlap fields describe phase intervals and can include waits.
`timing_eligible` specifies whether the per-request total Get workload
is comparable. The pipeline row also reports the comparison against
`cold_gm_fenced`.

### BS16 Cold Score with native P2 window visibility

This two-arm benchmark fixes BS16, Q4, history 131072, resident 16384,
miss 10%, four windows, seed 43 and the original prefetch policy.

```bash
export KVIP_A5_DEVICE=0
bash experiments/lifetch_mr23_service/run_mr28_cold_frontier.sh \
    --warmup 2 --iterations 10
```

The benchmark first runs native P2 with visibility capture enabled. Each
measured invocation records the two classified cursors actually observed
by Service for every request, Q and window. The candidate builds the full
raw Cold Score in GM and completes resident Top2048. In the original
Cold T2 scan and Service order, each replayed window waits for classification
through the saved frontier clipped to its own window cap. Ranking and next-Q
preloading then use those recorded values. The candidate still selects
candidates and submits Gets through the original Service.

Both measured arms also retain the actual source tokens before Terminal
retires them. `cold_frontier.json` records the evidence for each invocation.
`strict_comparison.json` reports any difference in the device-applied
frontiers, each request's per-window prefetch and final Get token
multisets, or per-Q final Get counts. The script exits nonzero on a
mismatch and preserves the raw KV-ready result in `summary.csv`.
The graph Event includes the experiment's capture and replay overhead.

## Small-batch single-change A5 comparison

Use the native P2 route from main d6cfd392 as the reference inside the same
experimental build. Each arm runs BS1 or BS8, Q4, history 131072, resident
16384, miss 10%, G32, seed 43, rank start 1024, and 32 AIC / 64 AIV.
Each keeps the original threshold-triggered prefetch with total budget 512
and window cap 128. The four experiments have exclusive switches.

| Arm | Scope | BS1 | BS8 |
| --- | --- | --- | --- |
| p2 | Native Hot, Threshold, Cold with rotating Service, and Terminal | W2 | W4 |
| cold | Subdivide each native Cold half into two, publish classified progress, and let the existing Service observe contiguous ready prefixes | P4 / W2 | P4 / W4 |
| windows | Native Score and Service, with one fewer window and the same Service AIV count | P2 already uses W2 | P2 / W3 |
| topk | Shard resident high-byte histogram across two AIVs and use the same per-Q cutoff, gather and output logic | T2 / W2 | T2 / W4 |
| get | Rebalance existing unique Final Get descriptors across four exclusive queues per request after placement, when it reduces the peak | W2 | W4 |

BS8 W3 keeps eight Service AIV per window and changes the token caps to
16384, 73728 and 131072. Its per-request sum of window budgets is at most
384. BS1 W1 would leave the P2 sequence-split route. Cold P4 adds four
producer AIVs on BS1 and 32 on BS8 relative to P2. P4 keeps the native
Service order and uses live classified prefixes. Its speculative fetch
workload can vary with scoreboard progress. Resident T2 adds one helper
launch and keeps the rolling and final Top2048 paths. Final Get leaves
prefetched entries and assigned destination slots in place.

In the existing shared worktree on the networked host, prepare dependencies
only if needed. Inside the container, build them once using the matching CANN
and Torch environments.

```bash
# Host, only for dependency preparation
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh --prepare-deps

# Container, only when dependencies are unbuilt
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh --build-deps
```

Run graph Event timings and checked trace captures separately.

```bash
export KVIP_A5_DEVICE=0
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh --warmup 1 --iterations 5
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh --trace --warmup 0
```

Either invocation accepts repeatable `--batch 1|8` and
`--variant cold|windows|topk|get|p2`. P2 is included for each selected
batch. Omitting both selectors runs nine arms. To run the first priority
only, pass `--variant cold --variant windows`. Reuse the printed output
folders; the timing and trace commands run workers sequentially. Each
invocation stops at the first failed arm and retains completed reports.
Use `--dry-run` to validate the matrix without A5 dependencies.

- Graph Event timings use normal graph replay and retain each arm's
  checked phase times and per-call/request Get counters. The summary has
  raw ratios and a separate ratio for equal per-request prefetch and
  Final Get counts. Counts alone do not establish identical token IDs.
- Marked traces include diagnostic capture overhead and each arm writes
  lifetch_msprof_trace.json, lifetch_business_trace.json, lifetch_handoff_report.json
  and its checked stage report. The extra Resident histogram is labeled
  in the native timeline. The separate pre-Cold resident replay compares
  scores, ordered IDs and the thresholds the Cold consumer reads.
- Trace summaries compare per-request/window prefetch identities, final
  Get identities and actual per-Q queue loads. Logical assignments are
  captured before Index retirement. The diagnostic checks queue ordinal
  coverage and validates Top2048, KV and Index after device completion.
  Compare WQE generation and Final CQ on the critical path together with
  the number of requests whose peak per-queue load actually fell.

### Follow-up pairs and offline attribution

Use the same A5 fixture and prefetch configuration as above. Select the two
short comparisons with one bridge build per run.

```bash
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh \
  --follow-up --warmup 1 --iterations 5 --repeats 2
```

Each round schedules BS1 P2/W2 and Cold P4/W2, followed by BS8 P2/W4 and
P2/W3. Each arm uses an independent worker. The bridge is shared across
rounds. The summary compares each candidate with the P2 row of its own BS and
round and records `raw_kv_ready_delta_us` and the request-count comparison.
Use `--batch 1` or `--batch 8` to select one pair. Add `--trace --warmup 0`
to capture one checked replay per arm, with one round by default.

The existing full trace run also contains the evidence for the second and
third optimization points. Analyze it without a new replay.

```bash
python3 -B experiments/lifetch_mr23_service/analyze_small_batch_trace.py \
  /home/w00568585/li-aligned-a5/results/mr28.2qKjVo --format csv
```

Use a single trace directory or a `repeat-NN` subdirectory as input. The
default JSON includes per-window Top2048 and per-request Final Get
details. The CSV gives the Resident, window and Final Top2048 source
scopes, and the Final Get queue and CQ summary. Compare the maximal
per-lane scope union and event coverage. Interpret these as measured
source scopes with possible overlap.

### Direction 1 batch and miss matrix

The matrix compares the main native P2 route in the shared bridge with one
optimization per batch. Both arms use the same fixture and threshold-driven
prefetch. Keep the 128K history, 16K resident, Q4/G32, seed 43, total
prefetch budget 512, per-window cap 128 and rank start 1024.

| Batch | P2 baseline | Candidate | Service AIV per window |
| --- | --- | --- | ---: |
| 1 | P2/W2 | Cold P4/W2 | 2 |
| 2 | P2/W4 | Cold P4/W4 | 2 |
| 4 | P2/W4 | Cold P4/W4 | 4 |
| 8 | P2/W4 | P2/W3 | 8 |
| 16 | P2/W4 | P2/W3 | 16 |
| 32 grouped | 2× BS16 P2/W4 | 2× BS16 P2/W3 | 16 per wave |

Supported native sizes run against requested miss rates 5%, 10%, 15%,
20%, 25% and 30%. The runner labels the six BS32 cells as
UNSUPPORTED_NATIVE by default. Add --group-bs32 for a separately
labeled comparison. This captures two independent BS16 states as serial
calls in one device graph, and reports its full Event time in kv_ready_us.
The per-call stage/Get statistics and native_subcall_kv_ready_us are per
BS16 wave. The grouped Get totals are in the _per_group columns. Grouped rows
are also written to bs32-grouped/summary.csv and bs32-grouped/summary.json
under the run directory. Grouped BS32 uses two captured calls and reuses
the same synthetic content pattern across waves.

```bash
# Preview all 36 scenario pairs, including grouped BS32.
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh \
  --direction1-matrix --group-bs32 --dry-run

# Run all 36 pairs in the A5 container.
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh \
  --direction1-matrix --group-bs32 --warmup 1 --iterations 5

# Start with one pair at a new native shape.
bash experiments/lifetch_mr23_service/run_small_batch_a5.sh \
  --direction1-matrix --batch 2 --miss-percent 10 --warmup 1 --iterations 5
```

Use repeatable --batch and --miss-percent to select subsets. Each arm
gets its own worker, with one bridge build per timing run. Summary rows
pair by repeat, batch and requested miss and include the checked route,
fixture hash, actual fixture miss per Q, KV ready and prefetch/final/total
Get. The requested rate controls residency of the four-query selected
union. The exact per-Q miss varies with Q overlap. The raw ratio includes
prefetch redistribution and differences in total Get. The matched-count
ratio requires equal early and final request counts for every request and
invocation.

### BS1 and BS2 Cold Score rerun

On A5, run the paired timing and msprof experiment from the same checkout.

```bash
export KVIP_A5_DEVICE=0
bash experiments/lifetch_mr23_service/run_small_batch_cold_a5.sh
```

The script runs two independent graph timing rounds (one warmup and five
invocations per graph), followed by one checked msprof replay per arm. Each
mode builds once and prints its own output directory. The timings use
`summary.csv`; open each `miss10/bs{1,2}/{p2,cold}/lifetch_msprof_trace.json` and
`lifetch_business_trace.json` in the trace output directory for attribution.
Add `--dry-run` to print both plans.

Both arms keep each batch's native window geometry and original prefetch.
BS1 uses W2 and moves from 2 to 4 Cold MIX groups. BS2 uses W4, two Service
AIV per window, and moves from 4 to 8 Cold MIX groups. The fixture is Q4,
history 131072, resident 16384, miss 10%, with a total prefetch budget of 512
and a per-window cap of 128. Compare each P4 row with its same-batch P2
row in the same round, including the per-request Get count comparison.

Pass the trace run's `miss10` directory to the offline analyzer.

```bash
python3 -B experiments/lifetch_mr23_service/analyze_small_batch_trace.py \
  "$TRACE_OUTPUT/miss10" --format csv
```

Set `TRACE_OUTPUT` to the second `Run output` directory printed by the script.

### Full 32 Cold MIX groups and balanced windows on A5

The paired P2 baseline uses main `df8ad259` with the same experimental
bridge build as the candidate. Each full-width arm uses 32 Cold MIX
groups (32 AIC and 64 AIV): P32 for BS1, P16 for BS2, and P4 for BS8.
The default run compares six arms at native W2/W4/W4. `--balanced`
compares fourteen arms per miss and repeat, with P2 and full width at
each legal window count. BS1 supports W2, and BS2/BS8 support W2–W4.

In the balanced run, `g = ceil(BS/2)` groups supply `2g` Service AIVs per
window. The window plan satisfies `W*g <= 2*BS <= 32` because the current
rotation keeps two logical leader groups per request. The 512 total
and 128 per-window prefetch limits stay fixed.

```bash
export KVIP_A5_DEVICE=0
runner=experiments/lifetch_mr23_service/run_full32_a5.sh
bash "$runner" --balanced --dry-run
bash "$runner" --balanced --warmup 2 --iterations 5
bash "$runner" --balanced --batch 2 --trace
```

Read `summary.csv` for checked graph Event latency, phase times, and
early/final Get counts. `raw_kv_ready_delta_us` uses native-window P2;
`same_window_raw_kv_ready_delta_us` uses P2 at the candidate's window
count. A trace run validates the actual Cold MIX counts independently.

### Cold part and window matrix with native Hot

`run_full32_a5.sh --cold-window-matrix` uses Q4, 128K history, 16K resident,
and miss 10% by default. Hot keeps two MIX groups per request, each
scoring all four query rows over its history slice. The two AIVs in each
group reduce two query rows each. Only the size of each Cold history slice and
the window layout vary. At each BS, every arm uses the same fixture,
binary, and Service AIV count per window.

| Batch | Effective windows | Cold parts per request |
| --- | --- | --- |
| 1 | W2 | P2, P4, P8, P32 |
| 2 | W2 and W4 | P2, P4, P16 |
| 8 | W2 and W4 | P2, P4 |

W includes resident W0. BS1 has two static Service owner groups, so its
W4 cell is reported as unsupported. Both window policies keep the 128
per-window and 512 per-request prefetch caps. W2 therefore offers at most
256 early Get slots per request, and W4 offers up to 512.

```bash
runner=experiments/lifetch_mr23_service/run_full32_a5.sh
bash "$runner" --cold-window-matrix --dry-run
bash "$runner" --cold-window-matrix --warmup 2 --iterations 5
bash "$runner" --cold-window-matrix --batch 1 --trace
```

In `summary.csv`, `same_window_*` compares Cold partitioning against P2
with the same effective W. `same_parts_w4_*` compares W2 against W4 at the
same P. Read these deltas with the early and final Get counts. The trace
run records resident seed hashes and WQE token sets alongside the
native kernel and business timelines.

For the two BS2 comparisons from the matrix, use the scoped msprof runner:

```bash
bash "$runner" --bs2-trace-pair both
# P2/W2 vs P4/W2
bash "$runner" --bs2-trace-pair parts
# P4/W4 vs P4/W2
bash "$runner" --bs2-trace-pair windows
```

`both` captures P4/W2 once. This mode enables trace capture automatically
and uses miss 10% and one checked replay per arm by default. Under the
printed run directory, `miss10/bs2/windows_w2` is P2/W2,
`miss10/bs2/cold_w2` is P4/W2, and `miss10/bs2/cold` is P4/W4.
Open `lifetch_business_trace.json` for per-core scopes and
`lifetch_msprof_trace.json` for the CANN timeline in each arm directory.
In the root `summary.csv`, the P4/W2 row reports marked stage deltas
against P2/W2 under `same_window_*` and against P4/W4 under
`same_parts_w4_*`. These are traced kernel measurements.

### BS2 P4/W4 with separate Service AIVs

Run both arms with the same Hot P2, eight Cold P4 Score groups, W4
and prefetch settings. The separate arm launches four extra MIX groups.
Their AICs return immediately. Eight extra AIVs each own one original
`(window, lane)` job. After the original driver's two Score cursors
reach its fragment end, the AIV runs the existing rotating Service with
the same logical owner, predecessor and prefetch policy. W0 starts from
the resident publication. Score producers continue to the next window.
The separate arm fences MTE3 before each fragment's final cursor write.
Candidate scoring can advance sooner, so actual prefetch counts may differ.

```bash
runner=experiments/lifetch_mr23_service/run_bs2_p4_service_a5.sh
bash "$runner" --dry-run
bash "$runner" --warmup 2 --iterations 5
# Optional separate invocations
bash "$runner" --perf-only --warmup 2 --iterations 5
bash "$runner" --trace-only
```

The default run prints separate output directories for the graph and
the checked msprof replays. In the graph summary, compare
`kv_ready_us` and the early/final Get counts. In the trace summary,
compare the sampled AIC credit wait, the Service predecessor wait and
the wait for the original Score driver. The two traced arms sample the
first three tiles and the final tile of each Cold fragment. Each wait
metric includes the largest union of matching spans on one logical lane
and the longest individual span. Per-window figures are in
`bs2_service_waits.json` next to each timeline.
