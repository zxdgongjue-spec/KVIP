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
