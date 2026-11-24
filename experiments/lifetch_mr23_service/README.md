# MR23 A5 Service comparison

This experiment is delivered on the MR25 merge tree `609fa26f`. Its launcher stages a separate checkout of `7b4e3d52` and applies `runner.patch` there. The comparison runner exports the MR23 baseline `a56154c3` from Git history. All three measured routes use the MR23 Final and completed Resolve/Get/Index tail.

| Route | Score and Service |
| --- | --- |
| `mr23` | Original P2 history split and rotating Service inside Cold |
| `window_budget` | Uniform Score per window, then a separate Service launch |
| `full_budget` | Uniform full-history Score, then threshold and a Cold T2 kernel with rotating Service |

The last route moves full-history scoring before the threshold. Its measured difference from `window_budget` includes the changed Score order, the Cold score reread and the removal of separate Service launches.

## Run on A5

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

### Direct launcher

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

## Interpret the result

Open `summary.csv` at the printed `Summary:` path. Use accepted rows and `fused_event_us` for complete graph Event time per call through KV completion. `over_mr23` is each route's ratio to the original baseline. `over_window_budget` compares the two new routes directly.

When `pipeline_count_status=PASS`, `prefetch_requests_per_call` and `final_get_requests_per_call` give actual early and final token Get requests per batch invocation. Check for a positive prefetch count in each route to confirm that all three posted early requests. Per-request counts are in `case-b*/fused/path_result.json` under `pipeline_counts.records`.

The full route requires four ordered LI kernels in its measured trace, Score, threshold, Cold T2 and Final. The runner rejects an independent Service launch in that route. `cold_t2_us` covers the combined reread, classification and Service work.
