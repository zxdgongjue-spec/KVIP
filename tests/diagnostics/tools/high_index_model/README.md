# Isolated HIGH Index maintenance on Ascend 950

Run the same 16-row Index maintenance workload against each KVIP worktree with CANN 9.2 and the patched KVAF AIV headers. The probe copies the committed kernel sources into the output directory and selects the production Index code. It supplies published event lists and successful Post statuses. The kernel includes the private plan when present, the transport status check, all tuple updates and their joins. No LI, network, CQ or service is involved.

On the target, set `ASCEND_RT_VISIBLE_DEVICES=4`. The executable opens logical device 0. Set `ASCEND_CANN_PACKAGE_PATH` to the CANN installation if needed.

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
bash tests/diagnostics/tools/high_index_model/run_case.sh npu \
    /path/to/source-worktree /path/to/patched-kvaf/access_operator/demo/aiv_batch_get/include \
    /path/to/empty-case-directory
```

- `even` and `skew` each issue 5,000 disjoint swaps across 16 rows, exactly 10,000 changed tuples. In `even` each row has 312 or 313 swaps. In `skew` four rows have 966 swaps each and the others have 94 or 95.
- Optional `INDEX_LAYOUTS="even-mixed skew-mixed"` includes destinations reused across prefetch and final phases. There are 5,440 events and 10,848 changed tuples in each.
- `DEVICE_INDEX_US` is the uninstrumented whole entry measured by ACL Events. It includes any early planning. The first launch warms the kernel, the next six contribute to the mean and median. Inputs are restored outside timing. A separate instrumented launch checks full output tuples, valid bits and per-core move counts against the CPU permutation. Both timed and instrumented outputs are checked for agreement.
- `INDEX_WORK` and `MODEL_EXACT_PASS` show device counter ticks from the separate diagnostic launch. Early planning is reported explicitly. On a simulator, use `sim` in place of `npu` and interpret ticks as model units. Hardware microseconds require the device event measurement on the target.
- The physical NPU uses `history=131328`, `resident=16384`, four queries and five ordered event lists, matching the service replay geometry. A coprime stride of 7,919 spreads each request's synthetic cold sources across the 114,944 nonresident positions. Baseline and candidate see identical ID arrays. The smaller CANN simulator fixture keeps `history=32768` and stride 7 for comparison with earlier model runs.
- On a v2 build, `COLD_PLAN_MODEL` reports the actual per-row fast and fallback choices and their event coverage after comparison with CPU eligibility. The optional `mixed-fallback` layout contains one cold source selected again in a later phase in each of four requests. The general planner resolves its current hot location and invalidates the freed hot slot. The other twelve requests remain eligible for the fast planner. The complete CPU permutation remains the output oracle. Use the real replay for the captured requests and Index/CQ overlap.
