# Four-window production LI chip-model replay

On an AArch64 Linux machine with CANN 9.2, build the production LIFetch bridge for the Ascend950DT_9582 target. Use the same toolkit, KVAF headers and compiler settings for the reference and candidate. The launcher selects the CANN DAV3510 PEM CPU simulator. Each run requires a new output directory and preserves its detailed instruction trace and library hashes.

```bash
testdir=tests/operators/lifetch/model_window
bash "$testdir/run.sh" "$REFERENCE_BRIDGE" "$KVIP_SOURCE" "$KVAF_INCLUDE" "$RESULTS/reference"
bash "$testdir/run.sh" "$CANDIDATE_BRIDGE" "$KVIP_SOURCE" "$KVAF_INCLUDE" "$RESULTS/candidate"
python3 "$testdir/inspect_model.py" "$RESULTS/reference" "$RESULTS/candidate"
```

Use a final argument of `2` for two consecutive replays on the same device addresses. For the per-query phase handshake implementation, set `EXPECT_FINAL_MAIL=1` to check that each Q reached phase 2 before its second replay. A row-broadcast implementation can use a different layout for its phase words.

The fixture runs the actual resident scoring, resident threshold and cold scoring device entries. Four Cube cores and eight Vector cores process two rows with four queries each, 2,048 resident tokens, 5,120 total tokens and four service windows. The two initial hot-valid masks use 4 and 103 missing positions, the median per-query eligible counts in the archived G2 and G7 rank fixtures. Synthetic signed Q, K and weight values make all 40,960 scores independently checkable. Actual input capture, workload distribution and overall graph performance require the NPU replay.

Window service writes its normal WQEs into simulator memory. The verifier checks the resident scores, seed IDs, thresholds, uniqueness of each source, destination addresses and the resulting Index entries. It prints per-window prefetch membership and order changes separately. Communication completions and final TopK/KV processing are outside these three launches. The combined span covers the earliest resident-task start through the latest cold-task completion across all twelve cores.
