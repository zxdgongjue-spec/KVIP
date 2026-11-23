# Four-window production LI chip-model replay

On an AArch64 Linux machine with CANN 9.2, build the production LIFetch bridge for the Ascend950DT_9582 target. Use the same toolkit, KVAF headers and compiler settings for the reference and candidate. The launcher selects the CANN DAV3510 PEM CPU simulator. Use a new directory on the Linux VM filesystem, such as a directory under `/tmp`. The default timing mode saves per-core traces and library/configuration hashes.

```bash
testdir=tests/operators/lifetch/model_window
bash "$testdir/run.sh" "$REFERENCE_BRIDGE" "$KVIP_SOURCE" "$KVAF_INCLUDE" "$RESULTS/reference"
bash "$testdir/run.sh" "$CANDIDATE_BRIDGE" "$KVIP_SOURCE" "$KVAF_INCLUDE" "$RESULTS/candidate"
python3 "$testdir/inspect_model.py" "$RESULTS/reference" "$RESULTS/candidate"
```

Use a final argument of `2` for two consecutive replays on the same device addresses. For the per-query phase handshake implementation, set `EXPECT_FINAL_MAIL=1` to check that each Q reached phase 2 before its second replay. A row-broadcast implementation can use a different layout for its phase words. For a penultimate-window helper implementation, `EXPECT_IDLE_HELPERS=1` also checks that a previously assigned and completed helper was left in place before the next replay. Each replay saves its per-Q offer, assignment and completion fields.

The fixture runs the actual resident scoring, resident threshold and cold scoring device entries. Four Cube cores and eight Vector cores process two rows with four queries each, 2,048 resident tokens, 5,120 total tokens and four service windows. The two initial hot-valid masks use 4 and 103 missing positions, the median per-query eligible counts in the archived G2 and G7 rank fixtures. Synthetic signed Q, K and weight values make all 40,960 scores independently checkable. Actual input capture, workload distribution and overall graph performance require the NPU replay.

To exercise adjacent BF16 cutoffs, set `KVIP_MODEL_SCORE_FIXTURE=adjacent-cutoff` for both the reference and the candidate. The first row starts with resident scores of 128 and 129, and has cold scores of 129, 130 and 132. The second starts with 128 and 131, and has cold scores of 131, 132 and 134. A single active dot product of 128 and an exact FP32 scale per token give an independent expected score for every position. Use the default input to measure a larger cutoff jump separately.

Window service writes its normal WQEs into simulator memory. The verifier checks the resident scores, seed IDs, thresholds, uniqueness of each source, destination addresses and the resulting Index entries. It prints per-window prefetch membership and order changes separately. Communication completions and final TopK/KV processing are outside these three launches. The combined span covers the earliest resident-task start through the latest cold-task completion across all twelve cores.


For a fast correctness run, set `KVIP_MODEL_MODE=fast`. The launcher uses four host simulator threads and disables per-core instruction dumps. It still checks all scores, private window winners, source requests, destination slots and Index entries. Compare its result with a timing-mode directory to check each window's token membership. Latency percentages require both directories to use the timing mode. The current mode and the actual configuration are preserved in the run directory.

For the first replay in timing mode, the executed instruction addresses also identify when each row owner passed its predecessor, finished loading the frontier, filled its Post workspace and began building WQEs. To reproduce the last two windows' per-row times, run the following with the matching unstripped linked device object and source checkout. Each marker contains the first and last observed model tick and instruction address. Row 0 and row 1 can finish at very different times. The optional `--writes` flag also extracts each actual persistent TopK score or ID write, with the transfer byte count and destination address.

```bash
python3 "$testdir/owner_phases.py" "$RESULTS/candidate" "$BUILD/v9_kernel_merge_obj_dir/device.o" "$KVIP_SOURCE" --queries
```
