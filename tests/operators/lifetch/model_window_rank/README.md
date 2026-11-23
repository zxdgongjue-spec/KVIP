# Check window shortlist selection and handoff on the local CPU-driven CANN model

The input is the extracted five-group eight-window audit. Each `g{group}` folder contains `manifest.json` and `probe_inputs.npz`. The model invokes the compiled production helpers. Build for `Ascend950DT_9582` with the CANN 9.2 `dav_3510` simulator and the included L2-disabled configuration. Run one simulation process at a time.

```bash
bash run.sh /path/to/window-audit-results/analysis /tmp/window-measurements
python3 trace_report.py /path/to/main-li-window-trace.tar.gz
```

The shortlist check uses the actual eight-window audit IDs and BF16 score keys at every second saved boundary, tags residency, and models a deterministic union of all four Qs' posts. The validated offline input has 16 rows, four Qs and four modeled boundaries for each of the five workloads. The observed B24 mainline recorded a startup window with no ranking, three live incremental windows and changing per-row score readiness. Those event boundaries cannot be reproduced exactly from the earlier eight-window audit. `trace_report.py` reads the actual B24 diagnostics separately.

The model measures four lists per case, one median eligible count per modeled boundary. It also tests padding, score ties, zero candidates, quota/floor boundaries and dense 2048-candidate lists. The scalar oracle and the vector output must agree for each emitted ID in order. The handoff model passes four Qs' variable lists to the production reader for one and two owned rows. It verifies counts and every valid ID at the physical stride the reader returns. The timed span includes writing the metadata back in both paths. Deduplication, Post, CQ and score multiplication are outside both timed spans.

Both variants warm once, then alternate A/B/B/A. The report averages the two warmed readings per input. Device model cycles refer to their helper boundaries. The B24 trace reports service and incremental time for each lane and can establish which lanes finished near terminal entry. The incremental interval also covers merging, selection, metadata and publishing. It has no separate per-Q shortlist timestamps.

For exhaustive component validation, `fixtures.py AUDIT OUT --all-model` exports 256 ordered lists per group. `probe OUT/g7 --once` checks each of those lists on both variants and emits their two model times. This abbreviated run has no separate warmup.

### Continue a full fixture sweep

The chunk scheduler runs on Linux inside the CANN container. Source the same CANN environment as the original build and put the simulator library on `LD_LIBRARY_PATH`. It links the host driver against the already compiled device archive. Keep that build, fixture files, and model configuration in place for the duration of the sweep.

```bash
python3 host_only.py "$old_build" "$new_host"
python3 schedule_chunks.py --samples "$full_fixtures" --host "$new_host" \
    --config "$model_config_dir" --out "$separate_results" --plan
nice -n 15 python3 schedule_chunks.py --samples "$full_fixtures" --host "$new_host" \
    --config "$model_config_dir" --out "$separate_results" --max-jobs 2 \
    --chunk 16 --poll 10 --clear-polls 3 --reserve-cores 3 --reserve-mib 1024
```

For a completed representative ABBA run, add `--prior-samples DIRECTORY --prior-logs DIRECTORY` on first use. The scheduler accepts an archived row only when its input and expected output match the full fixture bytes, and verifies that the model settings agree apart from the trace mask. To continue a running legacy full-group job, register it while it is alive using `--existing "7:PID:/absolute/path/model-g7.log"`. Add `--resume-existing` to allow the scheduler to resume that registered process after headroom checks. Its enclosing batch shell should stay out of the scheduling loop. Subsequent restarts read the registrations and prior paths from the output directory. Optional repeated `--wait-executable-prefix` arguments protect higher priority validation binaries. The CPU and memory reserves govern launching new jobs or resuming registered jobs. Active jobs run to completion.

`--plan` updates the receipts and `per-query.csv`, prints coverage, and leaves simulation paused. Re-run the regular command with the same paths after an interruption. Each worker and its model process hold a lock through exit. After a partial run the next job measures only missing variant/fixture pairs. A correctness mismatch stops scheduling for inspection. To exercise manager interruption and partial-run recovery without a device model, run `python3 check_resume.py`.

Each CSV row preserves the original group, window, row, Q, and variant. The underlying runner checks all emitted IDs in order against the stored oracle. The `source` column distinguishes prior warmed ABBA averages, full launches, and new chunks. The `first_in_launch` column identifies the first fixture of each variant launch. A new process or chunk changes startup and memory/cache history. Use within-chunk comparisons with comparable sources, and report first rows separately when interpreting timing. Exact output equivalence holds across chunk boundaries. The model ticks describe this window selection component only.

`cutoff_step_fixtures.py AUDIT OUT` reconstructs every admitted score from the original eight-window tensors, including admissions later dropped from the TopK. It checks all eight archived boundaries, then uses each pair of boundaries as one window and validates the ordered IDs, unsigned scores, and residency flags after every update. It writes five real rising-cutoff cases, five workload controls, and one synthetic fallback for the cutoff-step microkernel. Use `--all-fixtures` to export every nonempty compacted window by workload. The original scan geometry is recorded in the input manifest. B24's concurrent service snapshots have different boundaries.

With `--idle-cpu`, Linux schedules the model workers and their threads at SCHED_IDLE priority. The scheduler can read a live limit from `--cap-file PATH` up to `--max-jobs N`; updating the file affects future launches. `--telemetry` records measured CPU/memory headroom and complete output pairs by case on every poll.
