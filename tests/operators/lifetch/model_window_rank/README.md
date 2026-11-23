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
