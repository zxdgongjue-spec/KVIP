# Bounded final-window experiment (URMA)

Run `compare.py SNAPSHOT --kvaf-root KVAF --output NEW_DIRECTORY`.
The snapshot is a single-group `lifetch_failure_v1` dump; this prototype accepts
sequence-split LI and history up to 131328. Defaults: four windows, budget 1024,
initial rank 1024, two warmups and three profiled graph replays per variant.

A uses the checked-out operator unchanged. B is built from an isolated copy:

- Final TopK runs at the start of the streaming tail kernel; the separate final
  TopK launch is removed. Existing incremental TopK and classification are reused.
- A request qualifies if every Q has a valid compact miss list of at most 2048
  entries and the sum fits the remaining 2048-entry LRU allocation. Duplicate
  entries are included in this conservative capacity test.
- The intermediate `FastService` union and WQE posting implementation handles
  these misses, without the speculative rank or prefetch-budget cap. All selected
  misses must be mapped after posting; missing mappings fail completion.
- The original Index/output/CQ completion path remains. Nonqualifying requests
  retain the original resolver and Fetch implementation.
- Two all-AIV phase barriers remain: TopK publication and terminal service
  completion. This prototype does not yet start Q1 transfers before Q2–Q4 TopK.

This is not a complete replacement of resolver. The experiment asks whether
reusing intermediate posting plus eliminating the kernel handoff pays for the
remaining publication work. The new transfers count as prefetch internally;
`terminal_counts.json` distinguishes them from the intermediate-window traffic.

The runner reuses the temporary snapshot utility from historical commit c239791
(fetching its branch if necessary). It restores identical input state outside
measurement and profiles original A and experimental B in separate processes.
DDR payload is synthetic with captured address aliases; Q/Index and initial HBM
come from the snapshot. Compare sorted logical TopK and selected KV bytes, plus
completion/bounds checks. Timing is not accepted if either comparison fails.
`total_us` is the sum of profiled device kernel means, not graph-event latency.

Outputs: `ab_summary.json`, per-variant profiler traces, correctness artifacts,
terminal eligibility/post counts, and the complete isolated candidate source.
Only the experimental copy is patched; no installed serving library is replaced.

Terminal diagnostics: reason_bits 1=classifier error, 2=invalid count,
4=sum exceeds candidate capacity, 8=insufficient remaining LRU. Empty rows
do not count as eligible. Counts include every Q; fallback posted is zero.
Only the experimental final classifier exports beyond 64 misses; the original
resolver fast-path bound remains unchanged.

## Local device-model comparison

`model/run.py` builds the current experimental implementation and the previous
`c2c0a7b` experiment, then runs CANN's Ascend950PR_9599 simulator. Source the CANN
environment and put its simulator library directory on `LD_LIBRARY_PATH` first:

```bash
python tests/experiments/final_window/model/run.py \
  --kvaf-include /path/to/KVAF/access_operator/demo/aiv_batch_get/include \
  --output /tmp/lifetch-final-model-new
```

All three cases use history 131328, resident 16384, four Q, the same seeded
TopK and final 16384-score interval, miss counts `[11, 0, 94, 87]`, union 146,
and 84 already allocated LRU slots. The middle case uses one AIV for four Q;
final cases use four AIVs for TopK, then one AIV for posting, with a publication
barrier. The model checks exact expected TopK sets, distinct token mappings,
WQE source/destination addresses, posted counts, and final output locations.

The measured scope ends after posting and location materialization. It excludes
real NIC/CQ latency and the original downstream Index/output/completion workers,
which remain in the actual A/B operator. Simulator microseconds are reported
by the vendor profiler; they are not a conversion of device ticks or a prediction
of end-to-end hardware time. The older per-component probes are not added to
this combined-kernel measurement.

The optimized terminal path consumes packed `(token, original_position)` entries
directly, stages deduplicated posting IDs in the unused offsets buffer, and
updates only miss locations. It does not overwrite, back up, or restore the full
8192-position row. Admission metadata is computed once, not by all 256 threads.

Local combined-kernel results (one active request, bound batch 1):

| Path | Simulator duration, us | Posted | Check |
| --- | ---: | ---: | --- |
| Middle TopK + prefetch, detailed stamps off | 48.34 | 146 | PASS |
| Terminal at c2c0a7b | 57.96 | 146 | PASS |
| Terminal with scalar admission and hoisted miss counts | 50.25 | 146 | PASS |
| Current terminal, combined miss-location patch | 48.89 | 146 | PASS |

The terminal path saves 9.07 us (15.65%) versus c2c0a7b, leaving 0.55 us
versus the middle path with detailed stamps disabled. The earlier 49.13 us
middle measurement enabled service diagnostics and is superseded by 48.34 us
for this comparison.

The four miss counts and pointers are loaded once on the AIV. Admission no
longer launches a single-thread SIMT function. Location patching distributes
all four Q lists across the same thread group, instead of processing each Q
in sequence. Its independent location and counter writers require no internal
thread rendezvous: the final thread fence, VF completion and caller's all-core
publication barrier still precede downstream consumers. Detailed stamping is
compiled out of the normal service path.

The current device model verifies expected TopK, mappings, WQE addresses,
posted count and final locations (`mode=1 posted=146 bad=0`). The complete
experimental operator also builds with CANN, including diagnostic and normal
branches. These measurements exclude NIC/CQ and downstream completion, and do
not establish the improvement of the padded-batch-16 hardware tail (85.34 us
in the latest supplied trace). Rerun `compare.py` for that complete scope.

## Detailed hardware evidence

The A/B runner also captures a separate diagnostics-enabled graph after the
uninstrumented performance samples. Each `A/graph-captured` and
`B/graph-captured` directory retains `level3.json` (raw timing, flags, errors,
window events, rank records, row state, final descriptors and locations),
`window_events.bin`, `window_ticks.json`, and `final_detail.log`.

The console aggregates timings over active requests and service lanes rather
than dumping every core. Window rows show the combined TopK-through-service interval, TopK, service, scan, filter and post
mean/max; final rows show classification, terminal admission/post/location
patch, barriers, resolver, Index, output and Get phases. The counter rate is
measured on the device. Detailed replay timings are not substituted into the
uninstrumented A/B performance table. The extra terminal event is reported as
epoch `windows`; the preceding epochs are the intermediate windows.

Extra TopK cores still join completion barriers, but cannot write beyond the
original tail core-timing allocation. Raw terminal phase stamps use reserved
rank-record slots 16 through 24; window rank records remain unchanged.
