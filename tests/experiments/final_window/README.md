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
