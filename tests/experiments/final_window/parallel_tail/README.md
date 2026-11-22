# Four-core terminal pipeline experiment

Current generator: **v346**, four-core terminal prototype.
Posting + complete output + four-core join **10.2556 us**; non-CQ tail **28.9322 us**.
**10 us posting is not met.** Exact combined executable passes persistent SQ/head/doorbell
case15 twice (v347), including nonzero 64-bit head/ring wrap, and delayed Q4 case5 (v348).
Exact executable also passes nondefault ABC case11 (v378) and rotated miss distribution case16 (v379).
Input Index/scores are restored between launches; this is not evolving decode.
Compared with v310, output join uses an exact monotonic arrival count and WQE building
has a single-bank fast path with original multi-bank fallback. All full outputs remain.
The producer store and doorbell writeback remain at submission; software producer cache
writeback remains after the join and before Index/CQ, requiring exclusive queue ownership.
Production transport/CQ cache-line ownership must still be checked before integration.
The generator reproduces measured v346 kernel/post/emit/driver/prefetch sources exactly.
11 host tests pass. Production code is unchanged; no hardware or NIC acceptance.

The former v202 **11.7344 us** result used fixed ABC geometry and fails the new
nondefault ABC case (v233, four wrong output slots). Keep it only as historical
fixed-fixture evidence, not a general integration candidate.

Historical variants below are evidence records, not the current implementation.

Implements one request's terminal work as a single four-AIV kernel:

1. Four queries independently reuse the existing incremental TopK/export code.
2. Each AIV reads only its own compact candidates. Three warps register high-bit minimum-Q ownership markers for disjoint candidate subsets while the remaining warps prepare hot/ABC outputs. Existing prior-prefetch mappings remain unchanged. After earlier Q claim producers finish, only the owning Q retains each token.
3. Each AIV atomically reserves disjoint LRU slots and uses its own queue, reusing the original WQE constructor. Three warps prepare address/slot inputs and change records after warp0 publishes ownership and slot allocation through UB. The first hot-output warp preloads the immutable WQE template and current 64-bit SQ head. An explicit block-scope fence precedes compact-ID consumption; local barriers and prefix-Q ownership/mapping publications precede cold-output readers.
4. Emit each query's logical IDs and slots together. Compact candidates have a separate terminal buffer so output writes cannot overwrite later queries' inputs. All output readers rendezvous before Index metadata changes or TopK headers/progress are cleared.
5. Partition unique change records between four AIVs; reuse the original Index key/scale/address/logical swap helpers in one invocation per core.
6. Each AIV retires only the change records it consumed, after all output readers have finished. The four cores also partition retirement of the already-consumed prior admissions.

Index update reuses dead TopK scratch (20.5 KiB); it does not destroy/recreate the pipe. The original duplicate source/destination checks are retained. The fixture limits each slice to at most 512 records.

The fixture has history 131328, resident 16384, four Top2048 outputs, 84 prior admissions and candidate counts 11/0/94/87 (146 unique final admissions). Score values are constructed; this is not replay of the server's tensors. Prior Index updates are assumed complete, as requested. Used LRU slots are unique and cold sources/hot destinations disjoint. Capacity exhaustion/general permutation cases are not covered by this prototype.

CQ is modeled as ready 9000 simulator SYS_CNT ticks after the final post; native trace calibration must confirm this equals 5 us. Computation overlaps that delay. This does not emulate a real transport completion or validate real KV payload delivery; WQE source and destination fields are checked.

Per-request claim, mapping and output-ready sets occupy separate cache lines; each set contains four SIMT-only words and uses generation numbers. Scalar generation/publication lines are disjoint. The ready test uses a warp-wide ballot; staggered producer regression evidence is in `sync_probe/staggered_README.md`. No all-device barrier is added. The current harness uses `Args.eventsSrc` as logical output storage and is therefore an isolated simulator interface, not yet the production launch ABI.

## Run in CANN 9.1 simulator environment

```bash
python3 tests/experiments/final_window/parallel_tail/run.py \
  --output /tmp/new-parallel-tail \
  --kvaf-include /tmp/window-sync-perf/kvaf-include \
  --baseline-us 80
```

Cases: 15 = two independently audited launches retaining SQ heads, SQ contents and doorbells across a nonzero 64-bit head/ring-wrap boundary (Index/score inputs restored, not evolving decode); 12 = nonzero 64-bit SQ heads, ring wrap, and all fixed WQE fields checked; 11 = runtime ABC capacity384 with nonaligned begin and length123; 9 = two individually audited launches with prior-prefetch mappings; 1 = base, 2 = ABC/nonaligned query ends, 3 = two launches preserving synchronization generations, 4 = base with detailed device stamps, 5 = correctness stress delaying Q4 TopK by 20 us (excluded from performance gates). Normal performance runs disable detailed stamps. Native CANN profiler still measures every core's full duration.

The 80 us reference is the user's stated acceptance reference, not a newly measured matched baseline. The current gates are <=10 us for deduplication/posting including output preparation, full position output and synchronization, and <=35 us for the complete tail excluding measured CQ wait. The posting gate has NOT been met. Current and candidate measurements and their applicability are stated at the top of this document. `summary.json` must not be described as a fresh baseline A/B measurement.

Correctness covers all selected IDs, slots, logical outputs, WQE source/destination addresses, unique admissions, both sides of Index key/scale/address/logical swaps, and retired mapping/header state on the repeated-launch case. Inactive requests and multi-request contention still require integration validation.

Index key/scale and address/logical swaps share one SIMT invocation. This is safe here because all logical output IDs were emitted and all WQE remote addresses consumed before posting completion is published. Slot output does not read the mutable Index metadata.

## Latest measured progress

Native simulator timings; same constructed base fixture, detailed stamps enabled.

| Version | Full tail us | Non-CQ us | Posting + all outputs + sync us | Verification |
| --- | ---: | ---: | ---: | --- |
| v27 | 39.47 | 39.17 | 21.02 | Base, ABC, consecutive two launches PASS |
| v28 | 42.24 | 41.93 | 23.66 | PASS, performance regression rejected |
| v29 | 36.50 | 36.21 | 18.38 | Base, ABC, consecutive two launches PASS |
| v30 | 37.88 | 37.85 | 19.19 | PASS, direct GM output regression rejected |
| v31 | 36.16 | 36.16 | 18.38 | Base PASS; only merges private retirement fences |
| v38 | 35.38 | 35.36 | 17.63 | Base PASS; independent queues, protected candidate/header lifetimes |
| v41 | 35.99 | 35.97 | 18.06 | Base PASS; address preload and 32-bit logical gather did not improve v38 |
| v42 | 34.65 | 34.63 | 16.76 | Base, ABC/high-bit logical IDs, consecutive two launches PASS; 1024 output threads |
| v43 | 34.91 | 34.59 | 16.62 | Base PASS; merge deduplication and WQE input preparation |
| v45 | 35.64 | 35.38 | 17.52 | PASS; merging output with dedup regressed, rejected |
| v47 | 35.00 | 34.98 | 17.30 | PASS; queue metadata overlap did not improve the full path |
| v49 | 34.48 | 34.19 | 16.46 | Base/ABC PASS; retain full TopK IDs in UB |
| v52 | 34.00 | 33.68 | 16.06 | Base/ABC/consecutive two launches PASS; publish physical slots |
| v56 | 34.95 | 34.66 | 16.93 | PASS; combine queue/entity DMA wait, regression rejected |
| v57 | 35.50 | 35.48 | 17.68 | PASS; shared atomic admission bitmap, regression rejected |
| v58 | 42.44 | 42.42 | 24.98 | PASS; per-token GM CAS claim, regression rejected |
| v59 | 33.54 | 33.52 | 15.73 | Base PASS; pass request-local pointers directly |
| v60 | 34.42 | 34.12 | 16.08 | PASS; UB-only output fence removal did not improve full path |
| v61 | 34.00 | 33.68 | 16.06 | PASS; private queue cache lines and producer/doorbell agreement |
| v62 | 33.32 | 33.30 | 15.61 | Base, ABC/nonzero entity fields, consecutive two launches PASS |

The v39 Q4-delay stress also passed. v35 is not accepted despite its immediate-output check: relaxed dependencies exposed candidate-buffer/header lifetime risks, subsequently fixed in v38.

Previous accepted baseline v52: retain full TopK IDs in UB and publish physical slot IDs to avoid a second LRU lookup during output. Base, ABC/nonaligned/high-bit logical-ID and consecutive two-launch checks pass. The consecutive two-launch case measures 33.62 us without detailed stamps; it is not a matched profiling-overhead comparison.

v54 readiness publication inside slot preparation passed but regressed to 35.23 us full / 35.21 us non-CQ / 17.59 us posting and was rejected. Scalar-wait v53, count-in-ready v50 and token-striped union v51 also regressed and were rejected. Increasing output threads beyond 1024 did not improve the separate output microbenchmark. v55 reduced slot-preparation threads from 256 to 128, with matching loop stride: PASS, but 34.15 us full / 33.83 us non-CQ / 16.12 us posting showed no improvement and was rejected. The experimental generator now reproduces measured v62 sources exactly. It resolves request-local pointers before preparation and binds immutable WQE fields once per entity/output lease; the current SQ producer remains dynamic. This is an isolated harness ABI, not a production binding change. Consecutive two-launch validation with nonzero per-queue entity fields also passes (32.78 us without detailed stamps; not a matched profiling-overhead comparison). v63 reused the own-Q compact UB list, but measured 34.46 us full / 34.44 us non-CQ / 16.78 us posting and was rejected.

The base fixture meets the non-CQ 35 us gate; the 10 us posting gate remains unmet.

The prototype assumes previous admissions have already been updated and does not yet bind production mapping-journal lifetime or real CQ/residency publication. These results are not production end-to-end acceptance.


### Expanded prior-prefetch fixture (v68 onward)

Earlier base timings omitted the live 84-entry prior-prefetch mapping journal and
selected prior-prefetch tokens. They do not establish acceptance for that state.
The expanded fixture binds and retires all 84 historical mappings. Final admissions
retire through their unique change records rather than duplicating journal writes.

| Version | Full tail us | Non-CQ us | Posting + all outputs + sync us | Status |
| --- | ---: | ---: | ---: | --- |
| v68 | 38.27 | 38.00 | 15.67 | Base PASS; prior cleanup on one core |
| v71 | 33.57 | 33.26 | 15.49 | Base PASS; partition prior cleanup across four cores |

v71 snapshots journal count before the existing all-output rendezvous. It adds no
new whole-device barrier. Prior Index updates are still assumed complete. The
selected-prior-token regression and two-launch check remain pending; v69/v70
stopped at an invalid unstamped timing assertion and are not correctness verdicts.
The 10 us posting gate remains unmet.


Selected prior-prefetch regression completed: v72 reproduces four wrong logical
outputs (one per Q), with all other audits zero; v73 fixes the logical source to
the already-updated resident slot and passes all audits. v74 includes partitioned
journal cleanup and passes two consecutive launches with selected prior tokens;
first-launch retirement is checked before input restoration, full output/WQE/Index
checks run after the second launch. The generator reproduces v74 source hashes.
Cases 8 and 9 expose these single/repeated prior-token checks.

v75 flattens prior-Q candidate scanning: base PASS, full 33.18 us, measured non-CQ
32.87 us, posting including output/sync 15.16 us. It remains an isolated candidate;
its own expanded regression has not run. Neither v74 nor v75 meets the 10 us gate.


v75 is not accepted: flattened prior-Q batches invalidate the old assumption that
all lanes in a bitmap-word reduction contain distinct tokens. Summing duplicate
bits across Qs can carry into another bit. v76 groups by both Q and bitmap word
before reduction; v77 additionally combines readiness/header reads with candidate
processing and reads compact candidates directly from GM. Their model runs are
pending. Current reproducible validated source remains v74.


v76 base PASS but regressed to 34.08 us full / 33.77 us non-CQ / 15.78 us
posting. v77 direct-GM base PASS but regressed to 34.04 / 34.02 / 16.35 us.
Neither is adopted. Overlapping-Q fixture v78/v79 passes both old and corrected
variants; it did not expose the bitmap carry hazard in end-to-end execution and
must not be cited as reproducing it. The static hazard remains valid (adding two
identical bits changes their position); v75 remains excluded.

v80 overlapped immutable WQE template DMA within the measured posting interval.
Base PASS, but 34.61 us full / 34.59 us non-CQ / 15.73 us posting regressed; rejected.


| Variant | Full us | Non-CQ us | Posting + output + sync us | Finding |
| --- | ---: | ---: | ---: | --- |
| v81 | 33.29 | 33.27 | 15.52 | v74 with additional attribution only |
| v82 | 33.46 | 33.16 | 15.19 | Remove inter-warp barriers from one-warp polling |
| v83 | 33.99 | 33.67 | 15.49 | Skip zero-miss preparation and self-ready wait; not adopted |
| v84 | 33.95 | 33.65 | 15.72 | Assign whole prior Qs to warp groups; not adopted |
| v85 | 33.68 | 33.37 | 15.15 | Pad each prior Q to a warp and distribute warp batches |

All above base audits pass. v85 preserves per-Q uniqueness within each reduction;
`test_warp_partition.py` checks exact candidate coverage and single-Q warp ownership
for 1000 size combinations including zero, warp boundaries, and 2048 candidates.
The v86 combination of v82/v85 is undergoing base, delayed-Q4, and two-launch
prior-prefetch regression. No production source or binding is changed.


Current isolated generator: v86. Combined one-warp polling and warp-padded prior
candidate batches: 33.39 us full / 33.08 us non-CQ / 15.30 us posting. Same sources
pass prior-prefetch selected-token consecutive launches (v87) and delayed Q4 (v88).
The first launch's retirement and the second launch's full outputs/Index/WQEs are
verified. Case 10 exercises overlapping prior-Q candidates. No 10 us acceptance:
posting still exceeds the target by 5.30 us. v89 compile-time Q specialization
passes but regresses to 34.31 / 34.29 / 16.02 us and is rejected.

Earlier v75's 32.87/15.16 result is not an accepted baseline because of the
cross-Q bitmap reduction hazard described above. Current timings retain all
output preparation, output writes, and cross-core synchronization. This is still
one constructed request with 84 prior admissions and 146 unique final reads,
not production multi-request or real-NIC acceptance.


### Merged preparation and full output task

All output preparation remains inside the posting gate. v90 joins mapping
preparation and the full logical-ID/slot emitter in one SIMT task, completes output
DMA before reusing its UB for the original WQE writer, then joins all four cores.

| Variant | Full us | Non-CQ us | Posting + output + sync us | Finding |
| --- | ---: | ---: | ---: | --- |
| v90 | 32.53 | 32.51 | 13.96 | 1024 threads; base PASS |
| v91 | 32.61 | 32.59 | 14.25 | 512 threads; rejected |
| v92 | 35.35 | 35.33 | 16.87 | 256 threads; rejected |
| v93 | 32.14 | 32.12 | 13.72 | Remove duplicate barrier before Emit and at VF return; base PASS |
| v94 | 33.05 | 33.03 | 14.41 | Disjoint output DMA/WQE UB and overlap; rejected |

v93 retains the mapping publication barrier, the emitter's producer wait and
block rendezvous, output DMA completion, and the four-core output join. Expanded
regressions are pending; the reproducible generator remains v86 until they pass.


Current validated isolated generator: v93. Its same source passes selected prior
prefetch tokens across two launches (v96), delayed Q4 (v97), and ABC/nonaligned
query boundaries with high-bit logical IDs (v98). As before, the first launch's
retirement and the second launch's complete output/Index/WQE audit are checked.
Full timing is 32.14 us, non-CQ 32.12 us, posting including output and join 13.72 us.
The 10 us gate is still unmet by 3.72 us. The complete emitter now lives in
`emit.h` rather than being reconstructed through source-string replacements.

v95 merges candidate waiting/GM reads into the combined task: PASS but 32.76 us
full / 32.74 us non-CQ / 14.43 us posting, rejected. v99 restricts writer fences
to writing lanes and removes the UB-only emitter's global fence: PASS but
32.25 / 32.23 / 13.79 us, rejected; v93 retains the original fences.

### Publication/join experiment v100–v102

- v100 merges final output-ready publication with the existing warp join. Base fixture passes: posting including output/join 13.3211 us, full 32.58 us, measured non-CQ 32.2633 us. Posting improves over v93, but total does not.
- v101 additionally merges initial TopK-ready publication into the first wait: posting 14.9889 us, non-CQ 33.3017 us; rejected.
- v102 repeats v100 with selected prior-prefetch tokens and two launches (case 9); full output, WQE and retirement checks pass.
- Generator remains v93. Neither variant meets the 10 us posting target; no production integration or performance acceptance.

### v103: resolve hot/ABC output before mapping-ready wait

Base v100; emit independent hot/ABC outputs first, then wait for prefix mapping producers and fill cold outputs. Full output remains inside posting metric. Base fixture correctness PASS, posting 14.5961 us, non-CQ 32.9194 us. Rejected: additional traversal costs more than overlap saves. Generator unchanged.

### v104–v105: bitmap grouping and header acquisition

Both use v100 as baseline. v104 unrolls dense-word warp aggregation into four reductions; PASS, posting 13.5817 us, non-CQ 32.4256 us. v105 acquires candidate counts inside the initial ready-wait task, removing the header DMA; PASS, posting 13.5461 us, non-CQ 32.1244 us. Neither improves the best posting span (v100 13.3211 us). No adoption; generator remains v93. These are constructed base-case simulator results, not production acceptance.

### v106–v110: output acquisition scheduling

- v106 caches hot/ABC logical values in registers before the mapping-ready wait: base PASS, posting 13.6744 us, non-CQ 32.0678 us. Not adopted.
- v107 replaces the prefix mapping-ready wait with per-token mapping polling: base PASS, posting 13.2461 us, non-CQ 31.7517 us. v108 case9 repeated launches PASS.
- v109 delayed Q1 preparation but was terminated deliberately because Q1 candidates do not overlap later Qs in this fixture; no validation claim.
- v110 instead delays Q3 preparation by 36000 ticks after TopK publication. **FAIL: audit_output=46, WQE/TopK/count/Index/error audits all zero.** Thus v107 is unsafe under the tested producer delay and MUST NOT be adopted. Exact output discrepancy still requires diagnosis; no claim of a proven root cause. Generator remains v93.

### v111–v114: delayed producer control and warp-uniform polling

- v111 restores the v100 prefix-ready wait under the same delayed-Q3 workload: all audits PASS.
- v112 repeats per-thread polling with exact mismatch output: 46 Q4 slots remain UINT32_MAX while logical IDs are correct; WQE, TopK, counts and Index audits pass.
- v113 makes all lanes execute the mapping-poll loop with warp ballots and uniform termination before output. Delayed-Q3 case PASS (all audits zero). This demonstrates the per-thread implementation is unsafe; no claim about a compiler root cause is established.
- v114 measures v113 without injected delay: PASS, posting including output/join 15.3911 us, non-CQ 33.9739 us. Rejected on performance. Keep v93 generator; neither unsafe v107 nor slower v113 is adopted.

### v115–v116: two-phase minimum-Q ownership

Each core loads only its own candidates and atomically minimizes high-bit Q ownership markers into the token map. Existing prior mappings sort below the markers and remain intact. All four claim producers rendezvous before compaction/admission; the original prefix mapping-ready/output protocol remains.

- v115 base PASS: posting with output/join 13.1033 us, non-CQ 31.2550 us.
- v116 case9 PASS: two launches with selected prior-prefetch mappings; all WQE/TopK/count/output/Index/error audits zero.
- 1000 CPU random candidate/interleaving ownership checks PASS (semantic check only).
- Delayed producer and ABC device regressions still outstanding. Not adopted into the generator; 10 us posting target remains unmet.

### v119–v120: claim alternatives

- v119 first-winner CAS compacts directly and waits for all four mapping producers before Emit. Base PASS, posting 20.7761 us, non-CQ 39.2383 us; rejected.
- v120 keeps minimum-Q ownership but waits only for prefix claim producers. Later Q atomic-min cannot override either a lower-Q tag or its published physical slot. Base PASS, posting 13.0367 us, non-CQ 31.3072 us. Marginal change versus v115; delayed-producer/ABC/repeated regressions pending, not adopted.
- Generator remains v93 and neither posting target nor full production integration is complete.

### v121–v122: task size with minimum-Q prefix ownership

Base v120. Both retain all 2048 output entries per Q. v121 uses 512 threads: PASS, posting 13.2022 us, non-CQ 31.3156 us. v122 uses 256 threads: PASS, posting 15.5972 us, non-CQ 33.7406 us. Neither improves posting; not adopted. Existing BatchBuild<true> contains a reusable SIMT WQE field construction pattern, but merging WQE construction with output still requires a separate UB lifetime/layout proof and benchmark; no such improvement has been claimed.

### v123–v124: merged SIMT WQE field construction

Reuse the BatchBuild<true> 16-field layout inside the admission task, staging up to 256 WQEs at scratch words 6144..10239, disjoint from output 0..2047 and counters/errors near 4104. Larger batches retain the existing vector writer. Queue DMA and doorbell publication remain ordered. Host audit now also checks every immutable WQE field.

- v123 GM template loads: correctness PASS, posting 62.6217 us, non-CQ 80.4322 us.
- v124 preloads the 128-byte template into UB alongside candidates: correctness PASS, posting 15.9139 us, non-CQ 33.9567 us.
- Both regress versus the existing vector builder; rejected. No production edits or generator adoption. The fallback and alternate queue-head cases were not performance-qualified because the base optimization already failed its performance gate.

### v125: remove obsolete four-count array from task ABI

Base v120; pass own candidate count directly instead of four counts plus dynamic Q indexing. Base PASS, posting 13.9644 us, non-CQ 32.3028 us; rejected on measured performance. Fewer source operations did not translate to faster generated code.

Read-only instruction attribution for v120 Q4: Pack task first attributed instruction ~13.099 us, PrepareDirect/producer barrier reaches ~17.194 us, Emit completes ~20.107 us. These are overlapping source-line ranges, not additive exclusive phase measurements. WQE vector construction spans ~1.026 us. Further work should focus on the ~7 us admission/output task, not assume removing source instructions improves it.

### v126: per-warp slot reservation and immediate mapping

Base v120. Merge compact packing, per-warp LRU reservations, address load, mapping and change-record output; eliminates intermediate candidate reread and two block rendezvous. Base correctness PASS but posting 13.5272 us / non-CQ 31.9667 us regress versus v120; rejected. The extra per-warp slot atomics and changed schedule do not produce a measured win.

### v127–v128: v120 boundary regressions

- v127 delays Q3 after TopK publication and before candidate claim/prepare by 36000 ticks. Q4 shares 46 candidates with Q3. All WQE/TopK/count/output/Index/error audits PASS. Injected-delay timing is not a performance result.
- v128 runs the unchanged v120 binary with ABC/nonaligned/high-logical-ID case2: PASS.
- Exact v120 repeated-launch prior-prefetch regression remains pending (v116 validated v115, not v120). Generator still v93; no claim of production acceptance or 10 us target completion.

### Historical baseline v120 (v129 audit scope corrected below)

The generator now reproduces v120. Base timing: posting/output/join 13.0367 us; non-CQ 31.3072 us. v127 delayed-Q3, v128 ABC/nonaligned/high logical ID, and v129 case9 consecutive launches with selected prior mappings all PASS. Generated csrc, driver, helper and build sources match the measured snapshot. This is an isolated one-request simulation, not a production integration or a completed 10 us posting target. Earlier sections record historical experiments and do not override this status.

### v130–v131: no material improvement

- v130 retains compact candidate registers: correctness PASS, posting including output/join 13.5322 us, non-CQ 31.7906 us. Rejected.
- v131 explicitly unrolls the two output items per thread: correctness PASS, posting 13.0244 us, non-CQ 31.5522 us. Difference from v120 is immaterial; not adopted.
- Both use the unchanged measurement interval and the same constructed fixture. Neither meets the 10 us posting gate.

### v132: publish mapping before remote-address loads

Moved independent address loads after mapping publication; all existing fences and waits retained. Base audit PASS; posting/output/join 13.1417 us, non-CQ 31.3661 us. No improvement; rejected without broad regressions. Current generator remains v120; the 10 us posting target is still unmet.

### v133–v134: UB handoff and split admission tasks

- v133 directly reads the retained compact UB list, uses disjoint scratch for unique IDs, removes candidate reload. Base PASS; posting 14.1856 us, non-CQ 32.8528 us. Rejected.
- v134 runs admission in 32 threads and full output in a separate 1024-thread task, with explicit completion between tasks and unchanged mapping publication. Base PASS; posting 14.2361 us, non-CQ 32.4906 us. Rejected.
- No change to the accepted generator or production sources; expanded regressions intentionally omitted for slower variants.

### v135: 128-thread admission plus separate full output

Base PASS; posting 13.6722 us, non-CQ 32.1111 us. Better than the 32-thread split but still slower than v120; rejected. The retained combined task remains the baseline.

### v136: independent change-record writers

Move changes/changeSlots writes after the existing mapping rendezvous and split the two arrays across 512-thread groups. Mapping release remains intact; all records still precede the full-output join and Index readers. Base audit PASS, posting 12.8461 us, non-CQ 31.1600 us. Small improvement; expanded delayed-producer/ABC/repeated-state regressions pending, so not adopted. The generator remains v120.

### v137–v138: control-warp isolation and self-ready polling

- v137 excludes warp0 from change records, partitions each field over 480 threads. Base PASS; posting 12.9283 us / non-CQ 31.0756 us. CPU coverage checks pass for 17 counts through 2048. No posting improvement over v136; not adopted.
- v138 retains the local producer fences/barriers but polls only prior Q flags (excludes own flag). Base PASS; posting 13.2983 us / non-CQ 31.6389 us. Rejected.
- v136 remains the best new candidate at 12.8461 us; only base fixture verified. No production integration or 10 us acceptance; current reproducible generator remains validated v120.

### Audit correction and v139–v141

Review found that previous case3/case9 drivers audited full outputs only after the second launch, and checked first-launch final mapping retirement after restoring that mapping. Earlier reports saying verified_launches=2 establish two executions, not two complete output audits. Do not use them as full per-launch correctness evidence.

The driver now fully audits each launch before restoration and checks retirement before restoring mapping. The report rejects missing/failed per-launch results; regression tests cover first-fail/second-pass and missing-marker cases. v140 is retained as limited historical evidence, pending replacement by v143.

For v136 sources: v139 delays Q3 after TopK publication but before candidate claim by 36000 ticks; all full audits pass. v141 exact v136 binary ABC/nonaligned/high-logical-ID case2 passes. v143 strengthened consecutive-launch validation is pending.

### v142: retain own-Q Index inputs in UB

Remove final change-record GM writes and reuse the Q-local unique IDs/slots after the existing full-output join. Reuse the Index swap body with UB input lists, processing up to 512 records per tile; retire the own-Q mappings afterwards. Base full audit PASS; posting 13.5217 us, non-CQ 31.4300 us. This does not improve posting versus v136 and is rejected. No performance claim from merely deleting a GM round trip. Archived helpers.h contains the UB overload.

### v143: strengthened repeated-launch acceptance and current generator

Both launches report all audit classes zero and distinct verified_launch=1/2 markers before restoring inputs. The driver checks each launch's complete TopK, output slots/logical IDs, WQEs, Index swaps, errors and retirement. v143 uses the v136 kernel sources. The current generator now emits v136 kernels and this strengthened host driver; byte comparison against measured/verified source snapshots passes. This establishes the isolated fixture regression, not real NIC or general-capacity correctness, and does not satisfy the 10 us posting gate.

### v144–v146: control/data scheduling probes

All base audits pass with the strengthened driver. v144 puts slot allocation on thread1023: posting 12.7611 us, non-CQ 31.1267 us; small difference, not adopted. v145 moves claim/map publications to the last warp and excludes self-ready polls while preserving local barriers: posting 12.8533 us, non-CQ 31.0083 us; no posting gain, not adopted. v146 loads the exclusively owned queue head between the SIMT launch and its wait: posting 13.9528 us, non-CQ 31.7906 us; rejected.

### v147: full-quota bitmap compaction shortcut

When rank>=2048 and quota>=2048, PackUncovered emits eligible IDs directly and avoids the later two-pass score-threshold Emit. Other parameter combinations retain the original path through a false template specialization. Base audit PASS; posting 13.0389 us, non-CQ 31.2150 us; no improvement over v136, rejected without broad regression. Native Q4 trace contains no threshold-Emit instructions in v147. In v136 those Emit source lines span only about 10.411–10.620 us (0.209 us), so the assumption that removing them would recover most TopK skew was disproven. Source-line ranges are overlapping attribution, not additive phase costs. Production shared helper is unchanged.

### v148: replicated union/rank, rejected

All four cores independently merge the same candidates into exact union and lower-Q bitmaps, derive deterministic LRU ranks, and write disjoint change records. This removes final GM ownership/mapping publications but adds full-history local bitmap scans. Base audit passes (`mode=4 posted=146 bad=0`); posting including full outputs/join **17.2061 us**, complete non-CQ **34.9828 us**. Worse than v136; not adopted. An accidental mode-0 launch was terminated and is excluded; only the completed mode-4 run is reported.

### v149–v150: bounded replicated bitmaps, rejected

Both pass the base full audit. v149 scans only the actual candidate word span: posting/full output/join **15.0222 us**, non-CQ **32.6328 us**. v150 uses a single-warp prefix for spans <=32 words (retaining the general prefix otherwise): **14.6761 us**, **32.5628 us**. Neither improves v136 (12.8461 / 31.1600 us), so no broader regression runs or generator adoption. Native simulator timing only; no hardware performance claim.

### v151–v152: concurrent hot output and admission

v151 overlaps 256-thread claims with 768-thread hot/ABC output. Base PASS, posting/full output/join **13.7833 us**, non-CQ **31.8078 us**, rejected. v152 gives all admission phases to one warp and hot/ABC output to remaining warps; base PASS, posting **12.3978 us**, non-CQ **30.4511 us**. This is a candidate only: no generator adoption pending explicit UB handoff fence and ABC/repeat/delayed-producer regressions. The <=10 us posting target remains unmet.

### v153–v156: explicit handoff fence and address batching

v153 adds `asc_threadfence_block()` between compact-ID writes and consumption. Base full audit PASS; posting/full output/join **12.5278 us**, non-CQ **30.8261 us**. v154 runs its exact executable on ABC/nonaligned/high-logical-ID case2: PASS. v155 repeated-launch and v157 delayed-Q3 regressions remain pending at this entry. v156 batches four warp-sized remote/LRU reads before stores: base PASS, **12.5978 / 31.0272 us**, no improvement; rejected.

### v153 adoption

v155: both launches fully audited before restoration, PASS. v157: Q3 is delayed 36000 SYS_CNT ticks after TopK publication and before claims; full audit PASS, excluded from performance gates. The generator now uses v153 `parallel_post.h` and `emit.h`; other kernel/driver/helper code remains unchanged. v136 remains archived as the previous validated baseline. No production integration or <=10 us acceptance is claimed.

### v158–v159: prior bitmap and zero-count allocation

v158 imports earlier-Q candidates and constructs an exact UB bitmap instead of GM claim publications: base PASS but posting **18.1028 us**, non-CQ **36.4928 us**; rejected. v159 only guards shared slot allocation with `count != 0`; zero-count slots/records loops are empty and its `slotBase` has no downstream payload consumer. Base PASS, posting **12.3544 us**, non-CQ **30.4700 us**. Candidate only pending the current experiment comparison; small single-fixture point difference, no hardware acceptance.

### v160: all-thread address and record preparation

Keep claim/compact/allocation in the admission warp, then use a block rendezvous and all 1024 threads for remote/LRU reads and mapping/change records. Retain release fence and mapping-ready publication before cold outputs. Base full audit PASS; posting **12.5233 us**, non-CQ **30.8400 us**. Essentially unchanged from v153, not adopted.

### v161: reuse compact candidates in-place, rejected

Single-warp stable compaction writes only already-read positions; the exported compact buffer is consumed after TerminalQuery joins its DMA. A CPU reference checked 514 boundary/random candidate counts; device base full audit PASS. Posting **13.6117 us**, non-CQ **32.2528 us**: slower than v159 despite removing the reload, so not adopted. No compiler or UB-bank root cause is claimed from this timing alone.

### v162: parallel claims before hot-output overlap

All threads issue candidate claims before the existing initial block rendezvous; the admission warp then waits/compacts/allocates while other warps emit hot/ABC output. Base full audit PASS, posting **12.6333 us**, non-CQ **31.0317 us**. Slower than v159; rejected.

### v159 adoption

v163: two launches independently fully audited before restoration, PASS. v164: Q3 delayed after TopK publication before claims, PASS, excluded from timing gates. Generator now adds the zero-count slot-atomic guard from v159; all other v153 code is unchanged. The allocated base is unused when count==0; payload loops are empty, and output/publication/join behavior remains. <=10 us acceptance and production integration remain outstanding.

### v165–v166: ownership loads and initial barrier

v165 unrolls four ownership-map loads ahead of ballots/compact writes: base PASS, posting **12.6033 us**, non-CQ **31.0111 us**, rejected. v166 keeps hot-output errors thread-local until the existing final block rendezvous, removing the initial barrier: base PASS, posting **12.3128 us**, non-CQ **30.6317 us**. Posting differs from v159 by only ~0.04 us while total worsens ~0.16 us; no material gain and no expanded error-path qualification, so not adopted. Current generator remains v159.

### v167: fixed token-owner partition, rejected

Route each union token to token%4, deduplicate with returning UB atomic-OR, and keep per-Q output readers behind all four mapping publications. Base full WQE/output/Index audit PASS; CPU reference passes 500 shuffled duplicate/prior-admission combinations. Posts balance to **37/37/36/36**, but posting/full output/join is **13.7756 us**, non-CQ **32.3328 us**. Importing all four Q candidate sets introduces an all-TopK-ready dependency and cannot preserve early-Q admission; balancing alone does not overcome this cost. Rejected; generator remains v159. Covered bits alone were not substituted for the initial mapping check: the existing prior-admission fixture intentionally preserves mappings independently, and no proof of redundancy exists.

## v168: overlap cold logical-ID reads with admission — rejected

Prepare hot/ABC and cold logical IDs in independent output warps; preserve source tokens in separate UB storage and resolve pending physical slots after prefix mapping publication. Base full output/WQE/Index audit PASS. Posting including complete output and four-core join **12.3622 us**, non-CQ **30.5294 us**. No improvement over v159; no adoption or broader regression runs. Generator remains v159, and the 10 us target is unmet.

## v169–v171: queue publication and first-claim admission — rejected

v169 merges globally visible queue/doorbell writes and output-ready join in one SIMT task. Full output/WQE/Index audit passes, but scalar and SIMT write the same diagnostic cache line and post stamps are lost. Posting interval is 12.9056 us; CQ attribution is not accepted. v171 hands both post timestamps through UB to the scalar writer, and all eight clock samples match native SYS_CNT calibration (`clock_validation.json`). Posting/full-output/join regresses to **13.1567 us**; native whole-tail maximum is 31.69 us. The generic report intentionally leaves non-CQ attribution unavailable because it maps SYS_CNT instruction timestamps only. Neither variant is adopted.

v170 uses one admission warp with returning CAS to select the first claimant, while other warps prepare outputs. All four mapping producers are joined before cold output reads. Full base audit and 1000 CPU-reference claim/publication interleavings with duplicate/prior mappings pass, but posting is **31.9594 us**, non-CQ **50.1233 us**. Rejected.

The reporter now refuses non-CQ acceptance when the post timestamp is missing, chronologically invalid, or completion precedes the simulated five-microsecond minimum. Five report regression tests pass. Production sources remain unchanged.

## v172–v173: return-value atomics and compact ready words

v172 substitutes returning atomic-min with a common high marker for v170's CAS. Base audit passes but posting remains **31.9483 us**, non-CQ **50.1200 us**; rejected. Removing minimum-Q ownership does not overcome the cost of returning atomics in this implementation.

v173 groups each set of four claim/map/output ready words in one separate cache line. Every writer/reader of these words uses SIMT global stores/loads; scalar generation and TopK publication lines remain disjoint. No synchronization is removed. Base full audit PASS: posting **12.2389 us**, non-CQ **30.2717 us**. A small candidate improvement only; repeated and delayed-producer validation pending, generator stays v159.

## v174–v177: compact-ready regression coverage and load batching

v174 uses the exact v173 executable for selected prior mappings and two launches; both complete audits pass before input restoration. v175 delays Q3 after TopK publication but before ownership claims by 36000 SYS_CNT ticks; full audit passes. v177 uses the exact v173 executable for ABC/nonaligned/high-logical-ID coverage; full audit passes. v173 remains a qualified candidate, not adopted yet: its approximately 0.12 us improvement is too small to establish a robust performance gain from one timing sample.

v176 batches four unconditional in-range mapping loads before consuming their results, avoiding predicated tail loads. Full base audit passes; posting **12.3417 us**, non-CQ **30.3189 us**. No material posting gain versus v159 (12.3544 us); not adopted.

## v178: buffer diagnostic timestamps — rejected

Keep timestamp values in scalar locals and flush them after the measured work, retaining the same functional posting interval. Base full audit passes, but posting **12.6694 us** and non-CQ **31.7883 us** regress; not adopted. Moving timestamp stores does not explain or eliminate the remaining posting overhead.

## v179 rejected; v173 selected as the next isolated baseline

v179 moves the mutable Index-reader join into the output-preparation task, before each private output DMA/WQE post, and removes the later duplicate join. Full base audit passes, but posting/full output/sync is **13.3389 us**, non-CQ **31.6728 us**. Rejected.

After base, both-launch, delayed-Q3 and ABC/nonaligned coverage passed, the isolated generator selects v173's compact SIMT-only ready words. The small one-fixture timing difference is not claimed as a robust speedup; both requested thresholds must still be verified, and the posting threshold is still failed. No production integration or hardware performance acceptance is implied.

## v180–v181: reduce SIMT argument marshalling — rejected

v180 passes seven arguments and loads immutable addresses from a bound 128-byte table inside the posting interval. Full base audit passes, but posting **13.0789 us**, non-CQ **31.3583 us** regress. v181 also zeroes private counters/error via the existing input DMA completion, eliminating the initial block barrier; full audit passes but posting **13.7822 us**, non-CQ **31.7128 us** regress. Neither is adopted. Binding tables contain only addresses/geometry, not precomputed candidates or slots.

## v182–v183: central admission and post-publication waits — rejected

v182 keeps four TopK/output workers but assigns exact bit-set union, slot allocation and all 146 WQEs to Q1's core. Base full audit passes; posting **20.0211 us**, non-CQ **38.4328 us**. This central version fails both gates and is not adopted.

v183 retains payload fences before claim/map publication but relies on existing self-and-peer polling instead of additional fences after ready stores. Base full audit passes, yet posting **12.5300 us**, non-CQ **30.8878 us** regress. No broader synchronization tests or adoption: the original publication fences remain.

## v184–v185: address-read overlap

v184 stages LRU slots and preloads remote addresses with separate warps, then writes admission records in parallel. Full base audit passes; posting **12.9844 us**, non-CQ **30.8678 us**. Rejected. All added input DMA remains inside the measured interval.

v185 issues logical-ID and mapping reads independently in output emission, preserving the prior-admission physical-slot lookup. Full base audit passes; posting **12.1956 us**, non-CQ **30.4378 us**. The 0.0433 us posting difference is insufficient evidence for adoption; the 10 us gate still fails. Baseline remains v173.

## v186: combine adjacent block rendezvous — not adopted

Remove the block rendezvous immediately before Emit; retain Emit's producer-warp mapping wait and following block rendezvous before any output reads. Full base audit passes. Posting **12.2700 us**, non-CQ **30.4011 us**, full native maximum **30.70 us**. No timing improvement versus v173 and no broader qualification; root implementation unchanged.

## v187: remote address loads during ownership filtering — rejected

Load remote addresses alongside ownership-map reads and compact their low/high words with candidate IDs. Base full audit passes; posting **12.5700 us**, non-CQ **31.0594 us**. Extra speculative reads and changed instruction scheduling do not improve the baseline. No production changes.

## v188: direct SIMT full-output stores — rejected

Emit writes all 2048 logical IDs and slots directly to GM and fences before return; remove the separate output DMA/event handoff while retaining all-output readiness and the four-core join. Full base audit passes. Posting **14.6961 us**, non-CQ **32.7228 us**, full maximum **32.74 us**. Rejected; baseline output DMA retained.

## v189–v190: exact local prefix hash — rejected

v189 builds a 4096-entry local table of lower-Q candidates (up to 3072 entries), filters ownership locally, and retains mapping publication/output joins. Larger prefixes use atomic minimum ownership after earlier mappings publish. Base audit passes; posting **16.0639 us**, non-CQ **33.9522 us**. A CPU semantic check passes 1000 mixed-mode interleavings; it is not a device synchronization qualification.

v190 removes unused claim-ready publication in that protocol. Base audit PASS; posting **16.3678 us**, non-CQ **34.4722 us**. Neither replaces v173. Full sources, logs, reports and hashes are archived.

## v191–v193: output ownership and measurement controls

v191 assigns every output position to the same thread across preparation/emission, removes cross-warp handoff barriers and uses warp-uniform prefix-map readiness polling. Base audit PASS; posting **14.9544 us**, non-CQ **33.4550 us**. v192 keeps that ownership but restores one polling warp plus a block rendezvous: base PASS; **12.9994 / 31.1639 us**. Neither improves v173.

v193 runs the exact v173 executable in mode 0 (detailed stamps disabled); base full audit PASS. Native conditional-branch PCs at the original stamp3/stamp8 boundaries give **12.6006 us** versus **12.2522 us** with detailed stamps. The native boundaries precede original clock reads by up to **0.025001 us** (an initial 0.02 us validation bound was disproved; all exact offsets are recorded). Full native maximum is **29.79 us**, including the modeled CQ work/wait. No fixed 5 us is subtracted. This does not rescue the 10 us gate or imply a robust speed difference from one sample. See native_boundary_report.json and the identical-executable hash.

## v194: full-quota bitmap export on v173 — not adopted

Reuse v147's full-quota direct ID compaction and suppress unused score loads in that template specialization. The condition remains rank>=2048 and quota>=2048; bitmap exclusion and full TopK IDs are retained. Base full audit PASS. Q4 TopK falls from **9.1789 to 8.2644 us**, but posting is **12.4167 us**, non-CQ **30.7650 us**. Earliest Q2 preparation also starts earlier and subsequent work does not accelerate enough; the fixed complete posting interval does not improve. No production adoption.

## v195–v196: small-prefix comparison and compiler control — rejected

v195 stages at most 512 lower-Q candidates and assigns one warp per own candidate for exact parallel membership comparison. Larger prefixes/candidate sets use minimum ownership after lower-Q mappings publish. Full base audit PASS; posting **14.9172 us**, non-CQ **33.1322 us**. The extra candidate import and block rendezvous outweigh saved global ownership work. Not adopted; mixed-mode stress is not performance-qualified.

v196 changes only the v173 kernel compile option from -O2 to -O3. Base full audit PASS; posting **12.4606 us**, non-CQ **30.7828 us**. No gain, so production and generator retain -O2. Both requested performance gates retain their original meanings.

## v197–v200: overlapped ownership reads and qualification

v197 issues three independent ownership loads before ballots for candidate counts 65–96; other counts retain the original loop. Exact compact order passes 2000 CPU cases. Native trace confirms overlapping loads. Base full audit PASS, posting **12.0600 us**, non-CQ **30.2956 us**; the 10 us gate still FAILS. This is a small single-fixture simulator improvement, not a hardware claim.

v198 additionally batches remote-address/LRU reads: base PASS but **12.7656 / 31.3494 us**; rejected. v199 runs the exact v197 executable twice with retained generation flags and prior admissions; both launches independently PASS before input restoration. v200 delays Q4 (zero-based q==3) after TopK publication but before ownership work; full audit PASS. These stress runs do not supply performance acceptance. v201 ABC/nonaligned input audit also PASS. v197 now replaces v173 only in the isolated generator; regenerated source files match the measured candidate byte for byte. Production is unchanged. The posting target is still unmet.

## v202–v203: paired emission loads and empty queue control

v202 explicitly pairs the two output positions owned by each SIMT thread: issue both mapping loads, resolve both optional LRU mappings, then issue both logical-ID loads before stores. Retains exact output eligibility, errors, producer waits and fences. CPU 5000 paired cases PASS; base device audit PASS. Posting **11.7344 us**, non-CQ **30.0856 us**. Both measurements retain the original boundaries; posting still FAILS 10 us. v204 exact-executable repeated-launch qualification passes both launches independently; v205 delayed Q4 (zero-based q==3) after publication also passes. ABC/nonaligned qualification v207 also PASS. v202 now replaces v197 in the isolated generator; regenerated sources match the measured candidate byte for byte. Production unchanged.

v203 independently omits unchanged producer/doorbell writes when unique count is zero, retaining output and joins. Base PASS but **12.9844 / 31.0328 us**; rejected.

## v206: overlap logical-ID and mapping loads

Base full audit PASS. Posting **11.7344 us** (unchanged from v202), non-CQ **29.8322 us**. Retains ordinal-to-LRU fallback logical reads. Not adopted: no improvement to the binding posting gate, and this variant has only base-case validation.

## v208: single output-slot initialization — rejected

Write the final hot/ABC output slot once instead of first storing UINT32_MAX and then replacing it. Invalid and cold outputs retain sentinel writes and error handling; all barriers unchanged. Base full audit PASS, posting **12.0983 us**, non-CQ **30.4033 us**. No improvement over v202; generator unchanged.

## v209: group hot/ABC logical loads — rejected

Native v202 Q3 trace shows per-thread repeated logical loads (warp1 starts 13.2006, 13.8478, 14.4644 us; durations 0.5322, 0.5167, 0.0056 us). Overlapping warp times must not be summed. v209 computes up to three positions per output thread, batches their logical loads, then writes outputs. Base full audit PASS, posting **12.7267 us**, non-CQ **31.1011 us**. Rejected; source-level load overlap did not improve full measured interval. Exact regression attribution remains unproven.

## v210–v211: overlap complete output DMA and WQE build — rejected

Keep output slots at scratch offset 0; move WQE words to offset 7680 (4096 words, ending at 11776 within 12096-word scratch). Start complete slot/ID DMA before original WQE construction. v210 waits for MTE3 completion after the builder, including zero-WQE Qs. v211 relies on the builder's existing completion wait for nonzero Qs and retains an explicit output wait for zero Qs. Both retain all-output publication/join inside stamp3→stamp8. Internal stamp7 now follows the builder and must not be interpreted as exclusive output/preparation duration.

Base full audits PASS. v210 posting **12.7478 us**, non-CQ **31.4628 us**; v211 **13.7683 / 31.2328 us**. Neither improves the gate; not adopted. No broader correctness claim from these base-only runs.

## v212: one-batch-ahead address/LRU reads — rejected

Read next warp batch addresses and LRU slots before storing current batch records, with explicit candidate-bound guards and uniform batch iteration. Preserve ownership, publication, output and joining. Base full audit PASS; posting **13.3261 us**, non-CQ **31.7083 us**. Rejected; baseline v202 unchanged. The source-level pipeline adds control/register work and no critical-path gain is demonstrated.

## v213–v214: omit own-ready readback in current single-warp protocol — rejected

v213 waits for previous-Q map publications only in Emit, retaining own release fence/block rendezvous and full output join. v214 additionally excludes own claim-ready from prefix claim polling. This revisits v138 under the newer single-warp admission and paired-output implementation; previous-Q waits remain. Base audits PASS. Posting **11.9194 / 11.9439 us**, non-CQ **30.2189 / 30.2861 us** respectively. Neither improves v202; not adopted or broadly synchronization-qualified.

## v215–v216: identical-executable repeatability check

Both fresh base runs use the original v202 executable (SHA-256 `0c8afd9f4909e9a973efb435618e43b7155d07a6de249b2423511980067bd107`), without rebuilding. Both full audits PASS. Both posting intervals are **11.7344456 us**, non-CQ **30.0855545 us**, agreeing with v202 to trace float precision. The remaining 1.7344 us is reproducible in this simulator, not explained by run-to-run variation. This is not hardware repeatability evidence.

Initial report extraction lacked kernel sources in the fresh output folders, so phase detection rejected a negative `tail_setup`; copying the exact executable sources restored correct phase interpretation. Raw logs/traces were unchanged and no rerun was needed.

## v217: 512-thread merged admission/output task — rejected

Keep four Q cores, reduce the merged task from 1024 to 512 threads. Warp0 admission is unchanged; the remaining 480 threads prepare full output with stride 480, and paired Emit executes two 1024-position tiles. CPU coverage check confirms all 2048 positions exactly once in each pass. Base full device audit PASS. Posting **12.0400 us**, non-CQ **30.2728 us**. No gain versus v202, so retain 1024-thread baseline.

## v218: early immutable template DMA on paired-output baseline — rejected

Issue the 128-byte bound WQE template DMA beside compact candidate DMA, into existing postWork[6144]; retain its scalar completion handoff at writer consumption. Dynamic SQ producer reads stay per call. The complete posting interval still begins before both DMAs. Base full audit PASS; posting **12.5506 us**, non-CQ **30.6767 us**. No gain over v202; not adopted. Modified generated prefetch.h is archived alongside kernel sources.

## v219: scalar full-output join with separate cache lines — rejected

Replace the final 32-thread WaitFour task with existing scalar Signal/Await primitives. Four output-ready generations use separate lines flags[(5+q)*32]; previous scalar generation/TopK lines and SIMT ownership lines remain disjoint. All output completion precedes publication, and previous-Q reads invalidate cached lines. Base full audit PASS; posting **14.5722 us**, non-CQ **32.1022 us**. Rejected: scalar cache-control/polling is slower in this layout. No wider synchronization acceptance is claimed.

## v220: bypass prefix waits for zero input candidates — rejected

Only zero pre-dedup terminal candidates bypass claim-prefix/map-prefix waits. Qs with candidates but zero unique posts retain waits. Full output, publications and four-Q final join remain. Base full audit PASS, posting **12.2678 us**, non-CQ **30.6311 us**. Not adopted: no complete-interval improvement; no broad zero-candidate synchronization qualification claimed.

## v221: combine full-quota export with paired output — rejected

Apply the exact v194 ranked_candidates.h optimization to v202. Full bitmap exclusions remain; direct ID compaction is limited to full-rank/full-quota terminal export. Base full audit PASS; posting **12.0056 us**, non-CQ **30.3078 us**. The combination does not improve v202. Generator unchanged; modified header archived.

## v222: trim merged-task argument list — rejected

Pass only the current Q candidate count; remove unused width, bitmap and journal arguments from PackUniqueLocal. No replacement pointer table, extra loads, ownership or synchronization changes. Base full audit PASS. Posting **13.3028 us**, non-CQ **31.9422 us**. Rejected; fewer source-level arguments do not improve this compiled path.

### v222 compiler/native attribution follow-up

Compare Q3 using each run's own stage boundaries, not a common absolute time range. v202 TopK **8.7922 us**, dedup/output **6.9189 us**; v222 **9.1467 / 6.4372 us**. X30-relative scalar load/store counts in TopK are **222 vs 206**, and in dedup/output **18 vs 20**. Thus the reduced-argument task itself is faster, but surrounding work/arrival times regress; increased stack spilling is **not established** by these traces. An earlier fixed 2.8–11 us instruction window mixed different phases and must not support that claim. See compiler_audit.json. llvm-objdump reports unavailable instruction decoding for these device objects; native simulator events remain the instruction evidence.

## v223: reuse scalar doorbell address — rejected

Read entity[8] once into a scalar local for both doorbell store and cache cleaning. All dynamic queue reads, cache cleaning, full output and four-Q join remain. Base full audit PASS; complete posting **11.7650 us**, non-CQ **30.2483 us**. No improvement over v202 **11.7344 / 30.0856 us**; generator remains unchanged. A source-level repeated read cannot be treated as additive critical-path savings. No broader synchronization qualification is claimed for this rejected variant.

## v224: block fence at UB-only Emit return — not adopted

Replace only Emit final asc_threadfence with asc_threadfence_block. Preserve all ownership/map release fences, VF/output DMA completion waits, and four-Q full-output join. Full base audit PASS. Complete posting **11.7344456 us**, non-CQ **30.0855545 us**, identical to v202 repeat results to native trace precision. No measured benefit; retain v202. This removes the final Emit fence scope as a demonstrated explanation for the remaining 1.73 us; it does not prove other fences are free or removable.

## v225–v226: paired-output scheduling follow-ups

v225 revisits v186 on current v202: omit the block barrier immediately before Emit; its producer wait and subsequent block barrier remain. Base audit PASS; posting **11.9917 us**, non-CQ **30.0167 us**. Posting regresses; not adopted.

v226 preloads logical IDs for cold candidates in the hot/ABC preparation warps and stores them in disjoint scratch words [6144,8192). Emit uses the cached value only for direct physical mappings; prior ordinal mappings retain LRU/physical-logical lookup. Base full audit PASS, posting **11.7156 us**, non-CQ **29.9622 us**. The 0.0189 us posting improvement does not close the 1.7156 us deficit and is insufficient to justify adopting an unqualified variant. Additional scratch and load traffic are not integrated. No repeated-launch/delayed-producer/ABC qualification claimed for either variant. Baseline remains v202.

## v227: defer output error reporting — rejected

Keep output error detection and capture its UB result before scratch reuse, but combine GM reporting with Index status after update. Base full audit PASS. Posting **13.4217 us**, non-CQ **31.5022 us**; regression, not adopted. Native baseline Q3 line95 scalar reporting occupies about 0.03 us and overlaps other work; this was never evidence for a 1.73 us saving. Instruction evidence archived in baseline_error_write_audit.json. Production and generator remain unchanged.

## v228: exact-length DMA change records — rejected

Replace per-candidate SIMT changes/changeSlots stores with two DataCopyPad transfers from compact UB IDs and slots. Copy exactly own*4 bytes to each reserved range; skip zero count, retain mapping stores/release and wait for all output/record DMAs before publishing output readiness. Full base audit PASS. Posting **13.7367 us**, non-CQ **31.7722 us**; rejected. This complete-path result does not justify replacing the original stores. No broad boundary, repeated-launch, or multi-request qualification is claimed. Generator and production unchanged.

## v229: shared claim/map phase word — rejected

Encode claim-ready as 2*generation and map-ready as 2*generation+1 in the same SIMT-only word. Retain both publication fences and the original full-output join. 1620 abstract prefix-predicate checks PASS for generations up to 0x7fffffff; larger generations are outside this experimental encoding and would require a rollover design before integration. Base full device audit PASS. Posting **11.8133 us**, non-CQ **30.0367 us**. No posting improvement; not adopted and no broader memory-order qualification claimed. Original full-width generation protocol remains unchanged.

## v230: batch ownership and remote reads together — rejected

Apply early remote-address reads to the v202 specialized three-batch ownership path; compact address halves alongside token IDs and remove the later remote reads. Generic counts retain exact bounds; no extra barrier or timing-boundary change. Full base audit PASS. Posting **12.3161 us**, non-CQ **30.4989 us**. Not adopted: complete interval does not improve v202. No broad regression qualification claimed.

## v231: one remote-preload warp, original record writer

Warp1 preloads remote addresses while warp0 claims and compacts ownership; remaining 30 warps prepare hot/ABC output. Save compact original positions in postWork[6144..8192), remote64 in scratch[6144..10240); full-block rendezvous precedes consumption. Original warp0 still assigns/writes records and publishes mapping readiness. Base full audit PASS; posting **11.8244 us**, non-CQ **29.9917 us**. No <=10 us result or broader qualification; baseline unchanged.

## v232/v233: expose fixed ABC-boundary assumption

New case11 uses ABC capacity384, begin=history-333 and length123, retaining 146 unique cold reads. v233 runs unchanged v202 kernel with this driver: WQE/TopK/count/Index audits pass but **4 output slots fail**. v232 replaces hardcoded output ABC calculation with existing DirectTailSlot(context,0,resident,token): every audit passes, **mode=11 posted=146 bad=0**. This is an isolated single-request correctness repair, not production integration or a <=10 us performance claim. Original base performance and repeated-launch qualification pending.

## v234: original fixture with v232 executable

Exact v232 executable, mode4 original fixture, complete audit PASS. Posting **13.1378 us**, non-CQ **31.1661 us**. General ABC helper fixes demonstrated nondefault output error but adds cost; not yet adopted or synchronization-qualified. The v202 11.7344 us baseline is valid only for its tested fixed/no-ABC fixtures and must not be presented as general production readiness. Next optimization must preserve runtime ABC bounds/capacity and reduce repeated context loads instead of reinstating hardcoded geometry.

## v235/v236: one-thread runtime ABC cache — rejected scheduling

Thread0 reads ABC bounds/capacity into unused counter[2..4], shared after the existing initial block barrier. Reuses DirectTailSlot overload; no hardcoded bounds. Nondefault ABC case11 full audit PASS (v235). Exact same executable on original mode4 full audit PASS (v236), posting **13.4306 us**, non-CQ **31.4561 us**, worse than direct runtime-helper v234 **13.1378/31.1661 us**. Caching before the initial barrier did not recover cost; do not adopt this schedule. Runtime-parameter correctness requirement remains; fixed-geometry v202 is not a general integration candidate.

## v237/v238: scalar runtime ABC arguments — candidate

Read actual tail bounds and output capacity once after stamp3, pass three values into PackUniqueLocal, and reuse DirectTailSlot. No hardcoded geometry or timing exclusion. Same executable: nondefault case11 full audit PASS (v237); original mode4 full audit PASS (v238), posting **12.4444 us**, non-CQ **30.4983 us**. Improves direct runtime-helper v234 by ~0.6933 us posting, but still misses 10 us. Repeated-launch qualification pending in v239; not adopted yet.

## v240: packed scalar ABC range — candidate

Pack actual begin/end into one uint64 argument, keeping capacity as a separate runtime value. Unpack within existing SIMT task; reads remain after stamp3 and no synchronization changes. Original mode4 full audit PASS, posting **12.1061 us**, non-CQ **30.2544 us**. Nondefault ABC qualification is pending in v241; repeated-launch qualification not yet performed on this executable. Do not claim adoption or <=10 us.

## v239: v237 consecutive-launch qualification and adoption

The exact v237 executable passes mode9: both launches independently report posted=146 bad=0 before restoration, with verified_launch=1 and 2. This is the existing retained-generation/prior-mapping fixture, not an evolving real decode trace. Adopt v237 kernel/parallel_post/driver into the isolated generator and expose case11 through run.py. Production unchanged. The nondefault ABC repair is retained despite its cost; v240 remains an unqualified candidate.

## v241: v240 nondefault ABC qualification

Exact v240 executable passes case11, posted=146 bad=0; WQE, TopK, counts, all output slots/logical IDs, Index and error audits pass. Candidate still requires repeated-launch and delayed-producer qualification; current generator remains v237.

### Delay-label evidence correction

Source audit found v200/v205 used q==3, i.e. Q4, despite earlier prose saying Q3. Their successful audits remain valid for delayed Q4 only. v175 uses q==2 and truly delays Q3. v243 explicitly tests q==2 after TopK publication and before ownership; its complete audit passes (posted=146 bad=0).

## v243: delayed Q3 qualification for v240

Stress variant delays actual Q3 (q==2) after TopK publication and before ownership for 36000 SYS_CNT ticks. Full audit PASS, mode5 posted=146 bad=0; this is correctness evidence only, not a timing result. No synchronization was removed. v242 exact-executable consecutive-launch audit subsequently passed both launches independently; see evidence/v242/audit.json.

## v242: v240 consecutive-launch qualification and adoption

Exact v240 executable passes both mode9 launches independently before input restoration, with both verified_launch markers. Adopt v240 into the isolated generator after v241 nondefault ABC and v243 delayed Q3 also pass. Runtime ABC parameters stay inside the original posting interval. This replaces v237 **12.4444/30.4983 us** with **12.1061/30.2544 us**, but does not satisfy the posting gate or establish production/multi-request correctness.

## v244 / v245 runtime ABC arithmetic experiments

Both original-fixture mode4 full audits passed (146 unique admissions).
v244 runtime power-of-two fast remainder: posting12.0417us, non-CQ30.3161us.
v245 scalar base remainder with bounded per-token wrap: posting12.8689us, non-CQ30.7567us.
Neither meets10us; v244 saves only0.0644us posting and slightly worsens total.
Neither adopted or qualified on alternate ABC/repeated launch/staggered producer cases.
Current generator stays v240. A100000-case host arithmetic equivalence check passed;
this does not replace device correctness qualification.

## v246 / v247 full-output rendezvous experiments

v246 uses a uint64 cumulative arrival counter with target4*generation, retaining
release ordering and the complete-output gate. Base full auditPASS; posting12.0661us,
non-CQ30.1711us. v247 puts SIMT ready words on distinct cache lines; base full auditPASS,
posting12.0433us, non-CQ30.3306us. Neither reaches10us; neither is adopted or qualified
on repeated launches/delayed producers. Current generator remains v240.

## v248 / v249 admission and queue preparation overlap

v248 partitions address/LRU/record work across three warps after warp0 finishes
ownership and slot allocation. UB handoff markers and release fences retain ordering.
Base full auditPASS; posting11.8072us, non-CQ29.9450us. Broader qualification pending.
v249 reads the immutable WQE template and dynamic64-bit SQ head in the first hot-output
warp, then reuses the original vector WQE constructor and all output/post joins.
Base full auditPASS; posting11.8683us, non-CQ30.2611us. Neither adopted yet.

## v250 / v251 combined preparation and earlier join

v250 combines v248 admission writers with v249 queue/template preload; helper
warps execute a block-scope fence after their ready poll. Base full auditPASS:
posting11.6294us, non-CQ30.1111us. Broader qualification is pending; not yet adopted.
v251 keeps v240 preparation but moves the mutable-Index-reader join inside it;
private output DMA and WQE posting remain measured through stamp8. Base full
auditPASS, posting12.7222us, non-CQ30.8622us. Rejected; earlier join is slower.

## v252 output DMA overlap / v253 ABC qualification

v252 relocates complete output slots to TerminalWork[7168,9216), keeping the
original WQE bank. Output DMA overlaps WQE construction; builder completion
(or explicit empty-post completion) precedes doorbell/output join. Base auditPASS;
posting12.1894us, non-CQ30.4806us. Rejected. Phase names distinguish DMA issue
from completion; full gate unchanged. v253 exact v250 executable case11 auditPASS.

## v254–v256 qualification and adoption

v254: exact v250 executable, both launches independently verified before input
restoration, generations retained. v255: actual Q3 delayed after TopK publication
before ownership by36000 SYS_CNT ticks; full auditPASS. v256 keeps the same device
sources, adds a stronger host WQE audit and case12 with heads above2^32 near SQ
wrap boundaries, varied nonzero entity fields, full-width producer/doorbell checks,
and every immutable WQE field compared against its bound template. AuditPASS.
Previous audits covered WQE control/length/source/destination; do not describe
those as checks of every immutable field. v250 is now the isolated device baseline;
v256 host audit is retained. The10us goal and production integration remain open.

## v257 / v258 queue publication experiments

v257 merges SIMT queue/doorbell release and full-output rendezvous. Post clocks
return through UB and match native SIMT clock instructions against SYS_CNT
calibration; no CQ estimate is substituted. Base auditPASS; posting12.1339us,
non-CQ30.5528us. v258 writes only the8-byte producer field through DMA before
scalar doorbell release; base auditPASS, posting12.0317us, non-CQ30.2811us.
Both regress versus v250 and are rejected; no transport/extra-case qualification.
The report now validates SIMT clock evidence and distinguishes ready release.

## Additional rejected variants v259–v261

Same base fixture; each completed audit reported `mode=4 posted=146 bad=0`.

| Variant | Change | Posting including output/join us | Non-CQ tail us |
|---|---|---:|---:|
| v259 | Scalar UB counter initialization before SIMT | 12.3717 | 30.1700 |
| v260 | Vector UB counter initialization before SIMT | 12.0433 | 30.4322 |
| v261 | Consume retained compact UB candidates, retaining GM export | 12.4344 | 30.3928 |

None replaces v250. All initialization remains inside the posting interval.

## v262 measured candidate and qualification

Direct compact-UB consumption plus omission of terminal-only compact GM export: posting including output/join **10.9844 us**, non-CQ **29.2167 us**. Intermediate exports and full logical/slot output remain unchanged. Original attempt terminated with simulator signal11 and no profiling data; unchanged binary retry completed `mode=4 posted=146 bad=0`. Failure log is retained. Qualification completed as recorded at the top; it does not meet 10 us.

The v267 local-barrier removal passed the base audit but posting increased to 11.0917 us; it is not adopted.

## v268/v269 hot-output load scheduling — rejected

v268 unrolls the existing hot/ABC loop; v269 separates three positions classification, logical loads, and UB writes.
v268: 11.5322 us posting, 30.0056 us non-CQ; base audit PASS.
v269: 11.6700 us posting, 29.9017 us non-CQ; base audit PASS.
Neither improves the complete posting gate; generator remains v262.

## v270/v271 overlap change records with full output — rejected

Stage unique IDs back into dead compact UB after ownership. Mapping release stays before output reads; changes/changeSlots stores move inside Emit, before its loads (v270) or after paired logical loads (v271). All records complete before full-output publication.
v270: posting 11.4139 us, non-CQ 29.7789 us; PASS.
v271: posting 11.0794 us, non-CQ 29.6589 us; PASS.
No improvement; not adopted.

## v272 full-output join inside Index task — rejected

Move the unchanged four-Q release/acquire join into UpdateIndex, followed by its block rendezvous. Read total slot count and partition changes after that join on device. Capture the join boundary with thread0 clock64 in disjoint UB and copy the clock back only after VF completion. The archived parser matches that active-warp native clock against SYS_CNT calibration; inactive predicated clock events are excluded. Full base audit PASS; posting **11.8850 us**, non-CQ **34.0428 us**. Both regress versus v262, so neither device change nor experimental parser is adopted.

## v273/v274 address prefetch and distributed ownership claims

v273 preloads raw candidate remote addresses in a spare warp, stages original indices during ownership compaction, and uses explicit UB release/acquire before three writer warps consume them. v274 instead distributes atomic-min claims across the existing three writer warps; helper global fences and UB completion flags precede per-Q claim publication.
v273: posting 11.1317 us, non-CQ 29.5772 us; PASS.
v274: posting 10.8417 us, non-CQ 29.3589 us; PASS.
v273 is rejected. v274 is a small posting candidate, not adopted pending qualification; no <=10 us result. Host coverage verifies each candidate exactly once for counts 0..2048, not memory ordering.

## v275 combination / v276 delayed producer

v275 combines three-warp claims with speculative address preload: full audit PASS, posting 10.8589 us and non-CQ 29.1117 us. Posting is slightly worse than simpler v274, so combination is not adopted. v276 delays actual Q3 after TopK publication and before ownership by 36000 ticks on v274 source; complete audit PASS. Other v274 qualification remains pending.

## v277 per-Q compile-time specialization — rejected

Instantiate the same admission/output code for each constant Q, keeping all runtime inputs, sync and full output. Complete base audit PASS; posting 13.4244 us, non-CQ 32.0061 us. Larger specialized code does not improve the full path; not adopted.

## v274 qualification and isolated adoption

v278 exact-executable case9 verifies both launches before restoration; v279 case11 and v280 case12 pass. Together with delayed Q3 v276, adopt v274 parallel_post.h in the isolated generator. No production ABI or serving library changes.

## v281–v283 boundary-preserving follow-ups — rejected

v281 moves the all-mutable-reader join inside Pack before private output DMA/WQE posting. v282 publishes reader completion there but acquires it later through scalar read-only invalidation/polling after posting. Both retain full output and actual post before stamp8. v283 removes eight internal diagnostic timestamps at compile time while retaining original stamp3/stamp8 and CQ timestamps; it changes code layout, so is not a controlled measurement of instrumentation overhead.
v281: posting 11.8922 us, non-CQ 30.3828 us; PASS.
v282: posting 13.9106 us, non-CQ 32.1744 us; PASS.
v283: posting 11.9156 us, non-CQ 30.2922 us; PASS.
None improves v274. Runtime-switchable same-binary instrumentation A/B is prepared separately to isolate that effect.

## v284/v285 same-binary instrumentation control — not adopted

One compiled executable switches runtime mode14 (original stamp3/stamp8, actual post and exposed-CQ stamps only) versus mode4 (all original detail stamps). All full output/Index/WQE audits pass in both. Source input geometry and selected candidates match; actual binary SHA is recorded in both audit.json files. Minimal mode: posting **12.5683 us**, non-CQ **30.3233 us**. Detailed mode: posting **12.9350 us**, non-CQ **31.0733 us**. Within this executable detail stamps add ~0.3667 us to posting. The additional runtime measurement control changes compiled code and both modes are slower than v274. Do not subtract this delta from a different binary or call it a production optimization. Original parser intentionally marks minimal mode as lacking detailed stamps; its raw identical-boundary posting and CQ intervals remain in report.json. No new performance gate is accepted from this experiment.

## v286/v287 Q1 dependency and small-candidate specialization — rejected

v286 skips Q1 ownership claims, filters prior mappings and publishes Q1 claim readiness only after physical mapping writes; subsequent Qs retain their prefix dependency. Base audit PASS, posting 12.0706 us, non-CQ 30.4222 us. This lengthens the later-Q dependency. v287 guards a <=96 candidate specialized loop with a generic fallback; base audit PASS, posting 12.2633 us, non-CQ 30.6617 us. Neither is adopted; the specialized generic fallback was not separately qualified because its base performance already regresses.

## v288–v291 local control overhead — rejected

v288 avoids self-ready GM reads after own release fences: posting 11.1200 us, non-CQ 29.8278 us. v289 passes one candidate count instead of dynamically indexing four counts: 11.5583 / 29.7167 us. v290 skips empty-candidate template loads and zero-admission SQ/doorbell stores, retaining complete output/join: 11.0967 / 29.4683 us. v291 parallelizes initial UB control-word zeroing within the first warp and retains its block barrier: 11.0328 / 29.8633 us. All base audits PASS; none improves the complete posting gate versus v274, so none is adopted or further qualified.

## v292 queue/doorbell preload — qualified candidate, not adopted

Preload queue and doorbell addresses alongside the existing template and 64-bit SQ head in a helper warp; scalar submission consumes the UB addresses after VF completion. WQE construction, full output, cache cleaning and four-core join are unchanged. Base audit PASS: posting 10.7056 us; non-CQ 29.4628 us. Same binary passes case9 twice (v293), nonzero 64-bit SQ head/ring wrap (v294), and nondefault ABC (v296). Compared with v274, posting interval falls 0.1361 us but non-CQ rises 0.1039 us. Latest output/join stamp rises from 21.3456 to 21.4939 us while earliest prep stamp shifts from 10.5039 to 10.7883 us. Consequently this is not an overall critical-path improvement or a <=10 us acceptance. Keep v274 generator unchanged pending a clear net improvement.

## v295/v297 scalar address reads and compiler control — rejected

v295 consumes cached head/queue/doorbell as aligned uint64 UB values instead of pairs of uint32 values. Base audit PASS: posting 11.5978 us, non-CQ 29.8394 us. v297 compiles unchanged v274 device source with -O3 instead of -O2: base audit PASS, posting 11.1628 us, non-CQ 29.9522 us. Neither is adopted; production compiler options and isolated v274 baseline remain unchanged. All v286–v297 jobs have finished; no background simulation is left from this series.

## v298/v299 WQE construction in the admission VF — rejected

Reuse BatchBuild<true>'s 16-field wire layout, with its template already in UB, staging <=256 WQEs in a separate scratch bank. v298 constructs after Emit, v299 before Emit; both retain full outputs, SQ DMA/wrap, scalar producer/doorbell and the same join. Larger counts retain the original vector builder. Base audits PASS: v298 posting 11.0828 us, non-CQ 29.9428 us; v299 posting 11.5683 us, non-CQ 30.7328 us. Neither improves v274; not adopted. Larger-count fallback and wrap were not additionally qualified for these rejected performance candidates.

## v300/v301 complete WQEs written directly to SQ

Reuse the 16-field layout with the UB template and direct cache-bypassing SQ stores, retaining a global fence before VF completion/scalar producer and doorbell updates. Counts >256 use the original vector path. v300 accidentally retains two final fences: base audit PASS, posting 10.8772 us, non-CQ 29.5983 us. v301 uses one final fence: PASS, posting 10.7628 us, non-CQ 29.7339 us. The small interval gain is not a net complete-tail gain, no <=10 us result; neither adopted. Real NIC visibility, wrap, and large-count fallback need separate validation before any adoption. v302 was not run.

## v303/v304 overlap direct SQ construction with output warps — rejected

v303 has output warps wait on explicit UB writer/template releases before building full WQEs directly in SQ, before the existing full block/output phases. Complete base audit PASS, posting 13.8122 us, non-CQ 32.6233 us. v304 uses own GM map-ready plus the UB template release instead: the live process was observed until termination, not restarted; final audit PASS but posting 47.0222 us and non-CQ 66.1517 us. The broad GM polling variant is a severe regression, not a hang. Both remain isolated evidence and are not adopted.

## v306 cumulative ownership-claim publication — rejected

Starting from v274, each Q waits for the preceding Q's cumulative claim-ready before publishing its own. Payload/fences and full outputs remain intact. Complete base audit PASS; posting 11.5250 us, non-CQ 29.6372 us. The shortened read set does not beat the original prefix publication. Not adopted. v305 was not run. All jobs in this series are terminal; generator and production remain unchanged.

## v307/v308 read-only LRU staging — rejected

Bulk-copy the full 2048-entry LRU into disjoint scratch words [2048,4096), and consume it for admission/output lookup. Copy/fence are inside the unchanged posting gate. v307 base PASS, posting 12.7517 us, non-CQ 31.0428 us. v308 combines this with v275 remote-address preload: PASS, 11.9700 / 30.3928 us. Neither adopted.

## v309/v310 producer-state writeback scheduling

Keep the full SQ writes, producer store, doorbell store and doorbell cache writeback. Move only producer cache writeback later, retaining it in the complete non-CQ tail. v309 after modeled CQ: PASS, posting 10.9956 us, non-CQ 29.7128 us (rejected). v310 immediately after full-output join, before Index/CQ: PASS, posting 10.4111 us, non-CQ 29.3067 us. v310 is a pending candidate, not adopted or <=10 us accepted. This is justified only with exclusive queue ownership and no consumer of software producer state before the later commit; production CQ/transport ownership must be checked before integration.

## v311–v314 persistent queue verification

v311 moves both the producer scalar store and cache writeback after the output join: base PASS, posting 10.4122 us, non-CQ 29.6439 us. This does not improve v310. v312 runs this exact executable twice without resetting SQ heads, SQ contents or doorbells, starting at nonzero 64-bit heads and crossing ring boundaries: both complete audits PASS. v313 applies the same stronger host case15 to v310's unchanged measured device archive (host-only relink; hashes retained): both complete audits PASS. Input Index/score state is restored between launches; this is persistent transport-state coverage, not evolving decode. v314 preloads only the doorbell address on top of v310: base PASS, posting 10.8972 us, non-CQ 29.5522 us; rejected.

## v316 merged error publication — rejected

Retain classification error in a scalar until Index update finishes, then OR classification and Index errors into the original status once. No checks removed. Base complete audit PASS, but posting 11.7428 us and non-CQ 29.8744 us regress versus v310. Not adopted; fault-injection qualification was not run for this rejected candidate. v317 verifies nondefault ABC on v310's unchanged device archive: full audit PASS.

## v315 error-status scheduling — rejected

Move classification status publication after output join but before Index work, preserving checks. Base full audit PASS; posting 10.9522 us, non-CQ 29.2289 us. Posting regresses versus v310, so not adopted. v318 delayed Q4 and v319 same-device-archive repeat complete successfully; current generator retains v310. No live jobs remain from this series.

## v320/v321 status and event retirement — rejected

v320 replaces the first per-Q classification-error OR with assignment (the field is initialized to zero and has no intervening writer). Base full audit PASS; posting 10.9611 us, non-CQ 29.4067 us. No performance benefit; fault injection not run and variant not adopted. v321 keeps the original classification publication and moves TopK event-token retirement after the output join, retaining every wait in the full tail. Base full audit PASS; posting 10.7494 us, non-CQ 29.6328 us. Also rejected. Generator remains v310; both jobs terminal.

## v322/v323 polling and partial-tile reads — rejected

v322 changes helper-warp readiness polling from all-lane UB reads plus ballot to lane0 UB load plus uniform shuffle, retaining the acquire block fence. Base audit PASS; posting 10.8444 us, non-CQ 29.4161 us. v323 predicates the last mapping load in the 65–96-candidate compaction tile, retaining exact masks and ordering. Base PASS; posting 10.8000 us, non-CQ 29.5756 us. Neither improves v310; neither adopted or additionally stress-qualified.

## v324 per-Q candidate-count ABI — rejected

Pass only this Q candidate count into PackUniqueLocal and remove its four-count private array/dynamic indexing. Candidate work and all outputs unchanged. Base complete audit PASS; posting 11.3283 us, non-CQ 29.7722 us. No benefit, not adopted. All v322–v324 jobs finished; generator remains v310.

## v325/v326 WQE dependency waits — rejected

v325 retains between-bank MTE3_V and final MTE3_S but omits the redundant last-bank MTE3_V before MTE3_S. Base full audit PASS; posting 10.8311 us, non-CQ 29.4150 us. v326 omits terminal writer S_V because its UB inputs were produced by the completed VF and scalar only reads them. Base full audit PASS; posting 10.6972 us, non-CQ 29.5294 us. Neither improves v310, so both remain unadopted; no multi-bank/ring stress acceptance claimed for these rejected candidates.

## v327 single-bank WQE control path — candidate only

For count<=bankSize, invoke the unchanged FinalBuild and exact SQ wrap copies once without the loop; retain the original multi-bank path otherwise. All fences unchanged. Base audit PASS; posting 10.3811 us, non-CQ 29.4422 us. Posting improves only 0.0300 us versus v310 while full non-CQ tail worsens 0.1356 us. Not adopted and not <=10 us; multi-bank and ring qualification remains unproven. All v325–v327 jobs terminal.

## v328 combined single-bank waits — rejected; v329 order regression

v328 combines v327 single-bank control with omission of S_V and single-bank MTE3_V, retaining MTE3_S before the doorbell and the unchanged multi-bank loop. Base full audit PASS, but posting 11.4967 us and non-CQ 30.4922 us regress. Not adopted. v329 uses unchanged v310 device sources and adds mode16: per-Q candidates 94/87/0/11, starts 11/59/0/0, preserving union146 and prior84. All WQE, TopK, output, Index, count and error audits PASS. This is a different ordering correctness test, not a replacement performance acceptance. Mode16 is retained in the host driver; CLI now exposes both persistent-queue mode15 and ordering mode16. All jobs terminal.

## v330 complete-output rendezvous before WQE — rejected

Move the same WaitFour VF after complete logical/slot DMA but before WQE construction. Keep stamp8 after actual posting/event retirement and retain all output and global-read completion before Index mutation. Base full audit PASS; posting 12.5400 us, non-CQ 31.2122 us. The early rendezvous delays posting and does not improve the complete gate; not adopted or additionally stress-qualified. Job terminal; generator remains v310.

## v331/v332 intermediate thread counts — rejected

Keep the three ownership/write warps, use 768 threads (672 hot-output threads) or 896 threads (800 hot-output threads), and three guarded Emit positions per thread. Host enumeration proves exactly-once coverage of all 2048 hot-preparation/output positions. Both complete device audits PASS. v331 posting 11.0589 us, non-CQ 30.1083 us; v332 posting 11.0167 us, non-CQ 29.7983 us. Neither improves v310. No delayed-producer/repeated-launch qualification claimed; both unadopted and jobs terminal.

## v333/v334 completed-position early exit — rejected

After Emit mapping-ready/block synchronization, v333 returns a thread early when both full-output positions are already prepared; v334 exits only when the entire warp has no unresolved positions. Both preserve the final visibility fence on every path and complete DMA/join. Base full audits PASS. v333 posting 10.8217 us, non-CQ 29.7761 us; v334 posting 10.6700 us, non-CQ 29.5322 us. Neither beats v310; not adopted or further qualified. Jobs terminal.

## v335 defer Index-only records — rejected

Native source attribution from v319 is archived as source_line_ranges.json; ranges overlap across warps/pipelines and must not be summed. v335 keeps physical mapping publication on the posting path, stores unique IDs in disjoint scratch [8192,10240), and publishes changes/changeSlots at the start of UpdateIndex after posting/output join. Four-core records-ready publication precedes balanced Index consumers. All original output/Index/WQE audits PASS. Posting 12.0456 us and non-CQ 32.2128 us regress; not adopted or additionally stress-qualified. Job terminal, v310 retained.

## v336/v337 consolidated pointer handoff — rejected

Replace nine GM-pointer parameters with Context/Args and stage pointers in disjoint scratch counter[32..52) before the existing initialization barrier. v336 still loads row addresses per thread; v337 stages that pointer with a two-hop load too. Full base audits PASS. v336 posting 22.4467 us/non-CQ 40.5228 us; v337 15.8467/34.4028 us. Concentrating pointer preparation inside the VF severely regresses; neither adopted.

v338 selects the source field address per lane then performs one gathered load, retaining the v336 row-address reads and all dependencies. Full base audit PASS; posting 20.7767 us, non-CQ 39.5728 us. Also rejected. All v336–v338 jobs terminal, original v310 pointer calling convention retained.

## v339 logical-ID L2 retention hint — no simulated gain

Installed CANN device_functions_impl.h defines asc_ldcg as __ldg<NORMAL_FV,NON_CACHEABLE>. Change only logical-ID reads to NORMAL_LV, retaining NON_CACHEABLE L1 policy and all synchronization. Full base audit PASS; posting 10.4111118 us, non-CQ 29.3066648 us, identical to v319 within trace float precision. No measurable simulator gain; no hardware/cache-coherence generalization and not adopted. Job terminal.

## v340 single full-output traversal on current baseline — rejected

Keep three ownership/write warps and template preload but remove the early hot/ABC traversal. After mapping publication, paired Emit handles hot/ABC/cold and writes all logical IDs/slots in one traversal using actual tail bounds/capacity. Base full audit PASS; posting 11.9506 us, non-CQ 30.7189 us. Lost overlap outweighs reduced traversal in this measurement; not adopted or ABC/stress-qualified. Job terminal; v310 retained.

## v341 scalar binding reads before packing — rejected

Move queue/entity/output binding reads after stamp3 and before Pack; retain the same timing boundary, full outputs, barriers, WQE constructor, queue publication and CQ processing. Full base audit PASS. Posting 10.9950 us/non-CQ 30.2328 us; both regress from v310. Not adopted or further qualified. Job terminal; root device generator remains v310.

## v342 atomic output arrival count — candidate only

Replace four output-generation publications/reads with one monotonic arrival count. Each core completes output DMA before its release fence/atomic arrival; all cores wait for generation*4 before Index mutation. Base full audit PASS; posting 10.3339 us/non-CQ 29.1867 us. Still fails 10 us; repeated-launch/delayed-producer qualification pending. Unsigned >= is not wrap-safe at generation*4 overflow; do not integrate this variant. Root remains v310.

## v343 exact arrival count — candidate only

Use equality against generation*4 so previous-epoch arrival values cannot satisfy the wrap boundary early. Remove only the extra post-poll fence absent from original WaitFour; retain release before atomic arrival and after it, full output DMA and SIMD ballot. Base full audit PASS; posting 10.3328 us/non-CQ 29.2106 us. Root remains v310; candidate has no repeated-launch/delayed-core qualification yet and still misses 10 us. Neither v342 nor v343 is production-integrated. Jobs terminal.

## v344/v345 exact-arrival synchronization qualification

v343 unchanged executable passes both independently audited case15 launches preserving SQ/producer/doorbell state (v344), and case5 with Q4 delayed before TopK (v345). Every audit reports posted=146/bad=0. Inputs restore between case15 launches; no evolving-decode claim. No performance acceptance for these stress modes. v343 still misses 10 us; root generator remains v310 pending combined-path measurement.

## v346 exact-arrival join plus single-bank WQE path — candidate

Combine v343 synchronization with v327 single-bank WQE construction, preserving generic multi-bank path, ring split, all output and completion fences. Full base audit PASS; posting 10.2556 us/non-CQ 28.9322 us/native max28.95 us. Still fails <=10 us. Combined binary needs persistent-queue/wrap/delayed-core qualification; individual source ancestry is not that proof. Job terminal; root generator remains v310.

## v347/v348 combined-binary qualification

v346 exact executable passes both independent case15 audits with preserved SQ/head/doorbell/nonzero 64-bit wrap (v347), and delayed Q4 case5 (v348). Each audit posted146/bad0. Inputs restored between launches; not evolving decode. Performance remains 10.2556/28.9322 us from v346 base, still above posting target. Jobs terminal.

## v349 pre-arrival empty fence removal — not adopted

Retain output DMA completion and post-atomic fence, remove only the initial fence in the otherwise write-empty join VF. Base audit PASS; posting 10.2589 us/non-CQ28.8622 us. No posting improvement versus v346; not stress-qualified or adopted. v346 adopted in experiment generator after v347/v348 qualification; exact regeneration checked and 11 host tests PASS. All jobs terminal.

## v350/v351 single-lane arrival waiter — candidates

v350 uses lane0 only to increment/poll the exact arrival counter; other lanes perform no shared-memory work and VF completion waits for lane0. Both release fences and full output/post completion remain. v351 additionally checks atomic previous+1 to skip a load on last arrival. Both full base audits PASS; both posting10.1011 us/non-CQ28.2122 us/native max28.23 us. No measurable v351 gain. Repeated/delayed qualification pending for v350; no adoption or production claim, root remains v346. Jobs terminal.

## v352/v353 qualification rejects v350; v354 not accepted

v350 unchanged executable passes case15 twice (v352) but FAILS delayed-Q4 case5 (v353): total post count mismatch. Native durations first three cores24.06–26.52 us versus delayed Q446.96 us, inconsistent with the required all-reader join. Thus base10.1011 us is NOT an accepted optimization. Exact hardware/compiler cause is not yet established; do not generalize lane0 polling safety from base tests. v351 not independently delayed-qualified. v354 launches the same waiter with Dim3{1}/LAUNCH_BOUND(1), base identical10.1011/28.2122 us; no delayed qualification, not adopted. Acceptance sidecars retain rejection without altering raw measurements. Root remains qualified v34610.2556/28.9322 us. All jobs terminal.

## v355/v356 uniform broadcast waiter

Load arrival count in lane0 and broadcast with asc_shfl on every iteration; all lanes evaluate the same loop condition. v355 delayed-Q4 full audit PASS, unlike divergent v350; v356 same executable base full audit PASS, posting10.2372 us/non-CQ28.9222 us. No exact compiler root-cause claim for v350. Gain versus v346 is only0.0184 us posting; still misses10 us, repeated-launch qualification not performed, not adopted. Root remains v346. Jobs terminal.

## v357 skip untransmitted WQE scratch stores — rejected

Keep original vector field construction/transposition and exact-count SQ DMA; guard each trailing group of four StoreAlign operations (16 WQEs) by actual count. Host enumeration counts1..256 proves every transmitted word is written. Base device full audit PASS, but posting11.1083 us/non-CQ29.4239 us regresses vs v346. Fewer scratch stores do not imply critical-path improvement; conditional version not adopted or further qualified. Job terminal; root remains v346.

## v358 single-vector-batch WQE constructor — rejected

For count<=64 use a copy of original FinalBuild with j=0 and no loop; preserve all field calculations, transpose, stores, queue DMA/fences. Larger counts retain original builder. Base audits cover both <=64 and >64 branches and PASS, but posting10.7006 us/non-CQ29.7428 us regress. Not adopted or further qualified. Job terminal; root remains v346.

## v359 remove unused old TopK-ready publication — rejected

Source audit finds no consumer of flags[q*32]; current waits use claims20*32, mappings24*32 and output arrival5*32; generation[q*32+1] retained. Remove only Signal after TopK before prep stamp3. Base full audit PASS but full posting11.6644 us/non-CQ29.3022 us regress. Absolute start/end comparison saved; no claim that removing a pre-gate operation directly saves posting time. Not adopted or stress-qualified. Job terminal; v346 retained.

## v360 two candidate writer warps — rejected

Reduce candidate claims/record writers from96 to64 threads; poll one helper completion instead of two; use960 output threads. Preserve65..96 ownership compaction and all fences/joins. Host enumeration covers candidates0..2048 and every complete output position exactly once. Base full device audit PASS; posting11.0850 us/non-CQ29.9822 us regresses. Not adopted or additionally stress-qualified. Job terminal, v346 retained.

## v361 template preload on helper writer warp — rejected

Move template/head preload from first hot-output warp to third candidate writer warp after its records-ready publication. All block/output/queue barriers remain; disjoint cfg storage unchanged. Base full audit PASS but posting11.0667 us/non-CQ29.6522 us regress. Not adopted or further qualified. Job terminal; root remains v346.

## v362 skip empty queue publication — rejected

When own==0 omit producer/doorbell stores, doorbell flush and deferred producer flush; keep complete output, all arrival/Index synchronization and modeled timing. Base full audit PASS, but posting11.6756 us/non-CQ29.9183 us regresses. No wrap/persistent qualification claimed; not adopted. Job terminal; root remains v346.

## v363 uniform last-arriver fast return — not adopted

Broadcast atomic arrival previous value; if previous+1 equals generation*4 all warp lanes return together, otherwise retain original ballot waiter. Base full audit PASS; posting10.2311 us/non-CQ29.0622 us. Posting saves0.0244 us but non-CQ worsens0.1300 us and target remains unmet. Not delayed/repeated-qualified or adopted. Job terminal; root remains v346.

## v364 redistribute third output iteration — rejected

Rotate928 output lanes by736 so template warp handles only two output positions, moving third iteration to other warps. Host exact-once2048-position enumeration PASS; full device audit PASS. Posting11.0044 us/non-CQ29.8922 us regress. Not adopted or further qualified. Job terminal; root remains v346.

## v365 terminal rank arithmetic shortcut — rejected, narrow fixture only

Skip coverage/rank arithmetic at terminal epoch with rankTrace false and set2048. Equivalent for current fully covered long-history fixture; NOT generally equivalent when valid<=initial (original Rank returns start). Thus cannot integrate without boundary guard. Base full audit PASS; posting11.2778 us/non-CQ28.8478 us: slight whole-tail reduction but posting gate worsens. Not adopted or further qualified. Job terminal; root remains v346.

## v366 overlap owner GM completion with helper wait — not adopted

Move owner warp GM fence after helper UB-completion ballot, still before mapping-ready release. All fences retained; helper writes complete before their UB signal. Base full audit PASS; posting10.2444 us/non-CQ29.1522 us. Posting saves only0.0111 us but whole tail worsens0.2200 us. Not adopted or stress-qualified. Job terminal; root remains v346.

## v367 vector control initialization — rejected

Duplicate32 control/error words at postScratch[history/32], V_S wait before Pack, then omit lane0 zeroing and initial block barrier. All initialization remains after stamp3, full output and later fences unchanged. Base full audit PASS; posting11.0711 us/non-CQ30.3622 us regress. Not adopted or stress-qualified. Job terminal; root remains v346.

## v368 O3 on current combined implementation — rejected

Only device optimization option O2->O3 changes; current v346 algorithms/synchronization/input remain. Base full audit PASS; posting11.4139 us/non-CQ29.8733 us regress. This tests newer combined implementation, unlike old v297. Not adopted or stress-qualified. Job terminal; root remains v346/O2.

## v369 remove unused Pack arguments — rejected

Remove unused width/bits arguments only. Base full audit PASS; posting11.2100 us/non-CQ30.0617 us regress. Not adopted or stress-qualified; v346 remains baseline. Job terminal.

## v370 single-ballot small candidate compaction — rejected

Specialize counts<=32 to one ballot without generic loop, retaining all fences/output/join. Base audit posted146/bad0 PASS. Posting10.6483 us/non-CQ29.3622 us regress versus v346. Not adopted or stress-qualified; all jobs terminal.

## v371 uniform counter loads — rejected

Every lane reads the same arrival counter before ballot; retain exact-generation condition and release fences. Base posted146/bad0 PASS; posting10.3161 us/non-CQ29.2222 us regress versus v346. Not adopted or stress-qualified. Job terminal.

## v372 split output arrival from post-completion wait — rejected

Publish output arrival in a separate VF after complete output DMA, before WQE construction; wait for all arrivals after own post, retaining stamp8 and full gate. WQE inputs already local, no subsequent mapping reads. Base posted146/bad0 PASS; posting11.8328 us/non-CQ30.1022 us regress. Extra VF/handoff costs outweigh possible overlap in this fixture; not adopted or delayed-qualified. Job terminal; v346 remains baseline.

## v373 omit terminal intermediate-state persistence — not accepted

Keep final UB IDs/compact output and metadata updates, skip saved TopK score/ID GM writes only for terminal epoch. Full existing audit FAIL: bad6147 (TopK3/output6144), WQE/Index/error subchecks zero. Host oracle obtains selected token order from the saved prefix, so omission invalidates that oracle dependency; actual final output correctness remains unproven. No performance acceptance and no adoption. An independent input-derived final-output oracle is required, alongside next-round lifecycle verification. Job terminal; v346 retained.

## v374/v375 independent final-output oracle — candidate rejected

Host oracle builds expected token set from fixture inputs, maps each actual logical ID to its expected token, rejects duplicate/missing/foreign IDs, and checks corresponding physical slots against input mappings/LRU. It no longer derives token order from saved intermediate TopK. Original WQE/Index/error audits retained. Extracted exact oracle snippet passes host valid/permutation tests and rejects duplicate/foreign/sentinel/ambiguous IDs (output_oracle_test.cpp). v375 unchanged v346 device plus this host oracle: full PASS, posting10.2556/non-CQ28.9322 us, reproducing baseline. v374 v373 device plus same oracle: full PASS, posting11.5172/non-CQ29.5628 us; removing intermediate persistence regresses and is not adopted. This establishes only base final-output correctness, not next-round lifecycle safety or general production acceptance. All jobs terminal; root device generator remains v346.

## v376 XOR-sense output arrival — rejected

Each core atomically flips its own bit, full-warp ballot waits for all15 on odd generations or zero on even. Host enumeration of all24 arrival orders across both senses and uint32 wrap passes (requires previous kernel completion). Base full audit PASS, posting10.8456/non-CQ30.0322 us regresses. No repeated/delayed device qualification or adoption. Job terminal.

## v377 scalar atomic arrival and cache-refreshed wait — rejected

Replace output-join VF with scalar atomicAdd and exact-generation volatile polling with128-byte CleanCacheLines; full output DMA and WQE/doorbell completion precede arrival. Base full audit PASS, posting11.6956/non-CQ29.9511 us regresses. No stress qualification or adoption; scalar cache polling does not remove enough overhead. All jobs terminal; v346 retained.

## v378/v379 unchanged-v346 boundary qualification

v378 runs exact measured v346 executable in mode11: runtime ABC capacity384, nonaligned begin, length123, nonaligned query ends/high-bit logical IDs; full posted146/bad0 PASS. v379 same executable mode16 rotates candidates to94/87/0/11 while retaining union146 and prior84; full posted146/bad0 PASS. Native maxima28.62/28.54 us with detailed stamps OFF, therefore not comparable posting-gate measurements and not a claimed improvement over v346. SHA dd4248497d88e2d02e2f1151ff84f06d3e110293e99a04b2ccb1e133bac0f109 verified before launches. Both jobs terminal.

## v380 narrow Emit fence to block scope — no measured improvement

Emit writes only UB IDs/slots/error; GM mapping/change writes have preceding producer fences. Replace only its final asc_threadfence with asc_threadfence_block, retain output DMA completion and four-core join. Full base audit PASS. Posting10.2556/non-CQ28.9322 us equals v346 within printed timestamp precision; no performance gain, no new qualification/adoption. This does not prove identical generated instructions. Job terminal, v346 retained.

## v381 one slot store per hot-output item — rejected

Replace initial sentinel plus later physical-slot overwrite with a single final slot store; invalid/tagged-empty branches explicitly store sentinel. All output positions and fences retained. Base full audit posted146/bad0 PASS, posting10.3356/non-CQ29.2522 us regress versus v346. No adoption or stress qualification. Job terminal; baseline unchanged.

## v382 remove pre-allocation compaction fence — rejected

Retain block fence after compact IDs/count/base and before UB readiness release; omit earlier fence immediately before slot atomic allocation. Base full audit PASS, posting10.4800/non-CQ29.0622 us regress. Not adopted/stress-qualified.

## v383 reuse sparse covered-word gather for 128K..256K — not adopted

Route the existing 131072<history<=262144 range through the existing sparse GatherCoveredWords implementation instead of two full bitmap DMA/gather chunks. No filter/output/position/queue semantics removed. Base full audit PASS, posting11.4394/non-CQ28.3644 us. Whole tail improves0.5678 us vs v346, but required complete posting interval worsens1.1839 us; do not delay empty Q or change gate to claim acceptance. Not boundary/stress-qualified or adopted. Both jobs terminal; v346 retained.

## v384/v385 wider sparse covered-word gathering — candidates, not adopted

On v383 sparse path, v384 uses1024 threads/stride1024 instead of256/256, retaining2048 positions and all predicates. Full base audit PASS, posting11.4689/non-CQ28.0739 us. v385 explicitly pairs the two independent ID/covered-word loads before UB stores: full base audit PASS, posting11.0750/non-CQ27.8222 us. v385 reduces whole non-CQ tail1.1100 us vs v346, but complete posting gate is still0.8194 us worse and above10 us. No boundary/repeated/delayed qualification or production acceptance. Both jobs terminal; v346 retained.

## v386 explicit nonalias output/compact pointers — no measured gain

Add restrict to Pack input/fullIds/outputSlots and Emit IDs/locations only. Sources are separate incrementalIdsBuf/indicesOutBuf/TopK scratch; control/error scratch offsets are outside output range, shared ids/lo work pointers remain unrestricted. Base full audit PASS; posting10.2556/non-CQ28.9322 us identical to v346 within timestamp precision. No adoption or stress qualification. Job terminal.

## v387 terminal-only event priming/retirement — rejected

In generated isolated terminal header only, omit six score-stage SetFlag/WaitFlag pairs (Kscale, Qscale, VEC1 output ping-pong); retain three TopK pairs. No operation shifted outside gate; score-stage methods are not called by this harness. Base full audit PASS, posting10.9089/non-CQ29.8522 us regresses. Must never apply this edit to generic LI initialization used by score kernels. Not adopted or stress-qualified. Job terminal; v346 retained.

## v388 warp-all completion predicate — rejected

Replace ballot+UINT32_MAX comparison in WaitFour with asc_all on the same lane predicate, preserving release fences and exact-generation counter. Full base audit PASS, posting10.3944/non-CQ29.5222 us regresses. No delayed/repeated qualification or adoption. Job terminal; v346 retained.

## v389 distributed warp-local compaction and reservation — rejected

Three warps independently ballot fresh candidates in disjoint32-item chunks, atomically reserve local WQE offsets and global LRU ordinals, and write remote/slot/change records. Retain existing distributed claims/prefix-Q readiness, helper completion and full-output join; remove owner-to-helper compact-list handoff. Host enumeration covers indices exactly once for counts0..2048. Base full WQE/output/Index audit posted146/bad0 PASS. Posting10.8011/non-CQ29.9222 us regress versus v346; fewer handoffs do not outweigh added atomic/ready work in this fixture. No delayed/repeated qualification or adoption. Job terminal.

## v390 matched instrumentation-off native boundaries

Exact v346 executable, same base input case1 (only detailed-stamp runtime flag differs from case4). Full audit posted146/bad0 PASS. The archived native_boundary_audit.py uniquely matches original stamp3/stamp8 conditional-branch PCs across all four cores and validates enabled-clock offsets, with no clock reads when disabled. Instrumented branch span10.255555 us; disabled10.210556 us, improvement0.0450 us only. Posting still FAILS10 us. Native full duration28.12 us includes any exposed modeled CQ, already below35 without subtracting anything. No algorithm change or end-to-end hardware claim. Job terminal.

## v391 four writer warps — rejected

Expand candidate claims/payload writers from96 to128 threads, keep all three helper completion flags and shrink independent hot-output workers to896. Base full audit PASS, posting10.7772/non-CQ29.4822 us; slower than v346. No adoption or stress qualification.

## v392 conditional diagnostic count writes — instrumented baseline

Guard three diagnostic count stores with existing profile flag; actual output/WQE/Index audits unchanged. Instrumented base PASS, posting10.7600/non-CQ29.5806 us. Same-executable disabled pass v393 required; this is not an accepted optimization.

## v393 same-binary count-instrumentation-off — rejected

Exact v392 executable mode1. Full audit posted146/bad0 PASS. Native original stamp3/stamp8 branch-PC audit matches all four cores: instrumented10.7600, disabled10.6350 us. Worse than qualified v346/v390 disabled10.2106; diagnostic count gating not adopted. Both jobs terminal. No production change; goal remains unmet.

## v394 empty-candidate handoff bypass — rejected

For zero compact misses skip prefix claim wait, allocation, and payload helper rendezvous; retain own claim/mapping publications, full hot/ABC output, Emit prefix mapping wait and complete output/post join. Base full audit PASS, posting10.4778/non-CQ29.2622 us. Worse than v346; not adopted. No delayed/repeated qualification.

## v395 dedicated queue-template warp — rejected

Keep96 candidate writers; reserve warp96 for queue/template only.896 remaining threads cover all2048 hot-output positions exactly once (host enumeration). Base full audit PASS, posting10.5150/non-CQ29.3522 us. No adoption or stress qualification.

## v396 inline reader-arrival release — rejected

After Emit, block rendezvous ensures all mapping/logical reads and UB output values complete. Lane0 releases the existing arrival counter inside Pack. Later WaitFour only polls after own full output DMA, WQE and doorbell. No new VF or shifted timing endpoints. Base full audit PASS, posting10.5394/non-CQ28.9822 us, worse than v346. Not delayed/repeat-qualified or adopted. Both jobs terminal; original root generator unchanged.

## v397 event retirement at measured preparation start — rejected

Move all original FreeEventID waits immediately after stamp3, keeping them inside fixed gate. Base audit PASS, posting11.5444/non-CQ29.3444 us. Native v390 attribution shows apparent0.555us line117 instruction is scalar stack ST, not event WAIT; individual WAITs are0..0.00056us and ranges overlap. Evidence archived at v390/retirement_attribution.json. No adoption.

## v398 two-argument arrival helper — invalid

Reduce WaitFour to arrivals pointer and exact target; retain original fence/atomic/ballot source logic. Native live instruction logs on all four cores show repeated SIMT_VOTE_BALLOT with execMask1 while condition requires UINT32_MAX; no full correctness completion. Owned process56913 explicitly terminated after capturing repeated instruction evidence in stalled_ballot.json. This specialization cannot be adopted; source-level resemblance is insufficient to establish uniform warp control flow. v399 tests explicit shuffle-broadcast loop previously qualified in v355, with same minimal arguments.

## v399 two-argument uniform broadcast loop — rejected for performance

Replace v398 ballot loop with explicit lane0 load followed by whole-warp shuffle and uniform exit condition, reusing v355 source pattern. Full base audit PASS; posting10.5767/non-CQ29.1822 us. Resolves observed base stall but slower than v346; no delayed/repeat qualification or adoption. All current jobs terminal; generator remains v346.

## v400/v401 pending-output warp bitmap — rejected

Reserve dead scratch words8192..8255 for64 position masks. Hot/ABC output stays complete; only Emit warps with no unresolved position in either1024-item half return early after prefix-ready rendezvous. v400 initializes64words and atomic-ORs each pending position: base full audit PASS, posting14.0344/non-CQ32.8722 us. v401 uses uniform three-iteration hot-warp loops, one ballot and lane0 mask store per32 positions; host enumeration verifies all64 masks written exactly once. Full audit PASS, posting10.5328/non-CQ29.2022 us. Both slower than v346; no broader qualification/adoption. All jobs terminal.

## v402 last output warp publishes reader arrival — not adopted

Each warp finishes Emit and block-fences before a UB atomic increment at control+10. The last of32 warps publishes global reader arrival; later WaitFour only polls after own complete DMA/post. Avoids v396 added full-block barrier. Full base audit PASS, posting10.5606/non-CQ28.7522 us. Full tail improves0.1800us versus instrumented v346, but posting worsens0.3050us and fails10us. No delayed/repeat qualification or adoption.

## v403 scalar arrival after existing VF join — rejected

Publish global arrival using scalar atomicAdd immediately after existing V_S joining Pack. No new VF or moved gate endpoints; later WaitFour only polls. Full audit PASS, posting11.4406/non-CQ29.4994 us. Worse than baseline; not qualified/adopted. Both jobs terminal, generator remains v346.

## v404/v405 control scratch layout — rejected

v404 fixes control offset at2048 words immediately after complete slot output; terminal bitmap is dead and scalar own/error values are read before WQE scratch reuse. Full audit PASS, posting10.7422/non-CQ29.2522 us. v405 rounds original history-dependent control offset up to32words (4128 in fixture): full audit PASS, posting11.0306/non-CQ29.3783 us. Neither layout improves measured performance; no UB-bank root cause established, no adoption or expanded qualification. Both jobs terminal; root remains v346.

## v406/v407 reuse atomic-min returned owner — rejected

Store returned-owner eligibility in compact UB token high bit; skip later ownership reads for definitively earlier-owned/prior-mapped candidates. v407 additionally avoids Q1 re-read since no earlier Q can override it. Host exhaustive four-owner subset/permutation checks392 PASS; both full base device audits PASS. v406 posting19.8306/non-CQ38.4222, v407 posting20.5300/non-CQ39.6122 us. Native Q4 trace changes three SIMT_RED operations to SIMT_ATOM; longest atomic instruction grows from0.2978us to9.5517us. Ranges overlap and are not additive, but directly establish returned-value atomic penalty in this model. Neither adopted or stress-qualified; original non-returning atomic path retained. Both jobs terminal.

## v408/v409 separate only mapping-ready words — rejected

Keep packed claim words and exact atomic output-arrival join; spread only mapping-ready stores/loads by128B(v408) or64B(v409) within allocated32x32 flags. Full audits PASS. v408 posting10.6167/non-CQ29.4222; v409 posting10.5678/non-CQ29.1122 us. No evidence this sharing is a limiting bottleneck; neither adopted or broadly qualified. Both jobs terminal.

## v410/v411 smaller compiler optimization settings — unavailable

Keep v346 device source unchanged; change isolated kernel option only to-Os(v410) or-O1(v411). Both compiler backends fail with `Cmp instruction does not contain StepInst`; no simulator run/performance/correctness acceptance. Logs archived. No production/compiler setting change. All current experiments terminal, baseline unchanged.

## v412 four-argument uniform arrival helper — rejected

Remove only unused WaitFour arguments while retaining base/generation/limit arithmetic inside VF. Use v355 explicit lane0-load/shuffle whole-warp exit condition to avoid v398 divergent ballot specialization. Full base audit PASS, posting11.5872/non-CQ30.2528 us, worse than v346 and v399. No adoption or expanded qualification; argument-count reduction does not establish lower call cost in this compiler/model. Job terminal.

### v413 — existing queue-view numbering, isolated batch=1

Kept prefetch view 0 unused and assigned final Q1–Q4 to views 1–4.
Host layout proof for batches 1/2/4/8/16 gives disjoint even prefetch
views and four final views per row within 5*batch allocation. This proves
numbering only, not independent physical Jetty/CQ ownership. Production
CompletePrefetch(..., true) clears odd-view status: final ownership requires
that reset not run concurrently. No production changes made.

Simulator mode 4: correctness PASS, 146 posts; full posting/output/join
10.944999 us, native complete tail 29.60 us, exposed-CQ-adjusted
29.582779 us. Posting gate FAIL; not adopted. Original v346 retained.

### v414 — scalar completion wait for full output DMA

Replace only full logical/slot output MTE3_V Set/Wait with MTE3_S
Set/Wait. Writer retains S_V; full output completion and all-Q join
remain inside stamp3–8. Mode4 full audit PASS: posting10.730000 us,
non-CQ29.532224 us, native full29.55 us. Regresses v346; rejected,
no stress qualification. Root generator/device unchanged.

### v415 — single full-position output traversal

Remove hot/ABC traversal overlapping admission from Pack. Emit handles all
2048 positions once after mapping publication, preserving actual ABC bounds,
prior ordinal mapping, full logical/slot output, and all-Q join. Base full
audit PASS: posting12.092778 us, non-CQ31.152224 us, native31.17 us.
Rejected: removing duplicated traversal did not compensate for lost overlap
and changed generated code; no general causal split claimed. No broader
qualification. Root v346 unchanged.

### v416 — broadened baseline, four disjoint 128-miss Qs

Unchanged v346 device source; driver uses counts128/128/128/128 and
begins0/128/256/384, total512, prior84, enough LRU. Full WQE, TopK,
8192 output positions, counts, Index swaps and errors audits all zero;
mode4 posted512 bad0. Case-specific report requires512 rather than146.
Posting11.501666 us, non-CQ32.822222 us, native32.84 us. Correctness
extended to generic compaction loop; posting10us goal still FAIL.
Not a performance improvement or general production qualification.

### v417/v418 — four-chunk owner-load compaction through128

Issue four token loads, four owner loads, four ballots, then stable compact
in lane/chunk order for65..128 candidates. Larger counts use original loop.
Full audits PASS. v417 original146 fixture:10.685555us posting,29.462222
non-CQ (regression). v418 four disjoint128-miss Qs/512 total:
10.510000us posting,31.602222 non-CQ versus v41611.501666/32.822222.
Not adopted: original fixture regresses and neither reaches10us. Next
experiment preserves <=96 original specialization separately.

### v419/v420 — preserve96 specialization, add97..128 unrolled path

Full audits PASS. Original146 fixture(v419)10.789444/29.332222us
posting/non-CQ; disjoint512(v420)10.812223/32.232224us. Additional
branch does not preserve baseline timing and loses part of v418 gain.
Rejected; no stress qualification. All four jobs terminal; v346 remains
root implementation, no production changes.

### v421/v422 —128 admission threads with four-chunk compaction

Build on v417/418; four writer warps stride128,896 hot-output threads.
Claim helper flags move to control12..14, publish helpers3..5, error8
remains disjoint. Template warp128..159. Base full audits PASS.
v421 original146:10.694444us posting/29.532222non-CQ; v422
disjoint512:9.988334us posting/31.292224non-CQ, native31.31us.
First512 fixture gate pass, but only0.011666us margin, no delayed/repeated
qualification yet; original fixture fails. NOT adopted; v346 unchanged.

### v423/v424 — same v422 binary stress qualification

v423 mode15: two audited launches, each posted512 bad0, verified_launch1/2.
Persistent queue/doorbell state, nonzero64-bit counters and SQ wrap retained;
Index/scores restored between launches, not evolving decode. v424 mode5
delayed Q4: full audit posted512 bad0. Executable SHA256 recorded in
qualification.json. These establish narrow synchronization correctness,
not general performance or real transport. No adoption: low-miss gate fails.

### v425/v426 — runtime scalar dispatch between separate VF bodies

Keep v346 PackUniqueLocal and v421 PackUniqueLarge as separate VF bodies;
select large only for97..128 candidates, after stamp3. No case-based
compile-time dispatch. Full audits PASS. Original146 fixture(v425):
posting11.286110/non-CQ29.572779us;512fixture(v426):11.383889/32.522777.
Rejected: unified dispatch loses standalone gains. Cannot claim combined
10.21/9.99us performance. Root retained, all jobs terminal.

### v427/v428 — dynamic writer partition inside one VF

Use96 writers for<=96 candidates otherwise128, dynamic helper count and
hot-output stride, separated claim flags12..14/error8. Same four-chunk
compaction through128 as v421. Full audits PASS. Original146 fixture
11.694444us posting/30.512222non-CQ;512fixture10.641111/31.892222.
Rejected: internal dynamic partition fails both10us gates and does not
retain standalone gains. All jobs terminal; root v346 unchanged.

### v429/v430 — fixed128 writers with separate96/128 compaction

Keep v421 four-warp admission/template/output partition and separated
control flags, restore original three-chunk65..96 compaction before
four-chunk97..128 path. Full audits PASS. Original146 fixture:
10.774444us posting/29.682222non-CQ;512fixture10.407777/31.972222.
Does not improve v421/422; rejects duplicate compaction branch as remedy
for low-miss regression. Not adopted or stress-qualified; jobs terminal.

### v431/v432 — two hot-output positions per thread including writers

Based on v421/422.896 hot threads handle positions0..1791,128 writer
threads handle1792..2047 after their admission work. Host enumeration
proves exactly-once2048-position coverage; full device audits PASS.
Original146:posting12.301666/non-CQ30.882224us;512:10.946111/32.122222us.
Both regress; reject redistribution despite removing third hot iteration.
No stress qualification or adoption; original root unchanged.

### v433/v434 — parallel warp masks, one LRU allocation

For<=128 candidates each of4 warps ballots ownership, publishes mask and
ready in separate UB words16..23; prefix popcounts produce stable offsets.
Only lane0 reserves totalLRU once, then warps directly build payload/change
records and publish completion24..27. Original generic fallback remains.
Full base audits PASS.146:12.552222posting/31.192224non-CQ;512:
13.116111/34.352220us. Reject: parallel ownership with added handoffs
regresses both fixtures. No stress qualification, no adoption, jobs terminal.

### v435/v436 — rotate896 hot-worker positions by256

Based on fixed128-writer v421/422. Template warp handles two rather than
three hot-output positions. Host exact-once2048 coverage and full device
audits PASS. Original146 posting10.601111/non-CQ29.422222us,512
posting10.326666/non-CQ31.742224us. Small low-fixture gain vs v421 but
loses v422512 gate pass; no unified10us result. Not adopted/stress-qualified.
All jobs terminal, root v346 unchanged.

### v437/v438 — retain candidate count in QLIVector

Based on v421/422; ExportWindowCold retains count in a member and
TerminalCount returns it instead of reloading TopKHeader. GM metadata
publication remains, stamp endpoints unchanged. Full audits PASS.
Original146:11.029445posting/29.892777non-CQus;512:10.261111/31.722779.
Both regress versus v421/422. Not adopted or stress-qualified; source
headers archived, jobs terminal, root remains v346.

### v439/v440 —64-bit logical-table loads, retain low32 output

Native v421/422 inspection found hot logical SIMT_LDG instructions
lasting roughly0.63..0.66us across overlapping warps; not additive
critical-path attribution. Replace hot and Emit low32 strided loads with
uint64 loads cast to uint32; retain asc_ldcg and all reader joins. Full
audits PASS.146:10.856111posting/29.472222non-CQ;512:10.421111/31.662224.
Rejected versus v421/422; no stress qualification or production adoption.

### v441/v442 — logical-output DMA overlaps existing WQE buffer build

Slot DMA completes before scratch reuse; logical-output DMA is issued
after that wait and before WQE build. Nonempty writers retain final MTE3_S;
empty Q explicitly waits MTE3_S. No extra WQE buffer. Full audits PASS.
Stamp7 is issue point, NOT full output completion; gate3..8 retained.
146:posting11.216110/non-CQ29.377223/native29.69us;512:10.413889/31.691666/32.00.
Reject versus v421/422. Exposed CQ changes; never subtract fixed5us.
No stress qualification/adoption; jobs terminal.

### v443/v444 — existing BatchBuild wire layout fused into v421 Pack

Reuse v298 post-Emit SIMT WQE construction into scratch6144, and v299
exact ring DMA path; v421 four writers/atomic-arrival join unchanged.
Up to256 descriptors staged, larger counts retain original FinalBuild.
Full base audits PASS.146:11.402779posting/30.202777non-CQ;512:
10.328333/32.252779us. Composite does not improve v421/422 despite
one fewer VF; rejected, no stress qualification, jobs terminal.

### v445/v446 — offset complete slot output by32bytes

Based on v421/422; move only outputSlots pointer/DMA source from
scratch0 to scratch8 words, retaining aligned8192byte DMA and disjoint
control/error storage. Full audits PASS.146:11.401112posting/30.041667
non-CQ/native30.31us;512:10.181112/31.570001/native31.87. No gain
versus v421/422, no demonstrated bank-conflict root cause. Rejected,
no stress qualification/adoption, jobs terminal.

## v447–v450 synchronization placement experiments

Native simulator times; full gate remains earliest preparation start to latest complete
output/WQE/join. No production change and no candidate adopted.

| Variant | Fixture | Change | Posting us | Non-CQ us | Audit |
| --- | --- | --- | ---: | ---: | --- |
| v447 | 146 union | Only unresolved-output warps wait for prefix mapping | 11.027778 | 30.022222 | PASS |
| v448 | 512 union | Same, fixed four writer warps | 11.462222 | 32.772222 | PASS |
| v449 | 146 union | Move prefix wait before existing Pack barrier; remove redundant Emit barrier | 10.371667 | 29.162222 | PASS |
| v450 | 512 union | Same, fixed four writer warps | 10.065000 | 31.462222 | PASS |

Neither approach improves its qualified reference; both miss the 10 us posting gate.
Only base full-output/WQE/Index audits were run for these rejected candidates.

## v451–v452 disjoint WQE buffer / output DMA overlap

Move WQE scratch to postScratch[6144..10239], disjoint from full output
slots [0..2047] and control words. Retain the original FinalBuild, ring DMA,
output completion before doorbell, and four-core join.
Base 146-union: posting 11.571667 us, non-CQ 29.787778 us, native 30.10 us.
512-union: posting 10.643332 us, non-CQ 31.456668 us, native 31.77 us.
Both full host audits PASS. Neither meets the posting target; rejected.
Stamp7 is now DMA issue, not DMA completion; the complete stamp3-to-stamp8
acceptance interval is unchanged. No production or generator change adopted.

## v453–v454 invariant WQE template broadcasts

Hoist eleven invariant config broadcasts out of the FinalBuild 64-record loop,
copying registers before each destructive interleave. No field or completion-bit
calculation omitted. Full audit PASS on both fixtures.
146-union: posting 10.739445 us, non-CQ 29.062779 us, native 29.08 us.
512-union: posting 10.175000 us, non-CQ 31.012777 us, native 31.03 us.
Neither meets the complete posting gate. Not adopted; no production change.

## v455–v456 common claim/mapping phase flag

Use one per-Q SIMT-written flag: 2*generation-1 claims published,
2*generation mappings published. Retain all release fences and prefix-Q
conditions. No scalar-cache access to these flags. Base full audits PASS.
146-union: posting 10.471111 us, non-CQ 29.152222 us, native 29.17 us.
512-union: posting 10.181111 us, non-CQ 31.502222 us, native 31.52 us.
No improvement over respective references; rejected without additional stress
runs. Not adopted. Generation rollover would require explicit handling before
any integration of this rejected candidate.

## v457–v458 output work redistribution

Native v346 attribution shows late Pack completion on warps29/31 for Q3/Q4.
Try bijective i0=(tid*33)&1023 and i1=i0+1024 in Emit only; every output
position remains covered exactly once and all waits/fences stay unchanged.
Both device full audits PASS, but complete posting regresses substantially
(see evidence reports). Rejected. Late warp completion alone is not proof
that distributing sparse loads across more warps improves memory latency.
No source or production adoption.

## v459–v460 rotate complete output warps

Emit i0=(tid+896)&1023, i1=i0+1024 rotates warp ownership without
destroying consecutive32-index accesses. Host enumeration verifies exact
2048-position coverage and each warp's consecutive span. Both full audits PASS.
146-union: posting10.572777/non-CQ29.512222 us.
512-union: posting10.262777/non-CQ31.842222 us.
Neither beats its reference or meets10us. No adoption or further stress runs.

## v461–v462 select per-Q count without private array

Keep four-count ABI, replace private counts[4] dynamic indexing with one
conditional selection. All semantics and acceptance boundaries retained.
v461: audit PASS, posting 10.336666 us, non-CQ 28.902222 us.
v462: audit PASS, posting 10.440001 us, non-CQ 31.682224 us.
Not adopted: posting target remains unmet.

## v463–v464 compact unresolved output positions

After stamp3, SIMD compacts all non-hot non-padding output positions to
scratch[8192..10239], count at10240. Emit retains prefix mapping wait and
block rendezvous, then processes only the compact list (up to2048 positions).
Tail positions may be included but are excluded by the existing loc check.
Full2048 logical/slot output DMA and final join are retained.
Both complete audits PASS. 146-union posting12.348889/non-CQ31.882222 us;
512-union posting11.113890/non-CQ32.892222 us. Standalone position compaction
does not pay for itself. Not adopted, no stress qualification. A future fused
export must include its extra cost in the full tail, never hide standalone
compaction before stamp3 to claim posting acceptance.

## v465–v466 fuse pending-position export into bitmap gather

Reuse the first GatherCoveredChunk ID traversal to produce unresolved positions
in disjoint scratch8192..10239/count10240. Remove standalone PendingPositions
VF. This probe covers only the128K..256K bitmap-chunk geometry; other geometry
paths are NOT implemented and this is NOT an integration candidate.
Both131328-history complete device audits PASS.
146-union: posting12.472778/non-CQ31.322777 us; full31.34 us.
512-union: posting10.183333/non-CQ32.402779 us; full32.42 us.
No improvement over qualified references and10us gate fails. Rejected.
Do not move standalone compaction before gate or delay empty-Q completion to
claim a passing interval. Production and root generator unchanged.

## v467–v468 queue publication with current atomic arrival join

Combine producer/doorbell STCG and atomic-arrival join in one SIMT task.
Post clock64 is handed through UB to scalar stamps. Eliminate scalar queue
writeback to avoid stale producer overwrite. Preserve WQE/full-output waits.
Full audits PASS:146-union posting11.707222 us, native whole30.24 us;
512-union posting10.636666 us, native whole31.97 us.
Both fail posting and regress references. Rejected. Existing generic parser
does not calibrate this clock64 stamp, so non-CQ and5us delay acceptance are
UNAVAILABLE; do not substitute native total minus5. No stress qualification
or production integration. No need to qualify rejected clock path further.

## v469–v470 compile-time Q specialization

Instantiate PackUniqueLocal<Query> for Q0..3 in the same kernel, preserving
all prefix claims/mapping dependencies, outputs and four-core arrival join.
Both complete audits PASS but regress:146-union posting12.657777/non-CQ30.086110 us;
512-union posting11.468888/non-CQ32.555555 us. No adoption or stress extension.
Dynamic Q selection cannot be blamed for the remaining gap from source alone.

## v471–v472 512-thread preparation and four-position Emit

Retain96/128 candidate writers;416/384 remaining threads cover all hot
positions. Emit handles four positions per thread with staged map/slot/logical
loads. Host coverage and full device audits PASS.
v471: PASS; posting10.780555/non-CQ29.292222 us.
v472: PASS; posting10.232224/non-CQ31.522222 us.
No10us acceptance; not adopted or stress-qualified.

## v473–v474 scalar-built UB call descriptor

Stage31 call parameters in248bytes at postScratch[11000], after stamp3;
S_V precedes Pack, which loads the descriptor instead of31 direct arguments.
Scratch is disjoint from output/WQE/control. Both full audits PASS, but major
regression:146-union posting24.252778/non-CQ43.092777 us;512-union
posting22.390000/non-CQ43.452780 us. Both gates fail. Not adopted.
Both model processes terminated normally; this was a slowdown, not a hang.

## v475–v476 same-thread output preparation/completion

Hot workers retain identical ownership through Emit; host enumeration proves
each of2048 positions has the same producer/consumer thread. Remove two block
rendezvous, keep per-warp prefix mapping waits, all producer fences and final
VF/output/WQE/four-core join. Three staged output positions per hot thread.
Full audits PASS but severe regression:146 posting35.257776/non-CQ54.012224 us;
512 posting34.840555/non-CQ56.452224 us. Both models terminated normally, not
hangs. Rejected; no production or baseline adoption. Cause requires native
instruction analysis; do not attribute all regression to polling without it.

### v477: single-warp mapping wait with same-thread output ownership

Replaces v475 per-warp global polling with one full-warp poll followed by a block barrier; retains all output DMA, doorbell and four-core join. Full audit PASS. Native full posting 10.524444 us; non-CQ 28.962222 us; native whole 28.98 us. Removes severe polling regression but does not improve v346 or meet 10 us. Rejected; root implementation unchanged.

### v478: publishing warp directly waits predecessor mappings

v477 same-thread output ownership; move prefix join into the publishing warp branch and omit own mapping-ready read after own release. All producer-helper fences and final block/output/four-core joins retained. Base full audit PASS; full posting 10.723888 us, non-CQ 29.562222 us, native whole 29.58 us. Regresses v477 and v346; rejected. No production changes or acceptance claim.

### v479: larger miss-count scope check

Unchanged v346 device algorithm; four disjoint 256-miss Qs (1024 total, 84 prior, LRU sufficient). Full output/WQE/Index/retirement audit PASS. Full posting 13.294444 us; non-CQ 38.082224 us; native whole 38.10 us. Both performance gates fail. Per-Q dedup/output preparation 9.36–9.91 us; WQE construction 1.53–1.83 us; post-join Index/shared-retirement/CQ interval 11.38–12.07 us. The 146-miss fixture does not prove general performance acceptance. Next optimization must address candidate scaling and Index work, not only a fractional-microsecond join change.

### v480: staged ownership loads for 97..256 candidates

Eight independent warp-sized ownership loads precede ordered ballots/compaction. Original 65..96 path and larger fallback preserved. Same four-disjoint-256 candidate fixture as v479; all audited fields PASS. Full posting 12.105000 us (improves 1.189444), non-CQ 36.822220 us (improves 1.260004), native whole 36.84 us. Still fails both gates; candidate retained as experimental only, root/production unchanged. Needs varied count and overlap qualification before adoption.

### v481: 256 candidate writers with staged ownership loads

Based on v480; eight writer warps, 768 hot-output threads. Helper claim words12..18 and mapping completion20..26 are disjoint from error8 and count controls0..4. Four disjoint 256-miss Qs full audit PASS. Posting11.736667 us, non-CQ36.462224 us, native whole36.48 us. Improves v480 by0.368333/0.359996 us, still fails gates. Experimental only; no root or production adoption, small/overlap/repeat qualification pending.

### v482: double Index update threads

Same v481 candidate path; UpdateIndex launch and THREADS512→1024, shared layout/CHUNK unchanged. Full 1024-miss audit PASS. Posting11.736668 us unchanged, non-CQ36.802220 us vs36.462224 (0.34 us worse), native whole36.82. Rejected: more parallel threads did not reduce Index-tail elapsed time. Does not establish bandwidth cause without native attribution. Root and production unchanged.

### v483: paired Index key exchange loads

Based on v481, 512 update threads; preload two loop iterations before stores with guarded tail. Host exact traversal coverage checked for record counts1..512. Full 1024-miss device audit PASS. Posting11.736667 us unchanged; non-CQ36.932220 us (+0.469996), native whole36.95. Rejected. No production changes; v481 remains best larger-case experimental candidate.

### v484: 16-byte vector Index exchange

Native v481 longest Index instructions were STG at helpers.h:24 (~1.69us, overlapping across warps). Probe changes key exchange to ulonglong2, eight elements/row, preserving alignment and total bytes. Full 1024-miss audit PASS. Posting11.736667 us; non-CQ41.442220 us; native whole41.46. Regresses by4.98us; reject. Wider source vector operations do not imply faster simulator store execution. No adoption.

### v485: 32-bit Index key exchange

Same v481 posting; one warp spans one128-byte row using32-bit elements, total bytes unchanged. Full1024-miss audit PASS. Posting11.736667us; non-CQ37.722220us; native whole37.74us. Worse than64-bit v48136.462224us, better than128-bit v48441.442220us. Retain64-bit exchange. No production changes.

### v486: combine disjoint Index payload/metadata completion

Based on v481; immutable swap plan already published by validation barrier. Key/scale and address/logical arrays are disjoint. Remove fence after key/scale and intermediate block barrier; metadata phase retains per-thread fence covering preceding stores, followed by block barrier before retirement. Full1024-miss audit PASS. Posting11.736667us; non-CQ36.112224us vs36.462224us, native whole36.13us. Saves0.35us but both gates still fail. Experimental only; repeated/overlap/small qualification pending; root and production unchanged.

### v487: v486 two-launch SQ wrap qualification

Identical v486 executable SHA256 c4d17114fa8569b678f980d833b63f472249b54ad95f0fbfd2187fd2104b26af, mode15. Both launches posted1024 bad0, verified_launch1/2, terminal exit0. Retains64-bit producer between calls and audits ring owner/completion bits; verifies mapping/journal/TopK retirement before restoring inputs and Index. Does not verify evolving decode or real NIC/CQ. Qualification only; performance targets remain unmet.

### v488: preload unresolved logical IDs before mapping-ready wait

Based on v486. Output IDs/locations already published by Pack barrier; unresolved logical token reads precede mapping-ready join. Ordinal mapping path rereads logical at resolved slot; Index remains behind all-output four-core join. Full1024-miss audit PASS. Posting11.690000us (-0.046667), non-CQ35.362224us (-0.750000), native whole35.38us. Total improvement exceeds posting change: no claim that all0.75us is saved in posting. Experimental; small/overlap/repeated qualification pending; both gates still fail.

### v489: distributed initial control clearing

Based on v488, first27 lanes clear private control words0..26 including aliased error8; existing block barrier retained. Full1024-miss audit PASS. Posting11.570556us (-0.119444), non-CQ35.242224us (-0.12), native whole35.26us. Both gates still fail. Experimental only, root and production unchanged.

### v489 posting attribution follow-up

Complete interval11.400556→22.971111=11.570556us. Per-Q preparation through full output7.40/7.87/8.26/7.62us; WQE construction1.86/1.62/1.81/1.61us. Q2 is latest join, not Q3 whose preparation duration is largest. Do not sum per-Q times or overlapping native instruction durations. Native Q1 long instructions include GM change-record stores at parallel_post.h:109 (~0.627us) and hot logical reads atline124 (~0.611us). These observations support investigating publication scheduling, not claiming a proven removable delay.

### v490: publish mapping before change-record stores

Based on v489. For owned count<=256, each writer retains one token/ordinal/slot; mapping release precedes records; deferred records are fenced before original block rendezvous. Larger count retains original immediate records. Full1024-miss audit PASS. Posting12.063890us (+0.493334), non-CQ35.812224us (+0.57), native whole35.83us. Reject; moving records alone does not shorten this schedule. No root/production adoption.

### v491 invalid / v492 reduced Index hash sets

v491 stopped intentionally before acceptance: SET1024 but insert initial hash retained11 bits. No usable performance result. v492 masks initial bucket withSET-1, preserves separate SRC/DST offsets, clears1024 entries each. Full1024-miss audit PASS; posting11.570556us, non-CQ35.162224us, native whole35.18us. Saves0.08us; still misses both gates. Collision/maximum-count qualification pending, no root/production adoption.

### v493: latest large-miss algorithm on original 146-union fixture

Same device sources as v492; original v346 driver11/0/94/87 candidates, union146. Full audit PASS including emptyQ2 and overlapQ3/Q4. Posting10.586112us vsv34610.255555 (+0.330557); non-CQ28.542224us vs28.932222 (-0.389998). Thus latest changes improve total but regress the posting gate on small misses; not a universally improved replacement. No root or production adoption.

### v494: original96-writer partition with later optimizations

Original v346 candidate packing; first9 lanes clear controls, v488 logical-ID preloads, v486 combined Index completion, v492 hashSET1024. Original146-union fixture full audit PASS. Posting10.432221us vsv34610.255555 (worse0.176666), non-CQ28.212222us vs28.932222 (better0.72). Does not meet posting gate; no general adoption or production change.

### v495: speculative remote-address staging

Based on v492. For candidate count<=256, claim writers load remote addresses into UB lo[8192..8704), compaction stores original positions inlo[8704..8960); slot writers reuse staged addresses. Larger candidate counts retain original reads. Full1024-miss audit PASS. Posting13.518889us vs11.570556; non-CQ37.082224us vs35.162224. Reject: staging/compaction overhead outweighs removed address dependency. Root/production unchanged.

### v496: single-bank direct MTE3-to-scalar completion

Based on v492. Single-bank WQE copy waitsMTE3_S thenreturns, replacingMTE3_V plus trailingMTE3_S; multibank path unchanged. Full1024-miss audit PASS. Posting11.759444us, non-CQ35.959999us, native whole36.27us. Regresses; reject. Do not infer speed from fewer source-level waits. No production changes.

### v497: clear private Index sets during Pack

Based on v492. Relocate SRC/DST sets to scratchwords8192..10239; each of1024 Pack threads clears one word per set before original barrier. Remove later UpdateIndex clear, keep validation barriers. Sets do not overlap full output0..2047, WQEbank0..4095, controls4104..4130, plans4096..5119 or Indexcontrol5120. Full1024-miss audit PASS. Posting11.403335us, non-CQ35.072224us, native whole35.09. Small improvement; both gates still fail. Repeat/small qualification pending; no production adoption.

### v498: prior journal/TopK retirement before Index exchange

Based on v497. All-output join remains before UpdateIndex; move prior-journal retirement and own TopK/progress clear ahead of exchange; current change records retire afterward, final fence retained. Full1024-miss audit PASS. Posting11.403333us unchanged; non-CQ36.432224us vs35.072224 (+1.36), native whole36.45. Reject; earlier unrelated stores do not improve observed schedule. No production changes.

### v499: output-reader rendezvous before WQE construction

Based on v497; move existing four-core output-reader join before WQE, retaining posting gate through all doorbells. Full1024-miss audit PASS. Posting11.361112us vs11.403335, non-CQ36.002777us vs35.072224, native whole36.02us. Reject: negligible posting improvement and0.93us tail regression. Root/production unchanged.

### v500: 192 candidate writers / 832 output threads

Based on v497; helper completion counts and control clearing adjusted to six writer warps. Host traversal coverage verified for candidate counts0..2048 and2048 outputs. Full1024-miss simulator audit PASS. Posting11.692778us vs11.403335; non-CQ35.882224us vs35.072224, native whole35.90us. Reject: rebalancing toward full-output work worsens both gates. No production changes.

### v501: SIMT direct full-output writeback

Based on v497. Emit stores all2048 logical IDs and slots perQ directly toGM and fences before VF completion; remove two later DMA copies/events, preserve posting acceptance interval and all-output join. Full1024-miss audit PASS. Posting13.637222us vs11.403335; non-CQ37.372777us vs35.072224, native whole37.39us. Reject: direct SIMT stores regress relative to coalesced DMA. Production unchanged.

### v502: mapping publication after existing block barrier

Based on v497. Remove helper mapping-completion flags/poll; each writer retains GM fence, existing block barrier joins them, thread0 then publishes mapping generation. CrossQ Emit wait retained. Full1024-miss audit PASS. Posting11.422223us vs11.403335 (slightly worse); non-CQ34.772220us vs35.072224 (0.30us better), native whole34.79us. Non-CQ gate passes in this fixture, posting still fails. Repeat qualification pending. Production unchanged.

v502 same-binary mode15 qualification: terminal exit0; both launches posted1024 bad0, verified_launch1/2. See qualification.json/repeat.log. This qualifies restored-input repeated SQ-wrap execution, not evolving decode. Posting target remains unmet.

### v503: claim completion block rendezvous

Based on v502. Replace claim-helper polling with an additional block barrier after claims and hot-output work, then publish claim generation. Mapping completion barrier unchanged. Full1024-miss audit PASS. Posting13.956668us vs11.422223; non-CQ37.512224us vs34.772220. Reject: lost overlap exceeds polling savings. Native whole37.53us. No production changes. Retain asynchronous claim helpers.

### v504: separate claim-ready cache lines

Based on v502. Claim generations move from adjacent words to128-byte-spaced words20*32+q*32; readers match; no overlap with mapping-ready24*32+q or other live flags. Full1024-miss audit PASS. Posting11.548888us vs11.422223; non-CQ35.202224us vs34.772220. Reject. This layout alone does not improve measured performance. No production changes.

v502 delayedQ4 mode5 same-binary qualification: terminal exit0; posted1024 bad0, verified_launch1. Artificial delay excluded from performance acceptance. prepare_candidate.py now supportsv502; regenerated kernel/driver/headers byte-match tested sources and manifest checksums pass.

### v505: v502 device on original146-union fixture

Only driver replaced by v346 fixture; device sources unchanged. Full audit PASS including emptyQ2 and crossQ overlap. Posting10.949444us vs rootv34610.255555; non-CQ28.422222us vs28.932222, native whole28.44us. Total improves but posting materially regresses; do not adopt as general replacement.

### v506: v505 with96 candidate writers

Adjust writer loops/helper claim flags and928 output threads; retainv502 mapping publication barrier. Original146-union audit PASS. Posting10.742223us vs v50510.949444 androotv34610.255555; non-CQ28.422224us, native whole28.44us. Recovers only0.207us; still worse than root posting. No adoption.

### v507: helper count/base warp broadcast

Based on v502. Each helper warp lane0 loads UB count/base after ready, broadcasts through shfl; existing fences and mapping publication retained. Full1024-miss audit PASS. Posting11.547778us vs11.422223; non-CQ35.072224us vs34.772220. Reject: fewer UB loads do not reduce observed time. No production changes.

v502 native posting attribution: archive native_posting_attribution.json. Scalar stamp7 follows issuance of MTE3_V, not scalar DMA completion. ExampleQ2 output DMA19.323334..19.952778 overlaps scalar progression; stamp7=19.339. Report subinterval labels must not interpret stamp7 as physical output completion. End-to-end posting gate remains valid because WQE vector work respects MTE3 dependency and final all-output join follows completion. VF_SIMT Pack spans~6.89..7.73us; shared-bank WQE VF~0.62..0.66us. Overlapping instruction/event durations must not be summed.

### v508: independent128-WQE bank and output-DMA overlap

Based on v502. WQE bank scratch6144..8191 disjoint from full slots/control/hash. Remove output MTE3_V dependency; WQE path retains final MTE3_S and emptyQ explicitly waits output DMA. Full1024-miss audit PASS. Posting12.235001us vs11.422223; non-CQ35.992777us vs34.772220. Reject: extra128-bank construction/copies outweigh potential overlap. Full gate unchanged; no production changes.

### v509: independent256-WQE bank with later Index hash reuse

Based on v508; retain256-WQE single bank scratch6144..10239. Restore Index sets0/2048 and clear only after all-output/WQE join; remove early hash clear inPack. Full1024-miss audit PASS. Posting11.914999us vs v50211.422223; non-CQ35.532777us vs34.772220. Better than128-bankv508 but worse than shared-bankbaseline. Reject. No production changes.

### v510: preload same-call queue/doorbell pointers

Based on v502. Template warp stores queue pointer in cfg20/21 and doorbell pointer22/23, unused by FinalBuild; scalar posting consumes them. No cross-call caching. Full1024-miss audit PASS. Posting11.343889us vs11.422223 (-0.078334), non-CQ34.871113us vs34.772220 (+0.098893), native whole35.18us. Small posting improvement but not both-gate acceptance; repeated/small cases pending. No production adoption.

### v511: per-Q generation output join

Based on v502. Replace shared atomic arrival count with four GM generation words, one publisher perQ; full-warp ballot waits all4 with existing fences. Full1024-miss audit PASS. Posting11.419999us vs11.422223 (effectively unchanged); non-CQ35.202224us vs34.772220, native whole35.22us. Reject; no performance gain, no extra qualification pursued. No production changes.

### v512: same v502 binary with detailed stamps disabled

Mode1 full1024-miss audit PASS, terminal exit0. Match conditional-branch PCs for stamp3/stamp8 against mode4, validate enabled SYS_CNT reads immediately following baseline branches and absence when disabled. Posting11.425001us withoutstamps vs11.422223 withstamps; effectively unchanged. Native whole34.28us; no fabricated CQ subtraction without stamps. See native_boundary_report.json. Detailed timestamp writes do not explain remaining posting deficit.

### v513: per-Q prefix slot allocation

Based on v502. Publish deduplicated counts and generation separately; eachQ sums earlier counts and starts at immutable initialSlots; Q3 publishes final slots count. Replaces shared returning atomicAdd with prefix exchange. Full1024-miss audit PASS. Posting13.613333us vs11.422223; non-CQ36.892777us vs34.772220. Reject: count exchange dependency outweighs atomic allocation cost. Model initialSlots is fixture prior count; production integration not attempted.

### v514: combined64-bit count/generation prefix publication

Based on v513. Publish aligned64-bit generation/count, wait reads retain count and avoid second payload load. Full1024-miss audit PASS. Posting12.902778us vs v51313.613333 but v50211.422223; non-CQ36.182777us. Reject: improves prefix scheme but still worse than returning atomicAdd. Protocol repeated/delay/hardware publication qualification not claimed. Production unchanged.

### v515: queue/doorbell preload on root146 case

Based on v346; only applyv510 same-call pointer preloads and scalar consumers. Full146-union audit PASS. Posting11.676112us vs10.255555; non-CQ29.918890us vs28.932222. Reject: large-case marginal benefit does not transfer to root small-case path. Native whole30.17us; non-CQ deducts only measured exposed wait. No production changes.

### v516: distributed control clear on root146 path

Based on v346 only. First9 lanes clear private counters0..8 instead of lane0 serialstores; includes erroralias8, retains original barrier. Full146-union audit PASS. Posting10.518889us vs10.255555; non-CQ28.972224us vs28.932222. Reject; large-case distributed-clear benefit does not transfer in isolation. No production changes.

### v517: logical-ID preloads alone on root146 path

Based on v346; only overlayv488 emit.h. Full146-union audit PASS. Posting10.423889us vs10.255555; non-CQ28.992222us vs28.932222. Reject: isolated logical preloading does not improve root small case. No production changes.

### v518: unchanged v346 baseline reproducibility

Rerun identical binary SHA256dd4248497d88e2d02e2f1151ff84f06d3e110293e99a04b2ccb1e133bac0f109, same mode4/146fixture. Full audit PASS; posting10.255555us, non-CQ28.932222us, native whole28.95us exactly match archived baseline. No rebuild, no source/input changes. Recent regressions cannot be dismissed as baseline drift from this check. Report parser needs kernel source markers; linked original source before final extraction.

### v519: FinalBuild unroll2 hint (not applied)

Based on v346; addpragma unroll2 to shared SIMD FinalBuild loop. Compiler explicitly warns loop not unrolled / requested transformation unsupported. Full146-union audit PASS; posting10.255555us andnon-CQ28.932222us exactly unchanged. Not an implemented optimization; no adoption. Build log archived.

### v520: explicit two-group FinalBuild expansion

Based on v346. Explicit outerstep128 with original64-entry body twice, second guarded. Host group coverage exact for counts0..2048. Full146-union audit PASS. Posting11.737778us vs10.255555; non-CQ30.011667us vs28.932222. Reject; explicit expansion regresses. No production changes.

## v521: preload two ownership groups for counts 1..64

Full audit PASS; posting 10.618333 us, non-CQ 29.352222 us.
Rejected against v346 10.255555 / 28.932222 us. No production changes.
Fixed prepare_candidate v346 manifest selection and restored archived prefetch/CMake files; all six archived hashes match generated output.

## v522: latch observed producer readiness within each wait

Keep publication fences and the four-core join. Once a producer generation is observed, stop reloading its readiness within that wait. Basic complete audit PASS; same binary delayed-Q4 audit PASS (146 posts, zero bad). Posting 10.206666 us, non-CQ 29.112224 us. Still fails 10 us posting. Repeat-generation validation remains outstanding; root generator remains v346.

## v522 qualification and v523 self-read removal

v522 same binary mode15 passes two launches (146 posts, zero bad each); inputs/Index restored, queue state retained. Not evolving decode.
v523 skips own-Q mapping-ready read after its publication and existing block barrier. Full audit PASS, but posting 10.303333 us / non-CQ 29.262224 us; reject relative v522. No production changes.

## v524/v525: readiness-latch scope checks

v524 applies v522 claim/mapping readiness latch to v502 1024-miss fixture: audit PASS, posting 11.707223 us, non-CQ 35.552220 us. Reject versus v502 11.422223 / 34.772220.
v525 applies mapping readiness latch alone to v346: audit PASS, posting 10.461666 us, non-CQ 29.352224 us. Reject. The v522 small-fixture improvement does not generalize. Root generator unchanged.

## v526: ownership-ready latch only

Complete audit PASS; posting 10.237777 us, non-CQ 29.102222 us. Does not improve v522 combined 10.206666 us. The tested readiness-loop variants do not meet the posting gate. prepare_candidate now also reproduces v522; generated manifest hashes verified. No production changes.

## v527: logical-ID snapshot during hot-output preparation

Read logical IDs into postScratch[6144:8192] during preparation, then reuse after mapping publication; old ordinal mappings retain the physical-slot lookup. Full 146-miss audit PASS. Posting 10.271667 us, non-CQ 29.092222 us. Does not beat v522 posting 10.206666; reject. Snapshot lifetime ends before WQE scratch reuse and Index updates. No production changes.

## v528: combine readiness latch with delayed producer fence

Move own writer fence after helper completion poll, still before mapping-ready publication. Complete audit PASS; posting 10.460555 us, non-CQ 29.442222 us. Reject relative v522. The full posting gate retains earliest-Q start to latest-Q end, including cross-Q start skew; per-Q work durations cannot replace it. No production changes.

## v529: one candidate warp on current v522 pipeline

Remove helper claim/map handoffs, use32 candidate writers and992 hot-output threads. Host coverage verified counts0..2048 and all2048 output positions. Full audit PASS. Posting11.107223us, non-CQ29.972222us; reduced writer parallelism outweighs handoff removal. Reject; no production changes.

## v530: skip own claim-ready read after publication

Retain own release fence and lower-Q waits; exclude only own ready read in v522 latch. Full146-miss audit PASS. Posting10.395555us, non-CQ29.092224us. Reject versus v52210.206666us posting. No production changes.

## v531: remove unused Pack width and bits arguments

Only remove unused parameter/call operands, retain all processing and synchronization. Full146-miss audit PASS. Posting11.303332us, non-CQ29.802223us; reject versus v522. No production changes.

### v531 native attribution

Compared native trace X events inside stamp3..11 with v522. Q4 preparation span6.098->6.958us; SIMT source span5.512->6.131us. Scalar LD29->27 and ST21->19; SIMT LDS656->621, STK128 unchanged. This excludes increased argument instruction count as the cause; it does not prove register pressure or cache conflicts. Native intervals overlap and are not summed. Raw per-Q counts/spans in evidence/v531/native_attribution.json.

## v532: predicate unresolved token reads in Emit

Use token IDs only when use0/use1 require mapping resolution; full output stores and synchronization unchanged. Full146-miss audit PASS. Posting10.346667us, non-CQ29.282224us; reject versus v522. No claim of fewer native memory transactions without instruction attribution. No production changes.

## v533: 128 writers on v502 1024-miss fixture

128 candidate writers /896 hot-output threads; three helper claim publications. Host coverage0..2048 PASS; full1024-miss audit PASS. Posting11.984445us, non-CQ35.752224us; reject versus v50211.422223/34.772220. No production changes.

## Logical read cache policy audit

Installed intrinsic definitions distinguish NON_CACHEABLE asc_ldcg from CACHEABLE asc_ldca. Logical data is read-only before the output join but changes across calls. No L1 invalidation contract has been established for that reuse, so do not enable asc_ldca based on a single-launch timing. Details in evidence/cache_policy_audit/audit.json. This is not a performance result or a proof caching is impossible.

### v522 additional same-binary correctness qualification

Mode11 (ABC begin H-333, span123, capacity384, distinct Q ends/high logical bits) PASS; mode16 (candidate counts94/87/0/11) PASS. Both146 posts, zero audit errors, verified launch1. Binary hash remains7735cad9d05342511b742c791984f2a648d253e76f0d2a2545fbff51b1da40df. These are correctness checks, not evidence the posting gate is met or production hardware is qualified.

### v522 same-binary detailed-stamps-OFF check: reject adoption

Mode1 complete audit PASS, same executable SHA. Native boundary script verifies same JUMPC PCs at stamp3/8, enabled SYS_CNT immediately after and absent when disabled. OFF posting10.467222us versus ON10.206666us; native whole28.57us. Thus its small instrumented gain cannot be presented as production-mode improvement. v346 OFF10.210556us remains better. No artificial CQ subtraction for OFF run. Root remains v346.

### v494 same-binary OFF measurement

Mode1 full audit PASS. Same-PC stamp3/8 branch audit: ON10.432223us, OFF10.208333us; native whole27.06us. Still exceeds10us. Native whole includes exposed CQ; do not fabricate a non-CQ subtraction. Candidate for further qualification, no production adoption. Binary SHA5f10e97152ddb893a6d4f9d22110120575168f42110715ed73ac8d6ea034f94d.

### v494 same-binary repeat/delayed qualification

Mode15 two launches PASS,146 posts/zero bad each; mode5 delayed-Q4 PASS. Inputs/Index restored between launches, queue state retained; not evolving decode. Native OFF posting remains10.208333us, does not meet10us. No production integration.

## v534: issue LRU reads before remote-address reads

Based v494, swap two independent loads in both writer paths. ON audit PASS: posting10.445556us/non-CQ28.272222us. Same binary OFF audit PASS; same-PC boundary posting10.152777us, native whole27.13us. Small candidate improvement from v494OFF10.208333, still fails10us. Stress qualification pending; no production adoption.

### v534 same-binary qualification

Mode15 two launches PASS (146 posts,zero bad each); mode5 delayedQ4 PASS. Binary5f6ed8cc7aedf95c2fd585c9d6b8ade8ac2d176dd69f2de5859a65473aab248c. prepare_candidate supports v534 and generated checksums match. OFF posting10.152777us still fails target. No production changes.

## v535: GM records before UB WQE inputs

Based v534; reorder only independent stores, preserve publication fences. ON audit PASS10.562778us posting/28.452222us non-CQ. Same-binary OFF audit PASS10.207777us posting/native whole27.27us. Reject versus v534OFF10.152777us. No production changes.

## v536: terminal-only removal of unused VEC1 event pairs

Remove six VEC1 priming/wait pairs from isolated generated terminal QLIVector, retain TopK event pairs and full output/WQE/join. ON full audit PASS10.657778us posting/28.512222us non-CQ. OFF full audit PASS11.432223us posting/native whole27.82us. Reject; no production edits, v534 retained as candidate.

### v534 ABC and rotated-candidate qualification

Same binary mode11 and mode16 complete audits PASS,146 posted/zero bad each. Nonaligned ABC capacity384 and candidate redistribution94/87/0/11 covered. No production changes; posting gate still not achieved.

## v537: omit overwritten/unused initial control stores

Based v534; initialize counter2,3,4,6,7,8 only. Counter0/1 are written before ready publication; counter5 unused. ON complete audit PASS10.376667us posting/non-CQ28.042222us. OFF complete audit PASS10.038888us posting/native whole27.10us. Still exceeds10us, no rounding acceptance. Candidate pending stress qualification, no production changes.

## v538: six contiguous lanes initialize required control words

Same control indices as v537, mapped from lanes0..5. ON full audit PASS: posting10.378888us/non-CQ28.052222us. Same-binary OFF full audit PASS: posting10.214444us, verified native endpoint PCs. Reject versus v537OFF10.038888us. No production changes.

## v539: scalar own-Q candidate count

Based v537; replace four-element candidate count array with own-Q ternary selection. ON full audit PASS10.342778us posting/non-CQ28.102222us. Same-binary OFF complete audit PASS10.218889us posting with verified native endpoint PCs. Reject versus v537OFF10.038888us. No production changes.

### v537 same-binary correctness qualification

Mode15 two input-restored launches with persistent queue state PASS; mode5 delayedQ4 PASS; mode11 nonaligned ABC PASS; mode16 redistributed candidate counts PASS. All146 posts/zero audit errors. prepare_candidate supports v537; reconstruction source hashes verified. OFF posting10.038888us still fails10us. Constructed simulator fixture only, not production integration or true transport acceptance.

## v540: only live control words, 1024-miss fixture

Based v502; initialize2,8,12..18 only; keep publication and barriers. ON complete audit PASS11.338334us posting/34.992220us non-CQ. Same-binary OFF complete audit PASS11.379999us posting, same-PC boundaries verified. Small posting improvement versus v502OFF11.425001; still fails10us and full-tail margin is negligible. Not adopted, qualification beyond base pending.

## v541: single hot-slot output store on v537

Preserve invalid/error output stores, write valid slot once instead of invalid then valid. ON complete audit PASS10.375555us posting/28.352224us non-CQ. Same-binary OFF complete audit PASS10.243334us posting; reject versus v537OFF10.038888. No production changes.

## v542: preload stable mappings before existing ready join

Based v537,146-miss fixture. Full base audit PASS ON/OFF. ON posting10.470554us/non-CQ27.952224us; OFF native same-PC posting10.122778us. Retain all waits; unresolved mappings reloaded after join. No adoption or production changes; not qualified beyond base.

## v543: preload stable mappings before existing ready join

Based v502,1024-miss fixture. Full base audit PASS ON/OFF. ON posting11.426667us/non-CQ35.062220us; OFF native same-PC posting11.715555us. Retain all waits; unresolved mappings reloaded after join. No adoption or production changes; not qualified beyond base.

## v544: conditional participation waits

Skip cross-Q claim wait for empty own candidate list; retain mapping-ready/output joins. Fixture 146 misses. Full base audits PASS ON/OFF. ON posting10.622221us/non-CQ28.412222us; OFF native same-PC posting10.164445us. No adoption or production changes; no claim of delayed/repeat qualification.

## v545: conditional participation waits

Skip cross-Q claim wait for empty own candidate list; retain mapping-ready/output joins. Fixture 1024 misses. Full base audits PASS ON/OFF. ON posting11.298890us/non-CQ35.222220us; OFF native same-PC posting11.601666us. No adoption or production changes; no claim of delayed/repeat qualification.

## v546: conditional participation waits

Skip helper wait only when helper has no claim/payload indices; retain final block barrier and output join. Fixture 146 misses. Full base audits PASS ON/OFF. ON posting10.363888us/non-CQ28.272222us; OFF native same-PC posting10.187778us. No adoption or production changes; no claim of delayed/repeat qualification.

## v547: disjoint WQE bank at scratch word2048

Start both output DMAs, construct WQEs in [2048,6144), join DMA before publication completion. Slots [0,2048) and early Index hash [8192,10240) are disjoint; control words consumed before existing S_V fence. Full base audits PASS ON/OFF. ON posting11.639444us/non-CQ28.493887us; OFF native same-PC posting11.428889us. Reject; no production changes.

## v548: disjoint WQE bank at scratch word2048

Start both output DMAs, construct WQEs in [2048,6144), join DMA before publication completion. Slots [0,2048) and early Index hash [8192,10240) are disjoint; control words consumed before existing S_V fence. Full base audits PASS ON/OFF. ON posting11.602777us/non-CQ34.988889us; OFF native same-PC posting11.627777us. Reject; no production changes.

## v549: compact unresolved output positions during hot pass

Each hot warp compacts up to96 positions into disjoint postWork[6208..8992) (24 warps use less). Existing block barrier publishes counts and positions. Emit waits for original mapping readiness, then maps only these positions, including prior-window prefetched cold tokens. Dense2048 outputs and final join retained. Host coverage/bounds checks PASS; full base device audits PASS ON/OFF. ON posting11.456667us/non-CQ29.182222us; OFF same-PC posting11.118333us. Reject; no production changes or stress qualification.

## v550: compact unresolved output positions during hot pass

Each hot warp compacts up to96 positions into disjoint postWork[6208..8992) (24 warps use less). Existing block barrier publishes counts and positions. Emit waits for original mapping readiness, then maps only these positions, including prior-window prefetched cold tokens. Dense2048 outputs and final join retained. Host coverage/bounds checks PASS; full base device audits PASS ON/OFF. ON posting11.761667us/non-CQ35.402224us; OFF same-PC posting11.936666us. Reject; no production changes or stress qualification.

## v551: cached logical reads with in-gate DCI — INVALID

Use asc_ldca only for logical-ID reads, retain noncached mapping/readiness. Add dci plus PIPE_ALL inside posting gate; mode15 changes logical values before second launch. Simulator reports illegal SCALAR low-address accesses; base parser reports0 successful kernels. Repeated case also fails before first-launch audit. No valid timing or correctness claim. Exact failing child process was stopped; original sources untouched. DCI/live-cache interaction remains unproven; this interface must not be adopted based on declaration alone.

## v552: omit final terminal wrapper PIPE_ALL

Keep JoinColdOutput and pending-output waits, omit only trailing PIPE_ALL in isolated TerminalQuery. Generic LI and intermediate window methods unchanged. Full base audits PASS ON/OFF. ON posting10.320556us/non-CQ28.272224us; OFF same-PC posting10.411667us. Reject; no production changes or delayed/repeat qualification. Full final output/WQE join remains in original timing gate.

## v553: omit final terminal wrapper PIPE_ALL

Keep JoinColdOutput and pending-output waits, omit only trailing PIPE_ALL in isolated TerminalQuery. Generic LI and intermediate window methods unchanged. Full base audits PASS ON/OFF. ON posting11.724444us/non-CQ35.352224us; OFF same-PC posting11.387222us. Reject; no production changes or delayed/repeat qualification. Full final output/WQE join remains in original timing gate.
