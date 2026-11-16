# Native URMA rolling threshold and candidate refilter

This opt-in experiment extends `perf/urma-fused-scan-admission`, based on main
`532df2c8` plus PR15. It does not change graph routing, KVAF, Index layout, the
window count, or the final TopK implementation. All experiment options default off.

## Why this experiment

On the five BS16/Q4/656B snapshots, reverse scanning alone reduced mean device
chain time by 0.48–1.40%; forward repeat drift was +0.28–0.93%. Increasing the
request/window budgets from 512/64 to 1024/128 reduced G7 final repair from
3.922 to 1.814 MiB but increased waste from 1.108 to 1.936 MiB and time from
680.502 to 789.129 us. Moving reads earlier is not sufficient evidence of speed.

## Rolling state and cost

`KVIP_ROLLING_THRESHOLD` seeds a 256-bin reverse cumulative histogram from the
resident+recent score union already in UB. It adds one seed VF and histogram
work, but no new Q/Index DMA or matrix calculation. Each query retains 512
bytes of capped counts and a 256-byte broadcast seed key. The two-query AIV
allocation is 1536 bytes, appended after existing buffers.

Each cold tile updates the histogram inside the existing score/classify VF,
using its live sortable BF16 score register. This adds vector instructions,
UB count loads/stores and a store/load fence. It does not add a VF per tile.
Only native window boundaries select a new Kth. The seed recent interval is
excluded from subsequent counts, and causal/padding lanes do not raise Kth.

The range is seed..seed+255 in **sortable key units**, not floating-point score
units. Values above the range accumulate in its top bucket. Each CDF bin is
saturated at K=2048 after each <=128-lane update, so a long history cannot
overflow u16. Within the range this gives the exact observed-prefix Kth. If
the true Kth exceeds it, seed+255 remains a safe lower bound. The replay reports
the number of capped queries; coverage is measured, not assumed from the old
`seed_hot_key` trace column (which represented resident+previous TopK).

Classification retains the original `score > seed` candidate domain. Once Kth
increases it accepts `score >= current Kth`, retaining boundary ties rather
than introducing extra lost candidates through a strict comparison.

## Window refilter and cost

`KVIP_WINDOW_REFILTER` requires rolling thresholds. The producer writes a
32-byte threshold stripe in unused non-sequence `Progress` padding at the
seed and each window boundary. Existing progress DMA reads the stripe, so no
Context ABI extension or extra Fetch-side threshold DMA is needed. The initial
threshold diagnostic remains immutable. Maximum additional publication is
16 KiB for 64 queries and eight native epochs.

The existing candidate map-gather SIMT launch additionally reads at most four
32-bit score words for a still-cold candidate. Candidates below every eligible
query's latest published Kth are rejected in UB **before** the original
prefetch allowance is applied. Hot/already submitted candidates need no score
reads. No cache mapping is changed by this rejection. Original final repair
handles candidates that were never prefetched.

This is conservative Kth filtering, not a stable 2048-ID sort: equal-score ties
remain eligible. No new peer barrier is introduced. A slower query can expose
an older threshold, reducing filtering efficacy but not correctness. Score
reads are limited to the query's captured ready/causal prefix; reverse scan
positions are mapped back before looking up physical score addresses. A newer
threshold can safely filter an older ready candidate because it remains <=
the final Kth. Unready score rows are never read.

## Scope and validation

Persistent rolling state currently supports B1..16 with split-Q and the
non-sequence-split schedule, retaining its existing 16K scratch bank.
The native launcher rejects sequence splitting when rolling is enabled. The
replay also requires all queries to use PR15's exact resident+recent seed fast
path; it must not label a silent fallback as an optimization result.

Host tests execute the actual histogram helper and SIMT candidate-gather body
under lane/memory models, including >65535 observations, range overflow, ties,
recent de-duplication, reverse addressing and publication skew. They cannot
validate CANN compilation, generated instructions or NPU latency.

Remote comparison uses unchanged 512/64 budgets, five real-Q/Index inputs and
synthetic KV, measuring the complete native device graph:

1. forward;
2. reverse;
3. reverse + rolling;
4. reverse + rolling + refilter;
5. forward + rolling + refilter;
6. forward repeat.

Reported mean/p50, waste, final repair and total transfer bytes are necessary
together. Smaller repair alone is not a win. Initial/final thresholds and
ordered final TopK remain identical. An independent CPU prefix oracle checks
the actual last rolling key outside timing. Restore, CPU/H2D input preparation
and verification are excluded from the device-graph timing; all new seed,
counting, publication and refilter work is included. No NPU speedup for these
two additions has been measured locally.


## Batched counter update

The measured `bc2266f` five-group run passed the numerical checks but added
127–136 us of mean graph time for rolling thresholds. G7/G17 wasted bytes fell
21.72%/25.10%, while total bytes fell only 2.82%/3.64%. Refilter recovered
15–28 us but did not beat the no-rolling baseline. G2 traffic did not change;
its refilter timing gain cannot be explained by reduced reads alone. The
forward-repeat mean drift ranged from -0.35% to +0.44% (p50 -0.11% to +0.64%).
These options remain experiments, not enabled performance defaults.

`KVIP_ROLLING_BATCHED` keeps the same thresholds, scan order, candidates and
publication boundaries. It changes only how the counts are accumulated:

- Encode each live 128-score tile into 128 bin bytes in dedicated UB scratch.
- Flush every eight tiles, or immediately at a native window boundary. A
  partial batch cannot delay the boundary threshold or last-tile classification.
- Load the CDF once per batch. Feed two packed tiles (256 lanes) to each
  histogram instruction, retaining CDF registers across the batch.
- Saturate once at the end: previous count <=2048 plus <=1024 observations is
  safe in uint16. Saturated addition is associative, including BF16 ties.

For a full batch this halves histogram instructions, reduces CDF loads/stores
and its store/load fence to one eighth, and adds 128 bytes of UB writes/reads
per tile. The two-query AIV gains 2 KiB of scratch after existing allocations;
there are no new HBM transfers, launches, peer barriers or Index calculations.
The immutable seed is fenced once after initialization. Final TopK and the
512/64 admission limits are unchanged. Partial-batch overhead means the exact
instruction savings depend on native window lengths.

Host lane tests execute the real native helper in forward/reverse block order,
with odd 137-tile window boundaries, seed deduplication, partial causal tiles,
large histories, range overflow and ties. They check every window against an
independent exact Kth and count histogram calls. Device speed remains to be
measured, since additional branching/register pressure can offset savings.
The controlled replay retains the previous `reverse-refilter` implementation
and compares it against `reverse-batch-ref` in the same restored arena, plus
no-rolling and forward-repeat controls. All batching cost is inside timing.

## Window accumulation: move bookkeeping out of each score tile

`KVIP_ROLLING_WINDOWED` is an alternative to eight-tile batching, not an
additional nested batch. The original score VF writes directly into a dedicated
36 KiB UB buffer, and the original score DMA reads that same buffer. There is
no extra score copy per tile, no new Index read, and no new GM score pass.

At the existing window boundary, one VF reads the accumulated raw scores,
encodes two tiles together, and retains the CDF in registers until the whole
window is counted. Full cold tiles use Max/Sub/Min/Sub encoding; causal-tail
and recent-overlap tiles additionally mask ineligible lanes. The resident+recent
seed uses the same paired histogram helper. A scalar next-trigger counter
replaces the rolling code's repeated search over all window triggers. The
existing progress/Fetch schedule still uses its own unchanged trigger logic.

The last tile is classified again after selecting the new Kth, preserving the
old update-before-classify rule at each boundary. Counts are capped once per
window: previous count <=2048 plus <=18432 scores cannot overflow uint16.
Reverse addressing still maps complete 2048-position blocks; neither the
window count nor the 512/64 budgets change. Sorted final outputs remain the
responsibility of the original exact TopK path.

This opt-in variant requires H32, split-Q, no sequence splitting, and at most
18432 raw scores per cold window. The host launcher and replay reject other
shapes before launch. The raw buffer is not allocated for TopK-only kernels.
Its allocation follows existing scratch buffers and does not alias their
storage or change Fixpipe offsets.

For the recorded history=131328, resident=16384, eight native epochs, each
query has 898 cold tiles and seven cold-window boundaries. CDF loads/stores
fall from 898 to seven per query, compared with 114 for the eight-tile variant.
Histogram instructions are approximately halved, including the seed. These
are operation counts, not a claim of equal percentage latency savings. The
new UB reads, boundary VF, boundary bitmap reclassification, and additional UB
occupancy all remain inside device-graph timing. The boundary burst can delay
publication; only the full native replay can determine the net result.

The CPU audit checks all window thresholds using archived score keys and an
independent prefix TopK, without loading KV/Index tensors or allocating an NPU
arena. Host lane tests also execute the actual native helper under ASan/UBSan.
On the device, compare the original forward fused path at both ends, the prior
per-tile and eight-tile implementations, then window accumulation with/without
refilter and with/without reverse scan. Report waste, final repair, total
bytes, mean and median together. A candidate must beat the faster baseline
repeat in both mean and median by more than max(1%, observed repeat drift).
This is a conservative screening rule, not a confidence interval or automatic
runtime selection. An unsuccessful group stays on the original fused path.
All new options remain off by default until hardware evidence supports them.


## Window-cost follow-up (2026-11-16)

The five host20 snapshots passed the complete oracle at 8c82df1, but the
windowed rolling variants did not beat the original fused baseline. In
particular, G2 transferred effectively the same bytes and still paid 46 us
against the faster baseline repeat. Reduced histogram counts are not a speedup.

Two opt-in controls now separate exact execution savings from less frequent
threshold estimation. Both default OFF and retain the native 8-window protocol,
512/64 budgets, exact final TopK and final repair. Neither changes graph routing.

- `rolling_compact` requires windowed accumulation. At each boundary it computes
  `cutoff=max(seed,max(current,1)-1)` once. The live score classifier uses the
  original one-load, one-strict-compare operation. This is exactly equivalent to
  `score>seed && score>=current`, including BF16 boundary ties and zero keys.
  It also returns the Cube QK buffer on PIPE_V immediately after its last score
  consumer at a histogram boundary. The histogram uses separate score UB;
  score/bitmap DMA and Fetch progress are not published early. This removes
  unnecessary buffer retention; device overlap savings still require measurement.
- `rolling_first_window` additionally requires compact admission. It counts the
  seed and first cold window once, then freezes that conservative Kth. Remaining
  six cold windows use the original ping-pong score buffers, without collecting,
  histogramming or publishing another rolling threshold. This deliberately
  relaxes later admission, so additional waste/repair must be reported alongside
  the saved work. The independent oracle checks the frozen prefix, not full-prefix
  exactness. Final output selection remains exact.

Validation compares forward, reverse only, old windowed+refilter, compact,
first-window and forward repeat on each of G2/G7/G10/G17/G4. Compact controls
have identical Kth values; first-window values are bounded by seed and final Kth.
A candidate is retained only if full graph mean and median beat the faster
forward repeat by more than the measured drift/1% screening margin. Independent
instrumented replays are labelled raw cycles, not summed as graph microseconds.
CPU lane models verify arithmetic, masks and memory bounds. They do not model
CANN instruction scheduling or prove a positive NPU latency result.

## Visibility-safe second filtering and same-window refill

The legacy submission filter skips queries whose score publication has not
reached a candidate. If all other Q4 scores fall below the current Kth, it drops
that candidate even though an unpublished query may ultimately select it. Final
repair keeps outputs correct, hiding this admission false negative.

`refilter_safe` keeps an ID as soon as any causal-valid Q is unknown, without
waiting or reading its unpublished score. Causal-invalid queries are excluded;
known scores keep equality ties. The existing four-query gather loop does this
without a second pass. A candidate is rejected only when every valid query is
known and below its own safe lower-bound threshold. Thus a final-selected ID
cannot be rejected by this filter. Budget contention and service scheduling can
still defer an admitted candidate; the rule is not a completion-time guarantee.

`refilter_refill` additionally continues after a 2048-entry candidate/bitmap
scratch cutoff when the frozen window has unexamined candidates and admission
capacity remains. It does not resnapshot progress, wait, sort, or grow the 64/512
budgets. It appends to 480 existing UB words at `cfg[1536:2016]`, disjoint from
bitmap words, Q4 progress/thresholds and batched post metadata. No new memory
allocation or copy kernel is introduced. The host already limits rolling modes
to split-Q batches <=16; the public window limit is <=480.

Both options default off. Keep the legacy filter as an explicit control, then
measure safe-only and safe+refill separately. Refill can do additional bitmap
work; conservative unknown handling can admit more speculative traffic. Neither
is a speedup claim until complete graph mean and median improve beyond repeated
baseline drift. Report useful prefetch, final repair, waste and total bytes;
all-prefetch cold coverage must not be called pre-last-window coverage or link
utilization. Original forward and reverse-only controls retain their budgets.

Host tests run the actual gather, compact VF and refill orchestration with CPU
lane models, ASAN and UBSAN. They cover Q4 publication skew, ties, causal ends,
all publication masks, capacity cutoffs, unaligned append boundaries and budget
limits. Bitmap DMA enumeration is mocked in the refill test; remote CANN build
and five-snapshot output/threshold/transport validation remain required.

## Isolate compaction from admission policy

Host20 f72ce09 measurements do not establish a benefit over the original fused
path for either safe refilter or refill. Safe refilter improves useful prefetch
in G7/G10/G17 but its incremental mean changes are below 1%. Refill saves about
12--16 us against safe refilter, including 14 us in G2 with unchanged traffic.
It also reduces useful prefetch in G7/G10/G17. This is not evidence of a better
prefetch policy: refill also changed compaction from overlapping input/output
to disjoint UB storage, and asynchronous progress can differ between variants.

`compact_out_of_place` isolates that storage change. It needs no rolling score
buffer, refilter or refill loop. The same VF reads the same IDs and flags and
writes the same stable selected prefix into existing `cfg[1536:2016]` storage.
Submission reads this prefix directly; no extra copy kernel, allocation, sort,
budget or publication boundary is introduced. The original all-miss shortcut
continues to submit its unmodified input list. Default is OFF. Only H32 split-Q
B1..16 without sequence splitting is allowed until another layout is proved.
This is an alias/scheduling hypothesis, not a demonstrated device optimization.

Replay separately compares reverse-only, reverse plus disjoint compaction,
compact windowed safe refilter, and first-window-only correction with/without
safe refilter. Refill is disabled throughout. Both unchanged forward and
reverse-only controls run at the start and end. Headline gains are relative to
the faster unchanged-forward repeat, independently for mean and median; extra
gain beyond reverse-only is reported separately. A traffic reduction alone is
not enough to retain a candidate. CPU tests compare the actual compact VF with
an independent stable-selection oracle, in-place and disjoint destinations,
tail lengths, capacity limits and guard regions. Device compilation and timing
remain part of remote validation.

## Reuse the seed selector's histogram

The 5ade2b1 host20 run did not confirm disjoint compaction as a speedup: all
five means/medians were worse than reverse-only. First-window correction saves
29--43 us over compact all-window safe refilter but remains slower than the
unchanged fused baseline. G7/G17 also lose useful prefetch and increase waste
when later threshold updates are omitted. These controls stay opt-in.

`rolling_reuse_seed` removes the extra resident+recent histogram pass. The
exact seed selector has already computed the cumulative low-byte distribution
for the high byte containing Kth, plus the number of keys in earlier bytes.
Before discarding these registers, it exports a descending CDF into the
existing rolling state. Two 128-lane gathers and constant-size arithmetic
replace reloading/re-encoding/re-histogramming roughly 16K--18K seed scores per
query. The initial Kth and resident classification are unchanged. There is no
additional allocation, Index read, kernel or final TopK change. Memory barriers
separate the 512-byte temporary ascending CDF from its descending overwrite.

This reuses a narrower score interval: `[seed & 0xff00, seed | 0xff]`, whereas
the old histogram spans `[seed, seed+255]`. Within the reused byte, threshold
selection is exact. If a later prefix Kth exceeds it, the threshold stays at
the byte upper bound. It cannot over-tighten or remove a final-selected token;
the tradeoff is potentially more speculative traffic. Replay reports the cap
mode, capped query count, final-key gap, waste, repair and complete latency.
`bit_exact` means agreement with this explicitly capped reference, not an
assertion that the cap was never reached. First-window and all-window variants
are compared independently against their non-reuse controls and the original
forward/reverse controls. Both compaction separation and refill are off.

Native host lane tests execute the seed exporter and window accumulator under
ASan/UBSan. They compare the exported CDF with direct counts, test all later
prefix thresholds against an independent Kth, cover ties/padding/extreme keys,
reverse scan and first-window freezing, and verify that exporting the seed
adds zero histogram operations. They do not establish CANN compilation or a
positive NPU timing result. The option defaults off.
