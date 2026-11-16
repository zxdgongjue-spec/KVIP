# Native URMA rolling threshold and candidate refilter

This opt-in experiment extends `perf/urma-fused-scan-admission`, based on main
`532df2c8` plus PR15. It does not change graph routing, KVAF, Index layout, the
window count, or the final TopK implementation. Both new options default off.

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

Persistent rolling state currently supports the non-sequence-split schedule.
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
