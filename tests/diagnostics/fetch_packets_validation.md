# Producer-packet Fetch experiment

Native source: `bfc6510c7d0b8b5ccf86c9aa7d61b765a2b57aae`.
Run `run_list_segments_host20.sh packets` for G7/G2, physical NPU4,
BS16/Q4, 656-byte KV, eight cold windows plus the resident phase,
64 submissions per turn and 512 per request. No model service is started.

The candidate adds producer packets, budget-driven per-query rank catch-up,
and immediate submission of each ready request. Forward and reverse controls
keep their previous admission and batched-post code. All experiment flags are
off by default.

A packet contains the original T0-pass count and up to 16 score/offset pairs
for 128 scores. Overflow causes a full 2K score-block fallback, preserving
encounter order and ties. Normal packet consumption processes four tiles per
vector. Queries blocked by admission budgets leave their producer packets
and unsent winners intact; they catch up before their next admission.
The preceding service event still protects request state and transport queues.

## Local validation

CANN 9.2.0, dav_3510 simulator; compiled target `Ascend950DT_9582`.
The producer fixture invokes the same register packer as LI. The consumer
fixture invokes the production ranking and admission functions on real scores.

| Check | G7 | G2 |
| --- | ---: | ---: |
| Device producer tiles checked | 57472 | 57472 |
| Exact ordered Top2048 prefixes | 576/576 | 576/576 |
| Exact ordered Q4 submission lists | 144/144 | 144/144 |
| Demand-driven ranks materialized | 451 | 576 |
| Demand-driven submission lists | 144/144 | 144/144 |

A separate native test forced packet overflow on a valid tile and retained
all 36 query prefixes and 9 request submission lists for the tested request.
CPU tests additionally cover empty, sparse, overflowing and dense packets,
causal/seed masking, skipped queries, delayed frontiers, repeated frontiers,
Q4 deduplication and exhausted budgets. 97 relevant tests passed (the
96-test group and the additional lazy-rank regression test).

These are correctness results, not a measured URMA speedup. The simulator
fixtures do not reproduce live LI/Fetch scheduling or URMA completion latency.
In particular, G2 does not save rank updates through budget gating.

## Hardware decision

The runner keeps each variant in its own process. Its headline graph timing
includes LI packet production, rank maintenance, posting, final repair and
Index updates. Separate instrumented diagnostics run after headline timing,
reusing the restored DDR rather than launching another model or restoring it
for a separate diagnostic process. The final frozen fixture checks all
prefixes even when the live path did not need to materialize them.

The report prints two full-graph mean/P50 samples, drift, wasted-prefetch bytes,
repair bytes and total bytes. `GATE packets AHEAD` requires the slower
candidate measurement to beat the faster control measurement for mean and
P50 against **both** forward and reverse, after deducting the larger measured
repeat drift. `PERFORMANCE_GATE BOTH_AHEAD` requires both G7 and G2 to pass.
Otherwise the result is `KEEP_BASELINES`; reduced transfer volume alone is
not considered a performance improvement.
