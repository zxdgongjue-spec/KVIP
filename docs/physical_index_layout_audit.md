# Physical Index layout audit

This diagnostic separates two questions: does reversing physical blocks expose
the recorded final cold demand earlier, and does executing the reverse kernel
change persistent hot/Index state? It changes no native algorithm or defaults.

## CPU trace survey

`python -m kvip.diagnostics.layout_survey --root TRACE --result-dir NEW_DIR
--workers 16 --cold-windows 8` inspects the existing 8-DP, 101-round series and
analyzes rounds 2–101. Round 1 is the predecessor. All 78 layers are represented;
layers in one Index-sharing group share the same measured cache state. Q4 final
TopK positions are unioned per request, so duplicate KV demand is counted once.

Only ID/cache-state NPZ members are decompressed. Series/manifests, member CRC,
dtype/shape, full Index permutations and physical-to-logical TopK agreement are
checked. Scores, Q, real keys and payloads are not read; the whole archive SHA
is not recomputed. No NPU or model service is used in this phase.

The permutation is the native `cold_scan.h` rule: reverse complete 2048-slot
blocks valid for all four queries, preserving the resident prefix, order within
each block and the partial causal tail. It never sorts Index rows by token ID.
Window boundaries use 128-slot tiles. Eight cold windows are separate from the
resident publication. The native replay consequently requires **9 rounds**;
the former native value 8 represented resident + 7 cold windows.

`scanEarly` is the fraction of final cold demand encountered before cold window
8. `scanLead` weights each encountered token by the remaining scan fraction at
its publication boundary. Both are discovery opportunities: there is no budget,
threshold/admission, transport completion or latency model. Recent scores do not
advance candidate publication in this native fixed-threshold baseline. Invalid
resident slots and demand outside the prefetch domain are separately counted.

Outputs:

- `physical_blocks.csv.gz`: DP/request/round/group, physical block, cold demand,
  token-ID range, displaced entries and cross-block displacement.
- `requests.csv.gz`: per-request miss, displacement, adjacency breaks,
  position/token correlation, per-window discovery counts and predecessor changes.
- `round_layers.csv`: all analysis rounds and layers, aggregating the eight DPs.
- `summary.json`: layer-weighted totals, each group's first/last 20 rounds,
  gains, win/loss counts, quantiles and correlations.

Compare trends **within each group**. Declining discovery gain together with
increasing displacement is evidence of association, not proof of causation:
queries, cold demand and score distributions also change. The captured series
may already begin after substantial reordering. A frozen original state per
round is not a counterfactual rollout of the reverse policy.

## Native state audit

Use the existing `kvip.diagnostics.replay` entry with `--study scan-order
--scan-variants forward reverse reverse-end forward-end --cold-windows 8
--compare-layout`. Only the scan direction differs. Both directions use the
same 9-publication scheduling and budgets 64/512; this is not a fair timing
comparison to an earlier run with 8 native rounds.

The override changes only in-memory scheduling geometry; input files and trace
arrays stay intact. Full graph timing retains LI/TopK/Fetch/Index update/output
work. Separate post-state readback is outside timing. Existing oracles check
ordered TopK and thresholds, Index key/scale/address association, selected KV
bytes, and every valid hot KV byte.

The extra audit distinguishes physical layout equality from equal hot token
sets. It hashes causal Index IDs/keys/scales, relative DDR addresses, hot-valid
bits and valid hot payloads by slot and by token. Invalid hot payload bytes,
future positions and inactive owners are excluded. Difference counts include
changed Index positions, changed hot slots and hot-set symmetric difference.

Same-direction F/F and R/R repetitions reveal nondeterminism. A cross-direction
difference alone does not isolate the direction's effect when repetitions also
differ. Matching one-step post-state on five real inputs does not prove equality
over every future round; if states differ, subsequent closed-loop behavior needs
its own investigation. No live service or generated future query is invented.

`python -m kvip.diagnostics.layout_state RESULT.json` prints only the timing,
transfer and state summaries. Mean and median are separate; conservative gain
uses the slower reverse result and faster forward result. Both drift references
are reported.
