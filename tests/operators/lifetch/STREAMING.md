# Integrated streaming validation

Base: main `81b1472809b7a5f3469e26467feee0ca1c61d71a`.
Validated source: `056f02e` on `codex/pr11-resolve-fetch-early` (kernel
changes last updated at `5739e28`). This branch retains one streaming
operator, its PR11 prerequisites, correctness checks and saved-trace analysis.
Historical early-fetch, safe-slot, staged-only, fused-only and simulator copies
are intentionally not included. PR4 is not included.

## Supported path

URMA, Q4, BS1..16, history131072, resident16384, index_heads32,
prefetch budget at most512. The measured configuration uses budget512,
requested windows8, two Fetch workers per query and prepared Index metadata.
The actual window count remains dependent on BS, as in the tested version.

`--optimized-resolve` selects the measured options automatically. This remains
an explicit operator test path: the public `kvip.LIFetch` / vLLM service path
is not silently switched to this implementation. Model integration and general
runtime shapes are separate work. The URMA algorithm is preserved; the LI entry and sequence scan are shared with UBmem.

```bash
python tests/operators/lifetch/test_lifetch.py --transport urma \
  --optimized-resolve --device 4 --kvaf-root /path/to/KVAF \
  --kvaf-server 127.0.0.1:28885 --batches 16 --history 131072 \
  --resident 16384 --index-heads 32 --intermediate-fetches 8 \
  --prefetch-tokens 512 --miss-percent 10 --warmup 4 --iterations 20 \
  --trace-rounds 3 --timeout-seconds 900
```

`compare_matrix.py` retains the 20-shape, three-path comparison and reuses
one binary per source version. `support/summarize_core_timeline.py` and
`support/analyze_prefetch_pressure.py` analyze saved traces without a device.

## UBmem backend

The same `--optimized-resolve` option is now available with `--transport ubmem`.
The staged Resolver, output publisher and Index update are shared with URMA.
UBmem retains its existing CPU-service lease, mapped region and window-copy
implementation. Each published Q calls the existing KVAF `KvafBatchGetDevice`
UBmem overload; the overload drains its MTE pipeline before completion is released.
No new KVAF patch or worker-local allocator is required.

Both optimized transports compile the same LI entry translation unit and use
one host launch schedule, sequence-shard scan and Final TopK classification.
For aligned sequence windows, Final TopK preloads metadata and overlaps the
second metadata half with classification. Uneven window geometry and packed
query lengths use the common compatibility path, not a UBmem-specific path.
In this fixture BS2 has four uneven windows and uses that compatibility path
on BOTH transports. The baseline library remains unchanged for A/B.
The optimized path supports the same restricted geometry above and both 656/
2624-byte KV records. It remains an operator test path, not a vLLM default.

Two adjacent 656-byte slots can share one 32-byte write sector. Final Q copies
are therefore ordered Q1->Q2->Q3->Q4 **within each request**. Independent requests
remain sharded across 1/2/4 workers per Q, and copies overlap later Resolver and
Index work. Empty/error Qs publish completion too. Slot validity is published
only after copy completion and Index update; failures invalidate the owner.
UBmem has no URMA CQ. Do not interpret the compatibility trace's `*_cq_wait`
fields as communication measurements; `ubmem_queries` explicitly reports
Q-ready wait, previous-Q-copy wait, KVAF MTE Get and Index/completion join.

Run the baseline and optimized path with identical parameters:

```bash
for mode in baseline optimized; do
  options=()
  if [ "$mode" = optimized ]; then options+=(--optimized-resolve --q-workers 2); fi
  python tests/operators/lifetch/test_lifetch.py --transport ubmem \
    --kvaf-root /vllm-workspace/KVAF-ubmem --kvaf-server 127.0.0.1:28905 \
    --soc Ascend950PR_9599 --device 4 --batches 1,2,4,8,16 \
    --history 131072 --resident 16384 --index-heads 32 \
    --intermediate-fetches 8 --prefetch-tokens 512 --miss-percent 10 \
    --warmup 4 --iterations 20 --trace-rounds 3 --timeout-seconds 1800 \
    --output "tests/results/ubmem-streaming-$mode" "${options[@]}" || break
done
```

Use an already built UBmem KVAF checkout and CPU service (see the root README
for initialization). Each mode builds one library and reuses it across batches.
Both run KV-byte, TopK/SFA, Index mapping, repeated replay, owner permutation,
packed-query and padding checks. Profiling replays are separate from timing.
Full per-core/per-Q cycles are saved in `trace/*/samples.json`; graph stages and
counts are in `trace/*/attribution.json`, with raw `trace.json` retained.

Local validation covers device compilation and host control-flow tests. Hardware
correctness and performance of this new UBmem path still require the command above.

## Interpreting URMA baselines and saved traces

The default URMA operator on this branch includes the PR11 sequence-LI
prerequisites. It is not pristine main `81b1472`. Compare against a pinned main
checkout when reporting gains over main. The `prototype_with_update` metric is
the optimized full graph; `prototype_core` and standalone update measurements
are independent reference graphs. Full graph traces are printed separately.

`support/compare_transport_tail.py --urma URMA_RUN --ubmem UBMEM_RUN` reads saved
full attributions, reports per-BS fused medians and per-round last-finishing
cores, Fetch-core spans, counts and UBmem Q-ready/previous-Q/MTE/join costs.
It never runs the device. Pass one exact run directory for each transport.


## UBmem concurrent Q copies and BS2 aligned LI

`--optimized-resolve --ubmem-fetch-mode overlap` uses the existing KVAF device
Get/MTE pipeline on each Q worker. It copies disjoint target sectors first and
waits only for earlier Qs sharing a target sector before copying conflicts.
656-byte rows in adjacent even/odd slots share a 32-byte write sector; 2624-byte
rows do not. Classification checks all earlier Qs, including across empty Qs.
Copy completion and Index completion are still both required before publishing
valid slots. Errors and timeout publication remain enabled. URMA is unchanged.
`--ubmem-fetch-mode serial` retains whole-Q ordering as an A/B control, using the
same compiled library. BS1 still has one active request lane per Q; this change
exposes parallelism across Qs rather than repartitioning one Q's entries.

Run both modes, with full graph timing and separately profiled per-core/per-Q
stages, using an existing UBmem CPU service:

```bash
bash tests/operators/lifetch/compare_ubmem_fetch.sh "$UBMEM_KVAF_ROOT" 4 "$UBMEM_SERVER" 1,2,4,8,16
```

The summary reports safe/conflicting entry and Get-call counts, classification,
Q-ready wait, safe Get, dependency wait, conflicting Get, and Index/completion
join time. Get timings include the KVAF MTE drain; UBmem does not wait on URMA
CQs. Existing KV byte, native TopK/SFA, Index permutation, replay, owner remapping,
packed-query and padding checks run in both modes. An incomplete profile is not
performance evidence even if correctness passes.

For H131072/R16384 BS2, four windows no longer force combined-LI fallback.
Cold shard size is rounded up to 19200 tokens; the final shard clips at history.
Cold windows contain 38400,38400,37888 tokens, with ends 54784,93184,131072;
hot ends at16384. Both transports share this schedule and candidate scan bounds.
LI tiling, total history, window count and prefetch budget stay unchanged.
Other unsupported geometries still use the general path; it is not safe to delete
that path for head64 or irregular per-request lengths as part of a BS2 fix.
The A/B above isolates copy scheduling with this same LI schedule in both modes.
Use pinned commit d975d39 with the same hardware command for a pre-change LI
comparison; retain its checkout separately instead of replacing its sources.


The A/B script remembers the KVAF checkout, device, service and batches under
`tests/results/ubmem-fetch-settings.json`. After one invocation with explicit
arguments, subsequent runs are simply `git pull --ff-only` followed by
`bash tests/operators/lifetch/compare_ubmem_fetch.sh`. It loads the CANN environment
itself. The public test launcher prepends the selected KVAF Python directory for
its child process, so a previously sourced environment from another checkout
cannot override `--kvaf-root`. No KVAF rebuild or kernel change is involved.
