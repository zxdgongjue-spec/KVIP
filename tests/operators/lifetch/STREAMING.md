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
runtime shapes are separate work. The measured kernel source is preserved.

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

The baseline UBmem combined LI schedule is retained (no URMA sequence-split
transport is introduced). Final TopK runs the shared classification code.
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
