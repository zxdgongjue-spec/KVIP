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

## UBmem boundary

Existing main UBmem LIFetch remains supported through its existing entry point.
It does not use the streaming optimization. The new per-query Fetch worker
consumes URMA queue state, two Jettys per worker, WQE submission and CQ drain;
changing a transport argument cannot convert it to UBmem. Sharing Resolver,
output publishing and Index update requires a UBmem-compatible Fetch/completion
adapter and new correctness/performance validation. The CLI rejects the
unsupported combination rather than silently falling back.
