# Opt-in URMA work after the recent-index threshold

These experiments extend the consolidated recent-index change, based on
KVIP main `13d2898892652382ea36a0652030874efc31097d` and PR !15 commit
`4385ac75435c3e872b5e77a4b66e98f8fa5838a8`.
Every option defaults to **OFF**. None is a measured performance improvement
until the complete device graph has been compared on the target NPU.

| Build option | Change | Required invariant |
| --- | --- | --- |
| `KVIP_FINAL_TOPK_PRUNE` | Compact final score blocks to keys at or above the exact early Kth bound before histogram selection. | Keep every equal key, the original block order and GT/EQ tie order. Unsupported shapes and a zero bound use the original path. |
| `KVIP_SELECTED_METADATA` | Read hot-valid/reservation metadata only for selected physical IDs, instead of loading the entire resident table per query. | Join mapping publication; preserve ABC, reserved-slot, error and protection-bitmap behavior. |
| `KVIP_EARLY_FETCH_READY` | Publish descriptors and FinalPost before TND output DMA. | OutputWorker exclusively publishes flag 5 after all outputs; metadata readers join that flag. Errors release all required waiters. |
| `KVIP_COMPACT_INIT` | Fill with a 64 KiB buffer and clear only the streaming prefetch prefix of `changes`. | The mapping is still fully cleared. Streaming never consumes the extended `changes` suffix; legacy/UBmem paths retain full clearing. |

Python `kvip.runtime.build.build_libraries` accepts the corresponding boolean
arguments `final_topk_prune`, `selected_metadata`, `early_fetch_ready`, and
`compact_init`. Build cache keys include every option; the native bridge exports
their compiled values so replay can reject mismatched binaries before capture.
There are no extra kernels or new runtime tensors. UBmem does not enable these
experiments.

Final pruning currently targets the general BS16/Q4, 32-AIC, uint16-score,
K=2048, 16K-chunk path. It reuses the completed Cube result bank for offsets;
TopK-only/sequence-split and other unsupported configurations fall back. The
full LI scoring pass remains unchanged. The extra compaction/gather/remapping
work must be included when deciding whether pruning is beneficial.

For resident=16384, selective metadata reads at most 16 KiB of scalar payload
per query instead of loading 128 KiB. Cache-line traffic and random access may
offset that reduction. For BS16, compact initialization saves 0.5 MiB of
`changes` clearing; it does not remove the approximately 8 MiB mapping clear.
Early Fetch changes scheduling, not the number of bytes requested. Its existing
output-stage profile marker has a different boundary, so headline comparisons
must use the complete graph, not that marker's difference.

Validation uses actual helper code in CPU models, including delayed DMA/vector
writes, poisoned previous-round state, BF16 ties, fallback shapes and error
publication. These are correctness checks, not NPU timing simulators. CANN
compilation checks the native code. Target validation must additionally verify
real score bits, ordered TopK, thresholds, KV/Index outputs and full-graph
mean/p50 for G2/G7/G10/G17/G4. Keep the strong serial reference at 32 workers,
post size 128 and max inflight 2048, with LIDU and slot mapping included.

Fetch queue ownership, Get worker allocation, bounded ready batching and
adaptive rolling thresholds are unchanged. They require separate protocol and
performance evidence; these options do not claim to implement them.
