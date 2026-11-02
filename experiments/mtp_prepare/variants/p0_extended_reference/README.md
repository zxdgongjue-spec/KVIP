# A5 LIDU fused BatchGet prepare MTP

`A5LiduFusedBatchGetPrepareMtp` is an Ascend 950 custom operator.
It extends `A5LiduFusedBatchGetPrepare` from one TopK row per request to 2--4
packed MTP query rows per request, following the request grouping convention
used by `lidu_mtp_c8`.

## Inputs and outputs

- `topk_indices`: int32 `[T, 1, 2048]`, where `T` is the packed query count.
- `actual_seq_lengths_query`: cumulative int32 `[B]`; each adjacent difference
  must be in `[2, 4]` and the last value must equal `T`.
- `req_pool_entries`, `cache_tokens`, `candidate_lens`, and
  `logical_seq_lengths`: int32 `[B]`. Active `req_pool_entries` values must be
  unique within the batch.
- `token_to_slot_pool`: mutable int32 `[pool, source_capacity]`.
- `slot_to_token_pool`: mutable int32 `[pool, resident_capacity]`.
- `eviction_cursors`: mutable int32 `[pool]`; requests that select additional
  slots advance their pool-row cursor after a successful map update.
- `miss_slots`: int32 `[B, 2048]`; every request still supplies exactly 2048
  valid, unique slots.
- `miss_slot_bitmap`: int32 `[B, 512]`; bit `s` marks whether resident slot
  `s` belongs to `miss_slots`. The 512 words cover 16384 slots and include the
  UB-alignment padding above the supported 16256-slot resident domain.
- `resident_block_table`: int32 `[B, resident_block_capacity]`.
- `profile_enabled`: int32 `[1]`; `1` records per-lane request profiling and
  `2` enables graph-dummy TopK sanitization.

The public Torch API returns:

1. `destination_slots`: int32 `[T, 1, 2048]`.
2. `miss_counts`: request-level and total unique miss counts, int32 `[B+1]`.
3. `indices`: int32 `[B*8192]`.
4. `offsets`: int32 `[B*8192]`.
5. `output_indices`: int32 `[B*8192]`.
6. `resident_seq_lengths`: int32 `[B]`.

Only the globally compact descriptor prefix is written. The unused descriptor
tail remains caller-owned.

## Processing order

Each request is assigned a 1/2/4/8-AIV lane group, selected from the available
AIV count using the same rule as the non-MTP operator. If the batch is larger
than the available AIV count, the operator launches one lane per available AIV
and reuses the lanes in sequential waves:

1. Every lane DMA-loads the request's supplied `miss_slot_bitmap` into its
   private UB. The operator no longer constructs or merges membership from
   `miss_slots`.
2. Classify all 2--4 TopK rows against the old maps and membership, assigning
   complete query rows to lanes.
3. The leader lane compresses every query serially, preserving order in both
   groups: miss tokens first and hit tokens/hit slots last.
4. The leader lane DMA-loads each compacted miss prefix to UB, flattens the
   non-empty prefixes into one SIMT work domain, and deduplicates them with a
   UB source bitmap using `asc_atomic_or` and `asc_atomic_add`. The resulting
   unique-source list has no stable order.
5. When a request has more than 2048 unique misses, its leader merges every
   query's compacted hit-slot suffix into its private copy of the supplied miss
   bitmap and scans zero bits in circular cursor order. The first 2048 unique
   sources retain the input `miss_slots`; later sources use the selected
   additional slots. Every request finishes selection before any request
   changes a map.
6. All lanes scan the complete unique-miss list, but clear only victims in
   their cache-line-aligned forward-map domain. Installation uses the same
   forward-map ownership, with global barriers enforcing clear-before-install
   and install-before-refill ordering.
7. Refill complete query miss prefixes in parallel across lanes, with SIMT
   parallelism inside every lane-owned query. Clear and install use `asc_stcg`
   to update maps through L2 without retaining stale SIMT Data Cache lines and
   publish the stores with `asc_threadfence()` before the corresponding global
   barriers. Refill uses `asc_ldcg` to read the updated forward map directly
   from L2. The hit suffix is not rebuilt.
8. All operator cores globally repartition compact BatchGet descriptor cache
   lines, independently of request ownership.

In the sequential-wave fallback, all requests still complete the
classification/prepare phase before any request clears or installs a map. This
keeps the batch transaction boundary and clear-before-install ordering
unchanged while allowing `batch > aivCount`.

This means `miss_counts[b]` is the number of unique misses for request `b`.
`miss_counts[B]` is the total unique miss count across successful requests and
equals the valid compact descriptor prefix length. The internal per-query miss
counts determine the prefix refilled in each `destination_slots` row.

## Workspace layout

Every request reserves four independent int32 `[8192]` regions:

```text
classification       [B, 8192]
compacted_tokens      [B, 8192]
unique_miss_sources   [B, 8192]
resolved_miss_slots   [B, 8192]
request_control       [B, 16]
```

The compacted-token region never aliases or reuses classification storage.
Unused positions correspond to the fixed four-query capacity.

The request-level dedup bitmap, atomic count, and temporary unique-source list
are UB-only buffers. No request-level source bitmap or per-query unique list is
written to GM.

## More than 2048 unique misses

Every request can install up to 8192 unique misses. For a count `U > 2048`,
the operator constructs:

```text
resolved_miss_slots[0:2048] = miss_slots[0:2048]
resolved_miss_slots[2048:U] = additional unprotected cache slots
```

A slot is eligible when its bit is zero after OR-merging the supplied miss
bitmap with all 2--4 queries' hit slots. Selection scans `[0, cache_tokens)`
once, starting at `eviction_cursors[pool_row]`, and wraps at the cache boundary.
The cursor is committed only after clear, install, and refill succeed.

The request fails closed if there are not enough eligible slots or a resolved
slot is invalid. A defensive capacity check also rejects counts above the fixed
8192-entry MTP limit:

- maps and eviction cursors remain unchanged;
- every destination row for the request is filled with `-1`;
- `miss_counts[b]` is `-1` and `resident_seq_lengths[b]` is `0`;
- the failed request contributes `0` to `miss_counts[B]`;
- no descriptors are published for that request;
- profiling retains the observed unique count and reports `-106`, `-107`, or
  `-109` for capacity, eligible-slot, or slot-validation failures,
  respectively.

## Descriptor formulas

For compact ordinal `o`, request `b`, unique source token `t`, and resolved
destination slot `s`:

```text
indices[o]
  = (req_pool_entries[b] * num_layers + layer_id) * qcap + t / 128
offsets[o]
  = (t % 128) * 656
output_indices[o]
  = resident_block_table[b, s / 128] * 128 + s % 128
```

## Build and validation

On Ascend 950 with custom kernels enabled, the root vLLM-Ascend installation
automatically builds this operator into the packaged A5 OPP and embeds its
Torch registration in `vllm_ascend_C`:

```bash
python -m pip install --no-build-isolation -v -e .
```

For standalone operator development on an Ascend 950 machine with the CANN
9.1 environment loaded:

```bash
bash csrc/attention/nanovllm_a5_ops/a5_lidu_fused_batch_get_prepare_mtp/build_lidu_fused_batch_get_prepare_mtp.sh
```

CPU contract tests, which do not require `torch_npu`:

```bash
pytest -q csrc/attention/nanovllm_a5_ops/tests/test_lidu_fused_batch_get_prepare_mtp.py
```
