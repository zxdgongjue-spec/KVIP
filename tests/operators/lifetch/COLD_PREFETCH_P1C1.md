# P1-C1 — Real Cold KV Prefetch, mixed variable Q/K (controlled experiment)

Branch: `experiment/lifetch-prefetch-payload-p1c0` (continue on same branch).
Parent verified P1-C0: `ed238c89d44541acd588ca4ebdafadfc484ddc6f`.

P1-C0 demonstrated B4/H8192/R2048/G32/P64 two-round LI-only
Prefetch with 24/24 mapped tokens per round, CQ complete,
15,744 bytes DDR/HBM-equal per round, but **all tokens were resident**.
This C1 experiment constructs *real Cold TopK winners* and proves they
are transferred correctly from DDR and mapped into temporary HBM slots.

## Why changing miss-percent alone is not enough

Normal direct-hot LRU mode emits >2048 resident victim candidates,
scans from token index 0, and can exhaust its budget before reaching
cold history. For this focused diagnostic, leave the *production default*
LRU behavior unchanged, but launch a **cold-only probe** using an
opt-in `LIFetchBinding(enable_lru_slots=False)`:

- Set context `lruSlots=0` to skip victim classification and start the
  candidate scan at `resident`.
- Set `changeSlots=0` as well, because V9 Init otherwise copies a
  non-existent LRU table via `ctx->lruSlots`.
- PhysicalSlot then uses the logical slot directly, so speculative
  requests land in `owner.output[b, resident+offset]` instead of
  replacing Resident Token payload.
- Original semantics and launch arguments remain unchanged unless this
  *explicitly opted-in test* is used.

**This is not a production algorithm change** and is not yet a test of
Resident-victim vs Cold-prefetch budget competition.

## Deterministic Cold TopK fixture

`--cold-prefetch-probe` requires an already-enabled
`--li-only-varlen --prefetch-payload-audit --epoch-audit-profile` chain.
Set `--miss-percent 0.048828125`, exactly `100/2048` percent.

The fixture generates exactly one strong Cold Winner per Q:
`token = 2048 + 16*q`, `q=0,1,2,3`, all inside the first
Cold Tile (2048–2175). Its Query feature bands are independent,
so Native QLI and V9 QLI must both return one Cold Token per *active*
Query with actual K > 2048 and zero Cold Tokens when K = 2048.
All other TopK results remain Resident. Native/V9 complete TopK sets
must match, including padding checks, for both mixed-Q/K rounds.

The test still runs only `v9_init + v9_li + v9_wait` and does not
execute Resolver/Final Fetch/Index Update. No simulator/placeholder DMA
addresses are used: real KVAF URMA + real DDR payloads are exercised.

## Execution and evidence

On A5 (Ascend950DT_9582) in
`/home/z00945955/Fuse/gitcode_proj/third_party/KVIP`:

```bash
git pull --ff-only origin experiment/lifetch-prefetch-payload-p1c0
source scripts/env.sh
export ASCEND_RT_VISIBLE_DEVICES=6,7
export KVAF_SERVER=127.0.0.1:28999

python -m pytest \
  tests/operators/lifetch/test_cold_prefetch_probe.py \
  tests/operators/lifetch/test_prefetch_payload_audit.py \
  tests/operators/lifetch/test_epoch_audit.py \
  tests/operators/lifetch/test_rotation_epoch_schedule.py \
  tests/operators/lifetch/test_varlen_lengths.py -q

python tests/operators/lifetch/test_lifetch.py \
  --transport urma --device 0 --kvaf-server "$KVAF_SERVER" \
  --batches 4 --history 8192 --resident 2048 --index-heads 32 \
  --miss-percent 0.048828125 --prefetch-tokens 64 \
  --intermediate-fetches 8 --warmup 0 --iterations 1 \
  --trace-rounds 0 --li-only-varlen \
  --epoch-audit-profile --prefetch-payload-audit \
  --cold-prefetch-probe \
  --output /tmp/kvip-cold-prefetch-p1c1
```

For each round, examine:
- `li-varlen-stage.jsonl`: Native Cold Winner validation and V9 stage.
- `epoch-audit-mixed_qk_round{1,2}.json`: Core/Epoch/Selected counters.
- `prefetch-payload-mixed_qk_round{1,2}.json`: per-Row Token ID,
  logical and physical HBM Slot, CQ and byte-level payload checks.

Strict admission:
1. Native output has exactly one Cold Winner per active Q if K>Resident;
   correct physical ID `2048+16*q`.
2. Complete Native TopK sets and V9 TopK sets equal; all padding -1.
3. All Epoch-Lane events complete; zero device/Row/CQ errors.
4. `SelectedForPost = Row.requests = Mapping.count`;
   `CQ.completed_batch_count` is *group count*, not token count.
5. At least **one actual Cold Token in each round** has a valid Mapping,
   a unique temporary HBM Slot >= 2048, and an identical 656-byte DDR
   payload; **no Resident Token** is prefetched in cold-only mode.

If any criterion fails, report FAIL with JSON errors, do not silently
reinterpret the probe as successful. Until CQ is confirmed the original
KVAF quarantine mechanism must protect in-flight DMA.

Success establishes only isolated **Cold speculative Prefetch** and
one Cold TopK Winner per active Q. It does NOT verify that every
Cold TopK Winner was prefetched: skipped Epochs may leave some to the
Resolver/Final Fetch. It also does not demonstrate net TPOT improvement.

Next: P1-C2 introduces Resident/Cold competition, then full
LI + Resolver + Final Fetch + Index Update before GLM-5.3 four-layer
Index Share integration.
