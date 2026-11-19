# URMA dynamic Top2048 migration

This opt-in experiment ports dynamic TopK from `codex/lifetch-ranked-integration`
(`e2315cf9ff88939ff36ed291c59daa6423ca6994`) onto the current URMA replay source
(`bfc6510c7d0b8b5ccf86c9aa7d61b765a2b57aae`). It does not import the donor's
init/resolver, Index-update, KV transport, top1024 admission or per-query quota.

## Comparison contract

All three variants use BS16, Q4/MTP3, 656-byte single-layer KV, eight cold Fetch
windows, eight AIVs per window, and per-request budgets 64/window and 512 total.
Resident+recent seed scoring prepares the initial Top2048; it is not an extra
Fetch turn. Final repair is separate. Controls use the same corrected schedule,
so previous nine-turn/six-AIV timings must not be substituted as controls.

The candidate scans forward. The controls are the unchanged forward selection
policy and pure reverse scanning, both with dynamic TopK disabled. UBmem keeps
its original scheduling. No graph-route switching or end-to-end service changes
are included.

## Migrated computation

Each query retains its native Top2048 scores/physical IDs and its last consumed
score position. A Fetch lane owns two requests. It snapshots already-published
progress, updates each query independently using fresh above-threshold scores,
and invokes the existing LI TopK primitive. Unready queries do not block other
queries. A missed intermediate version is caught up directly; it is not replayed
one historical window at a time. Final selection reuses the accumulated result.

The next query's score transfer overlaps the current query's rank update. The
prefetch stage holds 8K score positions (22 KiB including the old scores and
bitmap), fitting alongside the unchanged TopK scratch. The donor's 40 KiB
stage would disable this overlap under the current source's UB capacity guard.
An interval larger than 8K is consumed by the existing chunk loop. The
initial-threshold bitmap remains the input filter. Newly rejected candidates
cannot re-enter later because the exact prefix Kth is monotone. Budget-deferred
IDs stay in the persisted Top2048 and are reconsidered while still selected.
Original slot reservation deduplicates Q4 admissions before posting.

## Rank and posting ownership

In the host20 results for native 7e3f73b, the mean rank body was 27.92 us on
G7 and 21.13 us on G2. Waiting for the previous service to finish accounted for
188.95 and 193.04 us per lane/window respectively (diagnostic replay).
Both that revision and donor 262bf7a joined the predecessor's complete event
before ranking. The donor tags resident IDs at seed and exports compact offers
with a score quota per query. The old port instead reclassified complete TopK
lists after rank publication. Its 30-us rank timer excluded that work.

The new producer tags only safe resident hits in its private seed ID copy.
Cold updates never introduce resident positions. The usual ID-remap vector
pass therefore packs untagged offers in order; a window reusing a prior rank
packs the saved list. Victims and invalid hot entries remain eligible, and
the map still handles ABC and prior admissions. Public output removes the
private tag and preserves padding. A single request still shares the 64-ID
window budget across its four queries.

Each epoch owns a separate offer region (4 MiB capacity at BS16). Only the
padded nonempty prefixes are written. Counts stay on the service AIV and a
zero count suppresses reads, so the arena needs no clearing. After publishing
rank state, the producer sets rankReady. The next owner can advance the rank
while the previous owner prepares and posts its WQEs. Posting still waits for
complete before consulting the mutable map, row or queue. Finalization joins
the last complete post. Ranked ABI 140 identifies the new offer-buffer pointer.

## Avoiding migration overhead

- Progress is captured once by ranking. Fetch does not read it again.
- Rank state is only 128 bytes/query. All-ready Q4 headers use one 512-byte row
  write; only a partially ready row writes individual ready query headers.
  This prevents overwriting concurrent seed publication without adding another
  readiness read or synchronization. A CPU regression executes this native copy
  loop with a simulated concurrent publisher and checks the common DMA count.
- Rank metadata no longer fetches unused row headers. No per-window clearing of
  whole TopK/state tables is added. Round initialization clears 8 KiB metadata.
- Existing seed input scores remain intact under native TopK, preserving the
  local resident protection-bit calculation.

## Validation and limits

CPU rank checks use actual G7/G2 sortable score keys, causal bounds and recent
bounds, including BF16 boundary ties and independently delayed queries. Native
CANN checks execute the migrated update with the same eight-window geometry,
two representative lanes (two requests each) and the same score inputs. They
validate functional prefix ranks and the next-query pipeline.
They do not simulate URMA transport or establish whole-graph performance.

The admission regression executes the native seed classifier on CPU. On frozen
G7/G2 rank lists, producing all eight offer buffers before draining them in
order reproduced all 512 ordered query/window admissions per group. Local CANN
probes exercise seed tagging, remap-and-pack, fallback packing and public ID
decoding on two full lists per group. Counts and ID order are checked against
the CPU reference. These probes do not run the URMA queue or measure overlap.

CANN 9.2.0 builds both feature-off and feature-on binaries. Local model tests use
Ascend950PR_9589; host20 is Ascend950DT_9582. This hardware/model difference and
preloaded score inputs prevent using model elapsed time as a hardware prediction.
The real replay checks LI scores, final rank validity, selected KV and Index
association against the frozen archive. BF16-equal IDs can differ, so traffic is
counted from each run's actual selection.

Headline timing covers LI, seed/rank work, Fetch, repair and Index update. Warmup,
restoration and diagnostic readback are excluded. Separate markers distinguish
rank predecessor wait, rank work (including offer export), post predecessor
wait, the sum of both waits, and the interval from open to the first Post or return. First Post is
neither necessarily useful nor completed. The donor's approximately 30 us rank
measurement must not be described as the complete window cost.

Each variant runs in a fresh process; first/reverse-order repeats expose baseline
drift. A candidate is kept only if both whole-graph mean and P50 beat both controls
with margin. Functional success does not establish a positive performance result.
