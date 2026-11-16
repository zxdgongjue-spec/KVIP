# Low-miss read-through in the fixed graph

This experimental build option changes cache admission in the low route. It is
disabled by default and does not replace the existing promotion path.

The previous successful round's feedback and the current request identities and
geometry still select a route on the device. Both routes use the same captured
graph and fixed tensor addresses. Request replacement invalidates old feedback;
neither route depends on a CPU readback of miss counts.

## Persistent-state contract

The resident KV and the hot-first Index are a matched pair. Skipping an Index
swap while overwriting its resident KV would break the next fused round. The
read-through route therefore changes neither side of that pair:

1. LI scans the complete valid history and computes the exact TopK.
2. A per-request producer identifies resident/ABC hits and deduplicates cold
   selections across Q4.
3. Cold values are fetched into temporary output slots, disjoint from hot/ABC.
4. Attention receives the exact hot, ABC or temporary slot for each selection.
5. Successful completion publishes actual miss feedback; resident KV, Index
   keys/scales, address order, logical IDs and hot-valid state stay unchanged.

The next high route can immediately use that resident/Index pair. No delayed
repair, Index permutation, or rebuild is owed. It performs the normal promotion
and Index update before another round starts. Temporary slots may be reused only
after their attention consumers have completed on the ordered execution stream.

## Removed work and remaining work

Low execution omits early-threshold/prefetch work, victim selection, resident
protection, promotion planning, Index movement and residency publication. Index
consumer/output-worker roles return without performing those tasks. The low Get
workers do not wait on an Index-update notification.

Exact selected-ID classification, cross-query deduplication, Get descriptors,
completion/error checks and attention slot outputs remain necessary. This is
not a claim that every launch or every device instruction has disappeared.

## Layout and prediction errors

Each owner reserves 8192 temporary slots: the maximum union of four TopK-2048
queries. A false-low prediction therefore changes cost, not correctness, even
when every selection is cold. The original ABC capacity is carried separately
from the enlarged output stride, so its ring modulo never includes scratch.
For 16 active requests and 656-byte tokens, scratch is 82 MiB; an allocation
with 17 stable owner rows reserves 87.125 MiB.

The binding validates the configuration before capture. Feature-off libraries
keep their original Context ABI. Feature-on libraries have a checked extra
configuration pointer and a distinct build-cache key.

## Measurement limits

Not promoting cold values is an admission-policy change. Repeated low rounds may
read the same cold value again, and ABC retirement may increase this effect.
Single-snapshot timings cannot establish an end-to-end gain. Compare:

- fresh strong-serial and previous-auto controls before and after the candidate;
- all-round and steady mean/median, including the initial high round separately;
- exact output payloads, persistent-state equality on low rounds, and a single
  captured graph carrying state through low/repacked-low/high without restoring;
- repeated-read exposure in the 100-round token trajectory.

The trajectory analysis is a traffic counterfactual, not a latency measurement
or a model of the complete automatic route policy. Local CPU tests establish
state and planning contracts; CANN compilation and NPU performance still need
target validation.
