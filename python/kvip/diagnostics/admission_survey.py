# SPDX-License-Identifier: Apache-2.0
"""CPU admission counterfactuals from complete score snapshots, without NPU timing."""

import argparse
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import array_digest, load_snapshot
from kvip.diagnostics.demand import SnapshotDemand
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.score_replay import score_value

TOPK = 2048
# name: (ordered-key boost, total/request, nominal window/request, carry credits)
PROFILES = {
    "baseline": (0, 512, 64, False),
    "carry": (0, 512, 64, True),
    "window128": (0, 512, 128, True),
    "total768": (0, 768, 128, True),
    "total1024": (0, 1024, 128, True),
    "window480": (0, 1024, 480, True),
    "boost1": (1, 1024, 480, True),
    "boost2": (2, 1024, 480, True),
    "boost4": (4, 1024, 480, True),
}


def _array(archive, name, shape, dtype):
    try:
        value = np.asarray(archive.array(name))
    except KeyError as error:
        raise ValueError(f"Missing {name}; regenerate a complete score snapshot with recent bounds") from error
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"Snapshot {name} has unexpected shape/dtype: {value.shape}/{value.dtype}")
    return value


def _requests(archive):
    g = archive.manifest["geometry"]
    if archive.manifest.get("content") != "real_li_synthetic_kv":
        raise ValueError("Admission survey requires a materialized real-LI score snapshot")
    batch, history, resident = (g[name] for name in ("batch", "history", "resident"))
    if history % 128 or resident % 128 or not TOPK <= resident < history:
        raise ValueError("Admission survey requires valid 128-token-aligned resident/history ranges")
    demand = SnapshotDemand(archive)
    metadata = _array(archive, "pre.metadata", (5, batch * 4), np.int64)
    if not np.all(metadata[3] == 1) or not np.array_equal(np.sort(metadata[2]), np.arange(batch * 4)):
        raise ValueError("Admission survey requires all four active queries per request")
    owners = metadata[4].view(np.int32)[:batch]
    ids = archive.array("pre.logical_ids")
    addresses = _array(archive, "pre.addresses", (len(ids), history), np.int64)
    ends = _array(archive, "pre.query_ends", (batch * 4,), np.int32).reshape(batch, 4)
    recent = _array(archive, "pre.threshold_bounds", (batch, 2), np.int32)
    abc = _array(archive, "pre.tail_bounds", (batch, 2), np.int32)
    scores = _array(archive, "post.score_keys", (batch, 4, history), np.uint16)
    output = _array(archive, "post.logical_output", (batch * 4, TOPK), np.int32)
    if np.any(ends < TOPK) or np.any(ends > history) or np.any(np.diff(ends, axis=1) < 0):
        raise ValueError("Invalid causal query ends")
    positions = np.arange(history)
    requests = []
    for row in range(batch):
        logical, forward, lru = demand.prefetch_requests[row]
        inverse = np.argsort(logical)
        begin, stop = map(int, recent[row])
        if not 0 <= begin <= min(int(abc[row, 0]), int(ends[row].min())) or stop != ends[row].max():
            raise ValueError("Recent bounds do not cover the retired ABC-to-current-query range")
        if not np.array_equal(logical[begin:stop], positions[begin:stop]):
            raise ValueError("Recent logical/physical Index positions differ")
        if np.any(np.maximum.accumulate(logical)[ends[row] - 1] >= ends[row]):
            raise ValueError("A physical causal prefix contains future logical IDs")
        slots = forward[logical]
        cold = slots < 0
        candidate_hot = (slots >= 0) & (slots < resident) & np.isin(slots, lru)
        readable = addresses[owners[row]] != 0
        eligible = (np.isin(positions, lru) | (positions >= resident)) & readable & (slots < resident)
        selected = np.zeros(history, dtype=bool)
        thresholds, final_keys, resident_keys = [], [], []
        for q in range(4):
            end = int(ends[row, q])
            keys = scores[row, q, :end]
            if np.isnan(score_value(keys)).any():
                raise ValueError("NaN score inside a causal query")
            chosen = output[np.flatnonzero(metadata[2] == row * 4 + q)[0]]
            if np.any(chosen < 0) or np.any(chosen >= end) or len(np.unique(chosen)) != TOPK:
                raise ValueError("Captured TopK IDs are not distinct causal tokens")
            physical = inverse[chosen]
            if np.any(physical >= end):
                raise ValueError("Captured selection lies outside the physical score prefix")
            final = int(np.partition(keys, len(keys) - TOPK)[-TOPK])
            picked = keys[physical]
            if np.any(picked < final) or np.count_nonzero(picked > final) != np.count_nonzero(keys > final):
                raise ValueError("Captured TopK omits a higher-scored token")
            seed = recent_positions(resident, recent[row], end)
            if len(seed) < TOPK or min(resident, end) < TOPK:
                raise ValueError("Threshold seed contains fewer than K positions")
            tau = int(np.partition(keys[seed], len(seed) - TOPK)[-TOPK])
            hot = int(np.partition(keys[:resident], min(resident, end) - TOPK)[-TOPK])
            if not hot <= tau <= final:
                raise ValueError("Resident/recent threshold is not a valid lower bound")
            thresholds.append(tau)
            resident_keys.append(hot)
            final_keys.append(final)
            selected[physical] = True
        needed = selected & (cold | candidate_hot)
        if np.any(needed & ~readable):
            raise ValueError("Selected cold/candidate-hot payload lacks a readable DDR address")
        order = positions.copy()
        order[:resident] += history
        order[lru] = np.arange(len(lru))
        requests.append(dict(scores=scores[row], ends=ends[row], eligible=eligible, order=order,
                             resident=resident,
                             selected=selected, needed=needed, cold=cold, candidate_hot=candidate_hot,
                             tau=np.asarray(thresholds), final=np.asarray(final_keys),
                             resident_tau=np.asarray(resident_keys)))
    return requests, demand.classify(output), array_digest(output)


def _gate(request, boost):
    gate = np.zeros(request["eligible"].shape, dtype=bool)
    for q, end in enumerate(request["ends"]):
        # Ordered BF16 key steps, not a relative score percentage. A boost is
        # allowed only for speculative admission, never final TopK pruning.
        lower = int(request["tau"][q])
        limit = min(65535, lower + boost)
        prefix = min(request["resident"], int(end))
        # Resident victim protection must use the exact, unboosted lower bound.
        gate[:prefix] |= request["scores"][q, :prefix].astype(np.int32) > lower
        gate[prefix:end] |= request["scores"][q, prefix:end].astype(np.int32) > limit
    return gate & request["eligible"]


def admit(gate, available, total, window, carry, weights, order=None):
    """Quota-rejected candidates go to final repair, not a later-window queue."""
    fetched = np.zeros(gate.shape, dtype=bool)
    cumulative = 0
    for phase, weight in enumerate(weights):
        quota = min(total, int(window * weight))
        cumulative += quota
        posted = int(fetched.sum())
        allowance = max(0, cumulative - posted) if carry else quota
        count = min(480, allowance, total - posted)
        candidates = np.flatnonzero(gate & (available == phase))
        if order is not None:
            candidates = candidates[np.argsort(order[candidates], kind="stable")]
        chosen = candidates[:count]
        fetched[chosen] = True
    return fetched


def survey(archive, profiles, phase_weights=None):
    requests, demand, selection_hash = _requests(archive)
    g = archive.manifest["geometry"]
    windows, resident, width = (g[name] for name in ("rounds", "resident", "token_bytes"))
    if not isinstance(windows, int) or windows <= 0:
        raise ValueError("Snapshot must declare a positive window count")
    weights = np.ones(windows) if phase_weights is None else np.asarray(phase_weights, dtype=float)
    if weights.shape != (windows,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Phase weights must contain one finite nonnegative value per window")
    result = []
    for name in profiles:
        boost, total, window, carry = PROFILES[name]
        counts = dict(eligible_candidates=0, useful_prefetch=0, unused_prefetch=0, final_missing=0,
                      cold_prefetch=0, candidate_hot_prefetch=0, selected_suppressed_by_boost=0,
                      selected_excluded_by_gate=0, boosted_above_final_queries=0)
        for r in requests:
            gate = _gate(r, boost)
            base = _gate(r, 0)
            positions = np.arange(len(gate))
            # Match Trigger(): epoch 0 covers resident only, followed by the
            # equal tile-progress cold windows. ScoreRecent publishes no hits.
            boundaries = resident + ((len(gate) - resident) // 128) * np.arange(windows) // max(1, windows - 1) * 128
            available = np.searchsorted(boundaries, positions, side="right")
            fetched = admit(gate, available, total, window, carry, weights, r["order"])
            useful = fetched & r["needed"]
            values = (gate.sum(), useful.sum(), (fetched & ~r["selected"]).sum(),
                      (r["needed"] & ~fetched).sum(), (fetched & r["cold"]).sum(),
                      (fetched & r["candidate_hot"]).sum(), (base & ~gate & r["needed"]).sum(),
                      (r["needed"] & ~gate).sum(), np.count_nonzero(r["tau"] + boost > r["final"]))
            for key, value in zip(counts, values):
                counts[key] += int(value)
        prefetched = counts["useful_prefetch"] + counts["unused_prefetch"]
        assert counts["useful_prefetch"] + counts["final_missing"] == demand["selected_read_demand"]
        result.append(dict(name=name, key_boost=boost, total_per_request=total, window_per_request=window,
                           carry_unused_credits=carry, **counts, prefetch_bytes=prefetched * width,
                           unused_bytes=counts["unused_prefetch"] * width,
                           final_bytes=counts["final_missing"] * width))
    tau = np.concatenate([r["tau"] for r in requests])
    final = np.concatenate([r["final"] for r in requests])
    hot = np.concatenate([r["resident_tau"] for r in requests])
    return dict(status="pass", group=archive.manifest.get("metadata", {}).get("group"), demand=demand,
                geometry={name: g[name] for name in ("batch", "history", "resident", "token_bytes", "rounds")},
                threshold_keys=dict(queries=len(tau), exact_final_queries=int(np.count_nonzero(tau == final)),
                                    recent_gain_mean_steps=float(np.mean(tau - hot)),
                                    recent_gap_mean_steps=float(np.mean(final - tau)),
                                    recent_gap_p95_steps=float(np.quantile(final - tau, .95))),
                selection_sha256=selection_hash, phase_weights=weights.tolist(), profiles=result,
                scope=("CPU counterfactual, frozen cache, exact captured TopK IDs. "
                       "No NPU time or transport performance."),
                ordering=("Equal tile-progress windows using Trigger boundaries; epoch 0 covers resident LRU order; "
                          "recent Index candidates appear only at their later physical window. Q4 union; cold physical "
                          "order; ABC excluded. Carry retains unused credits, capped at 480/window. Quota rejects "
                          "go to final repair, never revisited. Not actual producer timing/admission order."),
                threshold=("Resident+recent exact current-score Kth; ordered-key boosts affect cold speculative "
                           "strict-> admission only. Resident victim protection uses the unboosted threshold. "
                           "Full TopK is unchanged; ties use captured IDs, never recomputed tie choices."),
                limitation=("No closed-loop cache evolution, credit timing, URMA throughput, "
                            "or 100-round boost simulation from CSV summaries."))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    p.add_argument("--phase-weights", nargs="+", type=float)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    archive = None
    try:
        archive = load_snapshot(args.snapshot)
        report = survey(archive, args.profiles, args.phase_weights)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("ADMISSION_CPU_PASS frozen_cache=true exact_topk=true npu_timing=false")
        d = report["demand"]
        print(f"DEMAND cold={d['cold_demand']} candidate_hot={d['candidate_hot_refetch']} "
              f"miss={100*d['cold_miss_rate']:.4f}%")
        for row in report["profiles"]:
            print(f"{row['name']:<12} eligible={row['eligible_candidates']} useful={row['useful_prefetch']} "
                  f"unused={row['unused_prefetch']} final={row['final_missing']} "
                  f"boost_suppressed={row['selected_suppressed_by_boost']}")
    except (ValueError, KeyError, OSError) as error:
        p.exit(2, f"ADMISSION_CPU_FAILED: {error}\n")
    finally:
        if archive is not None:
            archive.close()


if __name__ == "__main__":
    main()
