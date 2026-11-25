#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay captured decode TopK on CPU using slot LRU and extent R1/R2/R4."""

import argparse
import csv
import json
import sys
import time
from collections import deque
from itertools import groupby
from pathlib import Path

import numpy as np

POLICIES = ("lru", "r1", "r2", "r4")
EMPTY = -1


def read_round(path):
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 4
        or manifest.get("kind") != "kvaf_decode_score_trace"
        or manifest.get("status") != "complete"
        or manifest.get("validation", {}).get("passed") is not True
    ):
        raise ValueError(f"Expected a validated decode score manifest: {path}")
    archive = (path.parent / manifest["archive"]["file"]).resolve()
    if not archive.is_relative_to(path.parent.resolve()) or archive.stat().st_size != manifest["archive"]["bytes"]:
        raise ValueError(f"Invalid archive path or size: {archive}")
    with np.load(archive, allow_pickle=False) as data:
        keys = ("pre.logical_ids", "pre.hot_valid", "pre.lru_slots", "pre.tail_bounds", "post.topk")
        arrays = [data[key] for key in keys]
    for key, array in zip(keys, arrays, strict=True):
        spec = manifest["arrays"][key]
        if list(array.shape) != spec["shape"] or array.dtype.name != spec["dtype"]:
            raise ValueError(f"Invalid {key}: {archive}")
    logical, valid, oldest, tails, topk = arrays
    if np.any(topk < 0) or np.any(topk >= logical.shape[1]):
        raise ValueError(f"TopK physical position outside captured history: {archive}")
    return logical, valid, oldest, tails, np.take_along_axis(logical[:, None, :], topk, axis=2)


class Associations:
    """Bounded logical-token co-occurrence, local to one request and Index group."""

    def __init__(self, neighbors, anchors, candidates, budget, seed):
        self.neighbors = neighbors
        self.anchors = anchors
        self.candidates = candidates
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.edges = {}
        self.previous = None

    def sample(self, tokens, limit):
        if len(tokens) <= limit:
            return tokens
        return [tokens[i] for i in self.rng.choice(len(tokens), limit, replace=False)]

    def update(self, demand, at_risk):
        active = set(demand)
        anchors = self.sample(demand, self.anchors)
        votes = {}
        for anchor in anchors:
            for token, score in self.edges.get(anchor, {}).items():
                if score >= 2 and token in at_risk and token not in active:
                    votes[token] = votes.get(token, 0) + score

        # Train A(t-1) -> B(t), restricted to B that was absent and at risk
        # when the decision was made. Learn only after forming current votes.
        if self.previous is not None:
            previous_anchors, previous_risk = self.previous
            positives = self.sample(sorted(active & previous_risk), self.candidates)
            for anchor in previous_anchors:
                neighbors = self.edges.setdefault(anchor, {})
                for token, score in list(neighbors.items()):
                    if token in previous_risk:
                        score += 1 if token in active else -1
                        if score:
                            neighbors[token] = min(score, 255)
                        else:
                            del neighbors[token]
                for token in positives:
                    if len(neighbors) >= self.neighbors:
                        break
                    neighbors.setdefault(token, 1)
        self.previous = anchors, at_risk - active
        return set(sorted(votes, key=lambda token: (-votes[token], token))[: self.budget])


class Cache:
    def __init__(self, policy, granularity, initial, oldest, budget, half_life, grace, associations=None):
        self.policy = policy
        self.associations = associations
        self.weak = set()
        self.saved = set()
        self.displaced = set()
        self.g = granularity
        self.slots = initial.copy()
        self.mapping = {int(token): slot for slot, token in enumerate(initial) if token != EMPTY}
        if len(self.mapping) != np.count_nonzero(initial != EMPTY):
            raise ValueError("Duplicate logical token in captured hot slots")
        self.budget = budget
        self.half_life = half_life
        self.grace = grace
        self.round = 0
        self.last = np.full(len(initial), -1, dtype=np.int64)
        self.last[oldest] = np.arange(-len(oldest) - 1, -1)
        self.last[initial == EMPTY] = np.iinfo(np.int64).min
        self.clock = 0
        self.count = np.zeros(len(initial), dtype=np.float64)
        self.born = np.full(len(initial) // self.g, -grace - 1, dtype=np.int64)

    def select(self, slot_order=None):
        if self.policy in ("lru", "assoc_lru"):
            order = np.argsort(self.last, kind="stable") if slot_order is None else slot_order
            if self.weak:
                mask = np.zeros(len(self.slots), dtype=bool)
                mask[[self.mapping[token] for token in self.weak]] = True
                protected = mask[order]
                order = np.concatenate((order[~protected], order[protected]))
            return order[: self.budget]
        size = len(self.born) * self.g
        recency = self.last[:size].reshape(-1, self.g).max(axis=1)
        if self.policy == "r1":
            score = recency
        elif self.policy == "r2":
            score = self.count[:size].reshape(-1, self.g).sum(axis=1)
        else:
            score = self.count[:size].reshape(-1, self.g).max(axis=1)
        protected = self.round - self.born < self.grace
        if protected.all():
            order = np.argsort(recency, kind="stable")
        else:
            order = np.lexsort((recency, np.where(protected, np.inf, score)))
        return (order[: self.budget // self.g, None] * self.g + np.arange(self.g)).ravel()

    def step(self, demand, future=None):
        # Model reads from the beginning-of-round mapping, then apply writes.
        # Only the used prefix of the reserved slots is physically overwritten.
        misses = [token for token in demand if token not in self.mapping]
        began = time.perf_counter_ns()
        slot_order = None
        if misses and self.associations is not None:
            slot_order = np.argsort(self.last, kind="stable")
        if misses and future is not None:
            order = np.argsort(self.last, kind="stable")[: self.budget]
            horizon = int(self.policy.removeprefix("oracle"))
            # Within the same victim budget, prefer empty slots then evict
            # the tokens whose next use is farthest away within this horizon.
            ranks = [
                horizon + 2
                if self.slots[slot] == EMPTY
                else min(future.get(self.slots[slot], horizon + 1), horizon + 1)
                for slot in order
            ]
            chosen = np.argsort(-np.asarray(ranks), kind="stable")[: len(misses)]
            victims = order[np.sort(chosen)]
        else:
            victims = self.select(slot_order)[: len(misses)] if misses else np.empty(0, dtype=np.int64)
        selection_ns = time.perf_counter_ns() - began if misses else 0
        evicted = 0
        protected_hits = 0
        current_hits = set(demand) - set(misses)
        saved_reuse = len(self.saved & current_hits)
        displaced_misses = len(self.displaced.intersection(misses))
        self.saved.difference_update(current_hits)
        self.displaced.difference_update(misses)
        saved_now = displaced_now = set()
        if slot_order is not None and self.weak:
            old_slots = set(slot_order[: len(victims)])
            new_slots = set(victims)
            saved_now = {int(self.slots[slot]) for slot in old_slots - new_slots} - {EMPTY}
            displaced_now = {int(self.slots[slot]) for slot in new_slots - old_slots} - {EMPTY}
        self.count *= 2 ** (-1 / self.half_life)
        for slot, token in zip(victims, misses):
            old = self.slots[slot]
            if old != EMPTY:
                del self.mapping[int(old)]
                self.saved.discard(old)
                evicted += 1
                protected_hits += old in current_hits
            self.slots[slot] = token
            self.mapping[token] = int(slot)
            self.count[slot] = 0
            self.born[slot // self.g] = self.round
        for token in demand:
            slot = self.mapping.get(token)
            if slot is not None:
                self.clock += 1
                self.last[slot] = self.clock
                self.count[slot] += 1
        self.saved.update(saved_now)
        self.displaced.update(displaced_now)
        predictions = len(self.weak)
        predicted_hits = sum(token in current_hits for token in self.weak)
        update_ns = 0
        if self.associations is not None:
            began = time.perf_counter_ns()
            # The next round can overwrite only the LRU victim prefix.
            risky_slots = np.argsort(self.last, kind="stable")[: self.budget]
            at_risk = set(self.slots[risky_slots]) - {EMPTY}
            self.weak = self.associations.update(demand, at_risk)
            update_ns = time.perf_counter_ns() - began
        self.round += 1
        runs = int(1 + np.count_nonzero(np.diff(victims) != 1)) if len(victims) else 0
        return (
            len(demand),
            len(misses),
            len(victims),
            len(misses) - len(victims),
            evicted,
            protected_hits,
            runs,
            selection_ns,
            predictions,
            predicted_hits,
            update_ns,
            len(saved_now),
            saved_reuse,
            len(displaced_now),
            displaced_misses,
        )


def resize_initial(initial, oldest, capacity):
    """Keep in-range slots and move live overflow tokens into free slots."""
    if capacity >= len(initial):
        return initial, oldest
    occupied = np.flatnonzero(initial != EMPTY)
    if len(occupied) > capacity:
        raise ValueError(f"Hot capacity {capacity} cannot hold {len(occupied)} captured live tokens")
    resized = initial[:capacity].copy()
    overflow = occupied[occupied >= capacity]
    free = np.flatnonzero(resized == EMPTY)[: len(overflow)]
    resized[free] = initial[overflow]
    remap = np.arange(len(initial))
    remap[overflow] = free
    oldest_live = oldest[initial[oldest] != EMPTY]
    return resized, remap[oldest_live]


ORACLE_HORIZONS = (1, 2, 4)


def demands_for_round(tails, topk):
    return [
        list(dict.fromkeys(int(token) for token in topk[row].flat if not begin <= token < end))
        for row, (begin, end) in enumerate(tails)
    ]


def replay_frames(references, lookahead):
    def load_round(group):
        frames = {}
        for ref in group:
            logical, valid, oldest, tails, topk = read_round(ref.manifest_path)
            frames[ref.dp_rank] = (ref, logical, valid, oldest, demands_for_round(tails, topk))
        return frames

    rounds = (load_round(group) for _, group in groupby(references, key=lambda ref: ref.source_round))
    queue = deque()
    for _ in range(lookahead + 1):
        frame = next(rounds, None)
        if frame is not None:
            queue.append(frame)
    while queue:
        current = queue.popleft()
        for rank, frame in current.items():
            ref, logical, valid, oldest, demands = frame
            future = [{} for _ in demands]
            for distance, following in enumerate(queue, 1):
                next_ref, _, _, _, next_demands = following[rank]
                if ref.request_signature != next_ref.request_signature:
                    raise ValueError("Request identity changed within lookahead")
                for row, demand in enumerate(next_demands):
                    for token in demand:
                        future[row].setdefault(token, distance)
            yield ref, logical, valid, oldest, demands, future
        frame = next(rounds, None)
        if frame is not None:
            queue.append(frame)


def simulate(args, references):
    caches = {}
    sample_counts = {}
    summaries = {}
    first_eviction = {}
    measured_evictions = {}
    avoided_misses = added_misses = 0
    header = (
        "dp",
        "round",
        "request",
        "policy",
        "extent",
        "scored",
        "hot_capacity",
        "demand",
        "misses",
        "writes",
        "bypass",
        "evicted",
        "evicted_current_hits",
        "destination_runs",
        "select_ns",
        "weak_from_previous_round",
        "weak_next_round_hits",
        "association_update_ns",
        "victims_skipped",
        "skipped_then_hit",
        "replacement_victims",
        "replacement_then_miss",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        lookahead = max(ORACLE_HORIZONS) if args.association else 0
        for ref, logical, valid, oldest, demands, future in replay_frames(references, lookahead):
            captured_capacity = valid.shape[1]
            capacity = args.hot_capacity or captured_capacity
            if capacity > captured_capacity:
                raise ValueError(f"Hot capacity {capacity} exceeds captured capacity {captured_capacity}")
            if not 0 < args.victims <= capacity or any(args.victims % g for g in args.extents):
                raise ValueError("Victim budget must fit replay capacity and be divisible by all extent sizes")
            for row, signature in enumerate(ref.request_signature):
                key = (ref.dp_rank, signature)
                if key not in caches:
                    initial = np.where(valid[row], logical[row, :captured_capacity], EMPTY)
                    initial, initial_oldest = resize_initial(initial, oldest[row], capacity)
                    live = np.count_nonzero(initial != EMPTY)
                    print(f"DP {ref.dp_rank} request row {row}: {live}/{capacity} hot slots occupied initially")
                    choices = [("lru", 1)] + [(policy, g) for g in args.extents for policy in POLICIES[1:]]
                    if args.association:
                        choices.extend((f"oracle{horizon}", 1) for horizon in ORACLE_HORIZONS)
                    caches[key] = {
                        (p, g): Cache(p, g, initial, initial_oldest, args.victims, args.half_life, args.grace)
                        for p, g in choices
                    }
                    if args.association:
                        model = Associations(
                            args.assoc_neighbors,
                            args.assoc_anchors,
                            args.assoc_candidates,
                            args.assoc_budget,
                            args.assoc_seed,
                        )
                        caches[key]["assoc_lru", 1] = Cache(
                            "assoc_lru", 1, initial, initial_oldest, args.victims, args.half_life, args.grace, model
                        )
                    sample_counts[key] = 0
                demand = demands[row]
                scored = sample_counts[key] >= args.warmup_rounds
                sample_counts[key] += 1
                if args.association and scored:
                    baseline = caches[key]["lru", 1].mapping
                    candidate = caches[key]["assoc_lru", 1].mapping
                    avoided_misses += sum(token not in baseline and token in candidate for token in demand)
                    added_misses += sum(token in baseline and token not in candidate for token in demand)
                for (policy, g), cache in caches[key].items():
                    if len(cache.slots) != capacity:
                        raise ValueError("Hot capacity changed during a request")
                    result = cache.step(demand, future[row] if policy.startswith("oracle") else None)
                    writer.writerow(
                        (ref.dp_rank, ref.source_round, signature[0], policy, g, int(scored), capacity, *result)
                    )
                    if result[4]:
                        first_eviction.setdefault((policy, g), ref.source_round)
                        measured_evictions[policy, g] = measured_evictions.get((policy, g), 0) + int(scored)
                    if scored:
                        summaries.setdefault((policy, g), []).append(result)
    if not summaries:
        raise ValueError("No measured rounds after warmup")
    print(f"Measured request-rounds per policy: {len(next(iter(summaries.values())))}")
    print(
        "Policy | Extent | Demand | Misses | Miss% | Writes | Bypass | Evicted | Hit evictions | "
        "Dest runs | First eviction round | Eviction request-rounds | Select p50 us | Weak | Weak hits"
    )
    for (policy, g), results in summaries.items():
        values = np.asarray(results, dtype=np.int64)
        demand, misses, writes, bypass, evicted, hits, runs = values[:, :7].sum(axis=0)
        rate = 100 * misses / demand if demand else 0
        select = values[:, 7]
        median = np.median(select[select > 0]) / 1000 if np.any(select > 0) else 0
        weak, weak_hits = values[:, 8:10].sum(axis=0)
        print(
            f"{policy:>6} | {g:6} | {demand} | {misses} | {rate:.2f} | {writes} | {bypass} | "
            f"{evicted} | {hits} | {runs} | {first_eviction.get((policy, g), '-')} | "
            f"{measured_evictions.get((policy, g), 0)} | {median:.2f} | {weak} | {weak_hits}"
        )
        if policy == "assoc_lru":
            hit_rate = 100 * weak_hits / weak if weak else 0
            update_us = np.median(values[:, 10]) / 1000
            skipped, reused, replacements, rereads = values[:, 11:15].sum(axis=0)
            print(
                f"At-risk outcomes: skipped victims={skipped}, later hits={reused}, "
                f"replacement victims={replacements}, later misses={rereads}"
            )
            print(
                f"Association vs LRU: avoided misses={avoided_misses}, added misses={added_misses}, "
                f"next-round weak hit rate={hit_rate:.2f}%, model update p50={update_us:.2f} us"
            )
    if args.association:
        print("Oracle1/2/4 use future accesses within the same LRU candidate window (offline hindsight references).")
    print(f"Per-round CSV: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True, help="Directory containing dp*/series.json")
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--round-range", required=True, help="Inclusive source rounds, e.g. 1:101")
    parser.add_argument("--dp", type=int, nargs="+", help="Only score these DP ranks")
    parser.add_argument("--hot-capacity", type=int, help="Hot slots per request (default: captured capacity)")
    parser.add_argument("--victims", type=int, default=2048)
    parser.add_argument("--extents", type=int, nargs="+", choices=(1, 4, 8, 16), default=(8, 16))
    parser.add_argument("--half-life", type=float, default=64)
    parser.add_argument("--grace", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--association", action="store_true", help="Compare LRU with one-round association protection")
    parser.add_argument("--assoc-neighbors", type=int, default=128, help="Maximum neighbors per logical token")
    parser.add_argument("--assoc-anchors", type=int, default=128, help="Active anchors sampled per round")
    parser.add_argument("--assoc-candidates", type=int, default=128, help="Active candidates sampled per round")
    parser.add_argument("--assoc-budget", type=int, default=128, help="Maximum weak protections per round")
    parser.add_argument("--assoc-seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        first, last = map(int, args.round_range.split(":"))
        if (
            not 1 <= first < last
            or not np.isfinite(args.half_life)
            or args.half_life <= 0
            or args.grace < 0
            or not 0 <= args.warmup_rounds <= last - first
            or (args.hot_capacity is not None and args.hot_capacity <= 0)
            or min(args.assoc_neighbors, args.assoc_anchors, args.assoc_candidates, args.assoc_budget) <= 0
            or args.assoc_seed < 0
            or len(set(args.extents)) != len(args.extents)
        ):
            raise ValueError("Require consecutive rounds, positive half-life, distinct extents and valid warmup/grace")
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "tests/operators/lifetch/support"))
        from workloads import select_series

        if not any(args.trace_root.glob("dp[0-9]*/series.json")):
            raise ValueError(
                f"No DP series under {args.trace_root}. Pass the original score trace root "
                "containing dp*/series.json. An individual suite case covers only one source round."
            )
        references = select_series(args.trace_root, args.group, (first, last))
        if args.dp is not None:
            if set(args.dp) - {ref.dp_rank for ref in references}:
                raise ValueError("Requested DP rank is absent from trace")
            references = [ref for ref in references if ref.dp_rank in args.dp]
        print(f"Group {args.group}, source rounds {first}–{last}, DP ranks {sorted({r.dp_rank for r in references})}")
        simulate(args, references)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"Replay failed: {error}\n")


if __name__ == "__main__":
    main()
