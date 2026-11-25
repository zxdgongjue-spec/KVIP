#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay captured decode TopK on CPU using slot LRU and extent R1/R2/R4."""

import argparse
import csv
import json
import sys
import time
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


class Cache:
    def __init__(self, policy, granularity, initial, oldest, budget, half_life, grace):
        self.policy = policy
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

    def select(self):
        if self.policy == "lru":
            return np.argsort(self.last, kind="stable")[: self.budget]
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

    def step(self, demand):
        # Model reads from the beginning-of-round mapping, then apply writes.
        # Only the used prefix of the reserved slots is physically overwritten.
        misses = [token for token in demand if token not in self.mapping]
        began = time.perf_counter_ns()
        victims = self.select()[: len(misses)] if misses else np.empty(0, dtype=np.int64)
        selection_ns = time.perf_counter_ns() - began if misses else 0
        evicted = 0
        protected_hits = 0
        current_hits = set(demand) - set(misses)
        self.count *= 2 ** (-1 / self.half_life)
        for slot, token in zip(victims, misses):
            old = self.slots[slot]
            if old != EMPTY:
                del self.mapping[int(old)]
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
        )


def simulate(args, references):
    caches = {}
    sample_counts = {}
    summaries = {}
    header = (
        "dp",
        "round",
        "request",
        "policy",
        "extent",
        "scored",
        "demand",
        "misses",
        "writes",
        "bypass",
        "evicted",
        "evicted_current_hits",
        "destination_runs",
        "select_ns",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for ref in references:
            logical, valid, oldest, tails, topk = read_round(ref.manifest_path)
            capacity = valid.shape[1]
            if not 0 < args.victims <= capacity or any(args.victims % g for g in args.extents):
                raise ValueError("Victim budget must fit captured capacity and be divisible by all extent sizes")
            for row, signature in enumerate(ref.request_signature):
                key = (ref.dp_rank, signature)
                if key not in caches:
                    initial = np.where(valid[row], logical[row, :capacity], EMPTY)
                    choices = [("lru", 1)] + [(policy, g) for g in args.extents for policy in POLICIES[1:]]
                    caches[key] = {
                        (p, g): Cache(p, g, initial, oldest[row], args.victims, args.half_life, args.grace)
                        for p, g in choices
                    }
                    sample_counts[key] = 0
                begin, end = tails[row]
                demand = list(dict.fromkeys(int(token) for token in topk[row].flat if not begin <= token < end))
                scored = sample_counts[key] >= args.warmup_rounds
                sample_counts[key] += 1
                for (policy, g), cache in caches[key].items():
                    if len(cache.slots) != capacity:
                        raise ValueError("Hot capacity changed during a request")
                    result = cache.step(demand)
                    writer.writerow((ref.dp_rank, ref.source_round, signature[0], policy, g, int(scored), *result))
                    if scored:
                        summaries.setdefault((policy, g), []).append(result)
    if not summaries:
        raise ValueError("No measured rounds after warmup")
    print(f"Measured request-rounds per policy: {len(next(iter(summaries.values())))}")
    print(
        "Policy | Extent | Demand | Misses | Miss% | Writes | Bypass | Evicted | Hit evictions | "
        "Dest runs | Select p50 us"
    )
    for (policy, g), results in summaries.items():
        values = np.asarray(results, dtype=np.int64)
        demand, misses, writes, bypass, evicted, hits, runs = values[:, :7].sum(axis=0)
        rate = 100 * misses / demand if demand else 0
        select = values[:, 7]
        median = np.median(select[select > 0]) / 1000 if np.any(select > 0) else 0
        print(
            f"{policy:>6} | {g:6} | {demand} | {misses} | {rate:.2f} | {writes} | {bypass} | "
            f"{evicted} | {hits} | {runs} | {median:.2f}"
        )
    print(f"Per-round CSV: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True, help="Directory containing dp*/series.json")
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--round-range", required=True, help="Inclusive source rounds, e.g. 1:101")
    parser.add_argument("--dp", type=int, nargs="+", help="Only score these DP ranks")
    parser.add_argument("--victims", type=int, default=2048)
    parser.add_argument("--extents", type=int, nargs="+", choices=(1, 4, 8, 16), default=(8, 16))
    parser.add_argument("--half-life", type=float, default=64)
    parser.add_argument("--grace", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
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
