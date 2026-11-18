# SPDX-License-Identifier: Apache-2.0
"""CPU reference for fetch-side, request-local window ranking. No time model."""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from kvip.diagnostics.incremental_topk import TOP, eligibility, inputs, scan_windows, select
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.window_audit import CachedArchive


class FetchRank:
    """Keep counts for all ranks, but identities only for fetch-eligible winners."""

    def __init__(self, scores, ids, eligible, k=TOP):
        self.k = k
        scores, ids, self.kth = select(scores, ids, k)
        self.seed_kth = self.kth
        self.counts = Counter()
        self.fast = 0
        self.winners = []
        self.add(scores, ids, eligible)

    def add(self, scores, ids, eligible):
        for score, token in zip(scores, ids):
            key = int(score)
            self.counts[key] += 1
            if eligible[token]:
                self.winners.append((key, int(token), self.counts[key]))

    def append(self, scores, ids, eligible):
        # Immutable producer filter; an old kth tie wins against any incoming tie.
        offers = scores > self.seed_kth
        kept = offers & (scores > self.kth)
        if np.any(kept):
            self.add(scores[kept], ids[kept], eligible)
            excess = int(kept.sum())
            if excess < self.counts[self.kth]:
                self.counts[self.kth] -= excess
                self.fast += 1
            else:
                for score in sorted(self.counts):
                    removed = min(excess, self.counts[score])
                    self.counts[score] -= removed
                    excess -= removed
                    if not self.counts[score]:
                        del self.counts[score]
                    if not excess:
                        break
                self.kth = min(self.counts)
            self.winners = [entry for entry in self.winners
                            if entry[0] > self.kth or
                            (entry[0] == self.kth and entry[2] <= self.counts[self.kth])]
        return offers

    def eligible_ids(self):
        return ([token for score, token, _ in self.winners if score > self.kth] +
                [token for score, token, _ in self.winners if score == self.kth])


def publication(lists, posted, epoch_budget, request_budget):
    # Preserve the reference Q-priority order. Budget applies after Q4 union.
    unique = dict.fromkeys(token for items in lists for token in items if token not in posted)
    limit = min(epoch_budget, request_budget - len(posted))
    submitted = list(unique)[:limit]
    posted.update(submitted)
    return submitted


class RequestWindows:
    """Per-request ordered input from four producers; no cross-request barrier."""

    def __init__(self, queries):
        self.ready = [deque() for _ in range(queries)]
        self.consumed = -1

    def offer(self, q, epoch, candidates):
        queue = self.ready[q]
        expected = queue[-1][0] + 1 if queue else self.consumed + 1
        if epoch != expected:
            raise ValueError(f"Query {q} publication {epoch} out of sequence, expected {expected}")
        queue.append((epoch, candidates))

    def advance(self):
        if not all(self.ready):
            return None
        self.consumed += 1
        return [queue.popleft()[1] for queue in self.ready]


def check_staggered(snapshots, oracle, epoch_budget, request_budget):
    # Alternate which query runs one epoch ahead. Test causal order and caps
    # independently of timing; the last opportunity also tests final catch-up.
    request = RequestWindows(len(snapshots[0]))
    delivered = [-1] * len(request.ready)
    posted, reference = set(), set()
    for epoch in range(len(snapshots)):
        laggard = epoch % len(request.ready)
        for q in range(len(request.ready)):
            end = epoch - int(epoch > 0 and epoch < len(snapshots) - 1 and q == laggard)
            for version in range(delivered[q] + 1, end + 1):
                request.offer(q, version, snapshots[version][q])
                delivered[q] = version
        lists = None
        while request.consumed < epoch:
            ready = request.advance()
            if ready is None:
                break
            lists = ready
        if request.consumed != min(delivered):
            raise AssertionError("Request consumed an incomplete version or stalled")
        # One physical opportunity has one budget even when it catches up
        # multiple versions. A repeated version can submit still-pending IDs.
        if lists is None:
            lists = snapshots[request.consumed]
        sent = publication(lists, posted, epoch_budget, request_budget)
        wanted = publication(oracle[min(delivered)], reference, epoch_budget, request_budget)
        if sent != wanted:
            raise AssertionError("Skewed Q4 submissions differ from the common-prefix oracle")
    if request.consumed != len(snapshots) - 1:
        raise AssertionError("Final producer version not consumed")


def collect_window(state, scores, ids, *, k=2048, capacity=2048):
    """Bounded sparse collector with re-filtering after an overflow flush."""
    values, selected, cutoff = state
    pending = []
    merges = nonempty = overflows = 0
    # A native DMA block ends at a physical 2048-position boundary.
    cuts = np.r_[0, np.flatnonzero(np.diff(ids // 2048)) + 1, len(ids)]
    for start, end in zip(cuts, cuts[1:]):
        offsets = np.arange(start, end)
        offered = offsets[scores[offsets] > cutoff]
        nonempty += bool(len(offered))
        if len(pending) + len(offered) > capacity:
            if not pending:
                raise ValueError("Collector capacity must fit one input block")
            values, selected, cutoff = select(
                np.r_[values, scores[pending]], np.r_[selected, ids[pending]], k)
            merges += 1
            overflows += 1
            pending = []
            offered = offsets[scores[offsets] > cutoff]
        pending.extend(offered.tolist())
    if pending:
        values, selected, cutoff = select(
            np.r_[values, scores[pending]], np.r_[selected, ids[pending]], k)
        merges += 1
    return (values, selected, cutoff), (nonempty, merges, overflows)


def check_group(scores, ends, bounds, masks, geometry, expected=None):
    batch, queries, history = scores.shape
    offers, seed_bins, touched = [], [], 0
    nonempty_tiles = total_tiles = tie_shortcuts = 0
    q_occurrences = unique_per_request = checks = 0
    collector_blocks = collector_merges = collector_overflows = 0
    window_budget, request_budget = geometry["window_limit"], geometry["prefetch_limit"]
    if (batch, queries, geometry["token_bytes"]) != (16, 4, 656):
        raise ValueError("Expected BS16 Q4 single-layer 656-byte workload")
    for row in range(batch):
        eligible = masks[row]
        windows = scan_windows(history, geometry["resident"], ends[row], True)
        rankers, dense, visited = [], [], []
        for q in range(queries):
            seed = recent_positions(geometry["resident"], bounds[row], int(ends[row, q]))
            seed_scores = scores[row, q, seed]
            rankers.append(FetchRank(seed_scores, seed, eligible))
            seed_bins.append(len(rankers[-1].counts))
            dense.append(select(seed_scores, seed))
            mask = np.zeros(history, dtype=bool)
            mask[seed] = True
            visited.append(mask)
        posted, oracle_posted = set(), set()
        snapshots, oracle = [], []
        for epoch, window in enumerate([None, *windows]):
            eligible_lists, oracle_lists = [], []
            for q, ranker in enumerate(rankers):
                old_scores, old_ids, kth = dense[q]
                if window is not None:
                    ids = window[(window < ends[row, q]) & ~visited[q][window]]
                    visited[q][ids] = True
                    new_scores = scores[row, q, ids]
                    passed = ranker.append(new_scores, ids, eligible)
                    offers.append(int(passed.sum()))
                    total_tiles += len(np.unique(ids // 128))
                    nonempty_tiles += len(np.unique(ids[passed] // 128))
                    touched += len(ids)
                    collected, work = collect_window(dense[q], new_scores, ids)
                    collector_blocks += work[0]
                    collector_merges += work[1]
                    collector_overflows += work[2]
                    old_scores, old_ids, kth = select(np.r_[old_scores, new_scores], np.r_[old_ids, ids])
                    if (collected[2] != kth or not np.array_equal(collected[0], old_scores)
                            or not np.array_equal(collected[1], old_ids)):
                        raise AssertionError(f"Window collector differs row={row} q={q} epoch={epoch}")
                    dense[q] = old_scores, old_ids, kth
                wanted = old_ids[eligible[old_ids]].tolist()
                got = ranker.eligible_ids()
                if kth != ranker.kth or got != wanted:
                    raise AssertionError(f"Rank or ordered eligible IDs differ row={row} q={q} epoch={epoch}")
                if expected is not None and not np.array_equal(expected[row, q, epoch], old_ids):
                    raise AssertionError(f"Archived reference differs row={row} q={q} epoch={epoch}")
                eligible_lists.append(got)
                oracle_lists.append(wanted)
                checks += 1
            snapshots.append(eligible_lists)
            oracle.append(oracle_lists)
            observed = publication(eligible_lists, posted, window_budget, request_budget)
            reference = publication(oracle_lists, oracle_posted, window_budget, request_budget)
            if observed != reference:
                raise AssertionError(f"Request submission differs row={row} epoch={epoch}")
            if epoch:
                q_occurrences += sum(map(len, eligible_lists))
                unique_per_request += len(set().union(*eligible_lists))
        check_staggered(snapshots, oracle, window_budget, request_budget)
        tie_shortcuts += sum(ranker.fast for ranker in rankers)
    return (f"checks={checks} skew=PASS collector_blocks/merges/overflow="
            f"{collector_blocks}/{collector_merges}/{collector_overflows} T0_offers/query/window={np.mean(offers):.2f} "
            f"p95={np.percentile(offers,95):.1f} max={max(offers)} "
            f"T0_pass={100*sum(offers)/touched:.3f}% "
            f"touched_tiles={100*nonempty_tiles/total_tiles:.2f}% "
            f"tie_shortcuts={tie_shortcuts}/{len(offers)} seed_score_bins_mean={np.mean(seed_bins):.1f} "
            f"Q4_duplicate_occurrences={100*(1-unique_per_request/q_occurrences):.2f}%"
            if q_occurrences else f"checks={checks} no_eligible_winners")


def run(root, group, audit):
    if audit:
        directory = root / f"g{group}"
        geometry = json.loads((directory / "manifest.json").read_text())["geometry"]
        with np.load(directory / "probe_inputs.npz", allow_pickle=False) as data:
            return check_group(data["scores"], data["query_ends"], data["threshold_bounds"],
                               data["eligible"], geometry, data["expected_ids"])
    directory = root / f"g{group}-input"
    archive = CachedArchive(directory, json.loads((directory / "manifest.json").read_text()))
    try:
        geometry, scores, ends, bounds = inputs(archive)
        masks = [eligibility(archive, row) for row in range(geometry["batch"])]
        return check_group(scores, ends, bounds, masks, geometry)
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--inputs", type=Path, help="Directory containing gN-input snapshots")
    sources.add_argument("--audit-root", type=Path, help="Prior window-audit/analysis directory")
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 7, 10, 17, 4])
    args = parser.parse_args()
    for group in args.groups:
        result = run(args.inputs or args.audit_root, group, args.audit_root is not None)
        print(f"G{group} PASS {result}", flush=True)


if __name__ == "__main__":
    main()
