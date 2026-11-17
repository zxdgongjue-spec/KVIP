"""CPU specification of producer-side incremental selection; no latency model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TOP = 2048
CHUNK = 2048


def select(scores, ids, k=TOP):
    """Native selector order: strict winners, then input-order kth-bin ties."""
    if len(scores) < k or len(np.unique(ids)) != len(ids):
        raise ValueError("Selection needs unique, sufficient candidates")
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    positions = np.r_[np.flatnonzero(scores > threshold), np.flatnonzero(scores == threshold)][:k]
    return scores[positions], ids[positions], int(threshold)


class IncrementalTopK:
    def __init__(self, scores, ids, *, k=TOP, chunk=CHUNK):
        self.k, self.chunk = k, chunk
        self.scores, self.ids, self.threshold = select(scores, ids, k)
        self.merges = 1
        self.compacted = 0

    def append(self, scores, ids):
        pending_scores, pending_ids = [], []
        def merge():
            if pending_ids:
                self.scores, self.ids, self.threshold = select(
                    np.r_[self.scores, np.asarray(pending_scores, dtype=scores.dtype)],
                    np.r_[self.ids, np.asarray(pending_ids, dtype=np.int64)], self.k
                )
                self.merges += 1
                pending_scores.clear()
                pending_ids.clear()
        tile_size = min(128, self.chunk)
        for start in range(0, len(ids), tile_size):
            if len(pending_ids) + tile_size > self.chunk:
                merge()
            values, positions = scores[start:start+tile_size], ids[start:start+tile_size]
            keep = values > self.threshold
            pending_scores.extend(values[keep])
            pending_ids.extend(positions[keep])
            self.compacted += int(np.count_nonzero(keep))
        merge()

    def pending(self, eligible, posted):
        # Budget-blocked entries remain here until selected out or submitted.
        return [int(token) for token in self.ids if eligible[token] and int(token) not in posted]


class WindowTopK(IncrementalTopK):
    """Fixed window positions, one boundary selection, no tile cutoff work.

    The simulator omits causal/seed exclusions, which the device zero-masks.
    Retained valid K entries precede those zeros, so even at Kth=0 invalid
    positions cannot enter K. Scores below the old Kth lose at selection.
    """
    def append(self, scores, ids):
        self.scores, self.ids, self.threshold = select(
            np.r_[self.scores, scores], np.r_[self.ids, ids], self.k)
        self.merges += 1



class PrefilterWindowTopK(WindowTopK):
    """Cheap tile cutoff, boundary compaction, bounded sparse selector input.

    More than CHUNK candidates use the full fixed-position window. This is a
    scratch-capacity fallback, never a candidate budget or a truncation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = []
        self.dense = self.empty = self.input_items = self.control_items = 0

    def append(self, scores, ids):
        keep = scores > self.threshold
        count = int(np.count_nonzero(keep))
        self.counts.append(count)
        self.compacted += count
        self.control_items += (self.k + len(scores) + 255) // 256 * 256
        if not count:
            self.empty += 1
            return
        if count <= self.chunk:
            scores, ids = scores[keep], ids[keep]
        else:
            self.dense += 1
        self.input_items += (self.k + len(scores) + 255) // 256 * 256
        super().append(scores, ids)


def verify_rank(scores, ids, k=TOP):
    if len(ids) != k or len(np.unique(ids)) != k:
        raise AssertionError("Duplicate or missing TopK IDs")
    kth = np.partition(scores, len(scores) - k)[len(scores) - k]
    if np.min(scores[ids]) != kth or np.count_nonzero(scores[ids] > kth) != np.count_nonzero(scores > kth):
        raise AssertionError("Incremental selection lost an exact TopK rank")


def simulate(scores, seed, windows, eligible, *, k=TOP, budget=512, window_budget=64, window_staging=False, prefilter=False):
    """Q4 union, immutable window lists, retained unsubmitted entries, final repair."""
    keeper_type = PrefilterWindowTopK if prefilter else WindowTopK if window_staging else IncrementalTopK
    keepers = [keeper_type(s[seed[q]], seed[q], k=k) for q, s in enumerate(scores)]
    seen = [set(map(int, seed[q])) for q in range(len(scores))]
    posted = set()
    thresholds, publication = [], []
    for window in [None, *windows]:
        if window is not None:
            for q, keeper in enumerate(keepers):
                ids = np.array([int(i) for i in window if i < len(scores[q]) and int(i) not in seen[q]], dtype=np.int64)
                keeper.append(scores[q][ids], ids)
                seen[q].update(map(int, ids))
                observed = np.array(sorted(seen[q]), dtype=np.int64)
                verify_rank(scores[q][observed], np.searchsorted(observed, keeper.ids), k)
        candidates = list(dict.fromkeys(t for keeper in keepers for t in keeper.pending(eligible, posted)))
        submitted = candidates[:max(0, min(window_budget, budget - len(posted)))]
        posted.update(submitted)
        publication.append(submitted)
        thresholds.append([keeper.threshold for keeper in keepers])
    selected = set()
    for q, keeper in enumerate(keepers):
        verify_rank(scores[q], keeper.ids, k)
        selected.update(int(i) for i in keeper.ids if eligible[i])
    return dict(
        thresholds=thresholds, publication=publication, selected=sorted(selected),
        prefetch=len(posted), useful=len(posted & selected), waste=len(posted - selected),
        repair=len(selected - posted), total=len(posted | selected),
        merges=sum(v.merges for v in keepers), compacted=sum(v.compacted for v in keepers),
        window_candidates=[n for v in keepers for n in getattr(v, "counts", [])],
        dense_windows=sum(getattr(v, "dense", 0) for v in keepers),
        empty_windows=sum(getattr(v, "empty", 0) for v in keepers),
        selection_items=sum(getattr(v, "input_items", 0) for v in keepers),
        dense_selection_items=sum(getattr(v, "control_items", 0) for v in keepers),
    )


def scan_windows(history, resident, ends, reverse):
    order = np.arange(resident, history)
    if reverse:
        lo = (resident + 2047) // 2048 * 2048
        hi = int(min(ends)) // 2048 * 2048
        mask = (order >= lo) & (order < hi)
        order[mask] = lo + hi - 2048 - (order[mask] // 2048 * 2048) + order[mask] % 2048
    tiles = (history - resident) // 128
    cuts = [tiles * e // 8 * 128 for e in range(9)]
    return [order[a:b] for a, b in zip(cuts, cuts[1:])]


def inputs(archive):
    g = archive.manifest["geometry"]
    scores = archive.array("post.score_keys")
    ends = archive.array("pre.query_ends").reshape(g["batch"], 4)
    bounds = archive.array("pre.threshold_bounds")
    if g["resident"] != 16384 or g["heads"] != 32 or g["batch"] > 16 or g["history"] % 128:
        raise ValueError("Incremental replay requires B<=16 H32 resident16384 aligned history")
    counts = np.maximum(0, np.minimum(bounds[:, 1, None], ends) - np.maximum(bounds[:, 0, None], g["resident"]))
    if np.any(ends < g["resident"]) or np.any(counts > 2048):
        raise ValueError("Every query must use the exact resident+recent seed, with at most 2048 recent scores")
    return g, scores, ends, bounds


def eligibility(archive, row):
    """Frozen physical Index eligibility; submitted-map exclusion is dynamic."""
    g = archive.manifest["geometry"]
    r, h = g["resident"], g["history"]
    tensors = archive.manifest["tensors"]
    if "pre.metadata" in tensors:
        # The service snapshot packs int32 owner IDs in metadata's int64
        # storage, just like SnapshotDemand and the native binding. A separate
        # row_owners tensor is optional and absent from the original captures.
        metadata = archive.array("pre.metadata")
        if metadata.shape != (5, g["batch"] * 4) or metadata.dtype != np.int64:
            raise ValueError("Invalid packed snapshot metadata")
        owners = metadata[4].view(np.int32)[:g["batch"]]
        if "pre.row_owners" in tensors and not np.array_equal(owners, archive.array("pre.row_owners")):
            raise ValueError("Snapshot row_owners disagrees with packed metadata")
    elif "pre.row_owners" in tensors:
        owners = archive.array("pre.row_owners")
    else:
        raise ValueError("Snapshot needs pre.metadata or pre.row_owners")
    hot_rows = archive.array("pre.hot_valid")
    if (owners.shape != (g["batch"],) or owners.dtype != np.int32 or
            hot_rows.ndim != 2 or hot_rows.shape[1] != r or
            np.any(owners < 0) or np.any(owners >= len(hot_rows))):
        raise ValueError("Snapshot owner IDs or hot-cache geometry are invalid")
    owner = int(owners[row])
    hot = hot_rows[owner] != 0
    reserved = archive.array("pre.reserved_hot")[row] >= 0
    eligible = np.ones(h, dtype=bool)
    eligible[:r] = ~hot | reserved
    begin, end = map(int, archive.array("pre.tail_bounds")[row])
    eligible[begin:end] = False
    return eligible


def check_publications(scores, ends, resident, bounds, lists, eligible, reverse):
    """Check every device window against full prefix ranks, outside timing."""
    from kvip.diagnostics.recent_threshold import recent_positions
    windows = scan_windows(scores.shape[-1], resident, ends, reverse)
    for q in range(4):
        seed = recent_positions(resident, bounds, int(ends[q]))
        seen = np.zeros(scores.shape[-1], dtype=bool)
        seen[seed] = True
        for epoch, window in enumerate([None, *windows]):
            if window is not None:
                seen[window[window < ends[q]]] = True
            observed = scores[q, seen]
            kth = int(np.partition(observed, len(observed)-TOP)[-TOP])
            header = lists[q, epoch, :32]
            count = int(header[0])
            if int(header[2]) != epoch+1 or int(header[1]) != kth or not 0 <= count <= TOP:
                raise AssertionError(f"Incremental prefix header/rank differs q={q} epoch={epoch}")
            ids = lists[q, epoch, 32:32+count].astype(np.int64)
            if len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= len(seen)):
                raise AssertionError("Incremental pending IDs are duplicate/out of range")
            if np.any(~seen[ids]) or np.any(~eligible[ids]) or np.any(scores[q, ids] < kth):
                raise AssertionError("Incremental pending IDs contain unscored/ineligible/non-TopK entries")
            strict = seen & (scores[q] > kth)
            if not set(np.flatnonzero(strict & eligible)) <= set(map(int, ids)):
                raise AssertionError("Incremental pending list dropped a strict TopK winner")
            tie_count = TOP - int(strict.sum())
            equal = seen & (scores[q] == kth)
            lower = int((strict & eligible).sum()) + max(0, tie_count-int((equal & ~eligible).sum()))
            upper = int((strict & eligible).sum()) + min(tie_count,int((equal & eligible).sum()))
            if not lower <= count <= upper:
                raise AssertionError("Incremental pending tie count cannot represent an exact TopK")


def verify(torch, bundle, archive):
    from kvip.diagnostics.archive import cpu_array
    g, scores, ends, bounds = inputs(archive)
    lists = cpu_array(torch, bundle.rotating.incremental_lists).reshape(g["batch"], 4, 9, 2080)
    digest = hashlib.sha256()
    for row in range(g["batch"]):
        check_publications(scores[row], ends[row], g["resident"], bounds[row], lists[row],
                           eligibility(archive, row), bundle.tuning.get("reverse_cold", False))
        for query in lists[row]:
            for window in query:
                count = int(window[0])
                digest.update(window[:3].astype("<i4", copy=False).tobytes())
                digest.update(window[32:32+count].astype("<i4", copy=False).tobytes())
    return dict(status="exact_prefix_ranks_and_pending", windows=9, queries=g["batch"]*4,
                prefix_lists_sha256=digest.hexdigest(), included_in_timing=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--window-staging", action="store_true")
    parser.add_argument("--prefilter", action="store_true", help="Validate sparse/dense boundary selection against the whole-window control")
    args = parser.parse_args()
    from kvip.diagnostics.archive import TensorArchive
    from kvip.diagnostics.recent_threshold import recent_positions

    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    archive = TensorArchive(args.snapshot, manifest)
    g, scores, ends, bounds = inputs(archive)
    for reverse in (False, True):
        merges = compacted = dense = empty = items = control_items = 0
        candidates = []
        for row in range(g["batch"]):
            valid = [scores[row, q, :int(ends[row, q])] for q in range(4)]
            seed = [recent_positions(g["resident"], bounds[row], len(s)) for s in valid]
            windows = scan_windows(g["history"], g["resident"], ends[row], reverse)
            result = simulate(valid, seed, windows, eligibility(archive, row),
                              window_staging=args.window_staging, prefilter=args.prefilter)
            if args.prefilter:
                reference = simulate(valid, seed, windows, eligibility(archive, row), window_staging=True)
                for key in ("thresholds", "publication", "selected", "prefetch", "useful", "waste", "repair", "total"):
                    if result[key] != reference[key]:
                        raise AssertionError(f"Prefilter changed exact whole-window {key} row={row} reverse={reverse}")
                candidates.extend(result["window_candidates"])
                dense += result["dense_windows"]
                empty += result["empty_windows"]
                items += result["selection_items"]
                control_items += result["dense_selection_items"]
            assert result["total"] == len(result["selected"]) + result["waste"]
            assert result["useful"] + result["repair"] == len(result["selected"])
            merges += result["merges"]
            compacted += result["compacted"]
        if args.prefilter:
            print(f"CPU_PREFILTER {'reverse' if reverse else 'forward'} PASS "
                  f"candidates_mean={np.mean(candidates):.1f} p95={np.percentile(candidates,95):.1f} "
                  f"max={max(candidates)} dense_fallback={dense}/{len(candidates)} empty={empty} "
                  f"selector_items_reduction={100*(1-items/control_items):.2f}% "
                  "window_thresholds_lists_traffic=IDENTICAL_TO_DENSE timing=NOT_SIMULATED", flush=True)
        print(f"CPU_INCREMENTAL {'reverse' if reverse else 'forward'} PASS queries={g['batch']*4} window_staging={args.window_staging} cold_windows=8 exact_score_ranks=YES retry_and_traffic_identity=YES timing=NOT_SIMULATED", flush=True)


if __name__ == "__main__":
    main()
