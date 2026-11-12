# SPDX-License-Identifier: Apache-2.0
"""CPU score-policy counterfactuals against a frozen, validated trace state.

Observed I/O comes only from recorded transfers. Policy candidates have no
window/capacity limits and never predict elapsed time or subsequent cache state.
"""

import numpy as np

from kvip.diagnostics.score_replay import _identity, score_value
from kvip.diagnostics.recent_threshold import range_state, recent_positions, threshold_bounds

TOPK = 2048
QUERIES = 4
TRUNK = 16384
POLICIES = ("hot", "recent", "seed_hot", "rolling", "final")


def seed_state(trace):
    """Retain only request identities and previous Q4 TopK IDs, not score arrays."""
    source, a = trace.manifest["metadata"], trace.arrays
    packed = a["pre.metadata"][2] // QUERIES
    return dict(
        dp_rank=source["dp_rank"],
        group=source["group"],
        scheduler_round=source["context"]["round"],
        layer_names=tuple(source["layer_names"]),
        recent=range_state(trace),
        requests={
            identity: np.unique(a["post.logical_output"][packed == row]).copy()
            for row, identity in enumerate(_identity(trace))
        },
    )


def _predecessor(trace, previous):
    source = trace.manifest["metadata"]
    if previous is None or (
        previous["dp_rank"] != source["dp_rank"]
        or previous["group"] != source["group"]
        or previous["scheduler_round"] + 1 != source["context"]["round"]
        or previous["layer_names"] != tuple(source["layer_names"])
    ):
        raise ValueError("Threshold survey requires the immediately preceding round of the same DP/group")
    identities = _identity(trace)
    if set(identities) != set(previous["requests"]):
        raise ValueError("Previous TopK request/owner lifetime differs from the current trace")
    return identities


def _kth(keys):
    if len(keys) < TOPK:
        raise ValueError("Threshold seed contains fewer than 2048 distinct causal positions")
    return int(np.partition(keys, len(keys) - TOPK)[-TOPK])


def _rolling_mask(keys, resident, initial):
    """Use each completed 16K-prefix Kth only for subsequently scanned positions.

    The resident victim bits keep the original threshold. Already admitted bits
    are never revisited. Histogram updates here are a CPU analysis shortcut,
    not a proposed native implementation or a measurement of its overhead.
    """
    end = len(keys)
    lower = min(resident, end)
    above = np.zeros(end, dtype=bool)
    above[:lower] = keys[:lower] > initial
    # BF16 scores often occupy a narrow range of sortable keys. Keep exact
    # integer counts without scanning all 65536 buckets at every checkpoint.
    minimum = int(keys.min())
    bins = int(keys.max()) - minimum + 1
    normalized = keys.astype(np.int32) - minimum
    histogram = np.bincount(normalized[:lower], minlength=bins)
    threshold = initial
    upper = (resident // TRUNK + 1) * TRUNK
    updates = 0
    while lower < end:
        stop = min(upper, end)
        above[lower:stop] = keys[lower:stop] > threshold
        if stop < end:
            histogram += np.bincount(normalized[lower:stop], minlength=bins)
            threshold = minimum + int(np.searchsorted(histogram.cumsum(), stop - TOPK + 1))
            updates += 1
        lower, upper = stop, upper + TRUNK
    return above, updates


def _finite(value):
    return float(value) if np.isfinite(value) else None


def analyze_trace(trace, previous_seed):
    """Return per-request and per-query rows; caller has validated the archive.

    All policies use the same readable candidate domain: resident LRU physical
    slots plus the physical cold suffix, excluding ABC and future positions.
    This is an explicit counterfactual, including when the recorded run used
    cold-only prefetch. The old schema does not record that switch.
    """
    manifest, a = trace.manifest, trace.arrays
    if manifest.get("content") != "scores":
        raise ValueError("Threshold analysis requires complete captured scores")
    identities = _predecessor(trace, previous_seed)
    recent_ranges = threshold_bounds(trace, previous_seed["recent"])
    g, source = manifest["geometry"], manifest["metadata"]
    b, h, resident = (g[name] for name in ("batch", "history", "resident"))
    layers = source["layer_names"]
    if g["token_bytes"] != 656 * len(layers):
        raise ValueError("Group payload width must equal 656 bytes times the captured layer count")
    summary = {row["packed_row"]: row for row in manifest["summary"]["requests"]}
    requests, queries = [], []
    packed_rows = a["pre.metadata"][2] // QUERIES
    positions = np.arange(h)
    for row, identity in enumerate(identities):
        ids = a["pre.logical_ids"][row]
        causal_maximum = np.maximum.accumulate(ids)
        inverse = np.empty(h, dtype=np.int32)
        inverse[ids] = positions
        begin, stop = map(int, a["pre.tail_bounds"][row])
        if not np.array_equal(ids[begin:stop], positions[begin:stop]):
            raise ValueError("ABC physical/logical coordinates differ; native candidate domain is ambiguous")
        abc = (ids >= begin) & (ids < stop)
        candidate = np.zeros(h, dtype=bool)
        candidate[a["pre.lru_slots"][row]] = True
        valid_hot = np.zeros(h, dtype=bool)
        valid_hot[:resident] = a["pre.hot_valid"][row] != 0
        cold = ~valid_hot & ~abc
        candidate_hot = valid_hot & candidate & ~abc
        eligible = (candidate | (positions >= resident)) & (a["pre.address_valid"][row] != 0) & ~abc
        selected = np.unique(a["post.logical_output"][packed_rows == row])
        selected_mask = np.zeros(h, dtype=bool)
        selected_mask[inverse[selected]] = True
        old_ids = previous_seed["requests"][identity]
        if np.any(old_ids < 0) or np.any(old_ids >= h):
            raise ValueError("Previous TopK logical IDs are outside the current history")
        # The same logical ID has moved physical slots since the prior round.
        seed_positions = np.unique(np.concatenate((positions[:resident], inverse[old_ids])))
        unions = {name: np.zeros(h, dtype=bool) for name in POLICIES}
        all_final_ties = np.zeros(h, dtype=bool)
        gaps = []
        equal = 0
        recent_gaps = []
        recent_equal = retired_total = retired_included = seed_total = seed_covered = 0
        recent_added = 0
        for q in range(QUERIES):
            end = int(a["pre.query_ends"][row, q])
            if causal_maximum[end - 1] >= end:
                raise ValueError("Physical causal score prefix contains a future logical token")
            keys = a["post.score_keys"][row, q, :end]
            records = a["post.threshold_records"][row, q]
            initial_records = records[np.isin(records[:, 3], (1, 2))]
            final_records = records[records[:, 3] == 4]
            if len(initial_records) != 1 or len(final_records) != 1:
                raise ValueError("Exactly one resident initial and one final threshold are required")
            initial, final = int(initial_records[0, 2]), int(final_records[0, 2])
            causal_seed = seed_positions[seed_positions < end]
            seeded = _kth(keys[causal_seed])
            if not initial <= seeded <= final:
                raise ValueError("Hot/previous union threshold must lie between resident and final thresholds")
            recent = recent_positions(resident, recent_ranges[row], end)
            recent_key = _kth(keys[recent])
            recent_added += len(recent) - min(resident, end)
            if not initial <= recent_key <= final:
                raise ValueError("Resident/recent threshold must lie between resident and final thresholds")
            recent_mask = np.zeros(end, dtype=bool)
            recent_mask[recent] = True
            retired = np.arange(int(recent_ranges[row, 0]), min(begin, end), dtype=np.int32)
            included = int(recent_mask[retired].sum())
            previous_positions = inverse[old_ids]
            previous_positions = previous_positions[previous_positions < end]
            covered = int(recent_mask[previous_positions].sum())
            retired_total += len(retired)
            retired_included += included
            seed_total += len(previous_positions)
            seed_covered += covered
            rolling, updates = _rolling_mask(keys, resident, initial)
            masks = dict(
                hot=keys > initial,
                recent=keys > recent_key,
                seed_hot=keys > seeded,
                rolling=rolling,
                final=keys > final,
            )
            ties = keys == final
            all_final_ties[:end] |= ties
            for name, mask in masks.items():
                admitted = mask & eligible[:end]
                unions[name][:end] |= admitted
            initial_value, final_value, seed_value = map(score_value, (initial, final, seeded))
            recent_value = score_value(recent_key)
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = _finite(final_value - initial_value)
                relative = _finite((final_value - initial_value) / abs(final_value)) if final_value != 0 else None
                recent_gap = _finite(final_value - recent_value)
                recent_relative = _finite((final_value - recent_value) / abs(final_value)) if final_value != 0 else None
            if gap is not None:
                gaps.append(gap)
            equal += initial == final
            if recent_gap is not None:
                recent_gaps.append(recent_gap)
            recent_equal += recent_key == final
            query_selected = a["post.logical_output"][a["pre.metadata"][2] == row * QUERIES + q].reshape(-1)
            query_cold = int(cold[inverse[query_selected]].sum())
            queries.append(
                dict(
                    dp_rank=identity[0],
                    group=source["group"],
                    scheduler_round=source["context"]["round"],
                    packed_row=row,
                    owner=identity[2],
                    request_id=identity[1],
                    query=q,
                    end=end,
                    initial_key=initial,
                    final_key=final,
                    seed_hot_key=seeded,
                    initial_value=_finite(initial_value),
                    final_value=_finite(final_value),
                    seed_hot_value=_finite(seed_value),
                    gap=gap,
                    relative_gap=relative,
                    initial_positions=min(resident, end),
                    initial_valid_hot=int(valid_hot[:end].sum()),
                    seed_hot_positions=len(causal_seed),
                    recent_key=recent_key,
                    recent_value=_finite(recent_value),
                    recent_gap=recent_gap,
                    recent_relative_gap=recent_relative,
                    recent_begin=int(recent_ranges[row, 0]),
                    recent_end=min(int(recent_ranges[row, 1]), end),
                    recent_positions=len(recent),
                    recent_added_positions=len(recent) - min(resident, end),
                    retired_abc_positions=len(retired),
                    retired_abc_included=included,
                    previous_topk_positions=len(previous_positions),
                    previous_topk_covered_by_recent=covered,
                    rolling_updates=updates,
                    topk_selected=len(query_selected),
                    topk_cold=query_cold,
                    topk_cold_miss_percent=100 * query_cold / len(query_selected),
                )
            )
        observed = summary[row]
        result = dict(
            dp_rank=identity[0],
            group=source["group"],
            scheduler_round=source["context"]["round"],
            packed_row=row,
            owner=identity[2],
            request_id=identity[1],
            layer_names="|".join(layers),
            layer_count=len(layers),
            token_bytes=g["token_bytes"],
            gap_query_count=QUERIES,
            gap_equal_queries=equal,
            gap_sum=sum(gaps),
            gap_max=max(gaps) if gaps else None,
            gap_finite_count=len(gaps),
            recent_equal_queries=recent_equal,
            recent_gap_sum=sum(recent_gaps),
            recent_gap_max=max(recent_gaps) if recent_gaps else None,
            recent_finite_count=len(recent_gaps),
            retired_abc_positions=retired_total,
            retired_abc_included=retired_included,
            previous_topk_positions=seed_total,
            previous_topk_covered_by_recent=seed_covered,
            recent_added_positions=recent_added,
            recent_range_positions_max=int(recent_ranges[row, 1] - recent_ranges[row, 0]),
            actual_prefetch=observed["reads"]["prefetch"],
            actual_final=observed["reads"]["final"],
            actual_unused_prefetch=observed["reads"]["unused_prefetch"],
            actual_read_bytes=observed["reads"]["bytes"],
            actual_unused_bytes=observed["reads"]["unused_bytes"],
        )
        for name in ("selected_unique", "cold_demand", "candidate_hot_refetch", "hot_hit", "abc_hit"):
            result[name] = observed[name]
        for category in ("cold", "candidate_hot", "other_hot", "abc"):
            result["actual_unused_" + category] = observed["unused_prefetch_" + category]
        for name in POLICIES:
            mask = unions[name]
            extra = mask & ~selected_mask
            count = int(extra.sum())
            # A boundary tie in any causal query is kept separate from tokens
            # strictly below all final query thresholds. This is a score class,
            # not identification of which query triggered the actual transfer.
            ties = int((extra & all_final_ties).sum())
            result.update(
                {
                    name + "_candidates": int(mask.sum()),
                    name + "_selected_candidates": int((mask & selected_mask).sum()),
                    name + "_extra_candidates": count,
                    name + "_extra_ties": ties,
                    name + "_extra_below": count - ties,
                    name + "_cold_extra": int((extra & cold).sum()),
                    name + "_candidate_hot_extra": int((extra & candidate_hot).sum()),
                    name + "_extra_bytes": count * g["token_bytes"],
                }
            )
        # Actual transfer records retain duplicates; policy sets do not.
        records = a["post.prefetch_records"]
        read_ids = records[records[:, 0] == row, 2]
        read_positions = inverse[read_ids]
        unused = ~selected_mask[read_positions]
        result["actual_unused_final_ties"] = int((unused & all_final_ties[read_positions]).sum())
        result["actual_unused_below_final"] = int((unused & ~all_final_ties[read_positions]).sum())
        if result["final_extra_candidates"]:
            raise ValueError("A strictly above-final candidate is missing from the captured Q4 TopK union")
        requests.append(result)
    return requests, queries
