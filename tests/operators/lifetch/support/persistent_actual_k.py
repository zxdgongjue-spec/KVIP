# SPDX-License-Identifier: Apache-2.0
"""P0-A2.2b: monotonic Actual-K growth with persistent Index/KV state.

The test fixture's initial physical row p contains the Index key for
base_physical_to_logical[p]. Direct Index Update moves complete Index rows;
current_physical_to_base therefore maps each current physical row back to
its original physical row. Compose the two mappings before evaluating the
independent score oracle.
"""
import numpy as np

TOPK = 2048
BLOCK = 128


def persistent_growth_schedule(initial_k, history, resident):
    """Advance each request monotonically through the initial K levels to H."""
    initial = tuple(int(value) for value in initial_k)
    if len(initial) != 4:
        raise ValueError("persistent_actual_k_requires_batch4")
    if (
        history < TOPK or history % BLOCK or resident < TOPK
        or resident > history
        or any(k < max(TOPK, resident) or k > history or k % BLOCK for k in initial)
    ):
        raise ValueError("persistent_actual_k_invalid_geometry")
    if max(initial) != history:
        raise ValueError("persistent_actual_k_requires_full_history_terminal")
    levels = sorted(set(initial))
    if len(levels) < 2:
        raise ValueError("persistent_actual_k_requires_multiple_initial_K_levels")

    schedule = []
    current = initial
    for stage in range(len(levels)):
        schedule.append(("initial" if stage == 0 else f"grow_{stage}", current))
        if stage + 1 == len(levels):
            break
        next_values = []
        for value in current:
            index = levels.index(value)
            next_values.append(levels[min(index + 1, len(levels) - 1)])
        current = tuple(next_values)

    if schedule[-1][1] != (history,) * 4:
        raise AssertionError("persistent_actual_k_schedule_did_not_reach_history")
    for (_, before), (_, after) in zip(schedule, schedule[1:]):
        if any(new < old for old, new in zip(before, after)):
            raise AssertionError("persistent_actual_k_schedule_not_monotonic")
    return schedule


def score_oracle_for_persistent_state(
    logical_scores,
    base_physical_to_logical,
    current_physical_to_base,
    lengths,
    resident,
):
    """Evaluate current physical K prefixes after persistent Index permutations."""
    scores = np.asarray(logical_scores)
    base_order = np.asarray(base_physical_to_logical)
    current_map = np.asarray(current_physical_to_base)
    if scores.ndim != 2 or scores.shape[0] != 4:
        raise ValueError("persistent_actual_k_oracle_expected_four_query_scores")
    history = scores.shape[1]
    if base_order.shape != (history,) or current_map.ndim != 2:
        raise ValueError("persistent_actual_k_oracle_invalid_mapping_shape")
    batch = current_map.shape[0]
    if current_map.shape[1] != history or len(lengths) != batch:
        raise ValueError("persistent_actual_k_oracle_invalid_batch_geometry")
    expected = np.arange(history, dtype=np.int64)
    if not np.array_equal(np.sort(base_order.astype(np.int64)), expected):
        raise ValueError("persistent_actual_k_base_order_not_permutation")

    reports = []
    selected_rows = []
    for row in range(batch):
        mapping = current_map[row].astype(np.int64, copy=False)
        if not np.array_equal(np.sort(mapping), expected):
            raise ValueError(f"persistent_actual_k_state_map_not_permutation:row={row}")
        key = int(lengths[row])
        if key < max(TOPK, resident) or key > history or key % BLOCK:
            raise ValueError(f"persistent_actual_k_invalid_key:row={row}:K={key}")

        # Current row -> original physical row -> logical score/token identity.
        current_order = base_order[mapping]
        chosen = []
        for query in range(4):
            prefix = scores[query, current_order[:key]]
            cutoff = np.partition(prefix, -TOPK)[-TOPK]
            indices = np.flatnonzero(prefix >= cutoff).astype(np.int32)
            if indices.size != TOPK:
                raise ValueError(
                    f"persistent_actual_k_boundary_tie:row={row}:query={query}:"
                    f"K={key}:winners={indices.size}"
                )
            chosen.append(indices)
        union = np.unique(np.concatenate(chosen))
        if union.size > resident:
            raise ValueError(
                f"persistent_actual_k_native_union_exceeds_resident:"
                f"row={row}:K={key}:union={union.size}:R={resident}"
            )
        cold = int(np.count_nonzero(union >= resident))
        reports.append({
            "request": row,
            "actual_key_length": key,
            "unique_topk": int(union.size),
            "unique_hot": int(union.size) - cold,
            "unique_cold": cold,
            "resident_capacity": int(resident),
            "spare_slots": int(resident - union.size),
            "boundary_tie_free": True,
        })
        selected_rows.append(np.stack(chosen))
    return reports, np.stack(selected_rows)
