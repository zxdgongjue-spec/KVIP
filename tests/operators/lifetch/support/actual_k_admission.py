"""P0-A2 test-only admission for fixed History and per-request BSND K.

The first actual K tokens are eligible. A four-query TopK union must fit
its request's physical resident slot capacity before Native LIDU runs.
"""
import numpy as np

TOPK = 2048
BLOCK = 128


def parse_actual_key_lengths(value, batch, history, resident):
    if value is None:
        return None
    try:
        fields = str(value).split(",")
        if len(fields) != batch or any(not x.strip() for x in fields):
            raise ValueError("wrong number of request K values")
        values = tuple(int(x.strip()) for x in fields)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"actual_key_lengths_invalid: expected {batch} comma-separated integers"
        ) from exc
    if any(k < max(TOPK, resident) or k > history or k % BLOCK for k in values):
        raise ValueError(
            "actual_key_lengths_invalid: max(2048, Resident) <= K <= History "
            "and all K values must align to 128"
        )
    return values


def score_oracle_for_actual_k(logical_scores, physical_order, lengths, resident):
    """Independent per-K BF16 score oracle; reject ambiguous boundary ties."""
    logical = np.asarray(logical_scores)
    order = np.asarray(physical_order)
    if logical.ndim != 2 or logical.shape[0] != 4:
        raise ValueError("actual_k_oracle_expected_four_query_scores")
    history = logical.shape[1]
    if order.shape != (history,) or not lengths:
        raise ValueError("actual_k_oracle_invalid_geometry")
    physical_scores = logical[:, order]
    reports = []
    selected_rows = []
    for row, key in enumerate(lengths):
        key = int(key)
        if key < TOPK or key > history:
            raise ValueError(f"actual_k_oracle_invalid_key: row={row} K={key}")
        chosen = []
        for query in range(4):
            prefix = physical_scores[query, :key]
            threshold = np.partition(prefix, -TOPK)[-TOPK]
            indices = np.flatnonzero(prefix >= threshold).astype(np.int32)
            if indices.size != TOPK:
                raise ValueError(
                    f"actual_k_oracle_boundary_tie: row={row} query={query} "
                    f"K={key} winners={indices.size}"
                )
            chosen.append(indices)
        union = np.unique(np.concatenate(chosen))
        cold = int(np.count_nonzero(union >= resident))
        if union.size > resident:
            raise ValueError(
                f"actual_k_native_union_exceeds_resident: row={row} K={key} "
                f"union={union.size} R={resident}"
            )
        reports.append({
            "request": row, "actual_key_length": key,
            "unique_topk": int(union.size),
            "unique_hot": int(union.size) - cold,
            "unique_cold": cold,
            "resident_capacity": int(resident),
            "spare_slots": int(resident - union.size),
            "boundary_tie_free": True,
        })
        selected_rows.append(np.stack(chosen))
    return reports, np.stack(selected_rows)
