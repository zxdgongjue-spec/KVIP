"""CPU oracle for Native MTP LIDU per-request unique Miss count.

ClassifyTopkSimt considers a selected token a Hit only when its initial
logical-token -> physical-slot mapping is valid AND that physical slot is
NOT marked in the input miss-slot/victim bitmap. Q1..Q4 share one union.
This oracle is strictly for test setup and must not run inside timed replays.
"""

import numpy as np


def native_miss_oracle(selected_ids, initial_forward_map, miss_slot_bitmap, resident_budget):
    ids = np.asarray(selected_ids, dtype=np.int64).reshape(-1)
    mapping = np.asarray(initial_forward_map, dtype=np.int32).reshape(-1)
    words = np.asarray(miss_slot_bitmap, dtype=np.int32).reshape(-1).view(np.uint32)
    budget = int(resident_budget)
    if budget <= 0 or words.size < (budget + 31) // 32:
        raise ValueError("native_miss_oracle_invalid_bitmap_geometry")
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= mapping.size):
        raise ValueError("native_miss_oracle_invalid_selected_ids")

    unique = np.unique(ids)
    slots = mapping[unique].astype(np.int64)
    mapped = (slots >= 0) & (slots < budget)
    protected = np.zeros(unique.size, dtype=np.bool_)
    valid_slots = slots[mapped]
    protected[mapped] = (
        words[valid_slots // 32] &
        (np.uint32(1) << (valid_slots % 32).astype(np.uint32))
    ) != 0
    return dict(
        selected_union=int(unique.size),
        unmapped_unique=int(np.count_nonzero(~mapped)),
        victim_resident_unique=int(np.count_nonzero(protected)),
        ordinary_resident_hits=int(np.count_nonzero(mapped & ~protected)),
        effective_unique_misses=int(np.count_nonzero(~mapped | protected)),
    )
