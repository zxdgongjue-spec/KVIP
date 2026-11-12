# SPDX-License-Identifier: Apache-2.0
"""Legacy snapshot inverse-map validation; never used in the service path."""

import numpy as np

QUERIES_PER_REQUEST = 4


def reserved_hot_reference(candidates, query_ends, resident):
    """Diagnostic CPU reference for init's compact LRU inverse mapping.

    Inactive packed rows are inert and canonicalized to -1 in snapshots. The
    service uses its existing init kernel, never this CPU reconstruction.
    """
    candidates, query_ends = np.asarray(candidates), np.asarray(query_ends)
    if (
        candidates.ndim != 2
        or candidates.dtype != np.int32
        or query_ends.size != candidates.shape[0] * QUERIES_PER_REQUEST
        or type(resident) is not int
        or resident < candidates.shape[1]
    ):
        raise ValueError("Reserved-hot reference requires int32 candidates, Q4 bounds and sufficient hot slots")
    ends = query_ends.reshape(candidates.shape[0], QUERIES_PER_REQUEST)
    if np.any(ends < 0):
        raise ValueError("Reserved-hot query bounds must be nonnegative")
    result = np.full((candidates.shape[0], resident), -1, dtype=np.int32)
    ordinals = np.arange(candidates.shape[1], dtype=np.int32)
    for row in np.flatnonzero(np.any(ends > 0, axis=1)):
        slots = candidates[row]
        if np.any(slots < 0) or np.any(slots >= resident) or len(np.unique(slots)) != len(slots):
            raise ValueError("Active LRU candidates must be unique in-range hot slots")
        result[row, slots] = ordinals
    return result
