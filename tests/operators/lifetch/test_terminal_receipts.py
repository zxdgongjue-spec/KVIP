# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from correctness import FETCH_AB_MAGIC, validate_terminal_receipts
from kvip.operators.lifetch.terminal_queues import final_queue


def test_request_and_completion_counts_on_bound_queues():
    # Row totals straddle the pacing boundary. Small Qs in a long row can
    # have fewer than four batches, and an empty Q has no CQ to consume.
    counts = np.array([[252, 1, 2, 0], [250, 1, 2, 3]])
    views = [final_queue(2, row, query) for row in range(2) for query in range(4)]
    statuses = np.zeros((10, 4), dtype=np.uint64)
    statuses[views, 0] = FETCH_AB_MAGIC
    statuses[views, 1] = counts.ravel()
    statuses[views, 3] = [1, 1, 1, 0, 4, 1, 2, 3]
    validate_terminal_receipts(counts, statuses, views)

    for field in range(4):
        faulty = statuses.copy()
        faulty[views[-1], field] ^= np.uint64(1)
        with pytest.raises(RuntimeError, match="row=1 Q=4"):
            validate_terminal_receipts(counts, faulty, views)
