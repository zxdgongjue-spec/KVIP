# SPDX-License-Identifier: Apache-2.0
"""Count real reciprocal swaps and prior core exits around the last row's Index join."""

import numpy as np


def survey_index_helpers(rows, marks):
    """Use existing RQI timestamps. One hypothetical donor must finish its own row duties first."""
    # block = q * 16 + row. Only reciprocal pairs are independent of every
    # other Index tuple. Long-cycle originals can live in the owner's private UB.
    pairs = marks[:, 16].reshape(4, 16).sum(axis=0)
    singles = marks[:, 17].reshape(4, 16).sum(axis=0)
    row_moves = 2 * pairs + singles
    index_end = rows[:, 8].reshape(4, 16).max(axis=0)
    last = int(index_end.argmax())
    heaviest = int(row_moves.argmax())
    # This is the latest lane to pass the row's validation/TND/Post join.
    # All role exit and Index end counters use the same device clock.
    admission = int(marks[:, 3].reshape(4, 16)[:, last].max())
    deadline = int(index_end[last])
    exits = marks[:, 1].reshape(4, 16)
    foreign = np.arange(16) != last
    # This is idle time that a donor could have used only if a yet unclaimed
    # exchange was available. It does not predict transport or execution time.
    windows = [max(0, deadline - max(int(t), admission)) for t in exits[:, foreign].flat]
    row_exit_gap = max(0, int(exits[:, last].max()) - int(exits[:, foreign].max()))
    return dict(last_index_row=last, heaviest_row=heaviest,
                last_row_pairs=int(pairs[last]), last_row_singles=int(singles[last]),
                heaviest_pairs=int(pairs[heaviest]), heaviest_singles=int(singles[heaviest]),
                freed_lanes=sum(w > 0 for w in windows), longest_free_window_ticks=max(windows),
                free_lane_ticks=sum(windows), last_row_only_exit_gap_ticks=row_exit_gap)
