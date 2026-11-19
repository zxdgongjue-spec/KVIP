# SPDX-License-Identifier: Apache-2.0

import numpy as np

from kvip.diagnostics.high_index_profile import describe_row_load, verify_permutation


def test_load_and_two_cycles_follow_final_permutation():
    flags = np.zeros((2, 16, 32), dtype=np.int32)
    flags[:, 0, 1] = [1, 1]
    flags[1, 12, 1] = 1
    src = np.full((2, 10240), -1, dtype=np.int32)
    dst = src.copy()
    src[:, 0], dst[:, 0] = 3, 0
    src[1, 2048], dst[1, 2048] = 3, 1
    before = np.tile(np.arange(4), (2, 1))
    # Same first swap. Row 1 subsequently moves that same original token
    # again, producing a 3-cycle. Neither unused padding nor owner order count.
    after = np.array([[1, 3, 2, 0], [3, 1, 2, 0]])
    owners = [1, 0]
    histograms = []
    assert verify_permutation(flags, src, dst, before, after, owners, cycle_histograms_out=histograms) == [2, 0]
    assert histograms == [{2: 1}, {3: 1}]
    load = describe_row_load(flags, src, dst, [1, 1, 1, 2], 2)
    assert load["row_moves"] == [2, 3]
    assert load["row_events"] == [1, 2]
    assert load["simple_swap_rows"] == 1
    assert load["dense_eligible_rows"] == 0

def test_cycle_histogram_covers_change_lengths_and_omits_unchanged_slots():
    import json
    from kvip.diagnostics.consumer_ab import cycle_move_bins

    lengths = [2, 2, 3, 4, 5, 8, 9, 107]
    size = sum(lengths)
    before = np.arange(size + 11)[None]
    src = np.zeros((1, 10240), np.int32)
    dst = np.zeros_like(src)
    after = before.copy()
    flags = np.zeros((1, 16, 32), np.int32)
    flags[0, 12, 1] = size
    start = 0
    for length in lengths:
        end = start + length
        positions = np.arange(start, end)
        shifted = np.roll(positions, 1)
        src[0, 2048+start:2048+end], dst[0, 2048+start:2048+end] = shifted, positions
        after[0, positions] = shifted
        start = end
    hist = []
    assert verify_permutation(flags, src, dst, before, after, [0], cycle_histograms_out=hist) == [4]
    assert hist == [{2: 2, 3: 1, 4: 1, 5: 1, 8: 1, 9: 1, 107: 1}]
    assert cycle_move_bins(json.loads(json.dumps(hist))) == "4/3/4/13/116"
