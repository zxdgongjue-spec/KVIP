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
    assert verify_permutation(flags, src, dst, before, after, owners) == [2, 0]
    load = describe_row_load(flags, src, dst, [1, 1, 1, 2], 2)
    assert load["row_moves"] == [2, 3]
    assert load["row_events"] == [1, 2]
    assert load["simple_swap_rows"] == 1
    assert load["dense_eligible_rows"] == 0
