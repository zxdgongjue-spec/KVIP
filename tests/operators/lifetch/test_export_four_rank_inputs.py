# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np

from export_four_rank_inputs import physical_arrays, array_digest


def test_donor_rows_stay_in_capture_order():
    logical = np.arange(12).reshape(3, 4)
    q = np.arange(48).reshape(3, 4, 4)
    layout = np.tile([2, 0, 3, 1], (3, 1))
    small = np.arange(3)[:, None]
    s = SimpleNamespace(geometry={"history": 4}, captured_layout=layout, logical_keys=logical[..., None],
                        logical_scales=logical, logical_address_valid=logical, logical_score_keys=q,
                        logical_score_valid=q, query=small, weights=small, query_scale=small, query_ends=small,
                        tail_bounds=small, captured_hot_valid=small, captured_lru=small)
    arrays = dict(physical_arrays(s, [2, 0]))
    assert arrays["rows"].tolist() == [2, 0]
    assert arrays["keys"][0, :, 0].tolist() == [10, 8, 11, 9]
    assert arrays["score_keys"][0, 3].tolist() == [46, 44, 47, 45]
    assert array_digest(arrays.items()) == array_digest(reversed(arrays.items()))
    assert array_digest(arrays.items()) != array_digest(physical_arrays(s, [0, 2]))
