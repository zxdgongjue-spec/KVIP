# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from long_history import max_history_sequence
from workloads import ReplayPagePlan


@pytest.mark.parametrize(("history", "visible"), [(524288, 524288), (524544, 524291)])
def test_max_history_oracle_and_last_page(tmp_path, history, visible):
    seq = max_history_sequence(tmp_path, history)
    row = next(iter(seq))
    assert np.array_equal(row.query_ends[0], np.arange(visible - 3, visible + 1))
    outputs = []
    for q, end in enumerate(row.query_ends[0]):
        # A unit query/scale/weight on every head reads exactly this one FP8 column.
        encoded = row.logical_keys[0, :, q]
        assert (encoded[end:] == 0x48).all()
        winners = np.flatnonzero(encoded[:end])
        assert len(winners) == 2048 and sum(winners >= row.geometry["resident"]) == 512
        assert winners[-1] == end - 1
        for boundary in (65536, 131072, 262144):
            assert np.isin([boundary - 1, boundary], winners).all()
        assert np.count_nonzero(encoded[:end] == 0x40) >= 64
        outputs.append(winners)
    selected = np.broadcast_to(outputs, (16, 4, 2048)).copy()
    row.validate_topk(selected)
    selected[0, 0, -1] = row.query_ends[0, 0]  # Has a higher score but is masked.
    with pytest.raises(AssertionError, match="causal prefix"):
        row.validate_topk(selected)
    pages = ReplayPagePlan(seq)
    assert pages.page_count == history // 128 + 15
    assert len(np.unique(pages.page_indices[:, (visible - 1) // 128])) == 16
