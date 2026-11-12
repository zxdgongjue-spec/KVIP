# SPDX-License-Identifier: Apache-2.0
"""P0-A2: valid-K Prefetch audit cannot count stale Mapping suffix."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from prefetch_audit import actual_reads  # noqa: E402


def _mixed_fixture():
    mapping = np.full((2, 16), -1, dtype=np.int32)
    mapping[0, 2] = 7
    mapping[0, 7] = 8
    mapping[1, 1] = 3
    mapping[1, 9] = 5
    # A preceding poison/reset step left an intentionally nonnegative
    # sentinel beyond actual K, not actual device Prefetch requests.
    mapping[0, 8:] = 1
    mapping[1, 12:] = 1
    topk = np.array([[2, 7, 7], [1, 8, 11]], dtype=np.int32)
    rows = np.zeros((2, 249), dtype=np.uint32)
    rows[:, 9] = [2, 2]
    rows[:, 247] = [1, 0]
    return mapping, topk, rows


def test_mixed_k_ignores_poisoned_suffix_but_checks_device_count():
    mapping, topk, rows = _mixed_fixture()
    actual = actual_reads(mapping, topk, rows, 4, (8, 12))
    assert actual == {
        "prefetch_hot": 2,
        "prefetch_cold": 2,
        "unused_hot": 0,
        "unused_cold": 1,
        "final": 1,
        "total": 5,
        "unused": 1,
        "ignored_out_of_K_mapping": 12,
    }


def test_legacy_full_history_still_rejects_count_mismatch():
    mapping, topk, rows = _mixed_fixture()
    with pytest.raises(ValueError, match="Prefetch mapping and device request counts disagree"):
        actual_reads(mapping, topk, rows, 4)


def test_mixed_k_does_not_hide_in_k_counter_mismatch():
    mapping, topk, rows = _mixed_fixture()
    rows[0, 9] += 1
    with pytest.raises(ValueError, match="mapping_valid_K=4 device=5"):
        actual_reads(mapping, topk, rows, 4, (8, 12))


def test_mixed_k_rejects_bad_lengths_and_invalid_topk():
    mapping, topk, rows = _mixed_fixture()
    for k in ((8,), (8, 17), (-1, 12), (8, "12")):
        with pytest.raises(ValueError, match="invalid per-request actual K"):
            actual_reads(mapping, topk, rows, 4, k)
    topk[0, 0] = 8
    with pytest.raises(ValueError, match="selected out-of-K Token"):
        actual_reads(mapping, topk, rows, 4, (8, 12))


def test_fully_active_k_remains_compatible():
    mapping, topk, rows = _mixed_fixture()
    rows[:, 9] = [10, 6]
    full = actual_reads(mapping, topk, rows, 4, (16, 16))
    assert full["ignored_out_of_K_mapping"] == 0
    assert full["prefetch_hot"] + full["prefetch_cold"] == 16
