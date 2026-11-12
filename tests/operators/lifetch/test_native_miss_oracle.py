# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for Native MTP LIDU Miss classification and Q4 union."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from native_miss_oracle import native_miss_oracle  # noqa: E402


def test_victim_resident_must_count_as_miss():
    # slots 0 and 2 are marked as victims. Token 5 and 6 are cold.
    mapping = np.array([0, 1, 2, 3, -1, -1, -1, -1], np.int32)
    bitmap = np.array([0b0101], np.int32)
    result = native_miss_oracle(
        [0, 1, 1, 2, 5, 5, 6], mapping, bitmap, 4
    )
    assert result == {
        "selected_union": 5,
        "unmapped_unique": 2,
        "victim_resident_unique": 2,
        "ordinary_resident_hits": 1,
        "effective_unique_misses": 4,
    }


def test_q4_reuses_same_token_without_double_counting():
    mapping = np.arange(16, dtype=np.int32)
    bitmap = np.array([0x0000000F], dtype=np.int32)
    # Q4 repeats token 2; the 4 victim slots count once each.
    q4 = np.tile(np.arange(8, dtype=np.int32), (4, 1))
    result = native_miss_oracle(q4, mapping, bitmap, 16)
    assert result["selected_union"] == 8
    assert result["effective_unique_misses"] == 4
    assert result["ordinary_resident_hits"] == 4


def test_bitmap_is_slot_indexed_not_token_indexed():
    mapping = np.array([7, 0, -1, 3, 4, 5, 6, 2], np.int32)
    bitmap = np.array([0b10000001], np.int32)
    result = native_miss_oracle(
        [0, 1, 2, 7], mapping, bitmap, 8
    )
    assert result["unmapped_unique"] == 1
    assert result["victim_resident_unique"] == 2
    assert result["ordinary_resident_hits"] == 1
    assert result["effective_unique_misses"] == 3


@pytest.mark.parametrize(
    "selected,bitmap,budget",
    [([-1], [0], 4), ([8], [0], 4), ([0], [], 4), ([0], [0], 0)],
)
def test_invalid_prelaunch_inputs_fail_closed(selected, bitmap, budget):
    mapping = np.array([0, 1, 2, 3, -1, -1, -1, -1], np.int32)
    with pytest.raises(ValueError, match="native_miss_oracle_invalid"):
        native_miss_oracle(
            selected, mapping, np.array(bitmap, np.int32), budget
        )
