# SPDX-License-Identifier: Apache-2.0
"""Explicit cross-query hazards for the staged resolver golden model."""
import pytest

from staged_resolver_reference import resolve


def test_cross_query_reuse_and_future_hot_protection():
    # Slot 4 is used by Q4 and must not be selected by earlier repairs.
    out, records, counts = resolve([[20, 21], [20, 22], [23, 20], [4, 23]],
                                   8, [0, 1, 2, 3], 1, {}, [1] * 8)
    assert counts == [2, 1, 1, 0]
    assert [token for token, _ in records] == [20, 21, 22, 23]
    assert out[0][0] == out[1][0] == out[2][1]
    assert out[3][0] == 4
    assert all(slot != 4 for _, slot in records)
    assert [slot for _, slot in records] == [1, 2, 3, 0]


def test_protect_prefetch_and_padding_and_repair_reserved_hot():
    out, records, counts = resolve([[9, 0, -1], [9], [8], [0]],
                                   8, [0, 1, 2, 3], 1, {9: 1}, [1] * 8)
    assert out[0][0] == out[1][0] == 1
    assert out[0][2] == -1
    assert out[0][1] == out[3][0]
    assert records == [(0, 2), (8, 3)]
    assert counts == [1, 0, 1, 0]


def test_insufficient_slots_is_error():
    with pytest.raises(ValueError, match='insufficient'):
        resolve([[3], [4], [], [1]], 2, [0], 0, {}, [1, 1])
