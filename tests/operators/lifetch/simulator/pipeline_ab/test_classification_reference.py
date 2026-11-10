import pytest
from classification_reference import MISSING, classify


def test_hot_reserved_prefetch_cold_and_padding():
    resident, history = 8, 32
    valid = [1] * resident
    reserved = [-1] * resident
    reserved[2] = 0
    mapping = [MISSING] * history
    mapping[2], mapping[10] = resident, resident + 1
    slots, bitmap = classify([1, 2, 10, 11, MISSING], resident, history, valid, reserved, mapping, [3, 4], 2)
    assert slots == [1, 3, 4, MISSING, MISSING]
    assert bitmap == [(1 << 1) | (1 << 3) | (1 << 4)]


def test_tail_and_invalid_hot_are_not_ordinary_hits():
    valid = [1, 0, 1, 1]
    reserved = [-1] * 4
    mapping = [MISSING] * 12
    mapping[1], mapping[2] = 4, 5
    slots, bitmap = classify([1, 2, 3], 4, 12, valid, reserved, mapping, [0, 1], 2, tail=(2, 3))
    assert slots == [0, 1, 3]
    assert bitmap == [0b1011]


def test_prefetch_absent_does_not_use_stale_mapping():
    slots, bitmap = classify([4], 4, 8, [1] * 4, [-1] * 4, [4] * 8, [2], 0)
    assert slots == [MISSING] and bitmap == [0]


def test_duplicate_slots_are_set_once():
    slots, bitmap = classify([1, 1, 1], 4, 8, [1] * 4, [-1] * 4, [MISSING] * 8, [], 0)
    assert slots == [1, 1, 1] and bitmap == [2]


@pytest.mark.parametrize('token', [-1, 8])
def test_invalid_token_is_rejected(token):
    with pytest.raises(ValueError):
        classify([token], 4, 8, [1] * 4, [-1] * 4, [MISSING] * 8, [], 0)
