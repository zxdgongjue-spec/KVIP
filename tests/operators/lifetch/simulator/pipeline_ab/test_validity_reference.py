import pytest
from itertools import permutations, product
from reference import SparsePermutation
from validity_reference import reconcile_hot_valid


def test_displaced_hot_index_without_kv_is_invalidated():
    state = SparsePermutation(16, 4)
    state.swap(6, 0)
    state.swap(2, 0)
    rows = [state.row_at.get(i, i) for i in range(16)]
    assert rows[:4] == [2, 1, 6, 3]
    assert reconcile_hot_valid(rows, [1] * 4, [(0, 2)], 4) == [1, 1, 0, 1]


def test_prefetch_replacement_is_valid_when_latest_record_matches():
    state = SparsePermutation(16, 4)
    state.swap(6, 0)
    state.swap(7, 0)
    rows = [state.row_at.get(i, i) for i in range(16)]
    assert reconcile_hot_valid(rows, [1] * 4, [(0, 7)], 4) == [1, 1, 1, 1]


def test_mismatched_destination_is_error_not_silently_valid():
    with pytest.raises(ValueError, match='does not match'):
        reconcile_hot_valid([5, 1, 2, 3, 4, 0], [1] * 4, [(0, 4)], 4)


def test_preexisting_invalid_slot_remains_invalid_until_fetch():
    assert reconcile_hot_valid(list(range(8)), [1, 0, 1, 1], [], 4) == [1, 0, 1, 1]


def test_final_residency_publication_does_not_revive_displaced_hot():
    """Exhaust small unique-token streams, including overwritten destinations.

    Prefetch/final dedup emits each original token once. Fetch writes each
    record's destination and publishes the destination valid after the index
    swaps. No still-valid hot row may point to a different token's KV.
    """
    resident, history = 3, 6
    for tokens in permutations(range(history), 3):
        for destinations in product(range(resident), repeat=3):
            state = SparsePermutation(history, resident)
            valid = [True] * resident
            kv_owner = list(range(resident))
            for token, destination in zip(tokens, destinations):
                source = state.position_of.get(token, token)
                state.swap(token, destination)
                if source != destination and source < resident:
                    valid[source] = False
                kv_owner[destination] = token
            for destination in destinations:
                valid[destination] = True
            for slot in range(resident):
                if valid[slot]:
                    assert state.row_at.get(slot, slot) == kv_owner[slot], (tokens, destinations, slot)
