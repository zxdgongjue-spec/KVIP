"""Correctness risks for producer/consumer incremental Index updates."""
import random

import pytest

from reference import SparsePermutation, materialized_oracle, sparse_oracle


@pytest.mark.parametrize("batches", [
    [[(9, 0), (10, 1)], [(11, 2)], [], [], []],
    [[(9, 0)], [(0, 0)], [], [], []],  # overwritten old-hot victim returns
    [[(9, 0)], [(10, 0)], [(0, 1)], [(9, 2)], []],  # multi-batch chain
    [[(9, 0), (10, 1)], [(0, 1), (1, 0)], [], [], []],
    [[], [], [], [], []],
    [[(0, 0)], [(0, 0)], [], [], []],
])
def test_sparse_matches_materialized(batches):
    expected, e_steps = materialized_oracle(32, 8, batches)
    actual, a_steps = sparse_oracle(32, 8, batches)
    assert actual == expected
    assert a_steps == e_steps
    assert sorted(actual) == list(range(32))


def test_random_repeated_cross_batch_swaps():
    rng = random.Random(71)
    for _ in range(40):
        batches = [[(rng.randrange(96), rng.randrange(16)) for _ in range(rng.randrange(12))]
                   for _ in range(5)]
        assert sparse_oracle(96, 16, batches) == materialized_oracle(96, 16, batches)


def test_invalid_inputs_are_rejected():
    state = SparsePermutation(16, 4)
    for source, destination in [(-1, 0), (16, 0), (0, -1), (0, 4)]:
        with pytest.raises(ValueError):
            state.swap(source, destination)


def test_ordered_frontier_preserves_cross_group_swaps():
    from reference import partition_ordered

    batches = [[(9, 0), (10, 1), (0, 0), (11, 2), (9, 3)],
               [(0, 2), (10, 0), (11, 1)], [], [], []]
    state = SparsePermutation(32, 8)
    sizes = []
    for batch in batches:
        sizes.extend(map(len, partition_ordered(state, batch)))
    assert state.commit() == {i: v for i, v in enumerate(materialized_oracle(32, 8, batches)[0]) if i != v}
    assert len(sizes) > 2  # the overlapping source/destination ends a frontier


def test_random_frontiers_preserve_full_permutation():
    from reference import partition_ordered

    rng = random.Random(83)
    for _ in range(40):
        batches = [[(rng.randrange(96), rng.randrange(16)) for _ in range(rng.randrange(20))]
                   for _ in range(5)]
        state = SparsePermutation(96, 16)
        for batch in batches:
            for _ in partition_ordered(state, batch):
                pass
        expected, _ = materialized_oracle(96, 16, batches)
        assert [state.row_at.get(i, i) for i in range(96)] == expected


def test_large_nonconflicting_group_and_boundary_overlap():
    from reference import partition_ordered

    state = SparsePermutation(4096, 1024)
    batch = [(1024+i, i) for i in range(512)] + [(0, 512), (2048, 513)]
    groups = list(partition_ordered(state, batch, limit=512))
    assert [len(x) for x in groups] == [512, 2]
    expected, _ = materialized_oracle(4096, 1024, [batch])
    assert [state.row_at.get(i, i) for i in range(4096)] == expected


def test_parallel_batch_tracks_cross_query_recovery():
    from reference import apply_parallel_batch

    state = SparsePermutation(32, 8)
    apply_parallel_batch(state, [(9, 0), (10, 1)])
    apply_parallel_batch(state, [(0, 1), (9, 2)])
    assert [state.row_at.get(i, i) for i in (1, 2)] == [0, 9]
    assert sorted(state.row_at.get(i, i) for i in range(32)) == list(range(32))


def test_parallel_batches_random_permutations_and_cycles():
    from reference import apply_parallel_batch

    rng = random.Random(723)
    for _ in range(100):
        state = SparsePermutation(96, 16)
        for _ in range(5):
            count = rng.randrange(17)
            incoming = rng.sample(range(96), count)
            targets = rng.sample(range(16), count)
            apply_parallel_batch(state, list(zip(incoming, targets)))
            rows = [state.row_at.get(i, i) for i in range(96)]
            assert sorted(rows) == list(range(96))
            assert all(rows[d] == source for source, d in zip(incoming, targets))
            assert all(state.position_of.get(row, row) == pos for pos, row in enumerate(rows))


def test_parallel_batch_rejects_duplicate_rows_or_destinations():
    from reference import apply_parallel_batch

    for events in ([(9, 0), (9, 1)], [(9, 0), (10, 0)]):
        with pytest.raises(ValueError):
            apply_parallel_batch(SparsePermutation(32, 8), events)


def test_real_geometry_prefetch_and_four_queries():
    from reference import apply_parallel_batch

    rng = random.Random(20261110)
    history, resident = 131072, 16384
    state = SparsePermutation(history, resident)
    batches = [list(zip(rng.sample(range(resident, history), 512),
                        rng.sample(range(resident), 512)))]
    for count in (141, 205, 0, 0):
        batches.append(list(zip(rng.sample(range(history), count),
                                rng.sample(range(resident), count))))
    for events in batches:
        apply_parallel_batch(state, events)
        rows = [state.row_at.get(i, i) for i in range(history)]
        assert sorted(rows) == list(range(history))
        assert all(rows[d] == source for source, d in events)
        assert all(state.position_of.get(row, row) == pos for pos, row in enumerate(rows))
