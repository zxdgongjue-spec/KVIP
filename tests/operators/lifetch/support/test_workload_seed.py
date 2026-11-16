import numpy as np

from workloads import prepare_full_chain_fixture


def test_seed_is_reproducible_and_changes_real_li_inputs(tmp_path):
    roots = [tmp_path / str(i) for i in range(3)]
    for root, seed in zip(roots, [43, 43, 44]):
        info = prepare_full_chain_fixture(root, 8192, 4096, 10, index_heads=32, seed=seed)
        assert info['seed'] == seed
    for name in ('arg1.bin', 'arg4.bin', 'logical_scores.npy'):
        assert (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes()
        assert (roots[0] / name).read_bytes() != (roots[2] / name).read_bytes()
    for root in roots:
        scores = np.load(root / 'logical_scores.npy')
        assert scores.shape == (4, 8192)


def test_exact_miss_boundaries_for_every_query(tmp_path):
    for heads in (32, 64):
        for miss in (0, 10, 100):
            root = tmp_path / f'{heads}-{miss}'
            info = prepare_full_chain_fixture(root, 131072, 16384, miss, index_heads=heads, seed=43)
            target = round(2048 * miss / 100)
            assert info['final_misses_per_query'] == [target] * 4
            scores = np.load(root / 'logical_scores.npy')
            order = np.load(root / 'physical_to_logical.npy')
            for query in scores:
                cutoff = np.partition(query, -2048)[-2048]
                assert np.count_nonzero(query >= cutoff) == 2048
                assert np.count_nonzero(query[order[16384:]] >= cutoff) == target
