import numpy as np

from real_fetch_inputs import prepare_full_chain_fixture


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
