import numpy as np
import pytest

from compact_fixture import prepare_compact_fixture, validate_topk


def test_compact_fixture_layout_and_residency(tmp_path):
    roots = [tmp_path/'low', tmp_path/'high']
    infos = [prepare_compact_fixture(root, 8192, 4096, miss, seed=17)
             for root, miss in zip(roots, (5, 50))]
    assert infos[0]['model'] == 'compact_builtin_v1'
    assert sum(infos[0]['final_misses_per_query']) < sum(infos[1]['final_misses_per_query'])
    for root in roots:
        scores = np.load(root/'logical_scores.npy')
        order = np.load(root/'physical_to_logical.npy')
        physical = scores[:, order]
        selected = np.argpartition(physical, -2048, axis=1)[:, -2048:]
        validate_topk(selected, physical)
        assert np.isfinite(scores).all()
        assert (root/'arg0.bin').stat().st_size == 4*32*128
        assert (root/'arg1.bin').stat().st_size == 8192*128
    assert (roots[0]/'arg0.bin').read_bytes() == (roots[1]/'arg0.bin').read_bytes()
    # Residency changes positions, not the score distribution.
    np.testing.assert_array_equal(np.sort(np.load(roots[0]/'logical_scores.npy'), axis=1),
                                  np.sort(np.load(roots[1]/'logical_scores.npy'), axis=1))


def test_score_oracle_accepts_ties_and_rejects_bad_or_duplicate_rows():
    scores = np.concatenate((np.full(100, 4.), np.full(2000, 2.), np.zeros(1000)))[None, :]
    valid = np.arange(2048)
    validate_topk(valid, scores)
    alternate = valid.copy()
    alternate[-1] = 2099
    validate_topk(alternate, scores)
    invalid = valid.copy()
    invalid[-1] = 3000
    with pytest.raises(AssertionError):
        validate_topk(invalid, scores)
    invalid[-1] = 0
    with pytest.raises(AssertionError):
        validate_topk(invalid, scores)
