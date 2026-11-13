"""Named check cases must prove their input and branch conditions actually ran."""
import numpy as np
import pytest

from correctness import boundary_evidence, require_boundary


def evidence(*, selected=(0, 1), valid=(1, 1, 1, 1), posted=0, final=(0, 0, 0, 0)):
    ids = np.tile(selected, (1, 4, 1))
    return boundary_evidence(ids, [valid], [range(4)], np.array([posted]), [final])


def test_hit_case_cannot_pass_with_reads_or_cold_inputs():
    require_boundary('all_hit_budget0', {'regular': evidence()})
    for rows in (evidence(posted=1), evidence(selected=(0, 5)), evidence(final=(1, 0, 0, 0))):
        with pytest.raises(AssertionError, match='zero reads'):
            require_boundary('all_hit_budget0', {'regular': rows})


def test_all_miss_distinguishes_full_unrelated_hot_from_cold_start():
    full = evidence(selected=(4, 5), final=(2, 0, 0, 0))
    require_boundary('all_miss', {'regular': full})
    cold = evidence(selected=(4, 5), valid=(0, 0, 0, 0), final=(2, 0, 0, 0))
    with pytest.raises(AssertionError, match='full unrelated'):
        require_boundary('all_miss', {'regular': cold})
    assert cold[0]['initial_cold'] == full[0]['initial_cold'] == 2


def test_zero_budget_still_requires_actual_final_repair():
    require_boundary('prefetch_zero', {'regular': evidence(selected=(0, 4), final=(1, 0, 0, 0))})
    with pytest.raises(AssertionError, match='final repair'):
        require_boundary('prefetch_zero', {'regular': evidence(selected=(0, 4))})


def test_cross_query_empty_repair_and_cold_start_have_separate_evidence():
    regular = evidence(selected=(0, 4), final=(1, 0, 0, 0))
    cold = evidence(selected=(0, 4), valid=(0, 0, 0, 0), final=(2, 0, 0, 0))
    assert regular[0]['cross_query_duplicates'] == [0, 2, 2, 2]
    extra = {'transfer': [dict(fallback_slots=1, conflicts=1)], 'padding': 'PASS'}
    require_boundary('boundary_regular', {'regular': regular, 'cold-start': cold, **extra})
    with pytest.raises(AssertionError, match='clear validity'):
        require_boundary('boundary_regular', {'regular': regular, 'cold-start': regular, **extra})
    with pytest.raises(AssertionError, match='no new repair'):
        require_boundary('boundary_regular', {'regular': evidence(final=(0, 0, 1, 0)), 'cold-start': cold, **extra})


def test_invalid_counter_data_is_not_a_correctness_assertion():
    with pytest.raises(ValueError, match='invalid counters'):
        evidence(posted=-1)


def test_unknown_boundary_and_empty_evidence_cannot_pass():
    import pytest
    from correctness import require_boundary
    with pytest.raises(ValueError, match='Unknown boundary'):
        require_boundary('typo', {'regular': []})
    with pytest.raises(AssertionError, match='no observed'):
        require_boundary('all_hit_budget0', {'regular': []})


def test_extended_contracts_use_production_validation():
    from correctness import check_contracts
    for transport in ('urma', 'ubmem'):
        checks = check_contracts(transport)
        assert all(row['correctness'] == 'PASS' for row in checks), checks
        names = {row['case'] for row in checks}
        assert {'reject_unaligned_history', 'reject_unaligned_resident',
                'reject_no_cold_history', 'reject_window_budget481'} <= names
        if transport == 'ubmem':
            assert {'reject_heads', 'reject_token_bytes', 'reject_soc'} <= names


def test_transfer_coverage_requires_real_outside_candidate_slots_and_conflicts():
    from correctness import transfer_coverage, require_transfer_coverage
    posts = np.zeros((1, 4, 6), dtype=np.int32)
    posts[0, 0] = [3, 0, 0, 0, 0, 2]
    result = transfer_coverage(posts, [0, 3, 1], [[0, 1]], 8, 4)
    assert result == [dict(request=0, final=3, conflicts=1, fallback_slots=1)]
    require_transfer_coverage(result)
    for key in ('conflicts', 'fallback_slots'):
        with pytest.raises(AssertionError):
            require_transfer_coverage([dict(result[0], **{key: 0})])
    posts[0, 0, 4] = 9
    with pytest.raises(RuntimeError, match='range'):
        transfer_coverage(posts, [0, 3, 1], [[0, 1]], 8, 4)
