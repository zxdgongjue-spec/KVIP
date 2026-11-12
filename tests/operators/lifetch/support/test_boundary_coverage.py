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
    require_boundary('boundary_regular', {'regular': regular, 'cold-start': cold})
    with pytest.raises(AssertionError, match='clear validity'):
        require_boundary('boundary_regular', {'regular': regular, 'cold-start': regular})
    with pytest.raises(AssertionError, match='no new repair'):
        require_boundary('boundary_regular', {'regular': evidence(final=(0, 0, 1, 0)), 'cold-start': cold})


def test_invalid_counter_data_is_not_a_correctness_assertion():
    with pytest.raises(ValueError, match='invalid counters'):
        evidence(posted=-1)
