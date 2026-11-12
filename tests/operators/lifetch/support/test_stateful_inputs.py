"""Controlled actual LI inputs and measured-state transition requirements."""
import numpy as np
import pytest

from correctness import stateful_inputs, require_stateful_round
from replay_inputs import ReplayPagePlan


def test_stateful_changes_query_only_and_has_exact_topk_without_ties(tmp_path):
    sequence = stateful_inputs(tmp_path)
    rounds = list(sequence)
    assert [row.source_round for row in rounds] == [1, 2, 3, 4]
    for inputs in rounds:
        assert np.array_equal(inputs.logical_keys, rounds[0].logical_keys)
        assert np.array_equal(inputs.captured_layout, rounds[0].captured_layout)
        assert inputs.geometry['resident'] == 4096
        # All operands are FP8 {0,1}; compute real positive dot products and
        # sum 32 identical heads, independently from the supplied score oracle.
        assert set(np.unique(inputs.query)) == {0, 0x38}
        assert np.all(inputs.query == inputs.query[:, :, :1])
        q = (inputs.query[:, :, 0] == 0x38).astype(np.float32)
        k = (inputs.logical_keys == 0x38).astype(np.float32)
        scores = np.einsum('bqd,btd->bqt', q, k) * 32
        selected = np.stack([np.stack([np.flatnonzero(query == 32) for query in row]) for row in scores])
        inputs.validate_topk(selected)
        assert all(len(np.unique(row)) == (4096 if inputs.source_round == 3 else 2048)
                   for row in selected)
    assert np.array_equal(rounds[0].query, rounds[1].query)
    assert np.array_equal(rounds[0].query, rounds[3].query)
    assert not np.array_equal(rounds[0].query, rounds[2].query)
    assert ReplayPagePlan(sequence).page_count == 2 * 12288 // 128


@pytest.mark.parametrize('round_number,reads,tokens', [
    (1, 2048, range(4096, 6144)), (2, 0, range(4096, 6144)),
    (3, 4096, range(6144, 10240)), (4, 2048, range(4096, 6144)),
])
def test_each_transition_requires_actual_reads_and_persistent_residency(round_number, reads, tokens):
    require_stateful_round(round_number, [tokens, tokens], [0, 0], [reads, reads])
    with pytest.raises(AssertionError, match='reads='):
        require_stateful_round(round_number, [tokens], [0], [reads + 1])
    with pytest.raises(AssertionError, match='persistent hot'):
        require_stateful_round(round_number, [[]], [0], [reads])


def test_eviction_cannot_be_faked_by_only_checking_new_tokens():
    with pytest.raises(AssertionError, match='evict A'):
        require_stateful_round(3, [range(4096, 10240)], [0], [4096])


def test_stateful_job_records_actual_fixed_geometry_and_budget():
    from correctness import check_jobs
    from workloads import configuration
    [(name, config)] = check_jobs(configuration(['check', '--suite', 'stateful']))
    assert name == 'stateful'
    assert (config.batches, config.history, config.resident, config.prefetch_tokens) == ((2,), 12288, 4096, 0)
    assert config.warmup == 0 and config.iterations == 1
