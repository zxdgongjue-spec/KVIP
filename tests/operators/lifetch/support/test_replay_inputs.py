import json

import numpy as np
import pytest

from workloads import assemble_round
from test_trace_inputs import capture_file
from workloads import CaptureReference


def reference(tmp_path, rank=0, number=1):
    directory = tmp_path / f'dp{rank}-round{number}'
    directory.mkdir()
    path = capture_file(directory)
    manifest = json.loads(path.read_text())
    manifest['geometry'].update(token_bytes=656, output_stride=2176)
    path.write_text(json.dumps(manifest))
    return CaptureReference(rank, number, 100 + number, 10, path, (('request', 0),))


def test_merge_preserves_dp_identity_and_query_bytes(tmp_path):
    first, second = reference(tmp_path), reference(tmp_path, rank=1)
    round_input = assemble_round([first, second])
    assert round_input.geometry['batch'] == 2
    assert round_input.identities == ((0, 'request', 0), (1, 'request', 0))
    assert round_input.page_keys[0][0] == (0, 0)
    assert round_input.page_keys[1][0] == (1, 0)
    assert round_input.weights.dtype == np.uint16  # Preserve BF16 bits.
    assert round_input.query.shape == (2, 4, 2, 128)


def test_current_path_layout_drives_index_not_captured_hot_layout(tmp_path):
    round_input = assemble_round([reference(tmp_path)])
    current = np.arange(4096, dtype=np.int64)[None]
    current[:, [0, 3000]] = current[:, [3000, 0]]
    keys, scales, valid = round_input.index_for_layout(current)
    assert scales[0, 0] == 1095 and scales[0, 3000] == 4095
    assert keys[0, 0, 0] == 1095 % 256
    assert valid.all()
    # Preparing another round must not mutate captured evidence or supplied state.
    assert round_input.captured_layout[0, 0] == 4095
    assert current[0, 0] == 3000
    current[0, 0] = 0
    with pytest.raises(ValueError, match='permutation'):
        round_input.index_for_layout(current)


def test_replay_rejects_unaligned_or_duplicate_dp_inputs(tmp_path):
    first = reference(tmp_path)
    second = reference(tmp_path, rank=1, number=2)
    with pytest.raises(ValueError, match='aligned'):
        assemble_round([first, second])
    with pytest.raises(ValueError, match='Duplicate DP'):
        assemble_round([first, first])


@pytest.mark.parametrize('mutation, message', [
    ('identity', 'Request identity'), ('bounds', 'bounds regressed'),
    ('pages', 'DDR page identity'),
])
def test_sequence_rejects_invalid_continuity(tmp_path, monkeypatch, mutation, message):
    import workloads
    first = reference(tmp_path)
    second = reference(tmp_path, number=2)
    initial = assemble_round([first])
    following = assemble_round([second])
    if mutation == 'identity':
        following.identities = ((0, 'another-request', 0),)
    elif mutation == 'bounds':
        following.query_ends[0, 0] -= 1
    else:
        following.page_keys = (((0, 999), *following.page_keys[0][1:]),)
    monkeypatch.setattr(workloads, 'select_series', lambda *args: [first, second])
    monkeypatch.setattr(workloads, 'assemble_round',
                        lambda refs: initial if refs[0].source_round == 1 else following)
    sequence = workloads.ReplaySequence(tmp_path, 10, (1, 2))
    iterator = iter(sequence)
    assert next(iterator) is initial
    with pytest.raises(ValueError, match=message):
        next(iterator)


def test_sequence_iteration_does_not_carry_test_state_between_repetitions(tmp_path, monkeypatch):
    import workloads
    references = [reference(tmp_path), reference(tmp_path, number=2)]
    monkeypatch.setattr(workloads, 'select_series', lambda *args: references)
    sequence = workloads.ReplaySequence(tmp_path, 10, (1, 2))
    first_pass = list(sequence)
    first_pass[0].captured_hot_valid[:] = 0
    first_pass[0].logical_keys[:] = 0
    second_pass = list(sequence)
    assert second_pass[0].captured_hot_valid.all()
    assert np.any(second_pass[0].logical_keys)
    assert [entry.source_round for entry in second_pass] == [1, 2]


def test_page_payload_preserves_aliases_and_later_abc_publication(tmp_path):
    from workloads import ReplayPagePlan
    first = assemble_round([reference(tmp_path), reference(tmp_path, rank=1)])
    # Duplicate logical pages within DP0 model a shared captured page alias.
    first.page_keys = (((0, 0), (0, 0), *first.page_keys[0][2:]), first.page_keys[1])
    later = assemble_round([reference(tmp_path, number=2), reference(tmp_path, rank=1, number=2)])
    later.page_keys = first.page_keys
    first.page_keys = ((None, *first.page_keys[0][1:]), first.page_keys[1])
    plan = ReplayPagePlan([first, later])
    assert plan.page_indices[0, 0] == plan.page_indices[0, 1]
    assert plan.page_indices[0, 0] != plan.page_indices[1, 0]
    np.testing.assert_array_equal(plan.token_payload(0, [0]), plan.token_payload(0, [128]))
    np.testing.assert_array_equal(plan.token_payload(0, np.arange(128)),
                                  plan.page_payload([plan.page_indices[0, 0]]))
    assert not np.array_equal(plan.token_payload(0, [0]), plan.token_payload(1, [0]))


def test_topk_oracle_accepts_cutoff_ties_but_not_bad_rank_or_duplicates(tmp_path):
    inputs = assemble_round([reference(tmp_path)])
    selected = np.broadcast_to(np.arange(2048, dtype=np.int32), (1, 4, 2048)).copy()
    inputs.validate_topk(selected)
    # All original scores tie; selecting a different set is valid.
    inputs.validate_topk(selected + 2048)
    inputs.logical_score_keys[0, 0, 3000] += 1
    with pytest.raises(AssertionError, match='ranking differs'):
        inputs.validate_topk(selected)
    selected[0, 0, 0] = 3000
    inputs.validate_topk(selected)
    selected[0, 0, 1] = 3000
    with pytest.raises(AssertionError, match='duplicates'):
        inputs.validate_topk(selected)
