# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from main_five_run import FeatureReplayChecks, parse_args


@pytest.mark.parametrize(('mode', 'expected'), [
    ([], 20),
    (['--history-cap', '131328', '--production-graph'], 20),
    (['--ablation-policy', 'single'], 10),
    (['--ablation-policy', 'paced'], 10),
])
def test_parent_and_device_worker_agree_on_replay_count(mode, expected):
    args = ['--main', '.', '--trace', '.', '--owner-archive', '.', '--kvaf-root', '.',
            '--output', '.', '--kvaf-server', '127.0.0.1:1', '--group', '7', *mode]
    assert parse_args(args).iterations == expected
    assert parse_args([*args, '--worker']).iterations == expected


@pytest.mark.parametrize('policy', ['single', 'paced'])
def test_feature_checks_follow_single_round_reset_boundaries(monkeypatch, policy):
    import terminal_batch_checks

    audit = Mock()
    monkeypatch.setattr(terminal_batch_checks, 'audit_queues', audit)
    state = SimpleNamespace(check_completion=Mock(), rotating=object(),
                            forward=SimpleNamespace(terminal=object()))
    full = Mock()
    sample = object()
    checks = FeatureReplayChecks(10, 4, policy)
    checks.validate('eager', state, sample, full)
    for i in range(10):
        checks.validate(f'pre_capture/input{i}', state, sample, full)
    for stage in ('warmup0', 'warmup1', 'warmup2', 'warmup3', 'measured', 'detail'):
        for i in range(10):
            checks.validate(f'{stage}/input{i}/round1', state, sample, full)
    receipt = checks.receipt()
    assert receipt['completion_and_private_CQ_checks'] == state.check_completion.call_count == audit.call_count == 50
    audit.assert_called_with(state.forward.terminal, state.rotating, policy=policy)
    assert receipt['full_state_checks'] == full.call_count == 21
    full.assert_called_with(sample)
    with pytest.raises(ValueError, match='Duplicate'):
        checks.validate('eager', state, sample, full)
    with pytest.raises(ValueError, match='Unexpected'):
        checks.validate('warmup0/input0/round2', state, sample, full)


def test_warmup_completion_error_invalidates_feature_check(monkeypatch):
    import terminal_batch_checks

    audit = Mock(side_effect=ValueError('bad CQ'))
    monkeypatch.setattr(terminal_batch_checks, 'audit_queues', audit)
    state = SimpleNamespace(check_completion=Mock(), rotating=object(),
                            forward=SimpleNamespace(terminal=object()))
    checks = FeatureReplayChecks(1, 1, 'single')
    with pytest.raises(ValueError, match='bad CQ'):
        checks.validate('warmup0/input0/round1', state, None, Mock())
    with pytest.raises(ValueError, match='Incomplete'):
        checks.receipt()
