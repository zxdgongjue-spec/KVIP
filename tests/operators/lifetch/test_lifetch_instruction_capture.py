# SPDX-License-Identifier: Apache-2.0
"""Offline checks of the native instruction capture handoff."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/lifetch_mr23_service'))
from lifetch_instruction_capture import (  # noqa: E402
    checked_collector, collector_options, instruction_plan, replay_once,
)


def test_operator_plan_uses_complete_application_replay(tmp_path):
    plan = instruction_plan('all', 'mte1|mte2|mte3')
    options = collector_options(plan, tmp_path)
    assert plan['launch_count'] == 4
    assert '--replay-mode=application' in options and '--warm-up=0' in options
    assert '--kernel-name=v9_li_kernel|v9_parallel_terminal' in options
    assert '--instr-timeline-pipe=mte1|mte2|mte3' in options
    assert '--aic-metrics=Default,InstrTimeline' in options
    assert instruction_plan('terminal', 'vector')['kernels'] == ['v9_parallel_terminal']
    with pytest.raises(ValueError, match='distinct instruction pipes'):
        instruction_plan('all', 'mte2|mte2')


def test_old_collector_fails_without_falling_back_to_kernel_replay(tmp_path, monkeypatch):
    monkeypatch.delenv('KVIP_MSOPPROF', raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda argv, **options: subprocess.CompletedProcess(
        argv, 0, '--replay-mode mode:application|kernel --warm-up', '',
    ))
    log = tmp_path / 'cli-help.log'
    with pytest.raises(RuntimeError, match='does not advertise InstrTimeline'):
        checked_collector('/profile/msprof', log)
    assert 'application' in log.read_text()


def test_collector_keeps_help_advertised_metric_spelling(tmp_path, monkeypatch):
    monkeypatch.delenv('KVIP_MSOPPROF', raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda argv, **options: subprocess.CompletedProcess(
        argv, 0, 'instrTimeLine --instr-timeline-pipe --replay-mode mode:application|kernel --warm-up', '',
    ))
    assert checked_collector('/profile/msprof', tmp_path / 'help.log') == (
        ['/profile/msprof', 'op'], 'instrTimeLine',
    )


def test_each_application_pass_uses_new_worker_directory(tmp_path, monkeypatch):
    template = tmp_path / 'config.json'
    template.write_text('{"msprof_instruction_replay": true}')
    paths = []

    def run(config):
        paths.append(config)
        assert config.read_bytes() == template.read_bytes()
        return 0

    monkeypatch.setitem(sys.modules, 'execution', SimpleNamespace(run_config=run))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert replay_once(template) == 0 and replay_once(template) == 0
    assert len(set(paths)) == 2
    status = json.loads((tmp_path / 'worker-result.json').read_text())
    assert status == {'exit_code': 0, 'attempt': str(paths[-1].parent)}
