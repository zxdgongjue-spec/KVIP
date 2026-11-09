# SPDX-License-Identifier: Apache-2.0
"""Host checks for the public transport-selecting test entry point."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'test_lifetch.py'
spec = importlib.util.spec_from_file_location('unified_lifetch_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_urma_passes_geometry_to_existing_graph_runner(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                        '--batches', '8,16', '--intermediate-fetches', '8', '--miss-percent', '30']) == 0
    cmd = commands[0]
    assert cmd[cmd.index('--batches') + 1] == '8,16'
    assert cmd[cmd.index('--miss-percent') + 1] == '30.0'
    assert cmd[cmd.index('--kvaf-root') + 1] == str(tmp_path)


def test_ubmem_requires_correctness_and_timing_markers(monkeypatch, tmp_path):
    commands = []
    def fake(command, log, timeout, label):
        commands.append(command)
        log.write_text('UBMEM_LIFETCH_PASS\nUBMEM_SFA_PASS\nUBMEM_GRAPH_EVENT_US p50=1\n')
        return 0
    monkeypatch.setattr(module, 'execute', fake)
    args = ['--transport', 'ubmem', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
            '--soc', 'Ascend950PR_9599', '--batches', '16', '--output', str(tmp_path / 'out')]
    assert module.main(args) == 0
    assert commands[0][commands[0].index('--batches') + 1] == '16'
    assert 'PASS: transport=ubmem' in next((tmp_path / 'out').glob('*/summary.log')).read_text()
    with pytest.raises(SystemExit):
        module.main(args + ['--intermediate-fetches', '4'])
