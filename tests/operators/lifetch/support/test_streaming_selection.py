"""Ensure migration selects only the retained pipeline and never falls back."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('streaming_selection', Path(__file__).resolve().parents[1] / 'test_lifetch.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_optimized_selects_measured_configuration(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda cmd: commands.append(cmd) or 0)
    assert module.main(['--transport','urma','--kvaf-server','host:1','--kvaf-root',str(tmp_path),
                        '--optimized-resolve']) == 0
    command = commands[0]
    for flag in ('--streaming','--streaming-fetch','--fused-final','--prepare-update-metadata','--optimized-resolve'):
        assert flag in command
    assert command[command.index('--q-workers')+1] == '2'


def test_ubmem_selects_streaming_backend(monkeypatch, tmp_path):
    commands = []
    def execute(command, log, timeout, label):
        commands.append(command)
        log.write_text('UBMEM_LIFETCH_PASS\nUBMEM_SFA_PASS\nUBMEM_GRAPH_EVENT_US\n')
        return 0
    monkeypatch.setattr(module, 'execute', execute)
    assert module.main(['--transport','ubmem','--kvaf-server','host:1', '--soc','Ascend950PR_9599',
                        '--kvaf-root',str(tmp_path),'--output',str(tmp_path/'out'),
                        '--trace-rounds','0','--optimized-resolve']) == 0
    assert '--optimized-resolve' in commands[0]
    assert commands[0][commands[0].index('--q-workers')+1] == '2'


@pytest.mark.parametrize('flag', ['--early-fetch','--avoid-prefetch-slots','--legacy-fetch-layout','--parallel-fetch-finalize','--streaming'])
def test_removed_experiments_fail_before_launch(flag):
    with pytest.raises(SystemExit):
        module.main(['--transport','urma','--kvaf-server','host:1',flag])
