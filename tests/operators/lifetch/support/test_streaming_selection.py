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


def test_ubmem_rejects_optimized_before_launch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('unsupported transport reached execution')
    monkeypatch.setattr(module, 'execute', unexpected)
    with pytest.raises(SystemExit):
        module.main(['--transport','ubmem','--kvaf-server','host:1','--optimized-resolve'])


@pytest.mark.parametrize('flag', ['--early-fetch','--avoid-prefetch-slots','--legacy-fetch-layout','--parallel-fetch-finalize','--streaming'])
def test_removed_experiments_fail_before_launch(flag):
    with pytest.raises(SystemExit):
        module.main(['--transport','urma','--kvaf-server','host:1',flag])
