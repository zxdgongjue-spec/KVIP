"""Ensure migration selects only the retained pipeline and never falls back."""
import importlib.util
import os
import subprocess
import sys
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


@pytest.mark.parametrize('flag', ['--early-fetch','--avoid-prefetch-slots','--legacy-fetch-layout','--parallel-fetch-finalize'])
def test_removed_experiments_fail_before_launch(flag):
    with pytest.raises(SystemExit):
        module.main(['--transport','urma','--kvaf-server','host:1',flag])


def test_explicit_kvaf_root_overrides_inherited_module(monkeypatch, tmp_path):
    selected = tmp_path / 'selected'
    wrong = tmp_path / 'wrong'
    for folder, marker in ((selected / 'build-npu/python', 'selected'), (wrong, 'wrong')):
        folder.mkdir(parents=True)
        (folder / 'kvaf.py').write_text(f"MARKER = {marker!r}\n")
    monkeypatch.setenv('PYTHONPATH', str(wrong))
    def execute(command, log, timeout, label):
        result = subprocess.check_output([sys.executable, '-c', 'import kvaf; print(kvaf.MARKER)'], text=True)
        assert result.strip() == 'selected'
        log.write_text('UBMEM_LIFETCH_PASS\nUBMEM_SFA_PASS\nUBMEM_GRAPH_EVENT_US\n')
        return 0
    monkeypatch.setattr(module, 'execute', execute)
    assert module.main(['--transport', 'ubmem', '--kvaf-server', 'host:1',
                        '--soc', 'Ascend950PR_9599', '--kvaf-root', str(selected),
                        '--output', str(tmp_path/'output'), '--trace-rounds', '0']) == 0


def test_ubmem_script_remembers_explicit_settings(tmp_path):
    import json
    script = (Path(__file__).resolve().parents[1] / 'compare_ubmem_fetch.sh').read_text()
    code = script.split("<<'PYSET'\n", 1)[1].split('\nPYSET', 1)[0]
    settings = tmp_path / 'settings.json'
    base = [sys.executable, '-c', code, str(settings), str(tmp_path)]
    expected = [str(tmp_path/'kvaf'), '4', '127.0.0.1:28905', '1,2,4,8,16']
    first = subprocess.check_output(base+expected, text=True).splitlines()
    again = subprocess.check_output(base, text=True, env={k:v for k,v in os.environ.items() if not k.startswith('UBMEM_')}).splitlines()
    assert first == again == expected
    assert json.loads(settings.read_text())['server'] == expected[2]
