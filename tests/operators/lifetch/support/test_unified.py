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
    assert '--internal-urma' in cmd and str(SOURCE) in cmd
    assert cmd[cmd.index('--batches') + 1] == '8,16'
    assert cmd[cmd.index('--miss-percent') + 1] == '30.0'
    assert cmd[cmd.index('--kvaf-root') + 1] == str(tmp_path)


def test_urma_fixture_can_import_native_log():
    assert callable(module.native_log)


def test_ubmem_requires_correctness_and_timing_markers(monkeypatch, tmp_path):
    commands = []
    def fake(command, log, timeout, label):
        commands.append(command)
        log.write_text('UBMEM_LIFETCH_PASS\nUBMEM_SFA_PASS\nUBMEM_GRAPH_EVENT_US p50=1\n'
                       + ''.join(f'UBMEM_TRACE round={i} resolve_us=10 fetch_us=20\n' for i in range(3)))
        return 0
    monkeypatch.setattr(module, 'execute', fake)
    args = ['--transport', 'ubmem', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
            '--soc', 'Ascend950PR_9599', '--batches', '16', '--output', str(tmp_path / 'out')]
    assert module.main(args) == 0
    assert commands[0][commands[0].index('--batches') + 1] == '16'
    assert 'PASS: transport=ubmem' in next((tmp_path / 'out').glob('*/summary.log')).read_text()
    with pytest.raises(SystemExit):
        module.main(args + ['--intermediate-fetches', '4'])


def test_urma_timeout_is_forwarded(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                        '--timeout-seconds', '90']) == 0
    command = commands[0]
    assert command[command.index('--timeout-seconds') + 1] == '90'


def test_supervisor_reports_timeout(tmp_path):
    import sys
    from runner import supervise

    assert supervise([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path,
                     heartbeat_seconds=0.1, timeout_seconds=0.3) == 1
    assert 'timeout=0.3s' in (tmp_path / 'summary.log').read_text()


def test_early_fetch_selects_urma_only(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                        '--early-fetch']) == 0
    assert '--early-fetch' in commands[0]
    with pytest.raises(SystemExit):
        module.main(['--transport', 'ubmem', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                     '--soc', 'Ascend950PR_9599', '--early-fetch'])


def test_baseline_unoccupied_selection_forwarded(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                        '--avoid-prefetch-slots']) == 0
    assert '--avoid-prefetch-slots' in commands[0]
    assert '--early-fetch' not in commands[0]


def test_final_workers_are_forwarded(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
                        '--avoid-prefetch-slots', '--batches', '1,2,4,8,16', '--final-workers', '2']) == 0
    assert commands[0][commands[0].index('--final-workers') + 1] == '2'
    assert '--avoid-prefetch-slots' in commands[0]
    with pytest.raises(SystemExit):
        module.main(['--transport', 'urma', '--kvaf-server', 'host:1', '--final-workers', '3'])


def test_fetch_only_requires_matching_two_worker_fused_path(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    common = ['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
              '--streaming', '--streaming-fetch', '--fused-final', '--q-workers', '2', '--fetch-only-compare']
    assert module.main(common) == 0
    assert '--fetch-only-compare' in commands[0]
    with pytest.raises(SystemExit):
        module.main(common[:-2] + ['4', '--fetch-only-compare'])


def test_optimized_resolve_requires_full_streaming_metadata_path(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(module.subprocess, 'call', lambda command: commands.append(command) or 0)
    base = ['--transport', 'urma', '--kvaf-server', 'host:1', '--kvaf-root', str(tmp_path),
            '--optimized-resolve']
    with pytest.raises(SystemExit):
        module.main(base)
    flags = ['--streaming', '--streaming-fetch', '--fused-final', '--q-workers', '2',
             '--prepare-update-metadata']
    assert module.main(base + flags) == 0
    assert '--optimized-resolve' in commands[0]
    internal = module.arguments(commands[0][commands[0].index('--internal-urma') + 1:])
    assert internal.optimized_resolve and internal.prepare_update_metadata
