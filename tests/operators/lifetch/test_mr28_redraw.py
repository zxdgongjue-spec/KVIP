# SPDX-License-Identifier: Apache-2.0
"""Verify that the container entry redraws complete saved traces offline."""

import json
import os
from pathlib import Path
import subprocess
import sys

from test_baseline_stage_trace import sample_buffer
from test_msprof_timeline import PHASES, PHASES_COLD_P4, fused_replay


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'experiments/lifetch_mr23_service/run_mr28_baseline_msprof.sh'


def write_arm(arm, route, *, batch=16, legacy=False):
    case = arm / 'worker' / 'miss10' / f'case-b{batch}'
    case.mkdir(parents=True)
    report = dict(correctness='PASS', execution='COMPLETED', counter_mhz=1000)
    if not legacy:
        report['cold_route'] = route
    (case / 'baseline_stage_report.json').write_text(json.dumps(report))
    parts = 4 if route == 'cold_p4' else 2
    (case / 'baseline_stage_trace.bin').write_bytes(sample_buffer(batch=batch, cold_parts=parts))
    native_path = arm / 'vendor.json'
    native_path.write_text(json.dumps(fused_replay(PHASES_COLD_P4 if route == 'cold_p4' else PHASES)))
    manifest = dict(status='PASS', worker_exit_code=0, msprof_timeline=str(native_path),
                    route='main_p2_parallel_terminal' if legacy else route)
    if not legacy:
        manifest['business_status'] = 'PASS'
    return manifest


def write_manifest(run, manifest, order):
    path = run / 'manifest.json'
    path.write_text(json.dumps(manifest))
    os.utime(path, (order, order))


def run_redraw(a5_root, *args):
    env = dict(os.environ, KVIP_A5_ROOT=str(a5_root), KVIP_PYTHON=sys.executable)
    env.pop('KVIP_AB_OUTPUT', None)
    return subprocess.run(['bash', str(RUNNER), '--redraw-latest', *args],
                          env=env, check=True, text=True, capture_output=True)


def test_redraw_latest_skips_newer_failed_and_incomplete_runs(tmp_path):
    root = tmp_path / 'results'
    old = root / 'mr28.old'
    completed = root / 'mr28.complete'
    incomplete = root / 'mr28.incomplete'
    failed = root / 'mr28.failed'
    for order, run in enumerate((old, completed, incomplete, failed), 1):
        manifest = write_arm(run, 'p2', legacy=run == completed)
        if run == failed:
            manifest['status'] = 'FAILED'
        if run == incomplete:
            (run / 'worker/miss10/case-b16/baseline_stage_trace.bin').unlink()
        write_manifest(run, manifest, order)

    original = (completed / 'manifest.json').read_bytes()
    result = run_redraw(tmp_path)
    assert f'Capture: {completed}' in result.stdout
    assert f'Skipping {incomplete.name}' in result.stderr
    output, = completed.glob('handoff-redraw.*')
    source = json.loads((output / 'handoff_source.json').read_text())
    device = json.loads((output / 'handoff_device.json').read_text())
    timeline = json.loads((output / 'lifetch_msprof_activity.json').read_text())
    assert source['cold_route'] == device['cold_route'] == 'p2'
    assert len(source['source_business_handoffs']) == len(device['native_device_handoffs']) == 3
    assert any(e.get('cat') == 'KVIP device handoff activity' for e in timeline['traceEvents'])
    assert not any('Physic Stream Id' in e.get('args', {}) for e in timeline['traceEvents'])
    assert (completed / 'manifest.json').read_bytes() == original


def test_redraws_comparison_arms_from_one_run(tmp_path):
    run = tmp_path / 'results/mr28.complete'
    routes = ('p2', 'cold_p4')
    children = {route: write_arm(run / route, route, batch=8) for route in routes}
    write_manifest(run, dict(status='PASS', planned_routes=routes, routes=children), 1)

    run_redraw(tmp_path)
    output, = run.glob('handoff-redraw.*')
    for route in routes:
        report = json.loads((output / route / 'handoff_source.json').read_text())
        assert report['cold_route'] == route
        assert (output / route / 'lifetch_business_activity.json').is_file()

    selected = run_redraw(tmp_path, '--route', 'cold_p4')
    assert 'cold_p4 reports:' in selected.stdout
    assert 'p2 reports:' not in selected.stdout
