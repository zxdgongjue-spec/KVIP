#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture one checked msprof replay of the current main P2 LIFetch route."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_mr25_handoff import KVAF, MAIN, ROOT, git, save


def has_task_events(path):
    document = json.loads(path.read_text())
    events = document.get('traceEvents', []) if isinstance(document, dict) else document
    return isinstance(events, list) and any(
        isinstance(event, dict) and event.get('ph') == 'X'
        and isinstance(event.get('dur'), (int, float)) and event['dur'] > 0
        for event in events
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument('--miss-percent', type=int, choices=range(101), metavar='0..100', default=10)
    parser.add_argument('--warmup', type=int, default=0, help='optional replays with the stage buffer disabled')
    parser.add_argument('--timeout-seconds', type=int, default=900, help='limit for capture, including first build')
    parser.add_argument('--dry-run', action='store_true', help='print the capture plan without CANN or KVAF')
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative and timeout must be positive')
    return args


def bench_args(args, kvaf):
    return [
        'bench', '--transport', 'urma', '--input', 'synthetic',
        '--kvaf-root', str(kvaf),
        '--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986'),
        '--device', os.environ.get('KVIP_A5_DEVICE', '0'),
        '--batches', str(args.batch), '--miss-percent', str(args.miss_percent),
        '--history', '131072', '--resident', '16384', '--queries', '4', '--index-heads', '32',
        '--intermediate-fetches', '4', '--prefetch-tokens', '512', '--prefetch-rank-start', '1024',
        '--seed', '43', '--warmup', str(args.warmup), '--iterations', '1',
        '--timeout-seconds', str(args.timeout_seconds),
    ]


def msprof_timelines(profiler, directory, log):
    """Keep the vendor's raw export and its capture clock unmodified."""
    profiles = sorted(path for path in directory.glob('PROF_*') if path.is_dir())
    with log.open('w') as stream:
        for profile in profiles:
            if not list(profile.rglob('msprof_*.json')):
                subprocess.run(
                    [profiler, '--export=on', '--output=' + str(profile)],
                    check=True, timeout=300, stdout=stream, stderr=subprocess.STDOUT,
                )
    timelines = sorted(path for profile in profiles for path in profile.rglob('msprof_*.json'))
    return [path for path in timelines if has_task_events(path)]


def capture(args, output, kvaf, arguments, manifest):
    # The CLI owns the actual workload process. Its log must contain the
    # existing checked-completion marker for the trace to be accepted.
    from execution import supervise, worker_environment
    from workloads import configuration

    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof'))
    if profiler is None:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    config = configuration(arguments)
    config.msprof_baseline = True
    worker = output / 'worker'
    raw = output / 'msprof'
    worker.mkdir()
    raw.mkdir()
    job_file = worker / 'config.json'
    save(job_file, {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()})
    command = [
        profiler, '--output=' + str(raw), '--task-time=l1', '--runtime-api=on',
        sys.executable, '-B', '-u', '-X', 'faulthandler',
        str(ROOT / 'tests/operators/lifetch/support/execution.py'), str(job_file),
    ]
    manifest['command'] = command
    save(output / 'manifest.json', manifest)
    manifest['worker_exit_code'] = supervise(
        command, worker, timeout_seconds=config.timeout_seconds, environment=worker_environment(kvaf),
        show_case_details=False,
    )
    case = worker / f'miss{args.miss_percent}' / f'case-b{args.batch}'
    stage_trace = case / 'baseline_stage_trace.json'
    stage_report = case / 'baseline_stage_report.json'
    manifest.update(stage_trace=str(stage_trace), stage_report=str(stage_report))
    if manifest['worker_exit_code'] or not stage_trace.is_file() or not stage_report.is_file():
        raise RuntimeError(f'Checked stage replay incomplete. See {worker / "summary.log"}')
    print(f'Stage timeline: {stage_trace}', flush=True)
    print(f'Stage summary: {stage_report}', flush=True)
    timelines = msprof_timelines(profiler, raw, output / 'msprof-export.log')
    if not timelines:
        raise RuntimeError(f'No vendor timeline was exported. See {raw} and {output / "msprof-export.log"}')
    manifest['msprof_timelines'] = [str(path) for path in timelines]
    for path in timelines:
        print(f'msprof timeline: {path}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    arguments = bench_args(args, kvaf)
    manifest = dict(
        status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
        kvaf_commit=KVAF, route='main_p2_parallel_terminal', local_counts=False,
        arguments=arguments, requested_stage_replays=1,
        clocks='independent: msprof device task clock and calibrated source counter',
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    output = Path(os.environ['KVIP_AB_OUTPUT']).resolve()
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'tests/operators/lifetch/support')]
    manifest['status'] = 'RUNNING'
    try:
        capture(args, output, kvaf, arguments, manifest)
        manifest['status'] = 'PASS'
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(status='FAILED', error=str(error))
        print(f'Capture incomplete: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
