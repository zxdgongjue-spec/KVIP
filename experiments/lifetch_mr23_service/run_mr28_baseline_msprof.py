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

from lifetch_msprof_timeline import publish_business_views, verified_vendor_timeline
from run_mr25_handoff import KVAF, MAIN, ROOT, git, read, save


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
    """Export each scoped pyACL collection with the CANN tools."""
    profiles = sorted(path for path in directory.glob('PROF_*') if path.is_dir())
    with log.open('w') as stream:
        for profile in profiles:
            if not list(profile.rglob('msprof_*.json')):
                subprocess.run(
                    [profiler, '--export=on', '--output=' + str(profile)],
                    check=True, timeout=300, stdout=stream, stderr=subprocess.STDOUT,
                )
    return sorted(path for profile in profiles for path in profile.rglob('msprof_*.json'))


def capture(args, output, kvaf, arguments, manifest):
    from execution import launch
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
    raw.mkdir()
    config.msprof_output = str(raw)
    manifest['msprof_output'] = str(raw)
    save(output / 'manifest.json', manifest)
    manifest['worker_exit_code'] = launch(config, worker)
    case = worker / f'miss{args.miss_percent}' / f'case-b{args.batch}'
    stage_trace = case / 'baseline_stage_trace.json'
    stage_report = case / 'baseline_stage_report.json'
    manifest.update(stage_trace=str(stage_trace), stage_report=str(stage_report))
    if manifest['worker_exit_code'] or not stage_trace.is_file() or not stage_report.is_file():
        raise RuntimeError(f'Checked stage replay incomplete. See {worker / "summary.log"}')
    print(f'Stage timeline: {stage_trace}', flush=True)
    print(f'Stage summary: {stage_report}', flush=True)
    report = read(stage_report)
    if report.get('correctness') != 'PASS' or report.get('execution') != 'COMPLETED':
        raise ValueError('Source trace lacks checked fusion output')
    capture_status = report.get('msprof_capture', {}).get('status', {})
    if capture_status != dict(init=0, start=0, stop=0, destroy_config=0, finalize=0):
        raise ValueError(f'Scoped CANN profiling did not complete: {capture_status}')
    manifest['msprof_capture'] = report['msprof_capture']
    timelines = msprof_timelines(profiler, raw, output / 'msprof-export.log')
    if not timelines:
        raise RuntimeError(f'No vendor timeline was exported. See {raw} and {output / "msprof-export.log"}')
    manifest['msprof_timelines'] = [str(path) for path in timelines]
    native_path, vendor, intervals = verified_vendor_timeline(timelines)
    business, device = publish_business_views(output, vendor, read(stage_trace), intervals, save)
    manifest.update(msprof_timeline=str(native_path), business_trace=str(business),
                    msprof_business_trace=str(device), kernel_intervals=intervals)
    print(f'Open fused stages: {business}', flush=True)
    print(f'Open msprof with fusion phase labels: {device}', flush=True)
    for row in intervals:
        print(f'  {row["name"]}: {row["duration_us"]:.2f} us', flush=True)
    print(f'Early prefetch / final Get: {report["total_prefetch"]} / {report["total_final_get"]}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    arguments = bench_args(args, kvaf)
    manifest = dict(
        status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
        kvaf_commit=KVAF, route='main_p2_parallel_terminal', local_counts=False,
        arguments=arguments, requested_stage_replays=1,
        collection='pyACL_prof_start_stop_around_fused_forward_only',
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
