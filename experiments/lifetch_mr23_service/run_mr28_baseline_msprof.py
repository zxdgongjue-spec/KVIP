#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture checked msprof replays of the fused LIFetch routes."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lifetch_cold_trace_comparison import REFERENCE, ROUTES as COLD_ROUTES, write_comparison
from lifetch_instruction_capture import STAGES, collect_instructions, instruction_plan
from lifetch_msprof_timeline import READY_SERVICE_ROUTES, publish_business_views, verified_vendor_timeline
from lifetch_ready_trace_comparison import write_ready_comparison
from lifetch_unified_trace_comparison import ROUTES as UNIFIED_ROUTES, write_unified_comparison
from run_mr25_handoff import KVAF, MAIN, ROOT, git, read, save


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument('--miss-percent', type=int, choices=range(101), metavar='0..100', default=10)
    parser.add_argument('--warmup', type=int, help='replays with the stage buffer disabled')
    parser.add_argument('--timeout-seconds', type=int, default=900, help='limit for capture, including first build')
    parser.add_argument('--instructions', action='store_true', help='also collect on-board msOpProf artifacts')
    parser.add_argument('--instruction-stage', choices=('all', *STAGES), default='all')
    parser.add_argument('--instruction-pipes', default='mte1|mte2|mte3',
                        help='pipe names separated by |, used with --instructions')
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--route', choices=('p2', 'dedicated', *READY_SERVICE_ROUTES), default='p2',
                            help='capture one stock or experimental fused forward')
    comparison.add_argument('--dynamic-compare', action='store_true',
                            help='compare fixed and dynamic Service on one BS16/miss10 case')
    comparison.add_argument('--cold-compare', action='store_true', help='compare native P2 and Cold P4 at BS8/miss10')
    comparison.add_argument('--unified-compare', action='store_true',
                            help='compare native P2, full Score P2 and full Score P4 at BS8/miss10')
    parser.add_argument('--local-counts', choices=('on', 'off'),
                        help='common comparison setting, defaults to on for Cold P4 and off for unified Score')
    parser.add_argument('--dry-run', action='store_true', help='print the capture plan without CANN or KVAF')
    args = parser.parse_args(argv)
    args.comparison_routes = (COLD_ROUTES if args.cold_compare else
                              UNIFIED_ROUTES if args.unified_compare else
                              READY_SERVICE_ROUTES if args.dynamic_compare else ())
    if args.batch is None:
        args.batch = 8 if args.cold_compare or args.unified_compare else 16
    if args.warmup is None:
        args.warmup = 2 if args.comparison_routes else 0
    if (args.cold_compare or args.unified_compare) and (args.batch != 8 or args.miss_percent != 10 or args.instructions):
        parser.error('Score comparison uses BS8, miss 10% and the scoped kernel/source trace')
    ready_service = args.dynamic_compare or args.route in READY_SERVICE_ROUTES
    if ready_service and (args.batch != 16 or args.miss_percent != 10 or args.instructions):
        parser.error('Dynamic Service uses BS16, miss 10% and the scoped kernel/source trace')
    if ready_service and args.local_counts == 'on':
        parser.error('Dynamic Service uses local counts OFF on both routes')
    if args.route == 'dedicated' and (args.batch != 16 or args.instructions):
        parser.error('Dedicated Service requires BS16 and the scoped kernel/source trace')
    if not args.comparison_routes and args.local_counts is not None:
        parser.error('--local-counts applies to a Score comparison')
    args.local_counts = args.local_counts == 'on' if args.local_counts is not None else args.cold_compare
    if args.warmup < 0 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative and timeout must be positive')
    try:
        args.instruction_plan = instruction_plan(args.instruction_stage, args.instruction_pipes) if args.instructions else None
    except ValueError as error:
        parser.error(str(error))
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


def capture_route(args, output, config, manifest, profiler, *, cold_route='p2'):
    from execution import launch

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
    native_path, vendor, intervals = verified_vendor_timeline(timelines, cold_route=cold_route)
    source = read(stage_trace)
    business, device = publish_business_views(output, vendor, source, intervals, save, cold_route=cold_route)
    manifest.update(msprof_timeline=str(native_path), business_trace=str(business),
                    msprof_business_trace=str(device), kernel_intervals=intervals,
                    handoff_report=str(output / 'lifetch_handoff_report.json'))
    print(f'Open fused stages: {business}', flush=True)
    print(f'Open msprof with fusion phase labels: {device}', flush=True)
    print(f'Kernel and business-scope handoffs: {manifest["handoff_report"]}', flush=True)
    for row in intervals:
        print(f'  {row["name"]}: {row["duration_us"]:.2f} us', flush=True)
    print(f'Early prefetch / final Get: {report["total_prefetch"]} / {report["total_final_get"]}', flush=True)
    manifest['business_status'] = 'PASS'
    return dict(report=report, source=source, intervals=intervals)


def capture_comparison(args, output, arguments, manifest, profiler):
    from workloads import configuration

    manifests, captures = {}, {}
    manifest['routes'] = manifests
    for route in args.comparison_routes:
        print(f'Running BS{args.batch} fusion trace {route}, local_counts={args.local_counts}', flush=True)
        arm = output / route
        arm.mkdir()
        arm_manifest = manifests[route] = dict(route=route, local_counts=args.local_counts, status='RUNNING')
        config = configuration(arguments)
        config.msprof_cold_route = route
        config.msprof_local_counts = args.local_counts
        config.msprof_build_dir = str(output / 'trace_build')
        try:
            captures[route] = capture_route(args, arm, config, arm_manifest, profiler, cold_route=route)
            arm_manifest['status'] = 'PASS'
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            arm_manifest.update(status='FAILED', error=str(error))
            print(f'{route} trace incomplete: {error}', file=sys.stderr, flush=True)
            if arm_manifest.get('worker_exit_code') != 0:
                break
        finally:
            save(arm / 'manifest.json', arm_manifest)
    if len(captures) != len(args.comparison_routes):
        raise RuntimeError('Every comparison arm requires checked source and native captures')
    if args.dynamic_compare:
        result = write_ready_comparison(output, captures)
        manifest.update(comparison=str(output / 'dynamic_trace_comparison.json'),
                        comparison_timeline=str(output / 'dynamic_trace_native.json'))
        for route, row in result['routes'].items():
            print(f"{route}: Resident+Cold+Service {row['resident_cold_service_device_us']:.2f} us, "
                  f"Hot to Terminal {row['hot_to_terminal_device_us']:.2f} us", flush=True)
        print(f"Get comparison: {result['get_comparison']['status']}; "
              f"native timeline: {manifest['comparison_timeline']}", flush=True)
        return
    if args.unified_compare:
        result = write_unified_comparison(output, captures, args.local_counts)
        manifest.update(comparison=str(output / 'unified_trace_comparison.json'),
                        comparison_timeline=str(output / 'unified_trace_native.json'))
        for route, row in result['routes'].items():
            cold = row['native_stages'][2]['duration_us']
            print(f"{route}: Cold+Service {cold:.2f} us, Score to KV ready "
                  f"{row['score_to_kv_ready_device_us']:.2f} us, Get {row['get_comparison']['status']}", flush=True)
        print(f"Compare complete native paths: {manifest['comparison_timeline']}", flush=True)
        print(f"Compare source phases and windows: {output / 'unified_trace_scopes.csv'}", flush=True)
        return
    result = write_comparison(output, captures, args.local_counts)
    manifest['comparison'] = str(output / 'cold_trace_comparison.json')
    print(f"Cold+Service native delta: {result['cold_service_native_delta_us']:+.2f} us; "
          f"Get comparison: {result['get_comparison']['status']}", flush=True)
    print(f"Compare per-window scopes: {output / 'cold_trace_scopes.csv'}", flush=True)


def capture(args, output, kvaf, arguments, manifest):
    from workloads import configuration

    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof'))
    if profiler is None:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    if args.comparison_routes:
        capture_comparison(args, output, arguments, manifest, profiler)
        return
    config = configuration(arguments)
    if args.route in ('dedicated', *READY_SERVICE_ROUTES):
        config.msprof_cold_route = args.route
        if args.route in READY_SERVICE_ROUTES:
            config.mr28_cold_trace = args.route
        config.msprof_local_counts = args.local_counts
        config.msprof_build_dir = str(output / 'trace_build')
    captured = capture_route(args, output, config, manifest, profiler, cold_route=args.route)
    if args.instruction_plan:
        instruction_output = output / 'instruction'
        manifest['instruction_status_file'] = str(instruction_output / 'instruction_status.json')
        save(output / 'manifest.json', manifest)
        manifest['instruction_capture'] = collect_instructions(
            profiler, instruction_output, kvaf, configuration(arguments), args.instruction_plan,
            captured['report']['fixture_sha256'],
        )


def main(argv=None):
    args = parse_args(argv)
    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    arguments = bench_args(args, kvaf)
    route = ('static_dynamic_service_comparison' if args.dynamic_compare else
             args.route if args.route in READY_SERVICE_ROUTES else
             'unified_score_p2_p4_comparison' if args.unified_compare else
             'cold_p2_p4_comparison' if args.cold_compare else
             'dedicated_service_parallel_terminal' if args.route == 'dedicated' else 'main_p2_parallel_terminal')
    planned_routes = args.comparison_routes or (args.route,)
    manifest = dict(
        status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
        worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=normal')),
        kvaf_commit=KVAF, route=route,
        local_counts=args.local_counts, arguments=arguments, requested_stage_replays=len(planned_routes),
        planned_routes=planned_routes,
        collection='pyACL_prof_start_stop_around_fused_forward_only',
        instruction_plan=args.instruction_plan,
        clocks='independent: msprof device task clock and calibrated source counter',
    )
    if args.cold_compare:
        manifest['historical_reference'] = REFERENCE
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
