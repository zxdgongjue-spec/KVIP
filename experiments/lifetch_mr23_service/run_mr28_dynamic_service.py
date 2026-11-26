#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark static versus dynamic BS16 Resident/Cold/Service on A5."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

from run_mr25_handoff import KVAF, MAIN, ROOT, bench_arguments, collect, git, save
from run_mr25_score import PAIRED_KEYS, compare_reads, completed_reads


BATCH = 16
ROUTES = ('static_ready', 'dynamic')
DYNAMIC_STAGES = ('hot_us', 'resident_cold_service_us', 'terminal_us')
ROLES = {
    'static_ready': dict(dedicated_service=True, scheduler_aic=0, cold_score_aic=24,
                         cold_score_aiv=48, cold_service_aiv=16),
    'dynamic': dict(dedicated_service=False, scheduler_aic=1, cold_score_aic=31,
                    cold_score_aiv=62, cold_service_aiv=64),
}
MATCH_FIELDS = (*PAIRED_KEYS, 'available_aiv', 'cold_mixed_groups')


def summarize(output, completed, modes):
    rows = {}
    for mode in modes:
        args = {'stage_fields': DYNAMIC_STAGES} if mode == 'dynamic' else {}
        row = rows[mode] = collect(output, mode, BATCH, **args)
        row['route_verified'] = (
            completed.get(mode) == 0 and row['accepted']
            and row.get('effective_aic') == 32 and row.get('available_aiv', 0) >= 64
            and row.get('windows') == 4 and row.get('local_counts') is False
            and row.get('cold_mixed_groups') == 32 and row.get('resident_aiv') == 64
            and all(row.get(field) == value for field, value in ROLES[mode].items())
        )
        row['get_comparison'] = 'unavailable'
        row['matched_kv_ready_ratio'] = None
        if not row['route_verified']:
            print(f"{mode}: checked run unavailable (worker rc={completed.get(mode)}, "
                  f"report={row['reason'] or 'unexpected route geometry'})", flush=True)
            continue
        if mode == 'static_ready':
            row['resident_cold_service_us'] = row['threshold_us'] + row['cold_service_us']
        print(f"{mode}: Hot {row['hot_us']:.2f}, "
              f"Resident+Cold+Service {row['resident_cold_service_us']:.2f}, "
              f"Terminal {row['terminal_us']:.2f} µs (kernel means); "
              f"KV ready {row['kv_ready_us']:.2f} µs (graph Event mean)", flush=True)
        print(f"  Prefetch/final/total Get per call "
              f"{row['prefetch_requests_per_call']:.2f}/{row['final_get_requests_per_call']:.2f}/"
              f"{row['total_get_requests_per_call']:.2f}", flush=True)

    comparisons = None
    passed = all(row['route_verified'] for row in rows.values())
    if len(rows) == 2 and passed:
        baseline, candidate = (rows[route] for route in ROUTES)
        mismatches = [key for key in MATCH_FIELDS
                      if baseline.get(key) is None or baseline.get(key) != candidate.get(key)]
        if mismatches:
            comparisons = dict(fixture_matched=False, mismatched_fields=mismatches)
            print(f'Pair rejected: different fixture/build/shape fields {mismatches}', flush=True)
            passed = False
        else:
            reads = {route: completed_reads(output, route, BATCH) for route in ROUTES}
            status, changed = compare_reads(*(reads[route] for route in ROUTES))
            comparisons = dict(fixture_matched=True, get_comparison=status,
                               get_vector_differences=changed,
                               raw_kv_ready_ratio=candidate['kv_ready_us'] / baseline['kv_ready_us'])
            baseline['get_comparison'] = 'baseline'
            candidate['get_comparison'] = status
            if status == 'exact':
                candidate['matched_kv_ready_ratio'] = comparisons['raw_kv_ready_ratio']
            print(f"Dynamic/static KV ready ratio {comparisons['raw_kv_ready_ratio']:.4f}x; "
                  f"request-by-window prefetch/Get {status}", flush=True)
            with (output / 'requests.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('route', 'source_round', 'invocation', 'request',
                                 *(f'prefetch_w{i}' for i in range(4)), *(f'final_get_q{i}' for i in range(4))))
                for route, records in reads.items():
                    for identity, (early, final) in sorted(records.items()):
                        writer.writerow((route, *identity, *early, *final))

    save(output / 'summary.json', dict(routes=rows, comparison=comparisons))
    columns = list(dict.fromkeys(key for row in rows.values() for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows.values())
    print(f'Summary: {output / "summary.csv"}', flush=True)
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--comparison-dry-run', action='store_true')
    parser.add_argument('--route', choices=('both', *ROUTES), default='both')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')

    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    arguments = bench_arguments(kvaf, (BATCH,))
    for option, value in (
        ('--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986')),
        ('--warmup', args.warmup), ('--iterations', args.iterations), ('--timeout-seconds', args.timeout_seconds),
    ):
        arguments[arguments.index(option) + 1] = str(value)
    modes = ROUTES if args.route == 'both' else (args.route,)
    completed = {}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_base=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=no')),
                    expected_kvaf=KVAF, arguments=arguments, variants=modes, local_counts=False,
                    cold_roles=ROLES,
                    service_gate='same request: Resident seed, whole current window and previous Service',
                    worker_exit_codes=completed)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    from execution import launch
    from workloads import configuration

    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr28_dynamic_service')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr28_dynamic_service.', dir=out_base))
    print(f'MR28 BS16 dynamic Service comparison: {output}', flush=True)
    try:
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        for mode in modes:
            config = configuration(arguments)
            config.mr25_handoff = mode
            config.mr28_dedicated_service = mode
            config.handoff_build_dir = output / 'build'
            print(f'Running BS16 {mode}, local_counts OFF', flush=True)
            completed[mode] = launch(config, output / mode)
            save(output / 'manifest.json', manifest)
            if completed[mode]:
                break
        passed = summarize(output, completed, modes)
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
