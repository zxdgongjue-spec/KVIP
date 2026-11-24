#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native P2 with dedicated Cold Service on A5 at BS16."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

from run_mr25_handoff import KVAF, MAIN, ROOT, bench_arguments, collect, git, save
from run_mr25_score import PAIRED_KEYS, compare_reads, completed_reads
from run_mr28_unified_score import prefetch_lead


BATCH = 16
MODES = ('p2', 'dedicated')
COLD_ROLES = {
    'p2': dict(cold_mixed_groups=32, cold_score_aic=32, cold_score_aiv=64,
               cold_service_aiv=16, dedicated_service=False),
    'dedicated': dict(cold_mixed_groups=32, cold_score_aic=24, cold_score_aiv=48,
                      cold_service_aiv=16, dedicated_service=True),
}
MATCH_FIELDS = (*PAIRED_KEYS, 'available_aiv', 'cold_mixed_groups', 'cold_service_aiv')


def summarize(output, completed, trace_prefetch):
    rows = [collect(output, mode, BATCH) for mode in MODES]
    counts = {}
    for row in rows:
        mode = row['variant']
        row['route_verified'] = (
            completed.get(mode) == 0 and row['accepted']
            and row.get('effective_aic') == 32 and row.get('available_aiv', 0) >= 64
            and row.get('windows') == 4 and row.get('local_counts') is False
            and all(row.get(key) == value for key, value in COLD_ROLES[mode].items())
        )
        if row['route_verified']:
            counts[mode] = completed_reads(output, mode, BATCH)
            if trace_prefetch:
                row['prefetch_lead_to_terminal_us'] = prefetch_lead(output, mode, BATCH)
        row.update(get_comparison='unavailable', paired=False, timing_eligible=False,
                   observed_kv_ready_ratio=None, get_vector_differences=None)

    baseline, candidate = rows
    mismatches = [key for key in MATCH_FIELDS if baseline.get(key) != candidate.get(key)]
    comparable = len(counts) == len(MODES) and not mismatches
    if comparable:
        status, changed = compare_reads(counts['p2'], counts['dedicated'])
        baseline.update(get_comparison='exact', paired=True, timing_eligible=True,
                        observed_kv_ready_ratio=1.0, get_vector_differences=0)
        candidate.update(get_comparison=status, paired=status == 'exact',
                         timing_eligible=status in ('exact', 'same_request_totals'),
                         observed_kv_ready_ratio=candidate['kv_ready_us'] / baseline['kv_ready_us'],
                         get_vector_differences=changed)

    for row in rows:
        mode = row['variant']
        if not row['route_verified']:
            print(f"{mode}: checked run unavailable (worker rc={completed.get(mode)}, "
                  f"report={row['reason'] or 'unexpected route geometry'})", flush=True)
            continue
        print(f"{mode}: Hot {row['hot_us']:.2f}, Threshold {row['threshold_us']:.2f}, "
              f"Cold+Service {row['cold_service_us']:.2f}, Terminal {row['terminal_us']:.2f} µs "
              f"(profiler kernel means); KV ready {row['kv_ready_us']:.2f} µs (graph Event mean)", flush=True)
        print(f"  Prefetch/final/total Get per call "
              f"{row['prefetch_requests_per_call']:.2f}/{row['final_get_requests_per_call']:.2f}/"
              f"{row['total_get_requests_per_call']:.2f}", flush=True)
    if comparable:
        print(f"Get comparison: {candidate['get_comparison']}; "
              f"observed KV ready ratio: {candidate['observed_kv_ready_ratio']:.4f}x", flush=True)
    elif mismatches and len(counts) == len(MODES):
        print(f'Pair rejected: different build or fixture fields ({", ".join(mismatches)})', flush=True)

    save(output / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    if counts:
        with (output / 'requests.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('variant', 'source_round', 'invocation', 'request', 'prefetch',
                             'final_get', 'total_get', *(f'prefetch_w{i}' for i in range(4)),
                             *(f'final_get_q{i}' for i in range(4))))
            for mode, records in counts.items():
                for identity, (early, final) in sorted(records.items()):
                    writer.writerow((mode, *identity, sum(early), sum(final), sum(early) + sum(final),
                                     *early, *final))
    print(f'Summary: {output / "summary.csv"}', flush=True)
    return comparable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--comparison-dry-run', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--prefetch-trace', action='store_true',
                        help='record a separate diagnostic replay for actual prefetch timestamps')
    args = parser.parse_args()
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
    if args.prefetch_trace:
        arguments[0] = 'diagnose'
        arguments.extend(('--level', '2'))
    completed = {}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_base=MAIN,
                    expected_kvaf=KVAF, arguments=arguments, variants=MODES, cold_roles=COLD_ROLES,
                    local_counts=False, prefetch_trace=args.prefetch_trace, worker_exit_codes=completed)
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    from execution import launch
    from workloads import configuration

    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr28_dedicated_service')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr28_dedicated_service.', dir=out_base))
    print(f'MR28 dedicated Service comparison: {output}', flush=True)
    try:
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        for mode in MODES:
            config = configuration(arguments)
            config.mr25_handoff = mode
            config.mr28_dedicated_service = mode
            config.handoff_build_dir = output / 'build'
            print(f'Running BS16 {mode}, local_counts OFF', flush=True)
            completed[mode] = launch(config, output / mode)
            save(output / 'manifest.json', manifest)
            if completed[mode]:
                break
        passed = summarize(output, completed, args.prefetch_trace)
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
