#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native P2 with dedicated Cold Service or W1 prescore at BS16."""

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
from terminal_handoff import P2_W1_KERNELS


BATCH = 16
MODES = ('p2', 'dedicated')
P2_W1_MODES = ('p2', 'p2_w1')
COLD_ROLES = {
    'p2': dict(cold_mixed_groups=32, cold_score_aic=32, cold_score_aiv=64,
               cold_service_aiv=16, dedicated_service=False),
    'dedicated': dict(cold_mixed_groups=32, cold_score_aic=24, cold_score_aiv=48,
                      cold_service_aiv=16, dedicated_service=True),
    'p2_w1': dict(cold_mixed_groups=32, cold_score_aic=32, cold_score_aiv=64,
                  cold_service_aiv=16, dedicated_service=False),
}
MATCH_FIELDS = (*PAIRED_KEYS, 'available_aiv', 'cold_mixed_groups', 'cold_service_aiv')


def summarize(output, completed, trace_prefetch, modes=MODES, *, p2_w1=False):
    rows = [collect(output, mode, BATCH) for mode in modes]
    counts = {}
    for row in rows:
        mode = row['variant']
        row['route_verified'] = (
            completed.get(mode) == 0 and row['accepted']
            and row.get('effective_aic') == 32 and row.get('available_aiv', 0) >= 64
            and row.get('windows') == 4 and row.get('local_counts') is False
            and all(row.get(key) == value for key, value in COLD_ROLES[mode].items())
            and (not p2_w1 or (row.get('w1_prescore') is (mode == 'p2_w1')
                              and row.get('prefetch_observed')))
        )
        if row['route_verified']:
            counts[mode] = completed_reads(output, mode, BATCH)
            if trace_prefetch:
                row['prefetch_lead_to_terminal_us'] = prefetch_lead(output, mode, BATCH)
        row.update(get_comparison='unavailable', paired=False, timing_eligible=False,
                   observed_kv_ready_ratio=None, get_vector_differences=None)

    mismatches = []
    comparable = len(rows) == len(counts) == 2
    if comparable:
        baseline, candidate = rows
        mismatches = [key for key in MATCH_FIELDS if baseline.get(key) != candidate.get(key)]
        comparable = not mismatches
    if comparable:
        status, changed = compare_reads(counts['p2'], counts[candidate['variant']])
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
        threshold = 'Threshold+W1 Score' if mode == 'p2_w1' else 'Threshold'
        print(f"{mode}: Hot {row['hot_us']:.2f}, {threshold} {row['threshold_us']:.2f}, "
              f"Cold+Service {row['cold_service_us']:.2f}, Terminal {row['terminal_us']:.2f} µs "
              f"(profiler kernel means); KV ready {row['kv_ready_us']:.2f} µs (graph Event mean)", flush=True)
        print(f"  Prefetch/final/total Get per call "
              f"{row['prefetch_requests_per_call']:.2f}/{row['final_get_requests_per_call']:.2f}/"
              f"{row['total_get_requests_per_call']:.2f}", flush=True)
    if comparable:
        ratio_label = ('observed KV ready ratio' if candidate['timing_eligible']
                       else 'raw KV ready ratio (different Get workload)')
        print(f"Get comparison: {candidate['get_comparison']}; "
              f"{ratio_label}: {candidate['observed_kv_ready_ratio']:.4f}x", flush=True)
    elif mismatches:
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
    return comparable if len(rows) == 2 else len(counts) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--comparison-dry-run', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--p2-w1', action='store_true',
                        help='compare native P2 with early W1 Score and rotating Service')
    parser.add_argument('--route', choices=('both', *MODES, 'p2_w1'), default='both',
                        help='run a single checked route for diagnosis or both routes for comparison')
    parser.add_argument('--staged-eager', action='store_true',
                        help='synchronize Hot, Threshold, Cold and Terminal separately on the first eager forward')
    parser.add_argument('--legacy-credit-tail', action='store_true',
                        help='diagnostic control that omits only the last two Dedicated AIV1 credit waits')
    parser.add_argument('--prefetch-trace', action='store_true',
                        help='record a separate diagnostic replay for actual prefetch timestamps')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')
    trial_modes = P2_W1_MODES if args.p2_w1 else MODES
    if args.route != 'both' and args.route not in trial_modes:
        parser.error(f'--route expects both or one of {", ".join(trial_modes)} for this comparison')
    if args.legacy_credit_tail and args.route != 'dedicated':
        parser.error('--legacy-credit-tail requires --route dedicated for the diagnostic control')

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
    modes = trial_modes if args.route == 'both' else (args.route,)
    completed = {}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_base=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=no')),
                    expected_kvaf=KVAF, arguments=arguments, variants=modes,
                    cold_roles={mode: COLD_ROLES[mode] for mode in modes},
                    local_counts=False, prefetch_trace=args.prefetch_trace, staged_eager=args.staged_eager,
                    dedicated_drain_aiv1=not args.legacy_credit_tail,
                    worker_exit_codes=completed)
    if args.p2_w1:
        manifest['expected_graph_kernels'] = {mode: P2_W1_KERNELS[mode] for mode in modes}
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    from execution import launch
    from workloads import configuration

    trial = 'mr28_p2_w1' if args.p2_w1 else 'mr28_dedicated_service'
    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', f'results/{trial}')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f'{trial}.', dir=out_base))
    comparison = 'P2 W1 prescore' if args.p2_w1 else 'dedicated Service'
    print(f'MR28 {comparison} comparison: {output}', flush=True)
    try:
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        for mode in modes:
            config = configuration(arguments)
            config.mr25_handoff = mode
            config.mr28_dedicated_service = mode
            config.mr28_p2_w1_experiment = args.p2_w1
            config.mr28_staged_eager = args.staged_eager
            config.mr28_dedicated_drain_aiv1 = not args.legacy_credit_tail
            config.handoff_build_dir = output / 'build'
            print(f'Running BS16 {mode}, local_counts OFF', flush=True)
            completed[mode] = launch(config, output / mode)
            save(output / 'manifest.json', manifest)
            if completed[mode]:
                break
        passed = summarize(output, completed, args.prefetch_trace, modes, p2_w1=args.p2_w1)
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
