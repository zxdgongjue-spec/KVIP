#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native P2 with dedicated Cold Service or W1 prescore."""

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
from terminal_handoff import COLD_GM_KERNELS, COLD_GM_MODES, COLD_GM_STAGE_FIELDS, P2_W1_KERNELS, STAGES
from kvip.operators.lifetch.geometry import li_geometry


BATCHES = (1, 2, 4, 8, 16)
MODES = ('p2', 'dedicated')
P2_W1_MODES = ('p2', 'p2_w1')
WINDOWS = 4
MATCH_FIELDS = (*PAIRED_KEYS, 'available_aiv', 'cold_mixed_groups', 'cold_service_aiv')


def build_case(kvaf, batch, modes, args):
    arguments = bench_arguments(kvaf, (batch,))
    for option, value in (
        ('--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986')),
        ('--warmup', args.warmup), ('--iterations', args.iterations), ('--timeout-seconds', args.timeout_seconds),
    ):
        arguments[arguments.index(option) + 1] = str(value)
    if args.prefetch_trace:
        arguments[0] = 'diagnose'
        arguments.extend(('--level', '2'))
    geometry = li_geometry(batch, intermediate_fetches=WINDOWS, aic_count=32)
    mixed_groups = geometry['rotating_aic']
    roles = {}
    for mode in modes:
        dedicated = mode == 'dedicated'
        score_groups = 24 if dedicated else mixed_groups
        roles[mode] = dict(cold_mixed_groups=mixed_groups, cold_score_aic=score_groups,
                           cold_score_aiv=2 * score_groups, dedicated_service=dedicated,
                           cold_service_aiv=geometry['service_aivs_per_window'])
    return dict(status='NOT_RUN', arguments=arguments, windows=geometry['intermediate_fetches'], cold_roles=roles,
                early_w1_mixed_groups=min(4, mixed_groups // 2) if 'p2_w1' in modes else 0,
                worker_exit_codes={})


def write_summary(output, rows):
    save(output / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Summary: {output / "summary.csv"}', flush=True)


def summarize(output, batch, case, trace_prefetch, modes, *, p2_w1=False, cold_gm=False):
    rows = [collect(output, mode, batch, stage_fields=COLD_GM_STAGE_FIELDS.get(mode, STAGES))
            for mode in modes]
    completed = case['worker_exit_codes']
    counts = {}
    for row in rows:
        mode = row['variant']
        row['route_verified'] = (
            completed.get(mode) == 0 and row['accepted']
            and row.get('effective_aic') == 32 and row.get('available_aiv', 0) >= 64
            and row.get('windows') == case['windows'] and row.get('local_counts') is False
            and all(row.get(key) == value for key, value in case['cold_roles'][mode].items())
            and (not p2_w1 or row.get('w1_prescore') is (mode == 'p2_w1'))
            and (not cold_gm or (row.get('cold_score_writeback') is (mode != 'p2')
                                 and row.get('cold_gm_row_consumer') is (mode == 'cold_gm_pipeline')))
        )
        if row['route_verified']:
            counts[mode] = completed_reads(output, mode, batch)
            if trace_prefetch:
                row['prefetch_lead_to_terminal_us'] = prefetch_lead(output, mode, batch)
        row.update(get_comparison='unavailable', paired=False, timing_eligible=False,
                   observed_kv_ready_ratio=None, get_vector_differences=None)

    comparable = len(rows) == len(counts)
    baseline = rows[0]
    if len(rows) > 1 and baseline['variant'] in counts:
        baseline.update(get_comparison='exact', paired=True, timing_eligible=True,
                        observed_kv_ready_ratio=1.0, get_vector_differences=0)
    for candidate in rows[1:]:
        mode = candidate['variant']
        if 'p2' not in counts or mode not in counts:
            continue
        mismatches = [key for key in MATCH_FIELDS if baseline.get(key) != candidate.get(key)]
        if mismatches:
            print(f'{mode} pair rejected: different build or fixture fields ({", ".join(mismatches)})',
                  flush=True)
            comparable = False
            continue
        status, changed = compare_reads(counts['p2'], counts[mode])
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
        labels = [('Hot', 'hot_us')]
        if mode == 'cold_gm_pipeline':
            labels.append(('Cold GM pipeline', 'cold_gm_pipeline_us'))
        else:
            phase = {
                'p2_w1': ('Threshold+W1 Score', 'threshold_us'),
                'cold_gm_fenced': ('Cold raw Score+Threshold', 'cold_raw_threshold_us'),
            }.get(mode, ('Threshold', 'threshold_us'))
            labels.extend((phase, ('Cold+Service', 'cold_service_us')))
        labels.append(('Terminal', 'terminal_us'))
        stages = ', '.join(f'{label} {row[key]:.2f}' for label, key in labels)
        print(f"{mode}: {stages} µs "
              f"(profiler kernel means); KV ready {row['kv_ready_us']:.2f} µs (graph Event mean)", flush=True)
        print(f"  Prefetch/final/total Get per call "
              f"{row['prefetch_requests_per_call']:.2f}/{row['final_get_requests_per_call']:.2f}/"
              f"{row['total_get_requests_per_call']:.2f}", flush=True)
    for candidate in rows[1:]:
        if candidate['observed_kv_ready_ratio'] is None:
            continue
        ratio_label = ('observed KV ready ratio' if candidate['timing_eligible']
                       else 'raw KV ready ratio (different Get workload)')
        print(f"{candidate['variant']}: Get comparison: {candidate['get_comparison']}; "
              f"{ratio_label}: {candidate['observed_kv_ready_ratio']:.4f}x", flush=True)

    if cold_gm and comparable and all(mode in counts for mode in COLD_GM_MODES):
        fenced, pipeline = (next(row for row in rows if row['variant'] == mode) for mode in COLD_GM_MODES)
        status, changed = compare_reads(counts[fenced['variant']], counts[pipeline['variant']])
        pipeline.update(get_vs_fenced=status, get_vector_differences_vs_fenced=changed,
                        timing_eligible_vs_fenced=status in ('exact', 'same_request_totals'),
                        observed_kv_ready_over_fenced=pipeline['kv_ready_us'] / fenced['kv_ready_us'])
        print(f"Pipeline vs fenced: {pipeline['observed_kv_ready_over_fenced']:.4f}x "
              f"(Get {status})", flush=True)

    if counts:
        with (output / 'requests.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('variant', 'source_round', 'invocation', 'request', 'prefetch',
                             'final_get', 'total_get', *(f'prefetch_w{i}' for i in range(WINDOWS)),
                             *(f'final_get_q{i}' for i in range(4))))
            for mode, records in counts.items():
                for identity, (early, final) in sorted(records.items()):
                    unused_windows = (None,) * (WINDOWS - len(early))
                    writer.writerow((mode, *identity, sum(early), sum(final), sum(early) + sum(final),
                                     *early, *unused_windows, *final))
    write_summary(output, rows)
    return comparable, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--comparison-dry-run', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument('--p2-w1', action='store_true',
                        help='compare native P2 with early W1 Score and rotating Service')
    experiments.add_argument('--cold-gm', action='store_true',
                             help='compare native P2 with fenced and per-request Cold GM consumers')
    parser.add_argument('--batch', action='append', type=int, choices=BATCHES,
                        help='repeat to compare multiple batch sizes with one build; default: 16')
    parser.add_argument('--route', choices=('both', 'all', *MODES, 'p2_w1', *COLD_GM_MODES), default='all',
                        help='run a single checked route or every route in the comparison')
    parser.add_argument('--staged-eager', action='store_true',
                        help='synchronize Hot, Threshold, Cold and Terminal separately on the first eager forward')
    parser.add_argument('--legacy-credit-tail', action='store_true',
                        help='diagnostic control that omits only the last two Dedicated AIV1 credit waits')
    parser.add_argument('--prefetch-trace', action='store_true',
                        help='record a separate diagnostic replay for actual prefetch timestamps')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')
    trial_modes = ('p2', *COLD_GM_MODES) if args.cold_gm else P2_W1_MODES if args.p2_w1 else MODES
    all_routes = args.route in ('all', 'both')
    if not all_routes and args.route not in trial_modes:
        parser.error(f'--route expects all or one of {", ".join(trial_modes)} for this comparison')
    if args.legacy_credit_tail and args.route != 'dedicated':
        parser.error('--legacy-credit-tail requires --route dedicated for the diagnostic control')
    batches = args.batch or (16,)
    if len(set(batches)) != len(batches):
        parser.error('Each batch can appear only once')
    if not (args.p2_w1 or args.cold_gm) and any(batch != 16 for batch in batches):
        parser.error('Dedicated Service supports only --batch 16')

    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    modes = trial_modes if all_routes else (args.route,)
    cases = {batch: build_case(kvaf, batch, modes, args) for batch in batches}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_base=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=no')),
                    expected_kvaf=KVAF, variants=modes, cases=cases,
                    local_counts=False, prefetch_trace=args.prefetch_trace, staged_eager=args.staged_eager,
                    dedicated_drain_aiv1=not args.legacy_credit_tail)
    if len(cases) == 1:
        manifest.update(cases[batches[0]])
    if args.p2_w1:
        manifest['expected_graph_kernels'] = {mode: P2_W1_KERNELS[mode] for mode in modes}
    if args.cold_gm:
        manifest['expected_graph_kernels'] = {mode: COLD_GM_KERNELS[mode] for mode in modes}
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    from execution import launch
    from workloads import configuration

    trial = 'mr28_cold_gm' if args.cold_gm else 'mr28_p2_w1' if args.p2_w1 else 'mr28_dedicated_service'
    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', f'results/{trial}')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f'{trial}.', dir=out_base))
    comparison = 'Cold GM' if args.cold_gm else 'P2 W1 prescore' if args.p2_w1 else 'dedicated Service'
    print(f'MR28 {comparison} comparison: {output}', flush=True)
    try:
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        passed = True
        rows = []
        for batch, case in cases.items():
            case_output = output if len(cases) == 1 else output / f'bs{batch}'
            case_output.mkdir(exist_ok=True)
            case['status'] = 'RUNNING'
            save(output / 'manifest.json', manifest)
            completed = case['worker_exit_codes']
            for mode in modes:
                config = configuration(case['arguments'])
                config.mr25_handoff = mode
                config.mr28_dedicated_service = mode
                config.mr28_p2_w1_experiment = args.p2_w1
                config.mr28_cold_gm_experiment = args.cold_gm
                config.mr28_staged_eager = args.staged_eager
                config.mr28_dedicated_drain_aiv1 = not args.legacy_credit_tail
                config.handoff_build_dir = output / 'build'
                print(f'Running BS{batch} {mode}, local_counts OFF', flush=True)
                completed[mode] = launch(config, case_output / mode)
                save(output / 'manifest.json', manifest)
                if completed[mode]:
                    break
            case_passed, case_rows = summarize(
                case_output, batch, case, args.prefetch_trace, modes, p2_w1=args.p2_w1, cold_gm=args.cold_gm)
            case['status'] = 'PASS' if case_passed else 'FAILED'
            passed = passed and case_passed
            rows.extend(case_rows)
            save(output / 'manifest.json', manifest)
            if any(completed.values()):
                break
        if len(cases) > 1:
            write_summary(output, rows)
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
