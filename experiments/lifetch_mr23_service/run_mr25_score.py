#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare Hot and Cold Score partitioning on the same MR28 library."""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

from run_mr25_handoff import KVAF, ROOT, bench_arguments, collect, git, read, save


MODES = ('p2', 'uniform')
STAGE_TO_MODE = {'hot': 'uniform', 'cold': 'cold', 'hot-cold': 'hot_cold'}
PARTITION_STAGES = {'p2': (False, False), 'uniform': (True, False),
                    'cold': (False, True), 'hot_cold': (True, True)}
MR28 = 'f29180a01867758c353724437c668e2c302ad45f'
PAIRED_KEYS = ('fixture_sha256', 'library_sha256', 'soc', 'effective_aic', 'windows',
               'score_parts', 'score_blocks', 'service_lanes', 'terminal_tasks', 'local_counts')


def completed_reads(output, mode, batch):
    result = read(output / mode / 'miss10' / f'case-b{batch}' / 'fused/path_result.json')
    return {
        (row['source_round'], row['invocation'], row['request']):
            (row['prefetch_by_window'], row['final_get_by_query'])
        for row in result['pipeline_counts']
    }


def compare_reads(base, candidate):
    if base.keys() != candidate.keys():
        return 'different_request_totals', None
    changed = [key for key in base if base[key] != candidate[key]]
    if not changed:
        return 'exact', 0
    if all(sum(map(sum, base[key])) == sum(map(sum, candidate[key])) for key in changed):
        return 'same_request_totals', len(changed)
    return 'different_request_totals', len(changed)


def matches_geometry(row, mode, batch):
    hot, cold = PARTITION_STAGES[mode]
    aics = row['effective_aic']
    hot_parts = min(aics // batch, 16384 // 256) if hot else 2
    # Dense Q4/G32/H128K/R16K with at most four windows has room for all
    # available even Cold partitions, including in the short last window.
    cold_parts = (aics // (2 * batch)) * 2 if cold else 2
    return (row['hot_score_parts'] == hot_parts and row['hot_score_blocks'] == batch * hot_parts
            and row['cold_score_parts'] == cold_parts and row['cold_score_blocks'] == batch * cold_parts)


def compare_arm(output, batch, base, row, completed):
    mode = row['variant']
    failures = [
        f"{arm['variant']}: {arm['reason'] or 'worker did not complete'}"
        for arm in (base, row)
        if completed.get(arm['variant']) != 0 or not arm['accepted']
    ]
    if not failures:
        failures.extend(key for key in PAIRED_KEYS if base[key] != row[key])
        if base['local_counts'] is not True:
            failures.append('local_counts must be ON')
        failures.extend(f"{arm['variant']}: unexpected Score geometry" for arm in (base, row)
                        if not matches_geometry(arm, arm['variant'], batch))
    if failures:
        row.update(comparison_reason=', '.join(failures), get_comparison='unavailable',
                   get_vector_differences=None, paired=False, timing_eligible=False)
    else:
        status, changed = compare_reads(completed_reads(output, 'p2', batch),
                                        completed_reads(output, mode, batch))
        row.update(comparison_reason='' if status == 'exact' else status,
                   get_comparison=status, get_vector_differences=changed,
                   paired=status == 'exact', timing_eligible=status != 'different_request_totals')


def summarize(output, batch, completed, modes=MODES):
    rows = [collect(output, mode, batch) for mode in modes]
    base, *candidates = rows
    for row in candidates:
        compare_arm(output, batch, base, row, completed)
    base.update(paired=all(row['paired'] for row in candidates),
                timing_eligible=all(row['timing_eligible'] for row in candidates),
                comparison_reason='', get_comparison='baseline', get_vector_differences=None)
    for row in rows:
        if row['accepted']:
            parts = row['hot_score_parts']
            row['hot_tokens_min'] = (16384 // 256 // parts) * 256
            row['hot_tokens_max'] = ((16384 // 256 + parts - 1) // parts) * 256
        for label, key in (('hot', 'hot_us'), ('cold_service', 'cold_service_us'), ('kv_ready', 'kv_ready_us')):
            row[f'{label}_over_p2'] = row[key] / base[key] if row['paired'] else None
            row[f'observed_{label}_over_p2'] = row[key] / base[key] if row['timing_eligible'] else None
    save(output / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    for row in candidates:
        mode = row['variant']
        if not row['paired']:
            print(f"BS{batch} {mode} strict pair rejected: {row['comparison_reason']} "
                  f"(changed request records={row['get_vector_differences']})", flush=True)
        if row['get_comparison'] != 'unavailable':
            def timing(metric):
                return f'{base[metric]:.2f} -> {row[metric]:.2f} us'

            compare_kind = {
                'exact': 'exact',
                'same_request_totals': 'observed, early/final Get placement differs',
                'different_request_totals': 'raw timing, per-request total Get differs',
            }[row['get_comparison']]
            print(f"BS{batch} {mode} ({compare_kind})\n"
                  f"  Hot {base['hot_score_blocks']} -> {row['hot_score_blocks']} MIX, {timing('hot_us')}\n"
                  f"  Cold {base['cold_score_blocks']} -> {row['cold_score_blocks']} MIX, "
                  f"Cold+Service {timing('cold_service_us')}\n  KV ready {timing('kv_ready_us')}", flush=True)
            for label in ('prefetch', 'final_get', 'total_get'):
                key = f'{label}_requests_per_call'
                print(f"  {key}: {base[key]:.2f} -> {row[key]:.2f}", flush=True)
    print('Summary: ' + str(output / 'summary.csv'), flush=True)
    gate = 'paired' if modes == MODES else 'timing_eligible'
    return all(row[gate] for row in candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', action='append', type=int, choices=(1, 2, 4, 8, 16), required=True,
                        help='batch size; repeat to run independent cases with one shared build')
    parser.add_argument('--comparison-dry-run', action='store_true')
    parser.add_argument('--reverse-order', action='store_true', help='run candidates before the baseline')
    parser.add_argument('--stage', action='append', choices=tuple(STAGE_TO_MODE),
                        help='candidate stage; repeat to share one P2 baseline; default: hot')
    args = parser.parse_args()
    if len(set(args.batch)) != len(args.batch):
        parser.error('Each batch can appear only once')
    stages = tuple(dict.fromkeys(args.stage or ('hot',)))
    modes = ('p2', *(STAGE_TO_MODE[stage] for stage in stages))
    order = modes[::-1] if args.reverse_order else modes
    needs_cold = any(PARTITION_STAGES[mode][1] for mode in modes)
    base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr25_hot_score')).resolve()
    base.mkdir(parents=True, exist_ok=True)
    prefix = 'mr28_score.' if needs_cold else 'mr25_hot_score.'
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', ROOT / 'third_party/KVAF')).resolve()
    cases = {batch: dict(arguments=bench_arguments(kvaf, (batch,)), completed={}) for batch in args.batch}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), mr28_base=MR28,
                    expected_kvaf=KVAF, cases=cases, variants=order, local_counts=True,
                    shared_build=str(output / 'build'))
    save(output / 'manifest.json', manifest)
    print('MR28 Score comparison: ' + str(output), flush=True)
    if args.comparison_dry_run:
        print(f'Planned batches {args.batch}, modes {list(order)}, '
              f'{len(order)} independent workers per batch, local_counts ON.', flush=True)
        return 0

    from execution import launch
    from workloads import configuration

    try:
        if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
            raise ValueError(f'Use the built KVAF checkout at {KVAF}')
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        passed = True
        for batch, case in cases.items():
            directory = output / f'bs{batch}'
            directory.mkdir()
            completed = case['completed']
            for mode in order:
                config = configuration(case['arguments'])
                config.mr25_handoff = mode
                hot, cold = PARTITION_STAGES[mode]
                config.mr25_hot_partition = hot
                config.mr25_cold_partition = cold if needs_cold else None
                config.handoff_build_dir = output / 'build'
                print(f'Running BS{batch} {mode}, local_counts ON', flush=True)
                completed[mode] = launch(config, directory / mode)
                save(output / 'manifest.json', manifest)
                if completed[mode]:
                    break
            passed = summarize(directory, batch, completed, modes) and passed
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
