#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the native P2 flow with full-history Score at P2 and uniform P."""

import argparse
import csv
import os
from pathlib import Path
import statistics
import sys
import tempfile

from run_mr25_handoff import KVAF, MAIN, ROOT, bench_arguments, collect, git, read, save
from run_mr25_score import PAIRED_KEYS, compare_reads, completed_reads


BATCHES = (1, 2, 4, 8, 16)
MODES = ('p2', 'full_p2', 'full_uniform')


def expected_parts(mode, batch, effective_aic):
    return min(16, effective_aic // batch) if mode == 'full_uniform' else 2


def prefetch_lead(directory, mode, batch):
    detail = directory / mode / 'miss10' / f'case-b{batch}' / 'fused/diagnose'
    leads = []
    for path in sorted(detail.glob('round-*/detail_samples.json')):
        for sample in read(path):
            diagnostics = sample['streaming_cycles'][0]
            prefetch = diagnostics['prefetch_bandwidth']
            if not prefetch['entries'] or 'first doorbell' not in prefetch['note']:
                continue
            timeline = diagnostics['terminal_timeline']
            first_terminal = min(core['stages'][0]['start_cycles'] for core in timeline['cores'])
            leads.append((first_terminal - prefetch['first_cycles']) / timeline['clock_mhz'])
    return statistics.mean(leads) if leads else None


def summarize(directory, batch, completed):
    rows = [collect(directory, mode, batch) for mode in MODES]
    baseline, full_p2, uniform = rows
    all_valid = True
    for row in rows:
        mode = row['variant']
        aic_count = row.get('effective_aic')
        expected = expected_parts(mode, batch, aic_count) if aic_count is not None else None
        row['route_verified'] = (
            completed.get(mode) == 0 and row['accepted']
            and expected is not None
            and row.get('local_counts') is False
            and row.get('unified_score') is (mode != 'p2')
            and row.get('producer_parts') == expected
            and row.get('producer_blocks') == batch * expected
        )
        if row['accepted']:
            row['score_us'] = row.pop('hot_us')
            row['score_scope'] = 'resident' if mode == 'p2' else 'full_history'
            row['cold_stage'] = 'Cold Score + Service' if mode == 'p2' else 'Cold T2 + Service'
            row['prefetch_lead_to_terminal_us'] = prefetch_lead(directory, mode, batch)

    for row in rows:
        mode = row['variant']
        reference = full_p2 if mode == 'full_uniform' else baseline
        comparable = (
            baseline['route_verified'] and reference['route_verified'] and row['route_verified']
            and all(baseline[key] == record[key]
                    for record in (reference, row) for key in PAIRED_KEYS)
        )
        all_valid &= comparable
        row.update(workload_verified=comparable, get_vs_native_p2='unavailable',
                   get_vs_full_p2='unavailable', paired=False,
                   observed_kv_ready_over_native_p2=None, score_over_full_p2=None)
        if comparable:
            counts = completed_reads(directory, mode, batch)
            status, changed = compare_reads(completed_reads(directory, 'p2', batch), counts)
            row.update(get_vs_native_p2=status, get_vector_differences=changed,
                       paired=status == 'exact',
                       observed_kv_ready_over_native_p2=row['kv_ready_us'] / baseline['kv_ready_us'])
            if mode == 'full_uniform':
                row['get_vs_full_p2'], _ = compare_reads(completed_reads(directory, 'full_p2', batch), counts)
                row['score_over_full_p2'] = row['score_us'] / full_p2['score_us']
            print(f"BS{batch} {mode}  Score {row['score_scope']} P{row['producer_parts']} "
                  f"{row['score_us']:.2f} µs  {row['cold_stage']} {row['cold_service_us']:.2f} µs "
                  f"KV ready {row['kv_ready_us']:.2f} µs  Get vs native {status}\n"
                  f"  prefetch/final/total per call {row['prefetch_requests_per_call']:.2f}/"
                  f"{row['final_get_requests_per_call']:.2f}/{row['total_get_requests_per_call']:.2f}",
                  flush=True)
        else:
            print(f"BS{batch} {mode} comparison unavailable "
                  f"(worker rc={completed.get(mode)}, report={row['reason']!r})", flush=True)

    save(directory / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary {directory / 'summary.csv'}", flush=True)
    return all_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=BATCHES, action='append',
                        help='repeat to select a subset, or omit to run all five')
    parser.add_argument('--comparison-dry-run', action='store_true')
    parser.add_argument('--prefetch-trace', action='store_true',
                        help='include a separate diagnostic replay for first-prefetch doorbell lead time')
    args = parser.parse_args()
    batches = args.batch if args.batch is not None else BATCHES
    if len(set(batches)) != len(batches):
        parser.error('Each batch can appear only once')
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', ROOT / 'third_party/KVAF')).resolve()
    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr28_unified_score')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr28_unified_score.', dir=out_base))
    cases = {}
    for batch in batches:
        arguments = bench_arguments(kvaf, (batch,))
        if args.prefetch_trace:
            arguments[0] = 'diagnose'
            arguments.extend(('--level', '2'))
        cases[batch] = dict(arguments=arguments, completed={})
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_base=MAIN,
                    expected_kvaf=KVAF, cases=cases, variants=MODES, local_counts=False,
                    prefetch_trace=args.prefetch_trace)
    save(output / 'manifest.json', manifest)
    print(f"MR28 unified Score comparison {output}", flush=True)
    if args.comparison_dry_run:
        print(f"Planned batches {list(batches)}, modes {list(MODES)}, miss 10%, pf512.", flush=True)
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f"Use the unified KVAF checkout at {KVAF}")
    from execution import launch
    from workloads import configuration

    passed = True
    try:
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        for batch, case in cases.items():
            directory = output / f'bs{batch}'
            directory.mkdir()
            completed = case['completed']
            for mode in MODES:
                config = configuration(case['arguments'])
                config.mr25_handoff = mode
                config.mr28_unified_score = mode
                config.handoff_build_dir = output / 'build'
                print(f"Running BS{batch} {mode}, local_counts OFF", flush=True)
                completed[mode] = launch(config, directory / mode)
                save(output / 'manifest.json', manifest)
                if completed[mode]:
                    break
            passed = summarize(directory, batch, completed) and passed
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
