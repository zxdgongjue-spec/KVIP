#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One isolated A5 worker per arm of the fixed batch-pipeline comparison."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MODES = ('native', 'grouped_serial', 'grouped_pipeline')
FIXTURE = dict(batch=16, queries=4, history=131072, resident=16384, index_heads=32,
               miss_percent=10, windows=4, prefetch_tokens=512, prefetch_rank_start=1024, seed=43)


def request_totals(report, *, placement=False):
    counts = []
    for invocation in report['get_requests']:
        if placement:
            counts.append([(row['prefetch'], row['final_get'], row['prefetch_per_window']) for row in invocation])
        else:
            counts.append([row['prefetch'] + row['final_get'] for row in invocation])
    return counts


def comparison(reports):
    rows = []
    baseline = reports.get('native')
    control = reports.get('grouped_serial')
    for mode in MODES:
        report = reports.get(mode)
        if report is None:
            rows.append(dict(mode=mode, correctness='NO_CHECKED_REPORT'))
            continue
        p50 = report['eager_kv_ready']['p50_us']
        same_fixture = baseline is not None and all(report[key] == baseline[key] for key in (
            'fixture_sha256', 'library_sha256', 'geometry', 'prefetch_tokens', 'prefetch_rank_start', 'soc', 'aics'))
        same_grouped_fixture = control is not None and all(report[key] == control[key] for key in (
            'fixture_sha256', 'library_sha256', 'geometry', 'prefetch_tokens', 'prefetch_rank_start', 'soc', 'aics'))
        calls = report['get_requests']

        def per_call(field):
            return sum(row[field] for call in calls for row in call) / len(calls)

        early, final = per_call('prefetch'), per_call('final_get')
        rows.append(dict(
            mode=mode, correctness=report['correctness'], p50_us=p50,
            p90_us=report['eager_kv_ready']['p90_us'], ratio_to_native=(
                p50 / baseline['eager_kv_ready']['p50_us'] if same_fixture else None),
            ratio_to_grouped_serial=(
                p50 / control['eager_kv_ready']['p50_us'] if same_grouped_fixture else None),
            prefetch_per_call=early, final_get_per_call=final, total_get_per_call=early + final,
            same_request_totals_as_native=(
                request_totals(report) == request_totals(baseline) if same_fixture else None),
            same_placement_as_native=(
                request_totals(report, placement=True) == request_totals(baseline, placement=True)
                if same_fixture else None),
            same_request_totals_as_grouped_serial=(
                request_totals(report) == request_totals(control) if same_grouped_fixture else None),
            same_placement_as_grouped_serial=(
                request_totals(report, placement=True) == request_totals(control, placement=True)
                if same_grouped_fixture else None),
            event_envelope_triple_overlap_us=report['event_envelope_triple_overlap_us'],
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dry-run', action='store_true', help='print the schedule without CANN or KVAF')
    parser.add_argument('--trace', action='store_true', help='collect one separate untimed CPU/NPU profiler trace per arm')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--timeout-seconds', type=int, default=900, help='limit per isolated worker')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')
    plan = dict(fixture=FIXTURE, modes=MODES, warmup=args.warmup, iterations=args.iterations,
                trace=args.trace, timeout_seconds=args.timeout_seconds,
                groups=[[0, 4], [4, 8], [8, 12], [12, 16]],
                queue_slot_offsets=[0, 24, 48], score_streams=1, service_terminal_slots=3,
                group_slot_sequence=[0, 1, 2, 0], timing='eager device events with host enqueue gaps')
    if args.comparison_dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    for name in ('KVIP_KVAF_ROOT', 'KVIP_KVAF_SERVER', 'KVIP_AB_OUTPUT'):
        if not os.environ.get(name):
            parser.error(f'{name} must be set inside the A5 container')
    for path in (ROOT / 'python', ROOT / 'tests/operators/lifetch/support'):
        sys.path.insert(0, str(path))
    from execution import launch
    from reporting import atomic_json
    from workloads import configuration

    config = configuration([
        'bench', '--transport', 'urma', '--input', 'synthetic',
        '--device', os.environ.get('KVIP_A5_DEVICE', '0'), '--batches', str(FIXTURE['batch']),
        '--kvaf-root', os.environ['KVIP_KVAF_ROOT'], '--kvaf-server', os.environ['KVIP_KVAF_SERVER'],
        '--history', str(FIXTURE['history']), '--resident', str(FIXTURE['resident']),
        '--queries', str(FIXTURE['queries']), '--index-heads', str(FIXTURE['index_heads']),
        '--seed', str(FIXTURE['seed']), '--miss-percent', str(FIXTURE['miss_percent']),
        '--intermediate-fetches', str(FIXTURE['windows']), '--prefetch-tokens', str(FIXTURE['prefetch_tokens']),
        '--prefetch-rank-start', str(FIXTURE['prefetch_rank_start']), '--warmup', str(args.warmup),
        '--iterations', str(args.iterations), '--timeout-seconds', str(args.timeout_seconds),
    ])
    output = Path(os.environ['KVIP_AB_OUTPUT']).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='lifetch-batch-pipeline-', dir=output))
    print(f'Run output: {run}', flush=True)
    atomic_json(run / 'plan.json', plan)
    config.batch_pipeline_build = run / 'build'
    config.batch_pipeline_trace = args.trace
    reports, outcomes = {}, {}
    fields = [
        'mode', 'correctness', 'p50_us', 'p90_us', 'ratio_to_native', 'ratio_to_grouped_serial',
        'prefetch_per_call', 'final_get_per_call', 'total_get_per_call', 'same_request_totals_as_native',
        'same_placement_as_native', 'same_request_totals_as_grouped_serial',
        'same_placement_as_grouped_serial', 'event_envelope_triple_overlap_us',
    ]
    for mode in MODES:
        config.batch_pipeline_mode = mode
        print(f'Running {mode}', flush=True)
        outcomes[mode] = launch(config, run / mode)
        path = run / mode / 'miss10' / 'case-b16' / 'pipeline.json'
        if outcomes[mode] == 0 and path.is_file():
            report = json.loads(path.read_text())
            if report['execution'] == 'COMPLETED' and report['correctness'] == 'PASS':
                reports[mode] = report
        rows = comparison(reports)
        atomic_json(run / 'summary.json', dict(plan=plan, rows=rows, worker_exit_codes=outcomes))
        with (run / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if mode in reports:
            row = next(row for row in rows if row['mode'] == mode)
            print(f"{mode} KV ready p50 {row['p50_us']:.2f} us, "
                  f"prefetch/Get {row['prefetch_per_call']:.2f}/{row['final_get_per_call']:.2f}", flush=True)
    print(f'Summary: {run / "summary.csv"}', flush=True)
    return int(len(reports) != len(MODES))


if __name__ == '__main__':
    raise SystemExit(main())
