#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair the main MR25 Terminal with local TopK-to-Service counts."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / 'tests/operators/lifetch/support'
sys.path[:0] = [str(SUPPORT), str(ROOT / 'python')]
from terminal_handoff import COUNT_COLUMNS, STAGES  # noqa: E402

MAIN = 'd6cfd392f722dd3f96ef6e05175f503aac511dbb'
KVAF = '813f1750a50b7e6c92d5f5397f2fbc05befef6da'
BATCHES = (1, 8, 16)
VARIANTS = ('main', 'local_counts')


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def apply_group_timing(row, timing, waves):
    """Report one device graph containing serial, independently checked forwards."""
    event_us = timing.get('graph_event_us')
    if (timing.get('measured_graphs') != 1 or timing.get('graph_calls') != waves
            or not isinstance(event_us, (int, float))
            or not math.isfinite(event_us) or event_us <= 0):
        raise ValueError(f'Grouped case requires one measured graph with {waves} calls')
    row['native_subcall_kv_ready_us'] = row['kv_ready_us']
    row['kv_ready_us'] = row['fused_event_us'] = event_us
    for field in COUNT_COLUMNS:
        row[f'{field}_per_group'] = sum(request[field] for request in row['per_call_requests'])


def collect(directory, mode, batch, *, stage_fields=STAGES, miss_percent=10):
    row = dict(variant=mode, batch=batch, miss_percent=miss_percent, accepted=False,
               reason='Missing completed report')
    folder = directory / mode / f'miss{miss_percent:g}' / f'case-b{batch}'
    if not (folder / 'report.json').is_file():
        return row
    case = read(folder / 'report.json')
    result = read(folder / 'fused/path_result.json')
    timing = result['all_rounds']
    calls = timing['graph_calls'] * timing['measured_graphs']
    stages, counts = result['handoff_stage_samples'], result['pipeline_counts']
    identities = {(r['source_round'], r['invocation'], r['request']) for r in counts}
    valid = (case['correctness'] == result['correctness'] == 'PASS'
             and case['execution'] == 'COMPLETED' and timing['event_mean_us'] > 0
             and len(stages) == calls and len(counts) == len(identities) == calls * batch)
    route = case['route']
    valid &= (route['mode'] == mode and route['score_parts'] == 2
              and route['score_blocks'] == 2 * batch and route['terminal_tasks'] == 4 * batch)
    row.update(accepted=valid, reason='' if valid else 'Correctness, route or sample coverage differs',
               kv_ready_us=timing['event_mean_us'], fused_event_us=timing['event_mean_us'],
               profiler_p50_us=timing['p50_us'], li_us=timing['stage_mean_us']['li'],
               inter_kernel_gap_us=timing['inter_kernel_gap_mean_us'],
               graph_edges_us=timing['outside_invocations_mean_us'],
               **{key: statistics.mean(r[key] for r in stages) for key in stage_fields},
               **{key + '_per_call': sum(r[key] for r in counts) / calls for key in COUNT_COLUMNS})
    row.update({key: value for key, value in route.items() if key not in ('mode', 'library')})
    row.update(prefetch_observed=row['prefetch_requests_per_call'] > 0,
               fixture_sha256=case['fixture_sha256'], library=route['library'])
    return row


def summarize(directory, completed):
    rows = []
    paired_keys = ('fixture_sha256', 'soc', 'effective_aic', 'windows', 'score_parts', 'service_lanes')
    for batch in BATCHES:
        baseline, candidate = (collect(directory, variant, batch) for variant in VARIANTS)
        paired = (all(completed.get(v) == 0 for v in VARIANTS)
                  and baseline['accepted'] and candidate['accepted']
                  and all(baseline[k] == candidate[k] for k in paired_keys))
        for row in (baseline, candidate):
            row['paired'] = paired
            row['over_main'] = row['kv_ready_us'] / baseline['kv_ready_us'] if paired else None
            row['both_posted_prefetch'] = (
                baseline['prefetch_observed'] and candidate['prefetch_observed'] if paired else None
            )
            rows.append(row)
        if paired:
            print(f"BS{batch} KV ready {baseline['kv_ready_us']:.2f} -> "
                  f"{candidate['kv_ready_us']:.2f} us ({candidate['over_main']:.4f}x)", flush=True)
    save(directory / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    print('Summary: ' + str(directory / 'summary.csv'), flush=True)
    return all(row['paired'] for row in rows)


def bench_arguments(kvaf, batches):
    return ['bench', '--transport', 'urma', '--input', 'synthetic',
             '--device', os.environ.get('KVIP_A5_DEVICE', '0'), '--kvaf-root', str(kvaf),
             '--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28985'),
             '--batches', ','.join(map(str, batches)), '--miss-percent', '10',
             '--queries', '4', '--history', '131072', '--resident', '16384',
             '--index-heads', '32', '--seed', '43', '--prefetch-tokens', '512',
             '--prefetch-rank-start', '1024', '--intermediate-fetches', '4',
             '--warmup', '2', '--iterations', '10', '--timeout-seconds', '1800']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dry-run', action='store_true')
    args = parser.parse_args()
    output = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr25_handoff')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr25_handoff.', dir=output))
    print('MR25 handoff comparison: ' + str(output), flush=True)

    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', ROOT / 'third_party/KVAF')).resolve()
    bench = bench_arguments(kvaf, BATCHES)
    completed = {}
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
                    expected_kvaf=KVAF, arguments=bench, variants=dict(main='OFF', local_counts='ON'),
                    completed=completed)
    save(output / 'manifest.json', manifest)
    if args.comparison_dry_run:
        print('Planned 2 variants x 3 batch sizes (6 cases).', flush=True)
        return 0

    from execution import launch
    from workloads import configuration

    try:
        if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
            raise ValueError(f'Use the built KVAF checkout at {KVAF}')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', MAIN, 'HEAD'], check=True)
        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        for variant in VARIANTS:
            config = configuration(bench)
            config.mr25_handoff = variant
            config.handoff_build_dir = output / 'build' / variant
            print(f'Running {variant} with KVIP_SERVICE_LOCAL_COUNTS={manifest["variants"][variant]}',
                  flush=True)
            completed[variant] = launch(config, output / variant)
            save(output / 'manifest.json', manifest)
            if completed[variant]:
                break
        passed = summarize(output, completed)
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
