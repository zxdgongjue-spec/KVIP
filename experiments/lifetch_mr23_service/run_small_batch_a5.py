#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one change at a time on the A5 BS1/BS8 P2 fixture."""

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from run_mr25_handoff import KVAF, MAIN, ROOT, bench_arguments, collect, git, read, save
from run_mr28_baseline_msprof import capture_route
from run_mr28_dedicated_service import write_summary
from small_batch import MODES, TRACE_ROUTES
from terminal_handoff import CHECKED_KERNELS, COLD_GM_STAGE_FIELDS, STAGES
from workloads import configuration


def plan(args, kvaf):
    batches = tuple(dict.fromkeys(args.batch or (1, 8)))
    selected = set(args.variant or MODES)
    for batch in batches:
        for mode in MODES:
            if (mode != 'p2' and mode not in selected) or (batch == 1 and mode == 'windows'):
                continue
            requested = 3 if mode == 'windows' else 4
            effective = min(requested, batch * 2)
            service_aics = batch * 2 // min(4, batch * 2)
            arguments = bench_arguments(kvaf, (batch,))
            for key, value in (
                ('--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986')),
                ('--warmup', args.warmup), ('--iterations', 1 if args.trace else args.iterations),
                ('--intermediate-fetches', requested), ('--timeout-seconds', args.timeout_seconds),
            ):
                arguments[arguments.index(key) + 1] = str(value)
            arguments.extend(('--prefetch-window-limit', '128'))
            yield dict(batch=batch, mode=mode, requested_windows=requested, effective_windows=effective,
                       service_aics=service_aics, cold_score_groups=batch * (4 if mode == 'cold' else 2),
                       arguments=arguments, kernels=CHECKED_KERNELS[f'small_{mode}'])


def trace_arm(output, case, config, profiler):
    args = SimpleNamespace(batch=case['batch'], miss_percent=10)
    case_output = output / f'bs{case["batch"]}' / case['mode']
    case_output.mkdir(parents=True)
    manifest = dict(**case, status='RUNNING')
    try:
        capture = capture_route(args, case_output, config, manifest, profiler,
                                cold_route=TRACE_ROUTES[case['mode']])
        report = capture['report']
        if (report.get('small_batch_mode') != case['mode'] or report['windows'] != case['effective_windows']
                or report['prefetch_window_limit'] != 128 or report['service_lanes'] != 2 * case['service_aics']):
            raise ValueError('Captured source geometry does not match the trial plan')
        manifest['status'] = 'PASS'
        return dict(
            variant=case['mode'], batch=case['batch'], accepted=True, windows=report['windows'],
            source=manifest['business_trace'], msprof=manifest['msprof_business_trace'],
            stage_report=manifest['stage_report'], fixture_sha256=report['fixture_sha256'],
            library_sha256=report['library_sha256'], requests=report['requests'],
            resident_seed_sha256=report['resident_seed_sha256'],
            native_first_kernel_to_terminal_us=capture['intervals'][-1]['end_us'] - capture['intervals'][0]['begin_us'],
            prefetch_requests_per_call=report['total_prefetch'],
            final_get_requests_per_call=report['total_final_get'],
            total_get_requests_per_call=report['total_prefetch'] + report['total_final_get'],
        )
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(case_output / 'manifest.json', manifest)


def perf_arm(output, case, config):
    from execution import launch

    case_output = output / f'bs{case["batch"]}'
    config.mr25_handoff = f'small_{case["mode"]}'
    code = launch(config, case_output / config.mr25_handoff)
    fields = COLD_GM_STAGE_FIELDS.get(config.mr25_handoff, STAGES)
    row = collect(case_output, config.mr25_handoff, case['batch'], stage_fields=fields)
    expected_parts = 4 if case['mode'] == 'cold' else 2
    if (code or not row['accepted'] or row.get('available_aiv') != 64 or row.get('effective_aic') != 32
            or row.get('windows') != case['effective_windows'] or row.get('service_lanes') != case['service_aics'] * 2
            or row.get('cold_score_parts') != expected_parts or row.get('small_batch_mode') != case['mode']
            or row.get('prefetch_window_limit') != 128):
        raise RuntimeError(f'{case["mode"]} did not complete as planned. See {case_output / config.mr25_handoff}')
    result_path = case_output / config.mr25_handoff / 'miss10' / f'case-b{case["batch"]}' / 'fused/path_result.json'
    result = read(result_path)
    row['per_call_requests'] = result['pipeline_counts']
    row['variant'] = case['mode']
    print(f'BS{case["batch"]} {case["mode"]}: KV ready {row["kv_ready_us"]:.2f} µs, '
          f'prefetch / final Get {row["prefetch_requests_per_call"]:.2f} / {row["final_get_requests_per_call"]:.2f}', flush=True)
    return row


def trace_compare(before, after):
    base = before['requests']
    candidate = after['requests']
    if any(left['request'] != right['request'] for left, right in zip(base, candidate, strict=True)):
        raise ValueError('Trace request IDs are not aligned')
    after.update(
        same_window_prefetch_tokens=all(left['prefetch_tokens_by_window'] == right['prefetch_tokens_by_window']
                                        for left, right in zip(base, candidate)),
        same_final_get_tokens=all(left['final_get_tokens'] == right['final_get_tokens']
                                  for left, right in zip(base, candidate)),
        same_final_get_total=all(left['final_get_requests'] == right['final_get_requests']
                                 for left, right in zip(base, candidate)),
        resident_seed_match=before['resident_seed_sha256'] == after['resident_seed_sha256'],
        requests_balanced=sum(max(req['final_posted_counts']) < max(req['final_original_owner_counts'])
                              for req in candidate),
    )
    if after['variant'] == 'topk' and not after['resident_seed_match']:
        raise ValueError('Resident Top2048 seed differs from P2')


def compare_request_counts(before, after):
    def keyed(row):
        return {(r['source_round'], r['invocation'], r['request']): r for r in row['per_call_requests']}

    base, candidate = keyed(before), keyed(after)
    if base.keys() != candidate.keys():
        raise ValueError('Measured request and invocation coverage differs from P2')
    keys = ('prefetch_requests', 'final_get_requests', 'total_get_requests')
    matches = {key: all(base[i][key] == candidate[i][key] for i in base) for key in keys}
    after['get_count_comparison'] = ('same_early_and_final_counts' if all(matches.values())
                                     else 'same_total_counts' if matches['total_get_requests']
                                     else 'different_request_totals')
    after['raw_kv_ready_ratio'] = after['kv_ready_us'] / before['kv_ready_us']
    if all(matches.values()):
        after['matched_get_count_kv_ready_ratio'] = after['raw_kv_ready_ratio']


def summarize(output, rows):
    write_summary(output, [{key: value for key, value in row.items()
                            if key not in ('requests', 'per_call_requests')} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1, 8), action='append')
    parser.add_argument('--variant', choices=MODES, action='append', help='include one single-change arm; P2 is automatic')
    parser.add_argument('--trace', action='store_true', help='capture one checked msprof replay per arm')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = 0 if args.trace else 1
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')

    a5 = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5 / 'KVAF-unified-813f1750'))
    cases = list(plan(args, kvaf))
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=normal')),
                    kvaf_commit=KVAF, measurement='marked_msprof' if args.trace else 'graph_event',
                    scopes=dict(queries=4, history=131072, resident=16384, miss_percent=10,
                                aic=32, aiv=64, prefetch_limit=512, per_window_limit=128, rank_start=1024),
                    cases=cases, bs1_windows='P2 already uses effective W2; fewer windows leave the sequence-split route')
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the pinned KVAF checkout at {KVAF}')
    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof')) if args.trace else None
    if args.trace and not profiler:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    output = Path(os.environ['KVIP_AB_OUTPUT']).resolve()
    print(f'Small batch A5 comparison: {output}', flush=True)
    rows, bases = [], {}
    manifest['status'] = 'RUNNING'
    try:
        for case in cases:
            config = configuration(case['arguments'])
            config.small_batch_mode = case['mode']
            config.small_batch_service_aics = case['service_aics']
            config.handoff_build_dir = output / 'bench_bridge'
            config.msprof_build_dir = output / 'trace_bridge'
            print(f"BS{case['batch']} {case['mode']} W{case['effective_windows']}", flush=True)
            row = trace_arm(output, case, config, profiler) if args.trace else perf_arm(output, case, config)
            rows.append(row)
            if row['variant'] == 'p2':
                bases[row['batch']] = row
            else:
                before = bases[row['batch']]
                if any(row[key] != before[key] for key in ('fixture_sha256', 'library_sha256')):
                    raise ValueError('Pair used different fixture or bridge')
                if args.trace:
                    trace_compare(before, row)
                else:
                    compare_request_counts(before, row)
            summarize(output, rows)
        manifest['status'] = 'PASS'
        return 0
    except BaseException:
        manifest['status'] = 'FAILED'
        if rows:
            summarize(output, rows)
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    raise SystemExit(main())
