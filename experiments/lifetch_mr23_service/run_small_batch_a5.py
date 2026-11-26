#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one change at a time on the native A5 P2 fixture."""

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from run_mr25_handoff import KVAF, MAIN, ROOT, bench_arguments, collect, git, read, save
from run_mr28_baseline_msprof import capture_route
from run_mr28_dedicated_service import write_summary
from kvip.operators.lifetch.geometry import li_geometry
from small_batch import (
    COLD_SCORE_PARTS, MODES, SUPPORTED_BATCHES, THREE_WINDOW_MODES, TRACE_ROUTES, WINDOW_SCOPE_PHASES,
)
from terminal_handoff import CHECKED_KERNELS, COLD_GM_STAGE_FIELDS, STAGES
from workloads import configuration


FOLLOW_UP_ARMS = {1: ('p2', 'cold'), 8: ('p2', 'windows')}
WINDOW_PROBES = {'bs1-p8': (1, ('cold_p8', 'cold_p8_bulk')), 'bs2-w3': (2, ('windows', 'windows_early'))}
DIRECTION1_MODES = {1: 'cold', 2: 'cold', 4: 'cold', 8: 'windows', 16: 'windows', 32: 'windows'}
DIRECTION1_MISSES = (5, 10, 15, 20, 25, 30)
NATIVE_BATCH_LIMIT = 16


def plan(args, kvaf):
    probe = WINDOW_PROBES.get(args.window_probe)
    default_batches = (probe[0],) if probe else DIRECTION1_MODES if args.direction1_matrix else FOLLOW_UP_ARMS
    batches = tuple(dict.fromkeys(args.batch or default_batches))
    misses = tuple(dict.fromkeys(args.miss_percent or (DIRECTION1_MISSES if args.direction1_matrix else (10,))))
    selected = set(probe[1]) if probe else set(args.variant or MODES) | {'p2'}
    for batch in batches:
        physical = min(batch, NATIVE_BATCH_LIMIT)
        waves = batch // physical
        if waves > 1 and not args.group_bs32:
            continue
        if probe:
            modes = probe[1]
        elif args.direction1_matrix:
            modes = ('p2', DIRECTION1_MODES[batch])
        else:
            modes = FOLLOW_UP_ARMS[batch] if args.follow_up else MODES
        service_aics = li_geometry(physical, intermediate_fetches=4, aic_count=32)['service_aics_per_window']
        iterations = 1 if args.trace else args.iterations
        for miss in misses:
            for mode in modes:
                if mode not in selected or physical not in SUPPORTED_BATCHES[mode]:
                    continue
                requested = 3 if mode in THREE_WINDOW_MODES else 4
                geometry = li_geometry(physical, intermediate_fetches=requested, aic_count=32,
                                       service_aics=service_aics)
                effective = geometry['intermediate_fetches']
                arguments = bench_arguments(kvaf, (physical,))
                for key, value in (
                    ('--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986')),
                    ('--miss-percent', miss),
                    ('--warmup', args.warmup), ('--iterations', waves if waves > 1 else iterations),
                    ('--intermediate-fetches', requested), ('--timeout-seconds', args.timeout_seconds),
                ):
                    arguments[arguments.index(key) + 1] = str(value)
                arguments.extend(('--prefetch-window-limit', '128'))
                measurement = 'serial_bs16x2_graph_event' if waves > 1 else 'native'
                parts = COLD_SCORE_PARTS.get(mode, 2)
                yield dict(batch=batch, physical_batch=physical, waves=waves, miss_percent=miss,
                           measurement_kind=measurement, mode=mode, requested_windows=requested, effective_windows=effective,
                           service_aics=service_aics, cold_score_parts=parts, cold_score_groups=physical * parts,
                           arguments=arguments, kernels=CHECKED_KERNELS[f'small_{mode}'])


def window_scopes(source, batch, windows):
    scopes = {}
    for event in source['traceEvents']:
        args = event.get('args', {})
        phase = args.get('phase_id')
        if (event.get('ph') != 'X' or args.get('kernel_id') != 22 or args.get('core_type') != 'AIV'
                or phase not in WINDOW_SCOPE_PHASES):
            continue
        window, lane = args['window_epoch'], args['service_lane']
        if lane not in range(batch) or window not in range(windows):
            continue
        name = WINDOW_SCOPE_PHASES[phase]
        key = (window, lane, name)
        if key in scopes:
            raise ValueError(f'Duplicate scope {key}')
        scopes[key] = dict(window=window, request=lane, scope=name, source_us=event['dur'],
                           begin_tick=args['begin_tick'], end_tick=args['end_tick'])
    return scopes


def trace_arm(output, case, config, profiler):
    args = SimpleNamespace(batch=case['batch'], miss_percent=case['miss_percent'])
    case_output = output / f'bs{case["batch"]}' / case['mode']
    case_output.mkdir(parents=True)
    manifest = dict(**case, status='RUNNING')
    try:
        capture = capture_route(args, case_output, config, manifest, profiler,
                                cold_route=TRACE_ROUTES[case['mode']])
        report = capture['report']
        if (report.get('small_batch_mode') != case['mode'] or report['windows'] != case['effective_windows']
                or report['cold_parts'] != case['cold_score_parts']
                or report['prefetch_window_limit'] != 128 or report['service_lanes'] != 2 * case['service_aics']):
            raise ValueError('Captured source geometry does not match the trial plan')
        probe = case['window_probe']
        scopes = window_scopes(capture['source'], case['batch'], report['windows']) if probe else {}
        if probe:
            names = [(w, b, 'window_ready') for w in range(report['windows']) for b in range(case['batch'])]
            names += [(w, b, 'previous_wait') for w in range(1, report['windows']) for b in range(case['batch'])]
            if case['mode'] != 'windows_early' and any(key[2] == 'early_header_enqueue' for key in scopes):
                raise ValueError('Unexpected read-ahead in the control or cursor probe')
            if probe == 'bs1-p8':
                names += [(w, 0, 'p8_cursor') for w in range(report['windows'])]
                if case['mode'] == 'cold_p8_bulk':
                    names += [(w, 0, 'bulk_cursor_mte2') for w in range(report['windows'])]
                elif any(key[2] == 'bulk_cursor_mte2' for key in scopes):
                    raise ValueError('Bulk cursor scope appeared in the scalar control')
            if any(key not in scopes for key in names):
                raise ValueError('Incomplete window probe source-scope coverage')
        manifest['status'] = 'PASS'
        return dict(
            variant=case['mode'], batch=case['batch'], accepted=True, windows=report['windows'],
            source=manifest['business_trace'], msprof=manifest['msprof_business_trace'],
            stage_report=manifest['stage_report'], fixture_sha256=report['fixture_sha256'],
            library_sha256=report['library_sha256'], requests=report['requests'],
            resident_seed_sha256=report['resident_seed_sha256'],
            window_scopes=list(scopes.values()),
            native_stages=capture['intervals'],
            early_header_enqueues=sum(key[2] == 'early_header_enqueue' for key in scopes),
            window_scope_count=len(scopes),
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
    physical = case['physical_batch']
    miss = case['miss_percent']
    row = collect(case_output, config.mr25_handoff, physical, stage_fields=fields, miss_percent=miss)
    expected_parts = case['cold_score_parts']
    if (code or not row['accepted'] or row.get('available_aiv') != 64 or row.get('effective_aic') != 32
            or row.get('windows') != case['effective_windows'] or row.get('service_lanes') != case['service_aics'] * 2
            or row.get('cold_score_parts') != expected_parts or row.get('small_batch_mode') != case['mode']
            or row.get('prefetch_window_limit') != 128):
        raise RuntimeError(f'{case["mode"]} did not complete as planned. See {case_output / config.mr25_handoff}')
    miss_dir = case_output / config.mr25_handoff / f'miss{miss:g}'
    result = read(miss_dir / f'case-b{physical}' / 'fused/path_result.json')
    row['per_call_requests'] = result['pipeline_counts']
    row['variant'] = case['mode']
    fixture = read(miss_dir / 'fixture/fixture.json')['full_chain_fixture']
    if fixture['requested_miss_percent'] != miss:
        raise ValueError('Fixture miss rate disagrees with the trial plan')
    for q, rate in enumerate(fixture['actual_miss_percent_per_query']):
        row[f'fixture_miss_percent_q{q}'] = rate
    if case['waves'] > 1:
        (timing,) = result['rounds'].values()
        if timing['graph_calls'] != case['waves'] or not timing['graph_event_us'] > 0:
            raise ValueError('Grouped case requires two checked BS16 calls in one device graph')
        row['native_subcall_kv_ready_us'] = row['kv_ready_us']
        row['kv_ready_us'] = row['fused_event_us'] = timing['graph_event_us']
        row.update({f'{field}_per_group': sum(record[field] for record in row['per_call_requests'])
                    for field in ('prefetch_requests', 'final_get_requests', 'total_get_requests')})
    row['batch'] = case['batch']
    timing_label, get_unit = ('2xBS16 graph', 'group') if case['waves'] > 1 else ('KV ready', 'call')
    early, final = (row[f'{field}_per_{get_unit}'] for field in ('prefetch_requests', 'final_get_requests'))
    print(f'BS{case["batch"]} {case["mode"]}: {timing_label} {row["kv_ready_us"]:.2f} µs, '
          f'prefetch / final Get per {get_unit} {early:.2f} / {final:.2f}', flush=True)
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
    if all('prefetch_tokens_in_submit_order_by_window' in row for row in base + candidate):
        after['same_window_prefetch_post_order'] = all(
            left['prefetch_tokens_in_submit_order_by_window'] == right['prefetch_tokens_in_submit_order_by_window']
            for left, right in zip(base, candidate))
    if after['variant'] == 'topk' and not after['resident_seed_match']:
        raise ValueError('Resident Top2048 seed differs from P2')


def write_window_probe_comparison(output, before, after):
    base = {(s['window'], s['request'], s['scope']): s for s in before['window_scopes']}
    deltas = []
    for scope in after['window_scopes']:
        key = (scope['window'], scope['request'], scope['scope'])
        if key in base:
            old = base[key]['source_us']
            deltas.append(dict(window=key[0], request=key[1], scope=key[2],
                               base_us=old, optimized_us=scope['source_us'], delta_us=scope['source_us'] - old))
    save(output / 'window_probe_comparison.json', {
        'control': before, 'experiment': after, 'source_scope_deltas': deltas,
        'source_scopes_unit': 'calibrated device microseconds',
    })


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
    summary = [{key: value for key, value in row.items()
                if key not in ('requests', 'per_call_requests', 'window_scopes', 'native_stages')} for row in rows]
    write_summary(output, summary)
    grouped = [row for row in summary if row.get('measurement_kind') == 'serial_bs16x2_graph_event']
    if grouped:
        directory = output / 'bs32-grouped'
        directory.mkdir(exist_ok=True)
        write_summary(directory, grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=tuple(DIRECTION1_MODES), action='append')
    parser.add_argument('--miss-percent', type=int, choices=DIRECTION1_MISSES, action='append',
                        help='select one miss rate in the direction-1 matrix')
    parser.add_argument('--group-bs32', action='store_true',
                        help='measure BS32 as one graph containing two serial BS16 forwards')
    route = parser.add_mutually_exclusive_group()
    route.add_argument('--variant', choices=MODES, action='append', help='include one single-change arm; P2 is automatic')
    route.add_argument('--follow-up', action='store_true', help='BS1 P2/P4 and BS8 W4/W3 only')
    route.add_argument('--direction1-matrix', action='store_true', help='sweep first-direction P4 or W3 against native P2')
    route.add_argument('--window-probe', choices=WINDOW_PROBES, help='run one fixed-geometry A5 A/B pair')
    parser.add_argument('--repeats', type=int, default=1, help='repeat the matrix with the same bridge and independent workers')
    parser.add_argument('--trace', action='store_true', help='capture one checked msprof replay per arm')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.window_probe and (args.batch or args.miss_percent or args.group_bs32):
        parser.error('--window-probe fixes the batch, miss rate and windows')
    if not args.direction1_matrix and (args.miss_percent or args.group_bs32
                                       or any(batch not in FOLLOW_UP_ARMS for batch in args.batch or ())):
        parser.error('Expanded batch sizes, miss rates and grouped BS32 require --direction1-matrix')
    if args.group_bs32 and args.trace:
        parser.error('Grouped BS32 uses a two-call graph Event, incompatible with the single-call trace')
    if args.warmup is None:
        args.warmup = 0 if args.trace else 1
    if args.warmup < 0 or min(args.iterations, args.repeats, args.timeout_seconds) < 1:
        parser.error('warmup must be nonnegative; iterations, repeats and timeout must be positive')

    a5 = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5 / 'KVAF-unified-813f1750'))
    matrix = list(plan(args, kvaf))
    cases = [dict(case, repeat=repeat, window_probe=args.window_probe)
             for repeat in range(1, args.repeats + 1) for case in matrix]
    unsupported = []
    if args.direction1_matrix and 32 in (args.batch or DIRECTION1_MODES) and not args.group_bs32:
        unsupported = [dict(batch=32, miss_percent=miss, accepted=False, status='UNSUPPORTED_NATIVE',
                            reason='Full-chain LIFetch supports up to BS16; use --group-bs32 for two serial BS16 calls')
                       for miss in dict.fromkeys(args.miss_percent or DIRECTION1_MISSES)]
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=normal')),
                    kvaf_commit=KVAF, measurement='marked_msprof' if args.trace else 'graph_event',
                    follow_up=args.follow_up, direction1_matrix=args.direction1_matrix,
                    window_probe=args.window_probe, repeats=args.repeats,
                    scopes=dict(queries=4, history=131072, resident=16384,
                                miss_percent=sorted({case['miss_percent'] for case in cases + unsupported}),
                                aic=32, aiv=64, prefetch_limit=512, per_window_limit=128, rank_start=1024),
                    cases=cases, unsupported_scenarios=unsupported,
                    bs1_windows='P2 already uses effective W2; fewer windows leave the sequence-split route')
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not cases:
        parser.error('No runnable scenario selected; BS32 requires --group-bs32')
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the pinned KVAF checkout at {KVAF}')
    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof')) if args.trace else None
    if args.trace and not profiler:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    output = Path(os.environ['KVIP_AB_OUTPUT']).resolve()
    print(f'Small batch A5 comparison: {output}', flush=True)
    rows, bases = list(unsupported), {}
    baseline = WINDOW_PROBES[args.window_probe][1][0] if args.window_probe else 'p2'
    manifest['status'] = 'RUNNING'
    try:
        for case in cases:
            config = configuration(case['arguments'])
            config.small_batch_mode = case['mode']
            config.small_batch_service_aics = case['service_aics']
            config.handoff_build_dir = output / 'bench_bridge'
            config.msprof_build_dir = output / 'trace_bridge'
            repeat = case['repeat']
            run_dir = output / f'repeat-{repeat:02d}' if args.repeats > 1 else output
            miss = case['miss_percent']
            if args.direction1_matrix:
                run_dir = run_dir / f'miss{miss:g}'
            print(f"Round {repeat}/{args.repeats}, BS{case['batch']} miss{miss:g} "
                  f"{case['mode']} W{case['effective_windows']}", flush=True)
            row = trace_arm(run_dir, case, config, profiler) if args.trace else perf_arm(run_dir, case, config)
            row.update(repeat=repeat, miss_percent=miss, physical_batch=case['physical_batch'],
                       waves=case['waves'],
                       measurement_kind=case['measurement_kind'])
            rows.append(row)
            pair = (repeat, row['batch'], miss)
            if row['variant'] == baseline:
                bases[pair] = row
            else:
                before = bases[pair]
                if any(row[key] != before[key] for key in ('fixture_sha256', 'library_sha256')):
                    raise ValueError('Pair used different fixture or bridge')
                if args.trace:
                    trace_compare(before, row)
                    if args.window_probe:
                        if not row['resident_seed_match']:
                            raise ValueError('Resident seed differs within the fixed-topology pair')
                        write_window_probe_comparison(run_dir, before, row)
                else:
                    compare_request_counts(before, row)
                    row['raw_kv_ready_delta_us'] = row['kv_ready_us'] - before['kv_ready_us']
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
