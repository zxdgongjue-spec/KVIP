#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one change at a time on the native A5 P2 fixture."""

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from run_mr25_handoff import KVAF, ROOT, apply_group_timing, bench_arguments, collect, git, read, save
from run_mr28_baseline_msprof import capture_route
from run_mr28_dedicated_service import write_summary
from kvip.operators.lifetch.geometry import li_geometry
from small_batch import (
    ADAPTIVE_ROUTES, FULL32_MODES, MODES, NATIVE_P2_BATCHES, P2_WINDOW_Q_WORKERS, P4_WINDOW_Q_WORKERS,
    PARTITIONED_MODES, SUPPORTED_BATCHES, TRACE_ROUTES, WINDOW_Q_BATCHES, WINDOW_Q_WORKERS,
    cold_score_parts, selected_windows,
)
from terminal_handoff import CHECKED_KERNELS, COLD_GM_STAGE_FIELDS, STAGES
from workloads import configuration


FOLLOW_UP_ARMS = {1: ('p2', 'cold'), 8: ('p2', 'windows')}
GRANULARITY_ARMS = {1: ('p2', 'cold', 'cold_p8'), 2: ('p2', 'windows')}
FULL32_ARMS = {batch: ('p2', 'cold_full') for batch in (1, 2, 8)}
WINDOW_ARMS = {batch: ('p2', 'windows', 'windows_w2') for batch in (2, 8)}
COLD_WINDOW_ARMS = {
    1: ('p2', 'cold', 'cold_p8', 'cold_full'),
    2: ('p2', 'cold', 'cold_full', 'windows_w2', 'cold_w2', 'cold_full_w2'),
    8: ('p2', 'cold_full', 'windows_w2', 'cold_full_w2'),
}
BS2_TRACE_BASELINES = {
    'parts': ('windows_w2',),
    'windows': ('cold',),
    'both': ('windows_w2', 'cold'),
}
BS2_TRACE_PREFIX = {'windows_w2': 'same_window_', 'cold': 'same_parts_w4_'}
ADAPTIVE_ARMS = {batch: ('p2', mode, 'adaptive') for batch, mode in ADAPTIVE_ROUTES.items()}
WINDOW_PROBES = {
    'bs1-p8': {1: ('cold_p8', 'cold_p8_bulk')},
    'bs2-w3': {2: ('windows', 'windows_early')},
}
WINDOW_SCOPE_PHASES = {
    151: 'previous_wait', 243: 'window_ready', 246: 'p8_cursor',
    247: 'early_header_enqueue', 250: 'bulk_cursor_mte2',
}
MAIN = 'df8ad2598e70e4b5e266b15028ab26c601a2bdf2'
DIRECTION1_MODES = {1: 'cold', 2: 'cold', 4: 'cold', 8: 'windows', 16: 'windows', 32: 'windows'}
DIRECTION1_MISSES = (5, 10, 15, 20, 25, 30)
NATIVE_BATCH_LIMIT = 16
WINDOW_Q_GROUPS = {20: (4, 5), 24: (8, 3), 32: (16, 2)}
BS16_W4_Q1_MISSES = (5, 10, 15)
W4_Q1_WIDE_MISSES = (*BS16_W4_Q1_MISSES, 20)
W4_Q1_ARMS = ('p2', 'p2_w4_q1', 'p2_w2_q1')


def balanced_arms(batch):
    # P4+ preserves the two native leader positions per request, even at 32 MIX.
    service_groups = (batch + 1) // 2
    owner_groups = 2 * batch
    arms = list(FULL32_ARMS[batch])
    for windows, pair in ((3, ('windows', 'cold_full_w3')), (2, ('windows_w2', 'cold_full_w2'))):
        if windows < min(4, owner_groups) and windows * service_groups <= owner_groups:
            arms.extend(pair)
    return tuple(arms)


def study_arms(args):
    if args.bs16_w4_q1:
        return {16: W4_Q1_ARMS}
    if args.w4_q1_wide:
        batches = NATIVE_P2_BATCHES if args.trace else (*NATIVE_P2_BATCHES, *WINDOW_Q_GROUPS)
        return {batch: ('p2', 'p2_w2_q1') if batch == 1 else W4_Q1_ARMS for batch in batches}
    if args.window_q_matrix:
        baseline = () if args.trace else ('p2',)
        if args.window_q_matrix != 'q1':
            arms = dict.fromkeys(WINDOW_Q_BATCHES, (*baseline, *P4_WINDOW_Q_WORKERS))
            arms[16] = (*baseline, *P2_WINDOW_Q_WORKERS)
            if args.window_q_matrix == 'wide':
                for batch, (physical, _) in WINDOW_Q_GROUPS.items():
                    arms[batch] = arms[physical]
            return arms
        return dict.fromkeys(WINDOW_Q_BATCHES, (*baseline, 'cold_w2', 'cold_w2_q1'))
    if args.bs2_window_q:
        return {2: tuple(P4_WINDOW_Q_WORKERS)}
    if args.adaptive_pw:
        return ADAPTIVE_ARMS
    if args.window_probe:
        return WINDOW_PROBES[args.window_probe]
    if args.bs2_trace_pair:
        return {2: (*BS2_TRACE_BASELINES[args.bs2_trace_pair], 'cold_w2')}
    if args.cold_window_matrix:
        return COLD_WINDOW_ARMS
    if args.full32 or args.balanced:
        return FULL32_ARMS
    if args.window_sweep:
        return WINDOW_ARMS
    if args.granularity_study:
        return GRANULARITY_ARMS


def plan(args, kvaf):
    study = study_arms(args)
    default_batches = (DIRECTION1_MODES if args.direction1_matrix else
                       study or FOLLOW_UP_ARMS)
    batches = tuple(dict.fromkeys(args.batch or default_batches))
    default_misses = (W4_Q1_WIDE_MISSES if args.w4_q1_wide else
                      BS16_W4_Q1_MISSES if args.bs16_w4_q1 else
                      DIRECTION1_MISSES if args.direction1_matrix else (10,))
    misses = tuple(dict.fromkeys(args.miss_percent or default_misses))
    selected = set(args.variant or (SUPPORTED_BATCHES if study else MODES)) | {'p2'}
    w4_comparison = args.bs16_w4_q1 or args.w4_q1_wide
    grouped_window_q = args.window_q_matrix == 'wide' or args.w4_q1_wide
    for batch in batches:
        physical = min(batch, NATIVE_BATCH_LIMIT)
        waves = batch // physical
        if grouped_window_q and batch in WINDOW_Q_GROUPS:
            physical, waves = WINDOW_Q_GROUPS[batch]
        elif waves > 1 and not args.group_bs32:
            continue
        if args.direction1_matrix:
            modes = ('p2', DIRECTION1_MODES[batch])
        elif args.balanced:
            modes = balanced_arms(batch)
        elif study:
            modes = study[batch]
        else:
            modes = FOLLOW_UP_ARMS[batch] if args.follow_up else MODES
        service_aics = li_geometry(physical, intermediate_fetches=4, aic_count=32)['service_aics_per_window']
        if args.balanced:
            service_aics = (physical + 1) // 2
        iterations = 1 if args.trace else args.iterations
        for miss in misses:
            for mode in modes:
                if mode not in selected or physical not in SUPPORTED_BATCHES[mode]:
                    continue
                native_windows = 2 if batch == 1 and (args.cold_window_matrix or args.adaptive_pw) else 4
                requested = selected_windows(mode, physical, native_windows)
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
                measurement = 'native'
                if waves > 1:
                    measurement = ('serial_equal_subbatches_graph_event' if grouped_window_q
                                   else 'serial_bs16x2_graph_event')
                parts = cold_score_parts(mode, physical)
                yield dict(batch=batch, physical_batch=physical, waves=waves, miss_percent=miss,
                           measurement_kind=measurement, mode=mode, requested_windows=requested, effective_windows=effective,
                           hot_score_parts=2, hot_score_aic=2 * physical, hot_score_aiv=4 * physical,
                           service_aics=service_aics, cold_score_parts=parts, cold_score_groups=physical * parts,
                           service_owner_groups=(service_aics * effective if args.window_q_matrix or w4_comparison
                                                 else 2 * physical),
                           service_group_slots=service_aics * effective,
                           max_requests_per_service_aiv=(physical + 2 * service_aics - 1) // (2 * service_aics),
                           window_probe=args.window_probe,
                           window_q=w4_comparison or (
                               bool(args.bs2_window_q or args.window_q_matrix) and mode in WINDOW_Q_WORKERS),
                           arguments=arguments, kernels=CHECKED_KERNELS[f'small_{mode}'])


def window_probe_scopes(source, case):
    names = WINDOW_SCOPE_PHASES
    counts = {name: 0 for name in names.values()}
    scopes = []
    for event in source['traceEvents']:
        detail = event.get('args', {})
        phase = detail.get('phase_id')
        if event.get('ph') != 'X' or detail.get('kernel_id') != 22 or phase not in names:
            continue
        window, lane = detail['window_epoch'], detail['service_lane']
        if lane >= case['physical_batch']:
            continue
        name = names[phase]
        counts[name] += 1
        scopes.append(dict(window=window, request=lane, scope=name, source_us=event['dur'],
                           begin_tick=detail['begin_tick'], end_tick=detail['end_tick']))
    if case['window_probe'] == 'bs1-p8':
        bulk = case['mode'] == 'cold_p8_bulk'
        windows = case['effective_windows']
        if counts['p8_cursor'] != windows or counts['bulk_cursor_mte2'] != (windows if bulk else 0):
            raise ValueError('P8 cursor trace does not show the planned copy path in every Service window')
    if case['mode'] != 'windows_early' and counts['early_header_enqueue']:
        raise ValueError('Header read-ahead unexpectedly ran in the control')
    return counts, scopes


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
        cold_entry = case['kernels'][-2]
        bodies = [stage for stage in report['stages']
                  if stage['kernel_entry'] == cold_entry and stage['phase_id'] == 1]
        groups = case['cold_score_groups']
        if [(body['logical_aic_lanes'], body['logical_aiv_lanes']) for body in bodies] != [(groups, 2 * groups)]:
            raise ValueError(f'{cold_entry} trace does not show the planned {groups} MIX groups')
        manifest['status'] = 'PASS'
        pool = read(case_output / 'worker/pool.json')
        row = dict(
            put_shutdown_retries=pool.get('shutdown_retries', 0),
            variant=case['mode'], batch=case['batch'], accepted=True, windows=report['windows'],
            cold_score_parts=report['cold_parts'], service_lanes=report['service_lanes'],
            source=manifest['business_trace'], msprof=manifest['msprof_business_trace'],
            stage_report=manifest['stage_report'], fixture_sha256=report['fixture_sha256'],
            library_sha256=report['library_sha256'], requests=report['requests'],
            resident_seed_sha256=report['resident_seed_sha256'],
            native_first_kernel_to_terminal_us=capture['intervals'][-1]['end_us'] - capture['intervals'][0]['begin_us'],
            prefetch_requests_per_call=report['total_prefetch'],
            final_get_requests_per_call=report['total_final_get'],
            total_get_requests_per_call=report['total_prefetch'] + report['total_final_get'],
        )
        fields = COLD_GM_STAGE_FIELDS.get(f"small_{case['mode']}", STAGES)
        for field, interval in zip(fields, capture['intervals'], strict=True):
            row[f'marked_{field}'] = interval['duration_us']
        if case['mode'] == 'adaptive':
            row['retired_service_job_count'] = len(report['retired_service_jobs'])
        if case.get('window_probe'):
            counts, row['window_scopes'] = window_probe_scopes(capture['source'], case)
            row.update({f'{name}_scopes': count for name, count in counts.items()})
        if case['window_q']:
            from window_q_trace import analyze_window_q, window_prefetch_get_counts

            windows = case['effective_windows']
            metrics, scopes = analyze_window_q(
                capture['source'], WINDOW_Q_WORKERS.get(case['mode'], 1), case['physical_batch'],
                windows=windows, parts=case['cold_score_parts'])
            window_gets = window_prefetch_get_counts(report['requests'], windows)
            if sum(window_gets.values()) != report['total_prefetch']:
                raise ValueError('Window Get counters do not match total prefetch')
            metrics.update(window_gets)
            row.update(metrics)
            save(case_output / 'window_q_report.json', dict(metrics=metrics, scopes=scopes, requests=report['requests']))
        return row
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
            or row.get('cold_score_blocks') != case['cold_score_groups']
            or row.get('prefetch_window_limit') != 128):
        raise RuntimeError(f'{case["mode"]} did not complete as planned. See {case_output / config.mr25_handoff}')
    miss_dir = case_output / config.mr25_handoff / f'miss{miss:g}'
    result = read(miss_dir / f'case-b{physical}' / 'fused/path_result.json')
    row['per_call_requests'] = result['pipeline_counts']
    row['put_shutdown_retries'] = read(miss_dir.parent / 'pool.json').get('shutdown_retries', 0)
    if case['window_q']:
        from window_q_trace import window_prefetch_get_counts

        timing = result['all_rounds']
        calls = timing['graph_calls'] * timing['measured_graphs']
        row.update(window_prefetch_get_counts(
            row['per_call_requests'], case['effective_windows'], calls=calls))
    row['variant'] = case['mode']
    fixture = read(miss_dir / 'fixture/fixture.json')['full_chain_fixture']
    if fixture['requested_miss_percent'] != miss:
        raise ValueError('Fixture miss rate disagrees with the trial plan')
    for q, rate in enumerate(fixture['actual_miss_percent_per_query']):
        row[f'fixture_miss_percent_q{q}'] = rate
    if case['waves'] > 1:
        (timing,) = result['rounds'].values()
        apply_group_timing(row, timing, case['waves'])
    row['batch'] = case['batch']
    timing_label, get_unit = ((f'{case["waves"]}xBS{physical} graph', 'group')
                              if case['waves'] > 1 else ('KV ready', 'call'))
    early, final = (row[f'{field}_per_{get_unit}'] for field in ('prefetch_requests', 'final_get_requests'))
    print(f'BS{case["batch"]} {case["mode"]}: {timing_label} {row["kv_ready_us"]:.2f} µs, '
          f'prefetch / final Get per {get_unit} {early:.2f} / {final:.2f}', flush=True)
    return row


def trace_compare(before, after, *, prefix=''):
    base = before['requests']
    candidate = after['requests']
    if any(left['request'] != right['request'] for left, right in zip(base, candidate, strict=True)):
        raise ValueError('Trace request IDs are not aligned')
    comparison = dict(
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
    if all('prefetch_tokens_in_submit_order_by_window' in item for item in base + candidate):
        comparison['same_prefetch_post_order'] = all(
            left['prefetch_tokens_in_submit_order_by_window'] == right['prefetch_tokens_in_submit_order_by_window']
            for left, right in zip(base, candidate, strict=True))
    after.update({f'{prefix}{key}': value for key, value in comparison.items()})
    if after['variant'] in (PARTITIONED_MODES | WINDOW_Q_WORKERS.keys() | {'topk'}) and not comparison['resident_seed_match']:
        raise ValueError(f"Resident Top2048 seed differs from {before['variant']}")


def compare_request_counts(before, after, *, prefix=''):
    def keyed(row):
        return {(r['source_round'], r['invocation'], r['request']): r for r in row['per_call_requests']}

    base, candidate = keyed(before), keyed(after)
    if base.keys() != candidate.keys():
        raise ValueError('Measured request and invocation coverage differs from P2')
    keys = ('prefetch_requests', 'final_get_requests', 'total_get_requests')
    matches = {key: all(base[i][key] == candidate[i][key] for i in base) for key in keys}
    after[f'{prefix}get_count_comparison'] = ('same_early_and_final_counts' if all(matches.values())
                                             else 'same_total_counts' if matches['total_get_requests']
                                             else 'different_request_totals')
    ratio = after['kv_ready_us'] / before['kv_ready_us']
    after[f'{prefix}raw_kv_ready_ratio'] = ratio
    if all(matches.values()):
        after[f'{prefix}matched_get_count_kv_ready_ratio'] = ratio


def compare_arms(before, after, *, trace, prefix=''):
    if any(not before[key] or before[key] != after[key] for key in ('fixture_sha256', 'library_sha256')):
        raise ValueError(f'{before["variant"]}/{after["variant"]} used different fixture or bridge')
    if trace:
        trace_compare(before, after, prefix=prefix)
    else:
        compare_request_counts(before, after, prefix=prefix)
        after[f'{prefix}raw_kv_ready_delta_us'] = after['kv_ready_us'] - before['kv_ready_us']


def compare_stage_trace(before, after, prefix):
    compare_arms(before, after, trace=True, prefix=prefix)
    for metric in ('native_first_kernel_to_terminal_us', *(f'marked_{stage}' for stage in STAGES)):
        after[f"{prefix}{metric.removesuffix('_us')}_delta_us"] = after[metric] - before[metric]


def summarize(output, rows):
    summary = [{key: value for key, value in row.items()
                if key not in ('requests', 'per_call_requests', 'window_scopes')} for row in rows]
    write_summary(output, summary)
    grouped = [row for row in summary if row.get('measurement_kind') == 'serial_bs16x2_graph_event']
    if grouped:
        directory = output / 'bs32-grouped'
        directory.mkdir(exist_ok=True)
        write_summary(directory, grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16, *WINDOW_Q_GROUPS), action='append')
    parser.add_argument('--miss-percent', type=int, choices=DIRECTION1_MISSES, action='append',
                        help='select one miss rate for a matrix or granularity study')
    parser.add_argument('--group-bs32', action='store_true',
                        help='measure BS32 as one graph containing two serial BS16 forwards')
    route = parser.add_mutually_exclusive_group()
    route.add_argument('--variant', choices=MODES, action='append', help='include one single-change arm; P2 is automatic')
    route.add_argument('--follow-up', action='store_true', help='BS1 P2/P4 and BS8 W4/W3 only')
    route.add_argument('--direction1-matrix', action='store_true', help='sweep first-direction P4 or W3 against native P2')
    route.add_argument('--granularity-study', action='store_true',
                       help='compare BS1 Cold P2/P4/P8 at W2 and BS2 P2/W4 versus P2/W3')
    route.add_argument('--full32', action='store_true',
                       help='compare stock P2 and Cold 32/BS at native windows for BS1, BS2, BS8')
    route.add_argument('--window-sweep', action='store_true',
                       help='compare native P2 W4/W3/W2 for BS2 and BS8 with fixed Service lanes')
    route.add_argument('--balanced', action='store_true',
                       help='compare P2 and Cold 32/BS across every feasible W2/W3/W4 at one request per Service AIV')
    route.add_argument('--cold-window-matrix', action='store_true',
                       help='keep Hot P2; sweep Cold parts at W2 for BS1 and W2/W4 for BS2/BS8')
    route.add_argument('--bs2-trace-pair', choices=BS2_TRACE_BASELINES,
                       help='capture BS2 msprof pairs: P2/P4 at W2, W4/W2 at P4, or both')
    route.add_argument('--bs2-window-q', action='store_true',
                       help='compare W1 serial Q4, Q2+Q2 and four Q1 at BS2 P4/W2, miss 10%')
    route.add_argument('--window-q1-matrix', dest='window_q_matrix', action='store_const', const='q1',
                       help='compare main P2, P4/W2 serial Q4 and four Q1 at BS1/2/4/8')
    route.add_argument('--window-q-matrix', action='store_const', const='all',
                       help='compare main P2 and W2 serial Q4, Q2+Q2 and four Q1 at BS1/2/4/8/16')
    route.add_argument('--window-q-wide', dest='window_q_matrix', action='store_const', const='wide',
                       help='add same-graph serial groups BS20 (5x4), BS24 (3x8) and BS32 (2x16)')
    route.add_argument('--bs16-w4-q1', action='store_true',
                       help='compare BS16 P2/W4 baseline, W4 last-window Q1 and W2 last-window Q1')
    route.add_argument('--w4-q1-wide', action='store_true',
                       help='compare P2/W4, W4 last-window Q1 and W2 Q1 for BS1–32 and miss 5/10/15/20')
    route.add_argument('--adaptive-pw', action='store_true',
                       help='compare native P2, fixed best P/W and adaptive P/W with retired AIV Service')
    route.add_argument('--window-probe', choices=WINDOW_PROBES,
                       help='compare fixed-geometry bulk cursor or early Top2048 header read')
    parser.add_argument('--repeats', type=int, default=1, help='repeat the matrix with the same bridge and independent workers')
    parser.add_argument('--trace', action='store_true', help='capture one checked msprof replay per arm')
    parser.add_argument('--main-root', type=Path,
                        help='separate, unmodified main checkout for W1 query matrix performance')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.bs2_trace_pair:
        args.trace = True
    if (args.bs2_window_q or args.window_q_matrix) and any(miss != 10 for miss in args.miss_percent or ()):
        parser.error('The W1 query trials use the miss 10% fixture')
    if args.bs16_w4_q1 and any(miss not in BS16_W4_Q1_MISSES for miss in args.miss_percent or ()):
        parser.error(f'--bs16-w4-q1 supports miss rates {BS16_W4_Q1_MISSES}')
    if args.w4_q1_wide and any(miss not in W4_Q1_WIDE_MISSES for miss in args.miss_percent or ()):
        parser.error(f'--w4-q1-wide supports miss rates {W4_Q1_WIDE_MISSES}')
    if args.w4_q1_wide and args.trace and any(batch in WINDOW_Q_GROUPS for batch in args.batch or ()):
        parser.error('Grouped batches use Graph Event timing; select their physical BS4/8/16 for a trace')
    w4_comparison = args.bs16_w4_q1 or args.w4_q1_wide
    study = study_arms(args)
    if study:
        if args.group_bs32 or any(batch not in study for batch in args.batch or ()):
            parser.error(f'This study supports BS values {tuple(study)}')
        if args.trace and args.window_q_matrix == 'wide' and any(
                batch in WINDOW_Q_GROUPS for batch in args.batch or study):
            parser.error('Grouped Graph Event timing requires a performance run; use --window-q-matrix --trace for native traces')
    elif args.direction1_matrix:
        if any(batch not in DIRECTION1_MODES for batch in args.batch or ()):
            parser.error(f'--direction1-matrix supports BS values {tuple(DIRECTION1_MODES)}')
    elif args.miss_percent or args.group_bs32 or any(batch not in FOLLOW_UP_ARMS for batch in args.batch or ()):
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
    cases = [dict(case, repeat=repeat) for repeat in range(1, args.repeats + 1) for case in matrix]
    unsupported = []
    if args.direction1_matrix and 32 in (args.batch or DIRECTION1_MODES) and not args.group_bs32:
        unsupported = [dict(batch=32, miss_percent=miss, accepted=False, status='UNSUPPORTED_NATIVE',
                            reason='Full-chain LIFetch supports up to BS16; use --group-bs32 for two serial BS16 calls')
                       for miss in dict.fromkeys(args.miss_percent or DIRECTION1_MISSES)]
    if args.cold_window_matrix and 1 in (args.batch or COLD_WINDOW_ARMS):
        unsupported = [dict(batch=1, miss_percent=miss, requested_windows=4, accepted=False,
                            status='UNSUPPORTED_NATIVE',
                            reason='BS1 static rotation has two Service owner groups; all Cold parts use effective W2')
                       for miss in dict.fromkeys(args.miss_percent or (10,))]
    if args.w4_q1_wide and 1 in (args.batch or study):
        unsupported = [dict(batch=1, physical_batch=1, waves=1, miss_percent=miss,
                            variant='p2_w4_q1', requested_windows=4, effective_windows=2,
                            accepted=False, status='UNSUPPORTED_NATIVE',
                            reason='P2 at BS1 has two MIX groups and cannot execute W3')
                       for miss in dict.fromkeys(args.miss_percent or W4_Q1_WIDE_MISSES)]
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=normal')),
                    kvaf_commit=KVAF, measurement='marked_msprof' if args.trace else 'graph_event',
                    follow_up=args.follow_up, direction1_matrix=args.direction1_matrix,
                    full32=args.full32, window_sweep=args.window_sweep, balanced=args.balanced,
                    cold_window_matrix=args.cold_window_matrix,
                    bs2_trace_pair=args.bs2_trace_pair,
                    bs2_window_q=args.bs2_window_q, window_q1_matrix=args.window_q_matrix == 'q1',
                    window_q_matrix=args.window_q_matrix in ('all', 'wide'),
                    window_q_wide=args.window_q_matrix == 'wide',
                    bs16_w4_q1=args.bs16_w4_q1, w4_q1_wide=args.w4_q1_wide,
                    trace_control=('same_mr_p2_option_disabled' if w4_comparison and args.trace else None),
                    main_root=str(args.main_root.resolve()) if args.main_root else None,
                    adaptive_pw=args.adaptive_pw, window_probe=args.window_probe,
                    granularity_study=args.granularity_study, repeats=args.repeats,
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
    uses_main_p2 = bool(args.window_q_matrix or w4_comparison)
    if uses_main_p2 and not args.trace:
        if args.main_root is None:
            parser.error('Window query matrix performance requires --main-root')
        args.main_root = args.main_root.resolve()
        if (git(args.main_root, 'rev-parse', 'HEAD') != MAIN
                or git(args.main_root, 'status', '--porcelain', '--untracked-files=no')):
            parser.error(f'--main-root must be a clean main checkout at {MAIN}')
        from window_q_main import measure_main
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the pinned KVAF checkout at {KVAF}')
    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof')) if args.trace else None
    if args.trace and not profiler:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    output = Path(os.environ['KVIP_AB_OUTPUT']).resolve()
    print(f'Small batch A5 comparison: {output}', flush=True)
    rows, bases, p4_arms, window_bases, w4_arms = list(unsupported), {}, {}, {}, {}
    focused_baselines = {}
    fixed_optimal = {}
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
            if args.direction1_matrix or study:
                run_dir = run_dir / f'miss{miss:g}'
            group = f" ({case['waves']}xBS{case['physical_batch']} serial graph)" if case['waves'] > 1 else ""
            print(f"Round {repeat}/{args.repeats}, BS{case['batch']} miss{miss:g} "
                  f"{case['mode']} P{case['cold_score_parts']}/W{case['effective_windows']} "
                  f"({case['cold_score_groups']} Cold MIX groups per call){group}", flush=True)
            if uses_main_p2 and not args.trace and case['mode'] == 'p2':
                row = measure_main(args.main_root, run_dir, case, MAIN)
            else:
                row = trace_arm(run_dir, case, config, profiler) if args.trace else perf_arm(run_dir, case, config)
                if uses_main_p2:
                    row['source_commit'] = manifest['commit']
            if w4_comparison:
                if case['mode'] == 'p2':
                    row['source_role'] = 'same_mr_p2_option_disabled' if args.trace else 'unmodified_main_p2'
                else:
                    row['source_role'] = 'mr_last_window_q1'
            row.update(repeat=repeat, miss_percent=miss, physical_batch=case['physical_batch'],
                       waves=case['waves'], requested_windows=case['requested_windows'],
                       effective_windows=case['effective_windows'], hot_score_parts=case['hot_score_parts'],
                       hot_score_aic=case['hot_score_aic'],
                       hot_score_aiv=case['hot_score_aiv'],
                       measurement_kind=case['measurement_kind'], cold_score_groups=case['cold_score_groups'],
                       cold_score_aic=case['cold_score_groups'], cold_score_aiv=2 * case['cold_score_groups'],
                       service_group_slots=case['service_group_slots'],
                       service_owner_groups=case['service_owner_groups'],
                       max_requests_per_service_aiv=case['max_requests_per_service_aiv'])
            rows.append(row)
            pair = (repeat, row['batch'], miss)
            if args.bs2_trace_pair:
                if row['variant'] == 'cold_w2':
                    for mode in BS2_TRACE_BASELINES[args.bs2_trace_pair]:
                        compare_stage_trace(focused_baselines[(*pair, mode)], row, BS2_TRACE_PREFIX[mode])
                else:
                    focused_baselines[(*pair, row['variant'])] = row
                summarize(output, rows)
                continue
            if args.window_probe:
                control = study[row['batch']][0]
                if row['variant'] == control:
                    focused_baselines[pair] = row
                else:
                    before = focused_baselines[pair]
                    if args.trace:
                        compare_stage_trace(before, row, '')
                        if not row['resident_seed_match']:
                            raise ValueError('Resident seed differs within the fixed-geometry window probe')
                        save(run_dir / 'window_probe_comparison.json', dict(control=before, experiment=row))
                    else:
                        compare_arms(before, row, trace=False)
                summarize(output, rows)
                continue
            if uses_main_p2 and not args.trace:
                if row['variant'] == 'p2':
                    bases[pair] = row
                    summarize(output, rows)
                    continue
                main = bases[pair]
                if main['fixture_sha256'] != row['fixture_sha256']:
                    raise ValueError(f'BS{row["batch"]}: fixture differs from main')
                compare_request_counts(main, row, prefix='vs_main_')
                row['vs_main_raw_kv_ready_delta_us'] = row['kv_ready_us'] - main['kv_ready_us']
                print(f'  KV ready / main: {row["vs_main_raw_kv_ready_ratio"]:.4f}x, '
                      f'Get {row["vs_main_get_count_comparison"]}', flush=True)
            if w4_comparison:
                if args.trace:
                    if row['variant'] == 'p2':
                        bases[pair] = row
                    else:
                        before = bases[pair]
                        compare_stage_trace(before, row, f'vs_mr_p2_w{before["windows"]}_')
                        if row['windows'] == before['windows'] == 4:
                            for metric in ('w3_after_last_aiv_score_scan_us', 'w3_max_owner_service_us',
                                           'w3_max_request_topk_elapsed_us', 'w3_max_owner_topk_join_us'):
                                row[f"vs_mr_p2_w4_{metric.removesuffix('_us')}_delta_us"] = row[metric] - before[metric]
                else:
                    for field in STAGES:
                        row[f"vs_main_{field.removesuffix('_us')}_delta_us"] = row[field] - main[field]
                summarize(output, rows)
                continue
            if args.bs2_window_q or args.window_q_matrix:
                row['w1_topk_workers_per_request'] = WINDOW_Q_WORKERS[row['variant']]
                if row['w1_topk_workers_per_request'] == 1:
                    focused_baselines[pair] = row
                else:
                    before = focused_baselines[pair]
                    if args.trace:
                        compare_stage_trace(before, row, 'vs_serial_q4_')
                        for metric in ('w1_after_last_aiv_score_scan_us', 'w1_max_owner_service_us'):
                            row[f"vs_serial_q4_{metric.removesuffix('_us')}_delta_us"] = row[metric] - before[metric]
                    else:
                        compare_arms(before, row, trace=False, prefix='vs_serial_q4_')
                summarize(output, rows)
                continue
            window_pair = (*pair, case['effective_windows'])
            parts = case['cold_score_parts']
            if (args.balanced or args.cold_window_matrix) and parts == 2:
                window_bases[window_pair] = row
            if row['variant'] == 'p2':
                bases[pair] = row
            else:
                before = bases[pair]
                if (args.granularity_study or args.cold_window_matrix) and row['variant'] == 'cold':
                    p4_arms[pair] = row
                compare_arms(before, row, trace=args.trace)
                if row['variant'] == 'cold_p8' and pair in p4_arms:
                    compare_arms(p4_arms[pair], row, trace=args.trace, prefix='vs_p4_')
                if ((args.balanced and row['variant'] in FULL32_MODES)
                        or (args.cold_window_matrix and parts > 2)):
                    compare_arms(window_bases[window_pair], row, trace=args.trace, prefix='same_window_')
                if args.adaptive_pw:
                    if row['variant'] == 'adaptive':
                        before = fixed_optimal[pair]
                        if args.trace:
                            compare_stage_trace(before, row, 'vs_fixed_optimal_')
                        else:
                            compare_arms(before, row, trace=False, prefix='vs_fixed_optimal_')
                    else:
                        fixed_optimal[pair] = row
            if args.cold_window_matrix and row['batch'] != 1:
                parts_pair = (*pair, parts)
                if case['effective_windows'] == 4:
                    w4_arms[parts_pair] = row
                else:
                    compare_arms(w4_arms[parts_pair], row, trace=args.trace, prefix='same_parts_w4_')
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
