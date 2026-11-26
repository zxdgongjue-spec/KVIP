#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one change at a time on the native A5 P2 fixture."""

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from run_mr25_handoff import KVAF, ROOT, bench_arguments, collect, git, read, save
from run_mr28_baseline_msprof import capture_route
from run_mr28_dedicated_service import write_summary
from kvip.operators.lifetch.geometry import li_geometry
from small_batch import (
    FULL32_MODES, MODES, PARTITIONED_MODES, SUPPORTED_BATCHES, TRACE_ROUTES, WINDOW_OVERRIDES, cold_score_parts,
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
MAIN = 'df8ad2598e70e4b5e266b15028ab26c601a2bdf2'
DIRECTION1_MODES = {1: 'cold', 2: 'cold', 4: 'cold', 8: 'windows', 16: 'windows', 32: 'windows'}
DIRECTION1_MISSES = (5, 10, 15, 20, 25, 30)
NATIVE_BATCH_LIMIT = 16


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
    misses = tuple(dict.fromkeys(args.miss_percent or (DIRECTION1_MISSES if args.direction1_matrix else (10,))))
    selected = set(args.variant or (SUPPORTED_BATCHES if study else MODES)) | {'p2'}
    for batch in batches:
        physical = min(batch, NATIVE_BATCH_LIMIT)
        waves = batch // physical
        if waves > 1 and not args.group_bs32:
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
                requested = WINDOW_OVERRIDES.get(mode, 4)
                if args.cold_window_matrix and batch == 1:
                    requested = 2
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
                parts = cold_score_parts(mode, physical)
                yield dict(batch=batch, physical_batch=physical, waves=waves, miss_percent=miss,
                           measurement_kind=measurement, mode=mode, requested_windows=requested, effective_windows=effective,
                           hot_score_parts=2, hot_score_aic=2 * physical, hot_score_aiv=4 * physical,
                           service_aics=service_aics, cold_score_parts=parts, cold_score_groups=physical * parts,
                           service_owner_groups=2 * physical, service_group_slots=service_aics * effective,
                           max_requests_per_service_aiv=(physical + 2 * service_aics - 1) // (2 * service_aics),
                           arguments=arguments, kernels=CHECKED_KERNELS[f'small_{mode}'])


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
        row = dict(
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
    after.update({f'{prefix}{key}': value for key, value in comparison.items()})
    if after['variant'] in (PARTITIONED_MODES | {'topk'}) and not comparison['resident_seed_match']:
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
    if trace:
        trace_compare(before, after, prefix=prefix)
    else:
        compare_request_counts(before, after, prefix=prefix)
        after[f'{prefix}raw_kv_ready_delta_us'] = after['kv_ready_us'] - before['kv_ready_us']


def summarize(output, rows):
    summary = [{key: value for key, value in row.items()
                if key not in ('requests', 'per_call_requests')} for row in rows]
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
    parser.add_argument('--repeats', type=int, default=1, help='repeat the matrix with the same bridge and independent workers')
    parser.add_argument('--trace', action='store_true', help='capture one checked msprof replay per arm')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    study = study_arms(args)
    if study:
        if args.group_bs32 or any(batch not in study for batch in args.batch or ()):
            parser.error(f'This study supports only native BS values {tuple(study)}')
    elif not args.direction1_matrix and (args.miss_percent or args.group_bs32
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
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'), main_commit=MAIN,
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=normal')),
                    kvaf_commit=KVAF, measurement='marked_msprof' if args.trace else 'graph_event',
                    follow_up=args.follow_up, direction1_matrix=args.direction1_matrix,
                    full32=args.full32, window_sweep=args.window_sweep, balanced=args.balanced,
                    cold_window_matrix=args.cold_window_matrix,
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
    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the pinned KVAF checkout at {KVAF}')
    profiler = shutil.which(os.environ.get('KVIP_MSPROF', 'msprof')) if args.trace else None
    if args.trace and not profiler:
        raise FileNotFoundError('msprof is missing from the active CANN environment')
    output = Path(os.environ['KVIP_AB_OUTPUT']).resolve()
    print(f'Small batch A5 comparison: {output}', flush=True)
    rows, bases, p4_arms, window_bases, w4_arms = list(unsupported), {}, {}, {}, {}
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
            print(f"Round {repeat}/{args.repeats}, BS{case['batch']} miss{miss:g} "
                  f"{case['mode']} P{case['cold_score_parts']}/W{case['effective_windows']} "
                  f"({case['cold_score_groups']} Cold MIX groups)", flush=True)
            row = trace_arm(run_dir, case, config, profiler) if args.trace else perf_arm(run_dir, case, config)
            row.update(repeat=repeat, miss_percent=miss, physical_batch=case['physical_batch'],
                       waves=case['waves'],
                       hot_score_parts=case['hot_score_parts'], hot_score_aic=case['hot_score_aic'],
                       hot_score_aiv=case['hot_score_aiv'],
                       measurement_kind=case['measurement_kind'], cold_score_groups=case['cold_score_groups'],
                       cold_score_aic=case['cold_score_groups'], cold_score_aiv=2 * case['cold_score_groups'],
                       service_group_slots=case['service_group_slots'],
                       service_owner_groups=case['service_owner_groups'],
                       max_requests_per_service_aiv=case['max_requests_per_service_aiv'])
            rows.append(row)
            pair = (repeat, row['batch'], miss)
            window_pair = (*pair, case['effective_windows'])
            parts = case['cold_score_parts']
            if (args.balanced or args.cold_window_matrix) and parts == 2:
                window_bases[window_pair] = row
            if row['variant'] == 'p2':
                bases[pair] = row
            else:
                before = bases[pair]
                if any(row[key] != before[key] for key in ('fixture_sha256', 'library_sha256')):
                    raise ValueError('Pair used different fixture or bridge')
                if (args.granularity_study or args.cold_window_matrix) and row['variant'] == 'cold':
                    p4_arms[pair] = row
                compare_arms(before, row, trace=args.trace)
                if row['variant'] == 'cold_p8':
                    compare_arms(p4_arms[pair], row, trace=args.trace, prefix='vs_p4_')
                if ((args.balanced and row['variant'] in FULL32_MODES)
                        or (args.cold_window_matrix and parts > 2)):
                    compare_arms(window_bases[window_pair], row, trace=args.trace, prefix='same_window_')
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
