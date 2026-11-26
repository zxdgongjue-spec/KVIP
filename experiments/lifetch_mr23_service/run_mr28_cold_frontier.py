#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native P2 with Cold GM using the native measured window frontiers."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from run_mr25_handoff import KVAF, ROOT, collect, git, read, save
from run_mr25_score import PAIRED_KEYS
from run_mr28_dedicated_service import build_case, write_summary
from cold_frontier import BATCH, FILE_NAME, GEOMETRY, ROUTES, compare_invocations, header, load_tapes
from terminal_handoff import COLD_GM_KERNELS, COLD_GM_STAGE_FIELDS, STAGES


def case_paths(output, route):
    return output / route / 'miss10' / f'case-b{BATCH}' / 'fused'


def collect_arm(output, route, case, calls):
    fields = COLD_GM_STAGE_FIELDS.get(route, STAGES)
    row = collect(output, route, BATCH, stage_fields=fields)
    row['worker_exit_code'] = case['worker_exit_codes'].get(route)
    row['route_verified'] = (
        row['worker_exit_code'] == 0 and row['accepted']
        and row.get('effective_aic') == GEOMETRY['aic'] and row.get('available_aiv') == GEOMETRY['aiv']
        and row.get('windows') == GEOMETRY['windows'] and row.get('local_counts') is False
        and row.get('cold_frontier_mode') == ROUTES[route]
        and row.get('cold_score_writeback') is (route == 'cold_gm_fenced')
        and row.get('cold_gm_row_consumer') is False
        and row.get('prefetch_observed')
        and all(row.get(key) == value for key, value in case['cold_roles'][route].items())
    )
    if row['route_verified']:
        timing = read(case_paths(output, route) / 'path_result.json')['all_rounds']
        row['route_verified'] = timing['graph_calls'] == calls and timing['measured_graphs'] == 1
    row['timing_eligible'] = False
    if row['accepted']:
        print(f"{route}: " + ', '.join(f'{key}={row[key]:.2f} µs' for key in fields), flush=True)
        print(f"  KV ready {row['kv_ready_us']:.2f} µs (graph Event mean; frontier and Get capture enabled)", flush=True)
        print(f"  Prefetch/final/total Get per call {row['prefetch_requests_per_call']:.2f}/"
              f"{row['final_get_requests_per_call']:.2f}/{row['total_get_requests_per_call']:.2f}", flush=True)
    return row


def summarize(output, case, calls):
    rows = [collect_arm(output, route, case, calls) for route in ROUTES]
    before, after = rows
    issues = [dict(route=row['variant'], reason=row['reason'] or 'worker or route verification failed')
              for row in rows if not row['route_verified']]
    if case.get('capture_failure'):
        issues.append(dict(route='p2', reason=case['capture_failure']))
    mismatches = [key for key in PAIRED_KEYS if before.get(key) != after.get(key)]
    if not issues and mismatches:
        issues.append(dict(reason='paired fields differ', fields=mismatches))
    if not issues:
        try:
            evidence = [read(case_paths(output, row['variant']) / FILE_NAME) for row in rows]
            for recorded, row in zip(evidence, rows):
                if recorded['header'] != header(row['fixture_sha256'], row['library_sha256'], calls):
                    raise ValueError(f"{row['variant']}: evidence header disagrees with measured report")
            issues.extend(compare_invocations(*evidence))
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(reason=str(error)))
    eligible = not issues
    if all(row['accepted'] for row in rows):
        after['raw_kv_ready_ratio'] = after['kv_ready_us'] / before['kv_ready_us']
    for row in rows:
        row['timing_eligible'] = eligible
    if eligible:
        after['strict_kv_ready_ratio'] = after['raw_kv_ready_ratio']
        print(f"Strict timing eligible: KV ready ratio {after['strict_kv_ready_ratio']:.4f}x", flush=True)
    else:
        print(f'Strict timing rejected ({len(issues)} issues); raw measurements retained', flush=True)
        if 'raw_kv_ready_ratio' in after:
            print(f"Raw KV ready ratio {after['raw_kv_ready_ratio']:.4f}x", flush=True)
    save(output / 'strict_comparison.json', dict(timing_eligible=eligible, differences=issues))
    write_summary(output, rows)
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--comparison-dry-run', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative, iterations and timeout must be positive')
    args.prefetch_trace = False

    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    case = build_case(kvaf, BATCH, ROUTES, args)
    manifest = dict(status='NOT_RUN', commit=git(ROOT, 'rev-parse', 'HEAD'),
                    worktree_dirty=bool(git(ROOT, 'status', '--porcelain', '--untracked-files=no')),
                    expected_kvaf=KVAF, geometry=GEOMETRY, variants=ROUTES, case=case,
                    expected_graph_kernels={mode: COLD_GM_KERNELS[mode] for mode in ROUTES},
                    timing='one profiled graph with frontier and Get token capture on both arms')
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
        raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
    from execution import launch
    from workloads import configuration

    out_base = Path(os.environ.get('KVIP_AB_OUTPUT', 'results/mr28_cold_frontier')).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr28_cold_frontier.', dir=out_base))
    print(f'MR28 Cold frontier comparison: {output}', flush=True)
    manifest['status'] = 'RUNNING'
    save(output / 'manifest.json', manifest)
    capture_path = case_paths(output, 'p2') / FILE_NAME
    try:
        for route, mode in ROUTES.items():
            config = configuration(case['arguments'])
            config.mr25_handoff = route
            config.mr28_dedicated_service = route
            config.mr28_cold_gm_experiment = True
            config.mr28_cold_frontier_mode = mode
            config.handoff_build_dir = output / 'build'
            if mode == 2:
                config.mr28_cold_frontier_tape = capture_path
            print(f'Running BS{BATCH} {route} (frontier mode {mode})', flush=True)
            result = launch(config, output / route)
            case['worker_exit_codes'][route] = result
            save(output / 'manifest.json', manifest)
            if result:
                break
            if mode == 1:
                row = collect(output, route, BATCH)
                try:
                    if not row['accepted']:
                        raise ValueError('Native P2 did not pass correctness and timing checks')
                    load_tapes(capture_path, header(row['fixture_sha256'], row['library_sha256'], args.iterations))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    case['capture_failure'] = str(error)
                    break
        eligible = summarize(output, case, args.iterations)
        manifest['status'] = 'PASS' if eligible else 'REJECTED'
        return int(not eligible)
    except BaseException:
        manifest['status'] = 'FAILED'
        raise
    finally:
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    raise SystemExit(main())
