#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native P2 prefetch, UB candidates, and batched posting on A5."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/lifetch_mr23_service'))
from run_mr25_handoff import KVAF, collect, git, read, save  # noqa: E402
from run_mr25_score import compare_reads, completed_reads  # noqa: E402
from kvip.operators.lifetch.geometry import li_geometry, prefetch_budget  # noqa: E402


BASE = '5db7ef34edb269db41206a1211044fb07bf1da94'
BATCHES = (1, 2, 4, 8, 16)
DEFAULT_BATCH = 16
REQUESTED_WINDOWS = 4
PREFETCH_LIMIT = 512
MODES = ('baseline', 'ub', 'batch')
SAME_WORKLOAD = ('fixture_sha256', 'soc', 'effective_aic', 'windows', 'score_parts',
                 'score_blocks', 'service_lanes', 'terminal_tasks', 'local_counts')


def arguments(args, kvaf, batch):
    return [
        'diagnose', '--level', '2', '--transport', 'urma', '--input', 'synthetic',
        '--kvaf-root', str(kvaf), '--kvaf-server', os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986'),
        '--device', os.environ.get('KVIP_A5_DEVICE', '0'), '--batches', str(batch),
        '--queries', '4', '--history', '131072', '--resident', '16384', '--index-heads', '32',
        '--miss-percent', '10', '--seed', '43', '--intermediate-fetches', str(REQUESTED_WINDOWS),
        '--prefetch-tokens', str(PREFETCH_LIMIT), '--prefetch-rank-start', '1024',
        '--warmup', str(args.warmup), '--iterations', str(args.iterations),
        '--timeout-seconds', str(args.timeout_seconds),
    ]


def expected_geometry(batch):
    geometry = li_geometry(batch, intermediate_fetches=REQUESTED_WINDOWS, aic_count=32)
    _, cap = prefetch_budget(PREFETCH_LIMIT, intermediate_fetches=REQUESTED_WINDOWS)
    windows, lanes = geometry['intermediate_fetches'], geometry['service_aivs_per_window']
    return dict(windows=windows, service_lanes=lanes, window_cap=cap,
                early_cap=min(PREFETCH_LIMIT, windows * cap))


def diagnostic_sample(sample, *, batch=DEFAULT_BATCH, post_chunk=0):
    if 'error' in sample['algorithm']:
        raise ValueError(f"Invalid admission journal: {sample['algorithm']['error']}")
    requests = sample['algorithm']['requests']
    scopes = sample['streaming_cycles']
    if len(requests) != batch or len(scopes) != batch:
        raise ValueError('Incomplete Terminal diagnostic request coverage')
    head = scopes[0]
    prefetch, final = head['prefetch_bandwidth'], head['final_get_bandwidth']
    if prefetch['entries'] <= 0 or final['entries'] <= 0 or 'first doorbell' not in prefetch['note']:
        raise ValueError('Expected positive early and final traffic with a captured first doorbell')

    geometry = expected_geometry(batch)
    service_lanes = geometry['service_lanes']
    windows = geometry['windows']
    split_supported = post_chunk and windows == REQUESTED_WINDOWS
    final_groups = final['groups']
    final_blocks = [group['block'] for group in final_groups]
    if (len(final_blocks) != len(set(final_blocks)) or any(not 0 <= block < 4 * batch for block in final_blocks)
            or any(group['entries'] <= 0 for group in final_groups)):
        raise ValueError('Invalid Final Get groups')
    duplicate_posts = duplicate_final_posts = overlap_posts = 0
    lane_windows = {}
    for b, request in enumerate(requests):
        transfers = request['prefetch_transfers']
        if len(transfers) != request['prefetch'] or len({t['position'] for t in transfers}) != len(transfers):
            raise ValueError('Admission journal differs from completed early posts')
        repeats = len(transfers) - len({t['token'] for t in transfers})
        audit = request['posted_identity_audit']
        if repeats != audit['early_duplicates']:
            raise ValueError('Early transfer identities disagree across admission audits')
        if (not 0 <= audit['final_duplicates'] <= request['final']
                or not 0 <= audit['early_final_overlap'] <= min(request['prefetch'], request['final'])):
            raise ValueError('Invalid Final Get duplicate counts')
        duplicate_posts += repeats
        duplicate_final_posts += audit['final_duplicates']
        overlap_posts += audit['early_final_overlap']
        next_slot = request['prefetch']
        for group in sorted((g for g in final_groups if g['block'] // 4 == b),
                            key=lambda g: g['slot_base']):
            if group['slot_base'] != next_slot:
                raise ValueError('Final Get source intervals are not contiguous')
            next_slot += group['entries']
        if next_slot != request['prefetch'] + request['final']:
            raise ValueError('Final Get source interval length differs from completed Get count')
        posted = [window['posted'] for window in request['windows']]
        if ([window['window'] for window in request['windows']] != list(range(1, windows + 1))
                or any(not 0 <= count <= geometry['window_cap'] for count in posted)
                or sum(posted) != request['prefetch']):
            raise ValueError('Invalid effective per-window prefetch counters')
        for epoch, count in enumerate(posted):
            if count:
                lane_windows.setdefault((b % service_lanes, epoch), []).append(count)
    if prefetch['entries'] != sum(row['prefetch'] for row in requests):
        raise ValueError('Early CQ scope does not match the admission journal')
    if final['entries'] != sum(row['final'] for row in requests):
        raise ValueError('Final CQ scope does not match the per-request count')

    timeline = head['terminal_timeline']
    cores = {core['block']: core['stages'] for core in timeline['cores']}
    if len(timeline['cores']) != batch * 4 or cores.keys() != set(range(batch * 4)) or timeline['clock_mhz'] <= 0:
        raise ValueError('Incomplete Terminal diagnostic timeline')
    frequency = timeline['clock_mhz']
    first_terminal = min(stages[0]['start_cycles'] for stages in cores.values())
    services = head['prefetch_services']
    first_service = min(event['begin_cycles'] for event in services)
    first_doorbell = prefetch['first_cycles']
    if not 0 < first_service <= first_doorbell <= first_terminal:
        raise ValueError('Prefetch doorbell timestamp is outside the expected early scope')

    def cq_join(block, phase):
        return next(stage['duration_us'] for stage in cores[block] if stage['name'] == phase)

    early_waits = [cq_join(b * 4 + 1, 'prefetch_CQ_join')
                   for b, request in enumerate(requests) if request['prefetch']]
    final_waits = [cq_join(group['block'], 'final_CQ_join') for group in final_groups]
    active = [event for event in services if event['posted_count']]
    event_keys = [(event['lane'], event['epoch']) for event in services]
    if (len(event_keys) != len(set(event_keys)) or
            any(not 0 <= lane < service_lanes or not 0 <= epoch < windows for lane, epoch in event_keys)):
        raise ValueError('Invalid Service lane/window coverage')
    active_counts = {(event['lane'], event['epoch']): event['posted_count'] for event in active}
    if active_counts != {key: sum(counts) for key, counts in lane_windows.items()}:
        raise ValueError('Service event posts differ from the request/window admission journal')
    doorbells = sum(event['doorbells'] for event in services)
    possible_doorbells = split_windows = grouped_fallback_windows = 0
    for event in services:
        counts = lane_windows.get((event['lane'], event['epoch']), ())
        grouped = len(counts)
        chunked = (sum((count + post_chunk - 1) // post_chunk for count in counts)
                   if split_supported else grouped)
        possible_doorbells += chunked
        if event['doorbells'] not in (grouped, chunked):
            raise ValueError(f'Doorbell count differs from posted chunks: {event}')
        if chunked > grouped:
            split_windows += int(event['doorbells'] == chunked)
            grouped_fallback_windows += int(event['doorbells'] == grouped)
        first, last = event['first_doorbell_cycles'], event['last_doorbell_cycles']
        if ((counts and not event['begin_cycles'] <= first <= last <= event['end_cycles'])
                or (not counts and (first or last))):
            raise ValueError(f'Invalid per-window doorbell stamps: {event}')
    if min(event['first_doorbell_cycles'] for event in active) != first_doorbell:
        raise ValueError('Earliest doorbell differs from the overall prefetch scope')
    ub_post_fraction = sum(event['posted_count'] for event in active if event['ub_candidates']) / prefetch['entries']
    return dict(early_entries=prefetch['entries'], final_entries=final['entries'],
                duplicate_early_posts=duplicate_posts,
                duplicate_final_posts=duplicate_final_posts, early_final_overlap=overlap_posts,
                duplicate_early_rate=duplicate_posts / prefetch['entries'],
                early_doorbells=doorbells, possible_chunked_doorbells=possible_doorbells,
                split_early_windows=split_windows, grouped_fallback_windows=grouped_fallback_windows,
                ub_early_post_fraction=ub_post_fraction,
                service_to_first_doorbell_us=(first_doorbell - first_service) / frequency,
                early_doorbell_to_terminal_us=(first_terminal - first_doorbell) / frequency,
                early_first_doorbell_cycles=first_doorbell,
                early_doorbell_to_cq_us=prefetch['span_us'],
                final_doorbell_to_cq_us=final['span_us'],
                early_cq_join_max_us=max(early_waits), final_cq_join_max_us=max(final_waits))


def diagnostics(folder, expected_samples, *, batch=DEFAULT_BATCH, post_chunk=0):
    sources = sorted(folder.glob('diagnose/round-*/detail_samples.json'))
    samples = []
    for path in sources:
        for sample in read(path):
            samples.append(dict(source=str(path), repeat=sample['round'],
                                source_round=sample['source_round'],
                                **diagnostic_sample(sample, batch=batch, post_chunk=post_chunk)))
    identities = {(row['source_round'], row['repeat']) for row in samples}
    if len(samples) != expected_samples or len(samples) != len(identities):
        raise ValueError(f'Expected {expected_samples} independent checked diagnostic samples, got {len(samples)}')
    save(folder / 'prefetch_protocol.json', samples)
    metrics = ('early_entries', 'final_entries', 'duplicate_early_posts', 'duplicate_early_rate',
               'duplicate_final_posts', 'early_final_overlap', 'service_to_first_doorbell_us', 'early_doorbell_to_terminal_us', 'early_doorbell_to_cq_us',
               'final_doorbell_to_cq_us', 'early_cq_join_max_us', 'final_cq_join_max_us',
               'early_doorbells', 'possible_chunked_doorbells', 'split_early_windows',
               'grouped_fallback_windows', 'ub_early_post_fraction')
    return {f'diagnostic_{key}_mean': statistics.mean(sample[key] for sample in samples) for key in metrics}


def verified_route(row, variant, batch):
    geometry = expected_geometry(batch)
    return (row['effective_aic'] == 32 and 'Ascend950' in row['soc']
            and row['windows'] == geometry['windows'] and row['service_lanes'] == geometry['service_lanes']
            and row.get('local_counts') is False
            and row.get('service_ub_candidates') is variant['service_ub_candidates']
            and row.get('service_post_chunk') == variant['service_post_chunk'])


def summarize_batch(output, variants, completed, iterations, batch):
    geometry = expected_geometry(batch)
    rows = []
    for mode in MODES:
        row = collect(output, mode, batch)
        row['worker_exit_code'] = completed.get(mode)
        row.update(route_verified=False, protocol_verified=False, optimization_exercised=None,
                   requested_windows=REQUESTED_WINDOWS, expected_window_cap=geometry['window_cap'],
                   expected_early_cap=geometry['early_cap'])
        if row['worker_exit_code'] not in (None, 0):
            row['reason'] = f"Worker exited with code {row['worker_exit_code']}"
        if row['worker_exit_code'] == 0 and row['accepted']:
            row['route_verified'] = verified_route(row, variants[mode], batch)
            if not row['route_verified']:
                row['reason'] = 'A5 P2 geometry or compiled Service variant differs'
            folder = output / mode / 'miss10' / f'case-b{batch}' / 'fused'
            try:
                row.update(diagnostics(folder, iterations, batch=batch,
                                       post_chunk=variants[mode]['service_post_chunk']))
                ub_used = row['diagnostic_ub_early_post_fraction_mean'] > 0
                if mode == 'ub':
                    row['optimization_exercised'] = ub_used
                    if not ub_used:
                        row['effect_note'] = 'All positive Service events used GM candidate handoff'
                elif mode == 'batch':
                    row['optimization_exercised'] = row['diagnostic_split_early_windows_mean'] > 0
                    if not row['optimization_exercised']:
                        if geometry['windows'] != REQUESTED_WINDOWS:
                            row['effect_note'] = f"Effective W{geometry['windows']} uses grouped posting"
                        elif row['diagnostic_grouped_fallback_windows_mean']:
                            row['effect_note'] = 'Eligible windows used grouped posting'
                        else:
                            row['effect_note'] = 'No request window posted more than 64 entries'
                row['protocol_verified'] = (row['diagnostic_duplicate_early_posts_mean'] == 0
                                            and row['diagnostic_duplicate_final_posts_mean'] == 0
                                            and row['diagnostic_early_final_overlap_mean'] == 0
                                            and row['prefetch_observed']
                                            and row['final_get_requests_per_call'] > 0
                                            and (mode == 'ub' or not ub_used))
                if not row['protocol_verified'] and not row['reason']:
                    row['reason'] = 'Duplicate posts, invalid traffic counts, or an unmatched UB path'
            except (ValueError, KeyError, StopIteration, OSError) as error:
                if not row['reason']:
                    row['reason'] = str(error) or 'Missing CQ join phase'
            if not row.get('library_sha256'):
                with Path(row['library']).open('rb') as binary:
                    row['library_sha256'] = hashlib.file_digest(binary, 'sha256').hexdigest()
        rows.append(row)

    baseline = rows[0]
    for row in rows:
        same_workload = (baseline['route_verified'] and row['route_verified']
                         and bool(baseline.get('fixture_sha256'))
                         and all(baseline.get(key) == row.get(key) for key in SAME_WORKLOAD))
        comparable = same_workload and baseline['protocol_verified'] and row['protocol_verified']
        row.update(workload_verified=same_workload, get_comparison='unavailable',
                   get_vector_differences=None, observed_kv_ready_ratio=None, matched_total_kv_ready_ratio=None)
        if comparable:
            base = completed_reads(output, 'baseline', batch)
            status, differences = compare_reads(base, completed_reads(output, row['variant'], batch))
            ratio = row['kv_ready_us'] / baseline['kv_ready_us']
            row.update(get_comparison=status, get_vector_differences=differences,
                       observed_kv_ready_ratio=ratio)
            if status != 'different_request_totals':
                row['matched_total_kv_ready_ratio'] = ratio
            print(f"BS{batch} {row['variant']} KV ready {row['kv_ready_us']:.2f} µs "
                  f"({ratio:.4f}× baseline), Get vectors={status}, protocol=PASS", flush=True)
            if row.get('effect_note'):
                print(f"  {row['effect_note']}", flush=True)
        else:
            reason = row['reason'] or baseline['reason'] or 'fixture or route mismatch'
            print(f"BS{batch} {row['variant']} comparison unavailable: {reason}", flush=True)
    return rows


def summarize(output, rows):
    save(output / 'summary.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Summary: {output / "summary.csv"}', flush=True)
    return all(row['workload_verified'] and row['protocol_verified'] for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=BATCHES, action='append',
                        help='repeat to select multiple BS cases; default: 16')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--timeout-seconds', type=int, default=1800, help='per arm and batch, including first build')
    parser.add_argument('--reverse-order', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print the plan without CANN or KVAF')
    args = parser.parse_args(argv)
    batches = args.batch or [DEFAULT_BATCH]
    if len(set(batches)) != len(batches):
        parser.error('each batch must appear only once')
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds < 1:
        parser.error('warmup must be nonnegative; iterations and timeout must be positive')

    a5_root = Path(os.environ.get('KVIP_A5_ROOT', '/home/w00568585/li-aligned-a5'))
    kvaf = Path(os.environ.get('KVIP_KVAF_ROOT', a5_root / 'KVAF-unified-813f1750')).resolve()
    variants = {
        'baseline': dict(service_ub_candidates=False, service_post_chunk=0),
        'ub': dict(service_ub_candidates=True, service_post_chunk=0),
        'batch': dict(service_ub_candidates=False, service_post_chunk=64),
    }
    order = MODES[::-1] if args.reverse_order else MODES
    cases = {batch: dict(arguments=arguments(args, kvaf, batch), geometry=expected_geometry(batch),
                         output=f'bs{batch}' if len(batches) > 1 else '.', worker_exit_codes={})
             for batch in batches}
    manifest = dict(status='NOT_RUN', source_commit=git(ROOT, 'rev-parse', 'HEAD'),
                    source_dirty=bool(git(ROOT, 'status', '--porcelain')), baseline_commit=BASE,
                    kvaf_commit=KVAF, local_counts=False,
                    order=order, variants=variants, batches=batches, cases=cases)
    if len(batches) == 1:
        case = cases[batches[0]]
        manifest.update(arguments=case['arguments'], worker_exit_codes=case['worker_exit_codes'])
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    base = Path(os.environ.get('KVIP_AB_OUTPUT', a5_root / 'results')).resolve()
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mr28_prefetch.', dir=base))
    print(f'A5 prefetch comparison: {output}', flush=True)
    try:
        if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
            raise ValueError(f'Use the unified KVAF checkout at {KVAF}')
        from execution import launch
        from workloads import configuration

        manifest['status'] = 'RUNNING'
        save(output / 'manifest.json', manifest)
        rows = []
        for batch, case in cases.items():
            directory = output / case['output']
            completed = case['worker_exit_codes']
            for mode in order:
                config = configuration(case['arguments'])
                config.mr25_handoff = mode
                config.handoff_build_dir = output / 'build' / mode
                config.service_ub_candidates = variants[mode]['service_ub_candidates']
                config.service_post_chunk = variants[mode]['service_post_chunk']
                print(f'Launching BS{batch} {mode}: {variants[mode]}', flush=True)
                worker = directory / mode
                completed[mode] = launch(config, worker)
                if completed[mode] and not (worker / 'worker-result.json').is_file():
                    # A timeout or a quarantined replay has no checked
                    # completion. Stop before reusing the same device.
                    manifest['stopped_after'] = dict(batch=batch, variant=mode, worker=str(worker))
                save(output / 'manifest.json', manifest)
                if 'stopped_after' in manifest:
                    break
            rows.extend(summarize_batch(directory, variants, completed, args.iterations, batch))
            if 'stopped_after' in manifest:
                print(f'A5 matrix stopped after an unchecked worker at {worker}', flush=True)
                break
        passed = summarize(output, rows) and 'stopped_after' not in manifest
        manifest['status'] = 'PASS' if passed else 'FAILED'
        return int(not passed)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(status='FAILED', error=str(error))
        print(f'Comparison incomplete: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        if manifest['status'] == 'RUNNING':
            manifest['status'] = 'FAILED'
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    sys.exit(main())
