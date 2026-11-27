# SPDX-License-Identifier: Apache-2.0
"""Four fused arms for fixed-score and last-window Q1 at unchanged geometry."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from statistics import mean
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = 'af52a02b97daddadd5d93b9d100f2ff1b03b499d'
ARMS = ('p2_q4', 'fixed_q4', 'p2_q1', 'fixed_q1')
FIXED_GROUPS_PER_REQUEST = {1: 8, 2: 8, 4: 8, 8: 4}
MISSES = (5, 10, 15, 20)


def plan(args):
    cases = []
    for batch in dict.fromkeys(args.batch or (1, 4)):
        windows = 2 if batch == 1 else 4
        service_lanes = 2 if batch == 1 else batch
        for miss in dict.fromkeys(args.miss_percent or (5, 15)):
            for arm in ARMS:
                cases.append(dict(batch=batch, miss_percent=miss, arm=arm, windows=windows,
                                  score_groups=batch * (FIXED_GROUPS_PER_REQUEST[batch]
                                                        if arm.startswith('fixed_') else 2),
                                  service_lanes=service_lanes, q_workers=4 if arm.endswith('_q1') else 1))
    return dict(base_commit=BASE, cases=cases, device=args.device, history=131072, resident=16384,
                queries=4, index_heads=32, requested_windows=4, prefetch_tokens=512,
                prefetch_window_limit=128, prefetch_rank_start=1024, seed=43,
                warmup=args.warmup, iterations=args.iterations, timeout_seconds=args.timeout_seconds)


def native_arguments(args, case):
    return [
        'diagnose', '--input', 'synthetic', '--transport', 'urma', '--level', '1',
        '--kvaf-root', str(args.kvaf_root), '--kvaf-server', args.kvaf_server,
        '--device', str(args.device), '--timeout-seconds', str(args.timeout_seconds),
        '--batches', str(case['batch']), '--miss-percent', str(case['miss_percent']),
        '--history', '131072', '--resident', '16384', '--queries', '4', '--index-heads', '32',
        '--seed', '43', '--prefetch-tokens', '512', '--prefetch-rank-start', '1024',
        # The native runtime clamps BS1 to W2 after deriving the W4 request's 128-entry cap.
        '--intermediate-fetches', '4', '--warmup', str(args.warmup), '--iterations', str(args.iterations),
    ]


def read_json(path):
    return json.loads(Path(path).read_text())


def files_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            part = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b''):
                part.update(block)
            digest.update(part.digest())
    return digest.hexdigest()


def read_counters(samples, case, iterations):
    if [sample['accounting']['invocation'] for sample in samples] != list(range(iterations)):
        raise ValueError('Missing or duplicated attributed graph invocations')
    batch, windows = case['batch'], case['windows']
    counts = []
    for sample in samples:
        algorithm = sample['algorithm']
        if 'error' in algorithm:
            raise ValueError(f"Invalid Get counters: {algorithm['error']}")
        requests = algorithm['requests']
        if [request['request'] for request in requests] != list(range(batch)):
            raise ValueError('Incomplete Get counters by request')
        call = []
        for request in requests:
            early = [window['posted'] for window in request['windows']]
            final = [query['final_posted'] for query in request['queries']]
            if (len(early) != windows or len(final) != 4
                    or sum(early) != request['prefetch'] or sum(final) != request['final']
                    or any(not 0 <= count <= 128 for count in early)):
                raise ValueError('Decoded windows or Get totals disagree with the arm')
            call.append(dict(prefetch_per_window=early, final_per_q=final))
        counts.append(call)
    return counts


def launch_spans(intervals):
    # Two overlapping AIC/AIV events belong to the same mixed launch.
    # Touching intervals still denote separate launches.
    spans = []
    for start, end in sorted(intervals):
        if spans and start < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return spans


def read_stages(samples):
    times = []
    for sample in samples:
        intervals = sample['accounting']['intervals_us']
        li = launch_spans(intervals['li'])
        terminal = launch_spans(intervals['topk_resolve_update'])
        # Graph attribution accepts only named kernels. Within each checked
        # invocation the small route always launches Hot, Threshold, Cold, Terminal.
        if len(li) != 3 or len(terminal) != 1 or li[-1][1] > terminal[0][0]:
            raise ValueError(f'Expected three ordered LI kernels and one Terminal: {intervals}')
        intervals = [*li, *terminal]
        if any(not math.isfinite(start) or not math.isfinite(end) or start >= end for start, end in intervals):
            raise ValueError(f'Invalid kernel spans: {intervals}')
        times.append([end - start for start, end in intervals])
    return dict(zip(('hot_us', 'threshold_us', 'cold_service_us', 'terminal_us'), map(mean, zip(*times))))


def read_arm(worker, case, iterations):
    folder = worker / f"miss{case['miss_percent']}"
    case_dir = folder / f"case-b{case['batch']}"
    report = read_json(case_dir / 'report.json')
    path = case_dir / 'fused/path_result.json'
    result = read_json(path)
    if (report['execution'] != 'COMPLETED' or report['correctness'] != 'PASS'
            or result['correctness'] != 'PASS'):
        raise ValueError(f'Unchecked benchmark result: {path}')
    route = result['combo']
    for field in ('arm', 'score_groups', 'windows', 'service_lanes', 'q_workers'):
        if route[field] != case[field]:
            raise ValueError(f"Unexpected {field} {route[field]!r} for {case['arm']}")
    timing = result['all_rounds']
    event = timing['event_mean_us']
    if (timing['graph_calls'] != iterations or timing['measured_graphs'] != 1
            or not math.isfinite(event) or event <= 0):
        raise ValueError(f'Incomplete per-call Graph Event timing: {path}')
    attribution = case_dir / 'fused/diagnose/round-1/full_attribution.json'
    data = read_json(attribution)
    samples = data['samples']
    if data['graph_calls'] != iterations or len(samples) != iterations:
        raise ValueError(f'Incomplete profiling samples: {attribution}')
    counts = read_counters(samples, case, iterations)
    prefetch = mean(sum(sum(row['prefetch_per_window']) for row in call) for call in counts)
    final = mean(sum(sum(row['final_per_q']) for row in call) for call in counts)
    fixture = folder / 'fixture'
    sources = [fixture / name for name in ('fixture.json', 'physical_to_logical.npy', 'logical_scores.npy',
                                           *(f'arg{i}.bin' for i in range(5)))]
    library = Path(route['library']).resolve()
    pool = read_json(worker / 'pool.json')
    return dict(**case, correctness='PASS', kv_ready_us=event, profiler_p50_us=timing['p50_us'],
                **read_stages(samples), prefetch_per_call=prefetch, final_get_per_call=final,
                total_get_per_call=prefetch + final, fixture_sha256=files_digest(sources),
                library_sha256=files_digest([library]), library=str(library), counts=counts,
                put_shutdown_retries=pool.get('shutdown_retries', 0),
                trace=data['trace'], report=str(path))


def match_counts(left, right):
    paired = [(a, b) for call_a, call_b in zip(left, right, strict=True)
              for a, b in zip(call_a, call_b, strict=True)]
    if all(a == b for a, b in paired):
        return 'same_window_and_final_q_counts'
    def totals(row):
        early = sum(row['prefetch_per_window'])
        final = sum(row['final_per_q'])
        return early, final
    values = [(totals(a), totals(b)) for a, b in paired]
    if all(a == b for a, b in values):
        return 'same_prefetch_and_final_counts'
    if all(sum(a) == sum(b) for a, b in values):
        return 'same_total_counts'
    return 'different_request_totals'


def compare_cell(arms):
    p2_q4, fixed_q4, p2_q1, fixed_q1 = (arms[arm] for arm in ARMS)
    for field in ('fixture_sha256', 'library_sha256'):
        if len({value[field] for value in arms.values()}) != 1:
            raise ValueError(f'The four arms used different {field}')
    a, b, c, d = (row['kv_ready_us'] for row in (p2_q4, fixed_q4, p2_q1, fixed_q1))
    return dict(batch=p2_q4['batch'], miss_percent=p2_q4['miss_percent'],
                fixed_score_savings_us=a - b, q1_savings_on_p2_us=a - c,
                q1_savings_on_fixed_us=b - d, combined_savings_us=a - d,
                combined_savings_percent=100 * (a - d) / a, interaction_savings_us=(b - d) - (a - c),
                q1_on_p2_get_match=match_counts(p2_q4['counts'], p2_q1['counts']),
                q1_on_fixed_get_match=match_counts(fixed_q4['counts'], fixed_q1['counts']),
                fixed_score_get_match=match_counts(p2_q4['counts'], fixed_q4['counts']),
                combined_get_match=match_counts(p2_q4['counts'], fixed_q1['counts']))


def write_reports(output, arms, cells):
    from reporting import atomic_json

    atomic_json(output / 'summary.json', dict(arms=arms, cells=cells))
    for name, rows in (('arms', arms), ('comparison', cells)):
        if rows:
            keys = [key for key in rows[0] if key != 'counts']
            with (output / f'{name}.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)


@contextmanager
def kvaf_service(args, output):
    host, port = args.kvaf_server.rsplit(':', 1)
    endpoint = (host, int(port))

    def listening():
        try:
            with socket.create_connection(endpoint, timeout=1):
                return True
        except OSError:
            return False

    if listening():
        yield
        return
    if host not in ('127.0.0.1', 'localhost'):
        raise RuntimeError(f'KVAF is unavailable at {args.kvaf_server}')
    service = args.kvaf_root / 'build-cpu/runtime/kvaf_cpu_service'
    log = output / 'kvaf-service.log'
    with log.open('w') as stream:
        process = subprocess.Popen([str(service), '--listen', host, '--port', port],
                                   stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            for _ in range(15):
                if process.poll() is not None:
                    break
                if listening():
                    yield
                    return
                time.sleep(1)
            raise RuntimeError(f'KVAF service did not start. See {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, action='append', choices=tuple(FIXED_GROUPS_PER_REQUEST))
    parser.add_argument('--miss-percent', type=int, action='append', choices=MISSES)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--device', type=int, default=os.environ.get('KVIP_A5_DEVICE', '0'))
    parser.add_argument('--timeout-seconds', type=int, default=2400)
    parser.add_argument('--kvaf-server', default=os.environ.get('KVIP_KVAF_SERVER', '127.0.0.1:28986'))
    parser.add_argument('--output', type=Path, default=os.environ.get('KVIP_AB_OUTPUT', ROOT / 'tests/results/lifetch-combo'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.warmup, args.device) < 0 or min(args.timeout_seconds, args.iterations) <= 0:
        parser.error('device and warmup must be nonnegative; iterations and timeout must be positive')
    try:
        host, port = args.kvaf_server.rsplit(':', 1)
        if not host or not 0 < int(port) <= 65535:
            raise ValueError
    except ValueError:
        parser.error('KVAF server must be HOST:PORT')
    matrix = plan(args)
    if args.dry_run:
        print(json.dumps(matrix, indent=2))
        return 0
    if not os.environ.get('KVIP_KVAF_ROOT'):
        parser.error('set KVIP_KVAF_ROOT')
    args.kvaf_root = Path(os.environ['KVIP_KVAF_ROOT']).resolve()
    sys.path.insert(0, str(ROOT / 'python'))
    from execution import launch
    from reporting import atomic_json
    from workloads import configuration

    args.output.mkdir(exist_ok=True, parents=True)
    output = Path(tempfile.mkdtemp(prefix='fixed-q4-combo.', dir=args.output.resolve()))
    print(f'Run output: {output}', flush=True)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    kvaf = subprocess.check_output(['git', '-C', str(args.kvaf_root), 'rev-parse', 'HEAD'], text=True).strip()
    atomic_json(output / 'manifest.json', dict(matrix, commit=commit, kvaf_commit=kvaf))
    rows, cells, arms = [], [], {}
    with kvaf_service(args, output):
        for case in matrix['cases']:
            batch, miss, arm = case['batch'], case['miss_percent'], case['arm']
            worker = output / f'bs{batch}' / f'miss{miss}' / arm
            print(f'BS{batch} miss{miss} {arm}: {case["score_groups"]} MIX, W{case["windows"]}', flush=True)
            os.environ['KVIP_FIXED_Q4_COMBO_ARM'] = arm
            code = launch(configuration(native_arguments(args, case)), worker)
            if code:
                raise RuntimeError(f'Failed arm {arm}. See {worker / "summary.log"}')
            row = read_arm(worker, case, args.iterations)
            if rows and rows[0]['library_sha256'] != row['library_sha256']:
                raise ValueError('The benchmark library changed within the matrix')
            arms[arm] = row
            rows.append(row)
            if len(arms) == len(ARMS):
                cell = compare_cell(arms)
                cells.append(cell)
                print(f"  Combined {cell['combined_savings_percent']:+.2f}% ({cell['combined_get_match']}), "
                      f"Q1 on fixed {cell['q1_savings_on_fixed_us']:+.2f} µs", flush=True)
                arms = {}
            write_reports(output, rows, cells)
            print(f"  KV ready {row['kv_ready_us']:.2f} µs, early/final "
                  f"{row['prefetch_per_call']:.2f}/{row['final_get_per_call']:.2f}", flush=True)
    print(f'Finished {len(rows)} checked arms. Comparisons: {output / "comparison.csv"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
