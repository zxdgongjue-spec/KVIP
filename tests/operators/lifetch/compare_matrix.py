#!/usr/bin/env python3
"""Compare native chain, current fusion and pinned main for 20 geometries."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from runner import supervise

MAIN = '81b1472809b7a5f3469e26467feee0ca1c61d71a'
BATCHES = (1, 2, 4, 8, 16)
MISSES = (3, 5, 10, 15)


def worker_pythonpath(code, kvaf, inherited):
    """Prioritize selected checkouts while retaining CANN/ACL environment paths."""
    return os.pathsep.join(filter(None, (str(code / "python"),
                                       str(kvaf / "build-npu/python"), inherited)))


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def summarize(run):
    rows, wide = [], []
    for batch in BATCHES:
        for miss in MISSES:
            pair = {}
            for variant in ('current', 'main'):
                reports = list((run / f'{variant}-b{batch}-m{miss}').glob('case*/report.json'))
                if len(reports) != 1:
                    continue
                report = json.loads(reports[0].read_text())
                if report['status'] != 'pass':
                    continue
                pair[variant] = report
                keys = [('current' if variant == 'current' else 'main_lifetch', 'prototype_with_update')]
                if variant == 'current':
                    keys.insert(0, ('serial_native', 'native_chain'))
                for name, key in keys:
                    timing = report['timing'][key]
                    rows.append(dict(batch=batch, miss_percent=miss, variant=name,
                                     p50_us=timing['p50_us'], p99_us=timing['p99_us'],
                                     status='PASS', report=str(reports[0])))
            if len(pair) == 2:
                if pair['current']['fixture'] != pair['main']['fixture']:
                    raise RuntimeError(f'Fixture metadata differs: BS={batch}, miss={miss}')
                if pair['current']['native_misses_per_request'] != pair['main']['native_misses_per_request']:
                    raise RuntimeError('Native miss counts differ between variants')
                serial = pair['current']['timing']['native_chain']['p50_us']
                current = pair['current']['timing']['prototype_with_update']['p50_us']
                main = pair['main']['timing']['prototype_with_update']['p50_us']
                native_main = pair['main']['timing']['native_chain']['p50_us']
                wide.append(f'{batch} | {miss} | {serial:.3f} | {current:.3f} | {main:.3f} | '
                            f'{(1-current/serial)*100:+.2f}% | {(1-current/main)*100:+.2f}% | '
                            f'{(native_main/serial-1)*100:+.2f}%')
    if rows:
        with (run / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = '\n'.join([
        'Graph p50 us; positive reduction = faster. Correctness checked outside timing.',
        'Operator-chain end-to-end, excludes model/SFA and CPU fixture setup.',
        'Native drift compares the extra native reference measured in main against current.',
        'BS | miss% | serial_native | current_fused | main_lifetch | vs serial reduction | vs main reduction | native drift',
        *wide, f'Completed comparison rows: {len(rows)}/60',
    ])
    (run / 'summary.log').write_text(text + '\n')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kvaf-root', type=Path, required=True)
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--server', default='127.0.0.1:28885')
    parser.add_argument('--soc')
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--trace-rounds', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=900, help='per geometry, not whole matrix')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.trace_rounds < 0 or args.timeout_seconds < 20:
        parser.error('invalid warmup, iterations, trace rounds or timeout')
    run = ROOT / 'tests/results/lifetch-matrix' / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    run.mkdir(parents=True)
    kvaf = args.kvaf_root.resolve()
    refs = {'current': git('rev-parse', 'HEAD'), 'main': git('rev-parse', MAIN)}
    (run / 'versions.json').write_text(json.dumps(dict(refs=refs, kvaf=str(kvaf), args=vars(args)), default=str, indent=2))
    print(f'Results: {run}\nVersions: {refs}', flush=True)
    codes, builds, digests = {}, {}, {}
    for variant, ref in refs.items():
        code = run / f'{variant}-source'
        subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(code), ref], check=True)
        vendors = code / 'csrc/operators/lidu/_custom_opp/vendors'
        vendors.parent.mkdir(parents=True, exist_ok=True)
        vendors.symlink_to(ROOT / 'csrc/operators/lidu/_custom_opp/vendors', target_is_directory=True)
        codes[variant] = code
    for batch in BATCHES:
        for miss in MISSES:
            # Alternate paired order to avoid always warming one variant first.
            order = ('current', 'main') if MISSES.index(miss) % 2 == 0 else ('main', 'current')
            for variant in order:
                folder = run / f'{variant}-b{batch}-m{miss}'
                folder.mkdir()
                if variant in builds:
                    (folder / 'universal-build').symlink_to(builds[variant], target_is_directory=True)
                    print(f'REUSE {variant}: {builds[variant]} sha256={digests[variant]}', flush=True)
                else:
                    print(f'COMPILE {variant}: first case only', flush=True)
                code = codes[variant]
                command = [sys.executable, '-u', '-X', 'faulthandler',
                           str(code / 'tests/operators/lifetch/test_lifetch.py'), '--internal-urma',
                           '--worker-run', str(folder), '--kvaf-root', str(kvaf), '--device', str(args.device),
                           '--kvaf-server', args.server, '--batches', str(batch), '--miss-percent', str(miss),
                           '--history', '131072', '--resident', '16384', '--index-heads', '32',
                           '--intermediate-fetches', '8', '--prefetch-tokens', '512',
                           '--warmup', str(args.warmup), '--iterations', str(args.iterations),
                           '--trace-rounds', str(args.trace_rounds)]
                if args.soc:
                    command += ['--soc', args.soc]
                if variant == 'current':
                    command += ['--streaming', '--streaming-fetch', '--fused-final', '--q-workers', '2',
                                '--prepare-update-metadata', '--optimized-resolve']
                old_path = os.environ.get('PYTHONPATH')
                os.environ['PYTHONPATH'] = worker_pythonpath(code, kvaf, old_path)
                print(f'=== {variant} BS={batch} miss={miss}% ===', flush=True)
                try:
                    status = supervise(command, folder, timeout_seconds=args.timeout_seconds)
                finally:
                    if old_path is None:
                        os.environ.pop('PYTHONPATH', None)
                    else:
                        os.environ['PYTHONPATH'] = old_path
                if status:
                    print(summarize(run))
                    raise SystemExit(f'Failed {variant} BS={batch} miss={miss}; stopped, logs={folder}')
                build = folder / 'universal-build'
                if variant in builds and list(folder.glob('case*/build.log')):
                    raise RuntimeError(f'Unexpected recompilation despite shared binary: {folder}')
                digest = hashlib.sha256((build / 'libv9_bridge.so').read_bytes()).hexdigest()
                if variant in digests and digests[variant] != digest:
                    raise RuntimeError('Binary changed during matrix')
                builds.setdefault(variant, build.resolve())
                digests[variant] = digest
                (run / 'binary_sha256.json').write_text(json.dumps(digests, indent=2))
                if variant == 'current' and args.trace_rounds:
                    with (folder / 'timeline-console.log').open('w') as output:
                        subprocess.run([sys.executable, str(ROOT / 'tests/operators/lifetch/support/summarize_core_timeline.py'),
                                        str(folder)], check=True, stdout=output)
                summarize(run)
    print(summarize(run))
    print(f'CSV: {run / "results.csv"}\nSummary: {run / "summary.log"}')


if __name__ == '__main__':
    main()
