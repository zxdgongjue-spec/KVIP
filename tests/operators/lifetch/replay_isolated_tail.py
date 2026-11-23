"""Replay only the isolated LI candidate on B24, then verify the loaded device code."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


GROUPS = (10, 4, 17, 7, 2)
MAIN = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
BEFORE = '22c4c4102dfe2b35dea1f90897a3fd918ae837f1'
FIXED = 'afe227c98ed17633f757163e3077fa090a743414'
READER = 'ff3622d5f4da96004e1e63c2621dadfd6804fc67'


def loaded_library(root, revision):
    arm = 'mainline' if revision == MAIN else 'li-window'
    tag = 'main.txt' if revision == MAIN else 'candidate.txt'
    if (root / tag).read_text().strip() != revision or (root / 'physical_npu.txt').read_text().strip() != '4':
        raise ValueError(f'Archive identity mismatch: {root}')
    builds = set()
    for group in GROUPS:
        results = root / f'g{group}' / arm / 'results'
        matches = list(results.glob('*/cases/diagnose/fused/mainline_checks.json'))
        if len(matches) != 1:
            raise ValueError(f'Expected one G{group} library receipt in {results}, found {len(matches)}')
        proof = json.loads(matches[0].read_text())
        if proof['production_revision'] != revision or proof['profiled_terminal_kernel'] != 'v9_parallel_terminal':
            raise ValueError(f'Unexpected production kernel in {matches[0]}')
        builds.add((Path(proof['loaded_library']), proof['library_sha256']))
    if len({digest for _, digest in builds}) != 1:
        raise ValueError(f'The five cases used different library contents in {root}')
    for path, digest in sorted(builds):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return path
    raise ValueError(f'No loaded library matches the recorded SHA-256 in {root}: {sorted(builds)}')


def run_fixed(repo, work):
    for revision in (READER, FIXED):
        subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', f'{revision}^{{commit}}'], check=True)
    source = subprocess.check_output(['git', '-C', str(repo), 'show',
                                      f'{READER}:tests/operators/lifetch/main_five_npu4.sh'])
    env = dict(os.environ, PEER_G2_WORK=str(work), PEER_PHYSICAL_NPU='4',
               ASCEND_RT_VISIBLE_DEVICES='4', PEER_LOCAL_ONLY='1')
    root = None
    with tempfile.TemporaryDirectory() as tmp:
        runner = Path(tmp) / 'replay.sh'
        runner.write_bytes(source)
        command = ['bash', str(runner), str(repo), READER, FIXED]
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                if line.startswith('RESULT_ROOT='):
                    root = Path(line.strip().split('=', 1)[1])
            if proc.wait():
                raise SystemExit(proc.returncode)
    if root is None:
        raise ValueError('Missing RESULT_ROOT in replay output')
    return root


def report_changes(before, fixed):
    for group in GROUPS:
        def load_times(root, revision):
            row = json.loads((root / f'g{group}/comparison.json').read_text())
            if row['group'] != group or row['production_revision'] != revision:
                raise ValueError(f'G{group} comparison identity mismatch in {root}')
            return row
        old, new = load_times(before, BEFORE), load_times(fixed, FIXED)
        if Path(old['archive']).resolve() != Path(new['archive']).resolve() or old['mainline'] != new['mainline']:
            raise ValueError(f'G{group} comparisons used different mainline archives')
        parts = []
        for key, label in [('graph', 'graph'), ('LI1_to_LI3', 'LI'), ('final_topk_through_tail', 'tail')]:
            avg, p50 = (a - b for a, b in zip(old['candidate'][key], new['candidate'][key]))
            parts.append(f'{label}_saved_mean/P50_us={avg:+.2f}/{p50:+.2f}')
        print(f'G{group} vs_before_isolation', *parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--result', type=Path, help='only verify this completed isolated replay')
    args = parser.parse_args()
    before = args.work / 'peer-baseline/li-window-five.DRHZaO'
    baseline = Path((before / 'reference.txt').read_text().strip())
    old_main = loaded_library(baseline, MAIN)
    old_candidate = loaded_library(before, BEFORE)
    fixed = args.result or run_fixed(args.repo, args.work)
    if Path((fixed / 'reference.txt').read_text().strip()).resolve() != baseline.resolve():
        raise ValueError('Isolated replay selected a different mainline archive')
    new_candidate = loaded_library(fixed, FIXED)
    check = Path(__file__).parent / 'support/check_isolated_device.py'
    subprocess.run([sys.executable, str(check), str(old_main), str(old_candidate), str(new_candidate)], check=True)
    report_changes(before, fixed)
    print(f'ISOLATION_VERIFIED RESULT_ROOT={fixed}', flush=True)


if __name__ == '__main__':
    main()
