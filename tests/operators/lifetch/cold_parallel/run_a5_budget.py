#!/usr/bin/env python3
"""Window-specific A5 shortlist using the existing full-chain runner.

Plan by default. --execute runs serial controls, including matched legacy P2,
staged-before, and direct/phase/inline controls; r0 timing is discarded.
"""
import argparse
from pathlib import Path
import sys

from explore_a5_budget import SHORTLIST
from run_controls import main as run_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--kvaf-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--aic-budget', type=int, default=32)
    parser.add_argument('--aiv-budget', type=int, default=64)
    parser.add_argument('--windows', choices=('2', '8', '2,8'), default='2,8')
    parser.add_argument('--prefetch', default='0,512')
    parser.add_argument('--fetch-workers', default='2')
    parser.add_argument('--candidate-posts', default='direct', help='direct,scatter for the secondary pack-grain comparison')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--kvaf-server')
    parser.add_argument('--valid-history', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.aic_budget <= 32 and 2 <= args.aiv_budget <= 64):
        parser.error('use an actually available AIC/AIV budget')
    capacity = min(args.aic_budget, args.aiv_budget // 2)
    selected = {int(w): [pair for pair in SHORTLIST[int(w)] if max(pair) <= capacity]
                for w in args.windows.split(',')}
    if any(not pairs for pairs in selected.values()):
        parser.error('budget is too small for this shortlist; use run_controls.py for smaller P')
    if args.execute and args.output.exists():
        parser.error('choose a fresh parent output directory')
    for windows, pairs in selected.items():
        argv = ['--matrix', 'tuning', '--repo', str(args.repo),
                '--kvaf-root', str(args.kvaf_root), '--output', str(args.output / f'W{windows}'),
                '--python', args.python, '--device', str(args.device),
                '--aic-budget', str(args.aic_budget), '--aiv-budget', str(args.aiv_budget),
                '--windows', str(windows), '--prefetch', args.prefetch,
                '--fetch-workers', args.fetch_workers, '--prepares', 'phase',
                '--candidate-posts', args.candidate_posts,
                '--partition-pairs', ','.join(f'{ph}:{pc}' for ph, pc in pairs),
                '--rounds', str(args.rounds), '--warmup', str(args.warmup),
                '--iterations', str(args.iterations), '--timeout', str(args.timeout)]
        if args.kvaf_server:
            argv += ['--kvaf-server', args.kvaf_server]
        if args.valid_history is not None:
            argv += ['--valid-history', str(args.valid_history)]
        if args.execute:
            argv.append('--execute')
        run_controls(argv)


if __name__ == '__main__':
    main()
