# SPDX-License-Identifier: Apache-2.0
"""Compare communication widths on the same binary and recorded inputs."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys

from ranked_five_report import fmt
from ranked_help_report import compare, load


def read_group(roots, group):
    reports, arms = {}, {}
    for width, root in roots.items():
        report = json.loads((root / f'g{group}' / 'comparison.json').read_text())
        record = report['arms']['candidate']
        if report['group'] != group or report['early_index']['q_workers'] * 4 != width:
            raise ValueError(f'G{group}: expected {width} communication cores at {root}')
        reports[width] = report
        arms[width] = load(root, group, 'candidate', record['revision'], 'joint')
    control = arms[8]
    for width, arm in arms.items():
        if arm['binary'] != control['binary'] or arm['revision'] != control['revision']:
            raise ValueError(f'G{group}: width {width} uses a different binary or source')
        comparison = compare(control, arm)
        if comparison['placement'] != 'matched' or not comparison['same_capture_sha256']:
            raise ValueError(f'G{group}: mismatched placement or capture for width {width}')
        yield width, reports[width], comparison['exact_topk_matched']


def run_width(candidate, reference, width, groups):
    runner = Path(__file__).resolve().parents[1] / 'peer_fused.sh'
    command = ['bash', str(runner), '--ranked-joint-local', candidate, '--reference', str(reference)]
    env = dict(os.environ, PEER_RANKED_Q_WORKERS=str(width // 4),
               PEER_RANKED_GROUPS=' '.join(map(str, groups)), PYTHONUNBUFFERED='1')
    print(f'REPLAY comm={width} groups={groups}', flush=True)
    last = deque(maxlen=12)
    root = None
    prefixes = ('CASE_BEGIN ', 'G2 PASS ', 'G4 PASS ', 'G7 PASS ', 'G10 PASS ', 'G17 PASS ',
                '  GRAPH ', '  GAIN ', '  STAGE ', '  MATCH ', '  DRIFT ', '  FOURTH ',
                '  EARLY_INDEX ', '  TAIL ', 'RESULT_ROOT=')
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            line = line.rstrip()
            last.append(line)
            if line.startswith(prefixes):
                print(line, flush=True)
            if line.startswith('REPLAY_COMPLETE RESULT_ROOT='):
                root = Path(line.partition('=')[2])
        if process.wait() or root is None:
            sys.exit('REPLAY_FAILED comm=' + str(width) + '\n' + '\n'.join(last))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for width in (4, 8, 16):
        parser.add_argument(f'--comm{width}', type=Path)
    parser.add_argument('--replay', metavar='CANDIDATE_SHA', help='Run the candidate on the existing snapshots')
    parser.add_argument('--reference', type=Path, help='Archived same-host original and 8-core control')
    parser.add_argument('--groups', type=int, nargs='+', default=[7, 4], choices=[7, 4, 10, 17, 2])
    args = parser.parse_args()
    roots = {w: getattr(args, f'comm{w}') for w in (4, 8, 16)}
    if args.replay:
        if not args.reference or any(roots.values()):
            parser.error('--replay requires --reference and obtains its own result directories')
        # Run the 8-core candidate once on all five loads. Screen alternative
        # widths on the requested loads; completed arms are reused on expansion.
        for width in (8, 4, 16):
            groups = [7, 4, 10, 17, 2] if width == 8 else args.groups
            roots[width] = run_width(args.replay, args.reference, width, groups)
    elif not all(roots.values()) or args.reference:
        parser.error('specify --replay and --reference, or three completed --comm directories')
    for marker in ('host.txt', 'physical_npu.txt', 'reference.txt'):
        if len({(p / marker).read_text().strip() for p in roots.values()}) != 1:
            parser.error(f'result roots differ in {marker}')
    positive = {4: set(), 16: set()}
    for group in args.groups:
        rows = list(read_group(roots, group))
        ref = rows[1][1]['arms']['candidate']
        for width, report, match in rows:
            arm = report['arms']['candidate']
            saved = [ref[k] - arm[k] for k in ('event_mean_us', 'profiler_p50_us')]
            same = match['span_p50_saved_us']
            if width != 8 and min(saved) > 0 and same is not None and same > 0:
                positive[width].add(group)
            orig = report['comparisons']['original']['saved_us']
            print(f'WIDTH G{group} comm={width} mean/p50_us={arm["event_mean_us"]:.2f}/{arm["profiler_p50_us"]:.2f} '
                  f'saved_vs_new8_us={saved[0]:+.2f}/{saved[1]:+.2f} '
                  f'same_TopK_vs_new8_p50_us={fmt(same)} pairs={match["n"]}/{match["before_calls"]} '
                  f'saved_vs_original_us={orig[0]:+.2f}/{orig[1]:+.2f}')
    non_small = set(args.groups) - {2}
    if non_small:
        consistent = [w for w, wins in positive.items() if non_small <= wins]
        print('WIDTH_SCOPE groups=' + ','.join(map(str, sorted(non_small))) +
              ' faster_than_new8_all_three_metrics=' + (','.join(map(str, consistent)) or 'none') +
              ' cross_run_drift=unmeasured')


if __name__ == '__main__':
    main()
