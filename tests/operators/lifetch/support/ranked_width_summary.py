# SPDX-License-Identifier: Apache-2.0
"""Compare communication widths on the same binary and recorded inputs."""

import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for width in (4, 8, 16):
        parser.add_argument(f'--comm{width}', type=Path, required=True)
    parser.add_argument('--groups', type=int, nargs='+', default=[7, 4], choices=[7, 4, 10, 17, 2])
    args = parser.parse_args()
    roots = {w: getattr(args, f'comm{w}') for w in (4, 8, 16)}
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
