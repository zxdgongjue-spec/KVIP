"""Measure a B20 candidate and compare its timings with the published B24 run."""
import argparse
from pathlib import Path

from li_restore_report import saved, stats
from li_window_report import KVAF, aggregate, report_workers
from main_five_report import GROUPS, MAIN, measured_artifacts, read
from portable_score_oracle import ARCHIVE, REQUEST_STATE
from reporting import atomic_json


def historical_times():
    archive = read(Path(__file__).with_name('b24_main_five_archive.json'))
    if archive['revision'] != MAIN or archive['physical_npu'] != 4:
        raise ValueError('Expected B24 physical NPU4 production mainline')
    return archive['groups']


def group_report(root, group, physical_npu=0):
    revision = (root/'candidate.txt').read_text().strip()
    if (revision == MAIN or (root/'main.txt').read_text().strip() != MAIN
            or (root/'kvaf.txt').read_text().strip() != KVAF
            or (root/'physical_npu.txt').read_text().strip() != str(physical_npu)):
        raise ValueError('Candidate revision, transport or physical NPU mismatch')
    folder, _, result, current, _, proof, _ = measured_artifacts(
        root, group, 'li-window', revision, verify_library=True, physical_npu=physical_npu)
    check = read(folder/'portable_score_check.json')
    if ((check['group'], check['calls'], check['queries']) != (group, 20, 1280)
            or check['score_multiset_sha256'] != ARCHIVE[group][1]
            or check['score_domain_sha256'] != ARCHIVE[group][0]
            or check['request_state_sha256'] != REQUEST_STATE[group]
            or check['topk_set_digests'] != result['topk_set_digests']
            or len(check['topk_set_digests']) != 20 or proof['runtime']['visible'] != str(physical_npu)):
        raise ValueError('Archived score verification and measured candidate disagree')
    previous = historical_times()[str(group)]
    host = proof['runtime']['host']
    print(f'G{group} CHECK TopK_scores_vs_verified_B24_archive=PASS calls=20/20 queries=1280/1280 '
          f'KV_and_Index_state=PASS measured_rows=320/320 ordered_Index=PASS separate_diagnostic_rows=320/320')
    print(f'G{group} B20_candidate', *(f'{k}_mean/P50_us={stats(v)}' for k, v in current.items()), f'host={host}',
          f'physical_npu={physical_npu}')
    print(f'G{group} B24_main_archive', *(f'{k}_mean/P50_us={stats(v)}' for k, v in previous.items()),
          'physical_npu=4')
    print(f'G{group} B24_archive_minus_B20_candidate cross_host',
          *(f'{k}_delta_mean/P50_us={saved(previous[k], v)}' for k, v in current.items()))
    report_workers(folder, group)
    atomic_json(root/f'g{group}/comparison.json', dict(group=group, production_revision=revision,
                host=host, physical_npu=physical_npu, mainline=previous, candidate=current))


def summary(root, physical_npu=0):
    rows = [read(root/f'g{group}/comparison.json') for group in GROUPS]
    revision = (root/'candidate.txt').read_text().strip()
    if any(r['production_revision'] != revision for r in rows) or len({r['host'] for r in rows}) != 1:
        raise ValueError('Cannot aggregate measurements from different candidates or hosts')
    # Older B20 result files were produced by the NPU0-only runner.
    if ((root/'physical_npu.txt').read_text().strip() != str(physical_npu)
            or any(row.get('physical_npu', 0) != physical_npu for row in rows)):
        raise ValueError('Cannot aggregate measurements from a different physical NPU')
    before, after, delta = aggregate(rows)
    print(f'FIVE_GROUP_EQUAL_WEIGHT cross_host B24_main_graph_mean_us={before:.3f} '
          f'B20_candidate_graph_mean_us={after:.3f} delta_us={before-after:+.3f} '
          f'delta_pct={delta:+.3f} same_host_10pct_target=UNVERIFIED '
          f'B20_physical_npu={physical_npu} B24_physical_npu=4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('group')
    sub.add_argument('root', type=Path)
    sub.add_argument('group', type=int, choices=GROUPS)
    total = commands.add_parser('summary')
    total.add_argument('root', type=Path)
    for command in (sub, total):
        command.add_argument('--physical-npu', type=int, choices=(0, 4), default=0)
    args = parser.parse_args()
    if args.command == 'group':
        group_report(args.root, args.group, args.physical_npu)
    else:
        summary(args.root, args.physical_npu)
