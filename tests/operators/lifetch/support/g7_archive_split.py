"""Read the three completed B24 G7 runs before collecting additional owner timings."""
import sys
from math import isclose
from pathlib import Path
from statistics import mean

from cutoff_tail_archive import graph_report, read_json


RUNS = (
    ('原始融合版', 'fused.m3IizR', '24c619b9581aad43fd53c815928def26a1e19560'),
    ('Q1提前交接版', 'fused.0SOIpe', '67b49a2748fde8f8779c4551410416596b35cf76'),
    ('Q1接手预取回执版', 'fused.Cmp6BN', '7f75960b903be1b2474c87cba89f28ce877593f6'),
)


def read_calls(area, name, folder, revision):
    root = area / folder
    if ((root / 'logical-ties.txt').read_text().strip() != revision or
            (root / 'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError(f'{name} 的归档版本或物理 NPU 不符：{root}')
    path, = (root / 'g7/logical-ties/results').glob('*/cases/diagnose/fused/path_result.json')
    run = read_json(path)
    if run['production_revision'] != revision or run['correctness'] != 'PASS' or run['failures']:
        raise ValueError(f'{name} 的实测归档未通过校验：{path}')
    profile = read_json(path.parent / 'diagnose/round-1/full_attribution.json')
    rows, _, _ = graph_report(7, profile)
    if not isclose(mean(r['tail'] for r in rows), run['all_rounds']['stage_mean_us']['resolve_update'],
                   abs_tol=0.01):
        raise ValueError(f'{name} 的逐调用尾段和汇总计时不一致')
    return rows


def split_calls(rows):
    return [[r for r in rows if (r['tail'] > 300) == slower] for slower in (False, True)]


def decompose(previous, current):
    """Exact symmetric decomposition of a two-bin mean difference, in microseconds."""
    previous_total = sum(map(len, previous))
    current_total = sum(map(len, current))
    share_change = within_change = 0.0
    for before, after in zip(previous, current):
        if not before or not after:
            return None
        w0, w1 = len(before) / previous_total, len(after) / current_total
        t0, t1 = mean(r['tail'] for r in before), mean(r['tail'] for r in after)
        share_change += (w1 - w0) * (t0 + t1) / 2
        within_change += (t1 - t0) * (w0 + w1) / 2
    return share_change, within_change


def report(area):
    runs = []
    for name, folder, revision in RUNS:
        rows = read_calls(area, name, folder, revision)
        bins = split_calls(rows)
        runs.append(bins)
        for label, calls in zip(('不超过300us', '超过300us'), bins):
            if not calls:
                print(f'G7 {name} {label} 调用=0/20')
                continue
            requests = mean(r['prefetch'] for r in calls), mean(r['final'] for r in calls)
            conflicts = [sum(r['columns'][q][2] for r in calls) for q in (0, 3)]
            print(f'G7 {name} {label} 调用={len(calls)}/20 尾段均值_us={mean(r["tail"] for r in calls):.2f} '
                  f'每调用预取/补取数={requests[0]:.1f}/{requests[1]:.1f} '
                  f'Q1/Q4冲突目的槽描述符总数={conflicts[0]}/{conflicts[1]}')
    terms = decompose(runs[1], runs[2])
    if terms is not None:
        mix, within = terms
        print(f'G7 Q1接手预取回执版减Q1提前交接版 尾段均值差_us={mix + within:+.2f} '
              f'两档调用占比变化_us={mix:+.2f} 档内耗时变化_us={within:+.2f}（描述性分解）')


if __name__ == '__main__':
    report(Path(sys.argv[1]))
