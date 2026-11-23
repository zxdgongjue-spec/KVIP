"""Read the two existing owner-TopK production archives. No device or replay is used."""
import argparse
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import mean

from cutoff_tail_archive import mean_median, query_counts, read_json
from owner_topk_report import times
from query_owner_report import checked_layout_plan, load


REVISION = '888db6b1603d22f4142bd2ccffe0518611856a31'
ARCHIVES = {2: 'fused.gbxKC9', 7: 'fused.N7n8ZP'}


def workload(group, sample, plan):
    number = sample['accounting']['invocation'] + 1
    algorithm = sample['algorithm']
    columns, largest_row, (row, query, largest_owner) = query_counts(group, number, algorithm)
    requests = algorithm['requests']
    totals = algorithm['totals']
    # Counts identify the amount assigned to an owner, not its completion time.
    return dict(call=number, tail=sample['accounting']['stages_us']['resolve_update'],
                prefetch=totals['prefetch'], final=totals['final'], active=64-sum(c[3] for c in columns),
                safe=sum(c[1] for c in columns), unsafe=sum(c[2] for c in columns), columns=columns,
                max_row=largest_row['final'], row=largest_row['request'],
                max_owner=largest_owner, owner=f'R{row}Q{query}', plan=plan,
                counts=tuple((r['prefetch'], tuple((q['final_posted'], q['final_safe']) for q in r['queries']))
                             for r in requests))


def archive_calls(root, group):
    if ((root / 'logical-ties.txt').read_text().strip() != REVISION or
            (root / 'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError(f'G{group}: expected the production owner-TopK on NPU4')
    run = load(root, group, 'logical-ties', row_dependencies=True)
    folder, = (root / f'g{group}/logical-ties/results').glob('*/cases/diagnose/fused')
    timing = times(folder, run[3:5], joined=True)
    score = read_json(folder / 'score_topk_check.json')
    if (score['group'], score['calls'], score['queries']) != (group, 20, 1280):
        raise ValueError(f'G{group}: missing measured TopK score check')
    samples = read_json(folder / 'diagnose/round-1/full_attribution.json')['samples']
    if len(run[11]) != len(samples):
        raise ValueError(f'G{group}: missing physical Index receipts')
    states = {}
    rows = [workload(group, s, checked_layout_plan(plans, states, f'G{group} call {i+1}'))
            for i, (s, plans) in enumerate(zip(samples, run[11]))]
    if not isclose(mean(r['tail'] for r in rows), timing['final_topk_through_tail'][0], abs_tol=.025):
        raise ValueError(f'G{group}: mismatched joined tail timings')
    return rows


def summary(group, label, rows):
    fields = ('tail', 'prefetch', 'final', 'active', 'safe', 'unsafe', 'max_row', 'max_owner')
    entries = ' '.join(f'{k}={mean_median([r[k] for r in rows])}' for k in fields)
    return f'G{group} {label} mean/P50 {entries}'


def matched_halves(fast, slow):
    """Count matched calls without reusing a match; per-row matches are independent."""
    def shared(left, right):
        return sum((Counter(left) & Counter(right)).values())

    return dict(
        full_plan=shared((r['plan'] for r in fast), (r['plan'] for r in slow)),
        row_plans=sum(shared((r['plan'][i] for r in fast), (r['plan'][i] for r in slow)) for i in range(16)),
        workload=shared((r['counts'] for r in fast), (r['counts'] for r in slow)))


def describe(group, rows):
    print(summary(group, 'PRODUCTION_20', rows))
    for q in range(4):
        cols = [r['columns'][q] for r in rows]
        print(f'G{group} Q{q+1} per_call_mean final={mean(c[0] for c in cols):.2f} '
              f'safe={mean(c[1] for c in cols):.2f} unsafe={mean(c[2] for c in cols):.2f} '
              f'active_owners_of_16={mean(16-c[3] for c in cols):.2f}')
    for r in rows:
        print(f'G{group} call={r["call"]} joined_tail_us={r["tail"]:.2f} '
              f'prefetch/final={r["prefetch"]}/{r["final"]} active_owners={r["active"]}/64 '
              f'safe/unsafe={r["safe"]}/{r["unsafe"]} max_final_row=R{r["row"]}:{r["max_row"]} '
              f'max_final_owner={r["owner"]}:{r["max_owner"]}')
    if group == 7:
        ordered = sorted(rows, key=lambda r: (r['tail'], r['call']))
        fast, slow = ordered[:10], ordered[10:]
        print(summary(group, 'FASTEST_10', fast))
        print(summary(group, 'SLOWEST_10', slow))
        match = matched_halves(fast, slow)
        print(f'G7 CROSS_HALF full_index_plan_matches={match["full_plan"]}/10 '
              f'row_plan_matches={match["row_plans"]}/160 owner_count_vector_matches={match["workload"]}/10')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('peer_baseline', type=Path)
    args = parser.parse_args()
    for group, name in ARCHIVES.items():
        root = args.peer_baseline / name
        print(f'ARCHIVE G{group} source={root}', flush=True)
        placement = read_json(root / f'g{group}/logical-ties/thp.json')
        print(f'G{group} PLACEMENT device={placement["ub_device"]} '
              f'numa_pages={placement["numa_pages_overlapping_vmas"]} '
              f'thp_min_pct={placement["coverage_min_pct"]:.2f}')
        describe(group, archive_calls(root, group))
