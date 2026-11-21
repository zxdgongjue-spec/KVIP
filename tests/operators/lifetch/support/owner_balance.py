"""Read an existing cutoff-tail archive and compare local AIV durations within each call."""
import json
import sys
from pathlib import Path
from statistics import median

from owner_tail_report import owner


def read(path):
    return json.loads(path.read_bytes())


def spread(values):
    return max(values) - min(values)


def summarize(group, data, archive_calls):
    samples = data['samples']
    if (data['schema_version'] != 2 or data['checked_calls'] != 10 or len(samples) != 5
            or len(archive_calls) != 20 or data['group'] != group):
        raise ValueError(f'Incomplete G{group} archive')
    archived = [set(call['layout_receipts'][r][0] for call in archive_calls) for r in range(16)]
    within_q = [[] for _ in range(4)]
    within_row = []
    barrier_share = []
    matches = 0
    print(f'G{group} PRIOR placement={data["plan_match"]["placement"]}')
    for sample in samples:
        if len(sample['ticks']) != 64 or len(sample['requests']) != 16:
            raise ValueError(f'Incomplete G{group} trace')
        metrics = [owner(ticks, i, data['ticks_per_us'])[0] for i, ticks in enumerate(sample['ticks'])]
        before = [m['owner_to_barrier'] for m in metrics]
        waits = [m['barrier'] for m in metrics]
        for q in range(4):
            within_q[q].append(spread(before[q::4]))
        within_row.append(median(spread(before[r:r+4]) for r in range(0, 64, 4)))
        barrier_share.append(100 * sum(waits) / (sum(waits) + sum(before)))
        last = min(range(64), key=waits.__getitem__)
        r, query = divmod(last, 4)
        matched = sample['layout_receipts'][r][0] in archived[r]
        matches += matched
        row = metrics[4*r:4*r+4]
        # Each quartet is Q1..Q4 on this one request. Waits on consecutive
        # owners can cover the same interval; show the chain without summing it.
        def quartet(key):
            return '/'.join(f'{m[key]:.2f}' for m in row)
        finals = '/'.join(str(q['final_posted']) for q in sample['requests'][r]['queries'])
        print(f'G{group} NEAR_LAST pair={sample["repeat"]+1} owner=R{r}Q{query+1} '
              f'row_plan_seen_in_archive={int(matched)} arrival_lag_upper_us={waits[last]:.2f} '
              f'Q1_to_Q4_final={finals} allocator_wait_us={quartet("resolver_handoff")} '
              f'adapt_us={quartet("adapt")} index_wait_us={quartet("index_handoff")} '
              f'index_us={quartet("index")} apply_us={quartet("index_apply")} '
              f'Q4_meta_commit_us={row[3]["meta_commit"]:.2f} '
              f'Q4_prefetch_flag/drain_us={row[3]["prefetch_flag"]:.2f}/{row[3]["prefetch_drain_call"]:.2f}')
    ranges = '/'.join(f'{median(v):.2f}' for v in within_q)
    print(f'G{group} BALANCE median_over_5_calls '
          f'Q1_to_Q4_same_role_local_prebarrier_range_us={ranges} '
          f'within_row_median_Q_range_us={median(within_row):.2f} '
          f'barrier_pct_of_summed_owner_time_to_barrier_exit={median(barrier_share):.2f} '
          f'near_last_row_plan_seen={matches}/5')


def main(root):
    archive = read(root / 'archive-calls.json')
    native = (root / 'logical-ties.txt').read_text().strip()
    if (archive['native'] != '24c619b9581aad43fd53c815928def26a1e19560'
            or native != '3f24b6ff66153da3605f0f6f24dacf69ce4b66c4'
            or archive['physical_npu'] != 4 or (root / 'physical_npu.txt').read_text().strip() != '4'):
        raise ValueError('Unexpected candidate or NPU')
    for group in (17, 4, 10, 7):
        data = read(root / f'g{group}/owner_tail.json')
        if data['native_revision'] != native:
            raise ValueError(f'G{group}: different native revision')
        summarize(group, data, archive['groups'][str(group)])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
