"""Summarize non-consuming CQ observations and preserve the original clocks."""
import argparse
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def quantiles(values):
    return '/'.join(f'{x:.2f}' for x in np.percentile(values, (50, 95)))


def credits(details):
    if len(details) != 20:
        raise ValueError('Expected 20 diagnostic calls')
    groups = defaultdict(list)
    for invocation, sample in enumerate(details):
        rows = sample['streaming_cycles']
        timeline = rows[0]['terminal_timeline']
        cores = timeline['cores']
        clock = timeline['clock_mhz']
        if (clock <= 0 or len(rows) != 16 or sample['round'] != invocation or
                [c['block'] for c in cores] != list(range(64))):
            raise ValueError('Incomplete native timing')
        critical_row = timeline['last_block'] // 4
        last_cq = max(range(4), key=lambda q: next(s['end_cycles'] for s in
            cores[critical_row*4+q]['stages'] if s['name'] == 'final_CQ_join'))
        for b, row in enumerate(rows):
            if row['batch'] != b or len(row['final_counts']) != 4:
                raise ValueError('Unexpected row or query order')
            for q, counts in enumerate(row['final_counts']):
                records = counts['credits']
                if sum(rec['entries'] for rec in records) != counts['total']:
                    raise ValueError('Incomplete batch records')
                idx = next(s['end_cycles'] for s in cores[b*4+q]['stages']
                           if s['name'] == 'index_update_and_join')
                for i, rec in enumerate(records):
                    floor = rec['last_absent'] or rec['post_begin']
                    sample_bounds = (
                        max(0, floor-rec['post_end']) / clock,
                        (rec['first_visible']-rec['post_begin']) / clock,
                        max(0, rec['drain_begin']-rec['first_visible']) / clock,
                        int(rec['first_visible'] <= idx),
                        int(rec['last_absent'] >= idx),
                        (rec['drain_end']-rec['drain_begin']) / clock)
                    if records and len(records) > 1:
                        groups[f'batch{i+1}'].append(sample_bounds)
                        if i == 0 and b == critical_row and q == last_cq:
                            groups['last_row_last_final_CQ_batch1'].append(sample_bounds)
                    else:
                        groups['single'].append(sample_bounds)
    if not groups['batch4']:
        raise ValueError('No four-batch G7 records')
    for name, items in groups.items():
        low, high, idle, early, late, drain = np.asarray(items).T
        if np.any(high < low) or np.any(early + late > 1):
            raise ValueError('Invalid CQ visibility interval')
        print(f'G7_CREDIT {name} n={len(low)} '
              f'post_to_visible_lower_P50/P95_us={quantiles(low)} '
              f'post_to_visible_upper_P50/P95_us={quantiles(high)} '
              f'already_visible_before_drain_P50/P95_us={quantiles(idle)} '
              f'drain_P50/P95_us={quantiles(drain)}' +
              (f' visible_by_index_end={int(early.sum())} absent_at_or_after_index_end={int(late.sum())} '
               f'uncertain={int(len(low)-early.sum()-late.sum())}' if name in
               ('batch1', 'single', 'last_row_last_final_CQ_batch1') else ''))


def main(root, group):
    base = root/f'g{group}/paced-tail/results/mainline/cases/diagnose'
    round_dir = base/'fused/diagnose/round-1'
    if group == 10:
        from li_restore_archive import dp_capture_id
        from ranked_integration_sets import INPUT_AND_RESULTS
        inputs = json.loads((base/'replay_input.json').read_text())
        digest = dp_capture_id(10, inputs)
        if digest is not None and digest != INPUT_AND_RESULTS[10][0]:
            raise ValueError('Prepared G10 inputs differ from the archived round-51 fingerprint')
    if group == 7:
        credits(json.loads((round_dir/'detail_samples.json').read_text()))
    files = [root/name for name in ('physical_npu.txt','candidate.txt','reader.txt','host.txt','machine_id.txt','boot_id.txt')
             if (root/name).is_file()]
    files += [base/'config.json', base/'replay_input.json', base/'fused/mainline_checks.json',
              base/'fused/score_topk_check.json', base/'fused/path_result.json',
              root/f'g{group}/paced-tail/thp.json']
    files += [round_dir/name for name in ('detail_samples.json','full_attribution.json','graph_event.json')]
    archive = root/f'g{group}-b20-npu5-tail-data.tgz'
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        partial = Path(tmp)/archive.name
        with tarfile.open(partial, 'w:gz') as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(root), recursive=False)
        os.replace(partial, archive)
    print(f'RESULT_BUNDLE={archive}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('group', type=int, choices=(7,10))
    args = p.parse_args()
    main(args.root, args.group)
