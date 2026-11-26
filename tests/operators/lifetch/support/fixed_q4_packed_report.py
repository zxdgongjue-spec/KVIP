# SPDX-License-Identifier: Apache-2.0
"""Compare the device event for ten serial calls with the event for ten fused calls."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from fixed_q4_packed_archive import NATIVE, same_production
from large_batch_placement import field, placement, same_card
from reporting import atomic_json

STAGES = ('graph', 'LI', 'tail')


def verify_source(out, archive, repo):
    if field(archive, 'native') != NATIVE or not same_card(archive, out):
        raise ValueError(f'Input archive is from another operator, host or physical card: {archive}')
    for key in ('kvaf', 'kvaf_tree'):
        if not field(out, key) or field(out, key) != field(archive, key):
            raise ValueError(f'Archive and current {key} differ')
    same_production(repo, NATIVE, 'HEAD')


def read_pair(pair, group, batch, cohort, root):
    rows = {}
    identities = []
    for arm in ('serial', 'fused'):
        directory = pair / arm
        row = json.loads((directory / 'packed_result.json').read_text())
        identity = row['identity']
        if (not (directory / 'placement_verified').is_file()
                or row['arm'] != arm or row['calls'] != 10 or row['correctness'] != 'PASS'
                or identity['group'] != group or identity['batch'] != batch
                or identity['physical_npu'] != int(field(root, 'physical_npu'))
                or identity['host'] != field(root, 'host') or identity['native'] != field(root, 'native')
                or row['checked_queries'] != 40 * batch or row['checked_rows'] != 10 * batch
                or (arm == 'serial' and row['get_workers'] != 2)):
            raise ValueError(f'Incomplete or mismatched packed result: {directory}')
        if batch <= 16 and identity['donor_rows'] != list(range(cohort, 16, 16 // batch)):
            raise ValueError(f'Wrong source requests for {directory}')
        times = {s: row['mean_us'][s] for s in ('LI', 'tail')}
        times['graph'] = row['graph_mean_us']
        if (not all(isfinite(v) and v > 0 for v in times.values()) or
                not isclose(times['graph'] * 10, row['graph_event_us'], rel_tol=1e-8)):
            raise ValueError(f'Invalid ten-call device event or stage measurements: {directory}')
        identities.append((identity, placement(directory / 'thp.json')))
        rows[arm] = times
    if identities[0] != identities[1]:
        raise ValueError(f'Serial and fused inputs, runtime or memory placement differ: {pair}')
    return rows


def cell_summary(root, group, batch):
    cell = root / f'g{group}/bs{batch}'
    observations = []
    cohort_count = max(1, 16 // batch)
    for cohort in range(cohort_count):
        directory = cell if cohort_count == 1 else cell / f'cohort{cohort}'
        for block in (1, 2):
            observations.append(read_pair(directory / f'block{block}', group, batch, cohort, root))
    times = {arm: {s: mean(r[arm][s] for r in observations) for s in STAGES}
             for arm in ('serial', 'fused')}
    saved = {s: times['serial'][s] - times['fused'][s] for s in STAGES}
    gain = 100 * saved['graph'] / times['serial']['graph']
    return dict(group=group, batch=batch, cohorts=cohort_count, blocks=2, calls_per_graph=10,
                mean_us=times, saved_us=saved, graph_gain_pct=gain,
                reached_20pct=times['fused']['graph'] <= 0.8 * times['serial']['graph'])


def report_cell(row):
    calls = row['cohorts'] * row['blocks'] * row['calls_per_graph']
    print(f"G{row['group']} BS{row['batch']} CHECK TopK_score/KV/Index=PASS "
          f"calls_per_arm={calls} queries_per_arm={4 * row['batch'] * calls}")
    for arm, values in row['mean_us'].items():
        print(f"G{row['group']} BS{row['batch']} {arm} graph_mean_us={values['graph']:.2f} "
              f"LI_mean_us={values['LI']:.2f} tail_mean_us={values['tail']:.2f}")
    saved = row['saved_us']
    print(f"G{row['group']} BS{row['batch']} serial_minus_fused graph_saved_us={saved['graph']:+.2f} "
          f"LI_saved_us={saved['LI']:+.2f} tail_saved_us={saved['tail']:+.2f} "
          f"graph_gain_pct={row['graph_gain_pct']:+.2f} target20={'PASS' if row['reached_20pct'] else 'MISS'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    source = subs.add_parser('verify-source')
    source.add_argument('out', type=Path)
    source.add_argument('archive', type=Path)
    source.add_argument('repo', type=Path)
    cell = subs.add_parser('cell')
    cell.add_argument('out', type=Path)
    cell.add_argument('group', type=int)
    cell.add_argument('batch', type=int)
    summary = subs.add_parser('summary')
    summary.add_argument('out', type=Path)
    summary.add_argument('--groups', type=int, required=True, nargs='+')
    summary.add_argument('--batches', type=int, required=True, nargs='+')
    args = parser.parse_args()
    if args.mode == 'verify-source':
        verify_source(args.out, args.archive, args.repo)
    elif args.mode == 'cell':
        report_cell(cell_summary(args.out, args.group, args.batch))
    else:
        cells = [cell_summary(args.out, g, b) for g in args.groups for b in args.batches]
        passed = sum(row['reached_20pct'] for row in cells)
        path = args.out / 'packed_summary.json'
        payload = dict(protocol='ten_independent_calls_one_device_event', blocks=2, cells=cells)
        archive = Path(field(args.out, 'packed_source'))
        old_path = archive / 'fixed_q4_acceptance.json'
        if 16 in args.batches and old_path.is_file():
            previous = json.loads(old_path.read_text())
            prior = {p['group']: p['new_packed_us'] for p in previous.get('packed', [])
                     if p.get('context_matched') and p.get('physical_npu') == 3}
            for cell in cells:
                group = cell['group']
                if cell['batch'] == 16 and group in prior and len(prior[group]) == 2:
                    historical = mean(prior[group])
                    if isfinite(historical) and historical > 0:
                        cell['historical_bs16_candidate_us'] = historical
                        print(f"G{group} BS16 historical_candidate_mean_us={historical:.2f} "
                              f"fresh_candidate_mean_us={cell['mean_us']['fused']['graph']:.2f}")
        atomic_json(path, payload)
        print(f'PACKED_TARGET cells_at_20pct={passed}/{len(cells)} SUMMARY={path}')


if __name__ == '__main__':
    main()
