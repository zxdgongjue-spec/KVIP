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


REFERENCE = '10442f1ffa314606c23f54b73e49ecfc372f1f9a'


def verify_source(out, archive, repo, *, candidate=False):
    if field(archive, 'native') != NATIVE or not same_card(archive, out):
        raise ValueError(f'Input archive is from another operator, host or physical card: {archive}')
    for key in ('kvaf', 'kvaf_tree'):
        if not field(out, key) or field(out, key) != field(archive, key):
            raise ValueError(f'Archive and current {key} differ')
    same_production(repo, NATIVE, 'HEAD', candidate=candidate)


def find_control(out, archive, repo, explicit):
    verify_source(out, archive, repo, candidate=True)
    wanted = {(g, b) for g in (10, 4, 17, 7, 2) for b in (1, 2, 4, 8, 16, 20, 24, 28, 32)}
    choices = [Path(explicit)] if explicit else list(out.parent.glob('fixed-q4-packed.*'))
    for ctl in sorted(choices, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        if (not (ctl / 'packed_summary.json').is_file() or field(ctl, 'native') != REFERENCE
                or not same_card(ctl, out) or Path(field(ctl, 'packed_source')) != archive
                or any(field(ctl, key) != field(out, key) for key in ('kvaf', 'kvaf_tree'))):
            continue
        summary = json.loads((ctl / 'packed_summary.json').read_text())
        cells = summary.get('cells', [])
        if (summary.get('protocol') == 'ten_independent_calls_one_device_event' and summary.get('blocks') == 2
                and len(cells) == len(wanted) and {(r['group'], r['batch']) for r in cells} == wanted):
            return ctl.resolve()
    raise ValueError('No completed matching 45-cell ten-call baseline. Resume the original run to completion.')


def read_arm(pair, group, batch, cohort, root, arm, *, traffic=False):
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
    if traffic:
        calls = row['traffic']
        if len(calls) != row['calls'] or any(type(c[k]) != int or c[k] < 0
                                             for c in calls for k in ('prefetch', 'final')):
            raise ValueError(f'Missing or invalid fused transfer counts: {directory}')
        for key in ('prefetch', 'final'):
            times[key] = mean(c[key] for c in calls)
        if all('first_prefetch' in c and 'first_nonempty_rows' in c for c in calls):
            times['first_prefetch'] = mean(c['first_prefetch'] for c in calls)
            times['first_nonempty_rows'] = mean(c['first_nonempty_rows'] for c in calls)
    # Each arm has already been checked against its own expected operator revision.
    identity = {k: v for k, v in identity.items() if k != 'native'}
    return times, (identity, placement(directory / 'thp.json'))


def cell_summary(root, group, batch):
    reference = field(root, 'reference')
    control = Path(reference) if reference else None
    candidate_blocks = int(field(root, 'candidate_blocks') or '2')
    if candidate_blocks not in (1, 2) or (candidate_blocks == 1 and control is None):
        raise ValueError('Single-block timing requires a two-block archived control')
    sources = {'serial': (control or root, 'serial', 2), 'fused': (root, 'fused', candidate_blocks)}
    if control:
        sources['previous_fused'] = (control, 'fused', 2)

    observations = {label: [] for label in sources}
    block_graphs = {label: [[] for _ in range(count)] for label, (_, _, count) in sources.items()}
    cohort_count = max(1, 16 // batch)
    for cohort in range(cohort_count):
        relative = Path(f'g{group}/bs{batch}')
        if cohort_count > 1:
            relative /= f'cohort{cohort}'
        expected = None
        for label, (source, arm, count) in sources.items():
            for block in range(1, count + 1):
                directory = source / relative / f'block{block}'
                values, identity = read_arm(directory, group, batch, cohort, source, arm,
                                            traffic=control is not None and arm == 'fused')
                if expected is not None and identity != expected:
                    raise ValueError(f'Serial and fused inputs, runtime or memory placement differ: {directory}')
                expected = identity
                observations[label].append(values)
                block_graphs[label][block - 1].append(values['graph'])
    times = {arm: {s: mean(r[s] for r in rows) for s in STAGES} for arm, rows in observations.items()}
    saved = {s: times['serial'][s] - times['fused'][s] for s in STAGES}
    gain = 100 * saved['graph'] / times['serial']['graph']
    calls = {arm: 10 * len(rows) for arm, rows in observations.items()}
    result = dict(group=group, batch=batch, cohorts=cohort_count, blocks=candidate_blocks, calls_per_graph=10,
                  calls_by_arm=calls, mean_us=times, saved_us=saved, graph_gain_pct=gain,
                  reached_20pct=times['fused']['graph'] <= 0.8 * times['serial']['graph'])
    if control:
        delta = {k: times['previous_fused'][k] - times['fused'][k] for k in STAGES}
        result['tokens_per_call'] = {arm: {key: mean(r[key] for r in observations[arm])
                                           for key in ('prefetch', 'final')}
                                     for arm in ('previous_fused', 'fused')}
        if all('first_prefetch' in r for r in observations['fused']):
            result['first_window'] = {k: mean(r[k] for r in observations['fused'])
                                      for k in ('first_prefetch', 'first_nonempty_rows')}
        previous = [mean(block) for block in block_graphs['previous_fused']]
        if candidate_blocks == 2:
            result['graph_saved_vs_previous_by_block_us'] = [
                old - mean(new) for old, new in zip(previous, block_graphs['fused'])]
        else:
            result['previous_graph_by_block_us'] = previous
        graph = times['fused']['graph']
        serial = times['serial']['graph']
        result.update(saved_vs_previous_us=delta, reached_15pct=graph <= 0.85 * serial,
                      gap_to_15_us=graph - 0.85 * serial, gap_to_20_us=graph - 0.80 * serial,
                      no_slowdown=delta['graph'] >= 0)
    return result


def report_cell(row):
    calls = row['calls_by_arm']
    if row['blocks'] == 1:
        counts = (f"candidate_calls={calls['fused']} archived_calls_per_arm={calls['serial']} "
                  f"candidate_queries={4 * row['batch'] * calls['fused']} "
                  f"archived_queries_per_arm={4 * row['batch'] * calls['serial']}")
    else:
        counts = f"calls_per_arm={calls['fused']} queries_per_arm={4 * row['batch'] * calls['fused']}"
    print(f"G{row['group']} BS{row['batch']} CHECK TopK_score/KV/Index=PASS {counts}")
    for arm, values in row['mean_us'].items():
        print(f"G{row['group']} BS{row['batch']} {arm} graph_mean_us={values['graph']:.2f} "
              f"LI_mean_us={values['LI']:.2f} tail_mean_us={values['tail']:.2f}")
    saved = row['saved_us']
    if 'saved_vs_previous_us' in row:
        delta = row['saved_vs_previous_us']
        old, new = (row['tokens_per_call'][arm] for arm in ('previous_fused', 'fused'))
        if row['blocks'] == 1:
            b1, b2 = row['previous_graph_by_block_us']
            block_text = f"previous_graph_block1/block2_us={b1:.2f}/{b2:.2f}"
        else:
            b1, b2 = row['graph_saved_vs_previous_by_block_us']
            block_text = f"previous_minus_new_block1/block2_us={b1:+.2f}/{b2:+.2f}"
        first = row.get('first_window')
        first_text = (f" first_window_GETs/active_rows={first['first_prefetch']:.1f}/"
                      f"{first['first_nonempty_rows']:.1f}" if first else '')
        print(f"G{row['group']} BS{row['batch']} saved_vs_serial_us={saved['graph']:+.2f} "
              f"gain_vs_serial_pct={row['graph_gain_pct']:+.2f} target15={'PASS' if row['reached_15pct'] else 'MISS'} "
              f"gap_to_15/20_us={row['gap_to_15_us']:+.2f}/{row['gap_to_20_us']:+.2f} "
              f"previous_minus_new_graph/LI/tail_us={delta['graph']:+.2f}/{delta['LI']:+.2f}/{delta['tail']:+.2f} "
              f"no_slowdown={'PASS' if row['no_slowdown'] else 'MISS'} "
              f"{block_text} "
              f"prefetch_old/new={old['prefetch']:.1f}/{new['prefetch']:.1f} "
              f"final_old/new={old['final']:.1f}/{new['final']:.1f}{first_text}")
        return
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
    ctl = subs.add_parser('reference')
    ctl.add_argument('out', type=Path)
    ctl.add_argument('archive', type=Path)
    ctl.add_argument('repo', type=Path)
    ctl.add_argument('explicit')
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
    elif args.mode == 'reference':
        print(find_control(args.out, args.archive, args.repo, args.explicit))
    elif args.mode == 'cell':
        report_cell(cell_summary(args.out, args.group, args.batch))
    else:
        cells = [cell_summary(args.out, g, b) for g in args.groups for b in args.batches]
        passed = sum(row['reached_20pct'] for row in cells)
        path = args.out / 'packed_summary.json'
        payload = dict(protocol='ten_independent_calls_one_device_event', blocks=cells[0]['blocks'], cells=cells)
        if field(args.out, 'reference'):
            payload['archived_blocks_per_arm'] = 2
        archive = Path(field(args.out, 'packed_source'))
        old_path = archive / 'fixed_q4_acceptance.json'
        if 16 in args.batches and old_path.is_file():
            previous = json.loads(old_path.read_text())
            prior = {p['group']: p for p in previous.get('packed', [])
                     if p.get('context_matched') and p.get('physical_npu') == int(field(args.out, 'physical_npu'))}
            for cell in cells:
                group = cell['group']
                if cell['batch'] == 16 and group in prior and len(prior[group]['new_packed_us']) == 2:
                    archived = prior[group]
                    historical = mean(archived['new_packed_us'])
                    refs = archived.get('reference_packed_us', [])
                    if archived.get('reference_kind') == 'same_card_live' and len(refs) == 2 and all(
                            isfinite(v) and v > 0 for v in refs):
                        pr30 = mean(refs)
                        delta = pr30 - cell['mean_us']['fused']['graph']
                        cell['same_card_archived_pr30_mean_us'] = pr30
                        cell['archived_pr30_minus_fused_us'] = delta
                        print(f"G{group} BS16 same_card_archived_PR30_us={pr30:.2f} "
                              f"PR30_minus_candidate_us={delta:+.2f}")
                    if isfinite(historical) and historical > 0:
                        cell['historical_bs16_candidate_us'] = historical
                        print(f"G{group} BS16 historical_candidate_mean_us={historical:.2f} "
                              f"fresh_candidate_mean_us={cell['mean_us']['fused']['graph']:.2f}")
        if field(args.out, 'reference'):
            hits = sum(c['reached_15pct'] for c in cells)
            steady = sum(c['no_slowdown'] for c in cells)
            both = sum(c['reached_15pct'] and c['no_slowdown'] for c in cells)
            bs16 = [c for c in cells if c['batch'] == 16]
            compared = [c['archived_pr30_minus_fused_us'] for c in bs16 if 'archived_pr30_minus_fused_us' in c]
            parity = sum(v >= 0 for v in compared)
            payload['acceptance'] = dict(cells=len(cells), gain15=hits, gain20=passed, no_slowdown=steady, both=both,
                                         bs16=len(bs16), bs16_pr30_compared=len(compared), bs16_pr30_parity=parity,
                                         all_pass=both == len(cells) and parity == len(bs16))
            atomic_json(path, payload)
            print(f'AIV_TARGET gain15={hits}/{len(cells)} gain20={passed}/{len(cells)} no_slowdown={steady}/{len(cells)} '
                  f'both={both}/{len(cells)} bs16_PR30_parity={parity}/{len(bs16)} '
                  f'bs16_PR30_compared={len(compared)}/{len(bs16)} SUMMARY={path}')
            return
        atomic_json(path, payload)
        print(f'PACKED_TARGET cells_at_20pct={passed}/{len(cells)} SUMMARY={path}')


if __name__ == '__main__':
    main()
