"""Compare a five-case LI candidate with the saved B24 parallel-terminal measurements."""
import argparse
from pathlib import Path
from statistics import mean

from li_restore_report import saved, stats
from main_five_report import archived_oracle, case_folder, GROUPS, MAIN, measured_parallel, read
from peer_delta import pair_placement, single_numa_node
from reporting import atomic_json

KVAF = 'a2857b460a74704114fd30a849ae38666cc2c372'


def details(folder):
    profile = read(folder / 'diagnose/round-1/full_attribution.json')
    phases = []
    fetches = []
    for row in profile['samples']:
        phases.append([stop - start for start, stop in sorted(row['accounting']['intervals_us']['li'])])
        totals = row.get('algorithm', {}).get('totals', {})
        pair = (totals.get('prefetch'), totals.get('final'))
        if all(isinstance(v, int) and v >= 0 for v in pair):
            fetches.append(pair)
    return [mean(series) for series in zip(*phases)], fetches


def critical_window_workers(folder):
    evidence = folder/'window_workers.json'
    if not evidence.is_file():
        return
    calls = read(evidence)['calls']
    details = read(folder/'diagnose/round-1/detail_samples.json')
    if len(calls) != 20 or len(calls) != len(details):
        raise ValueError('Incomplete final-window worker trace')
    late, dispatch, work = [], [], []
    for rows, detail in zip(calls, details):
        head = detail['streaming_cycles'][0]
        events = [e for e in head['prefetch_services'] if e['epoch'] == 3]
        if len(rows) != 16 or len({e['lane'] for e in events}) != 16:
            raise ValueError('Incomplete final-window service lanes')
        lane = max(events, key=lambda e: e['end_cycles'])['lane']
        mhz = head['terminal_timeline']['clock_mhz']
        arrival, snapshot, begin, end, *_ = max(rows[lane], key=lambda query: query[3])
        late.append(max(0, arrival-snapshot)/mhz)
        dispatch.append((begin-max(arrival, snapshot))/mhz)
        work.append((end-begin)/mhz)
    return late, dispatch, work


def penultimate_worker_counts(calls):
    if len(calls) != 20 or any(len(rows) != 16 for rows in calls):
        raise ValueError('Expected 20 penultimate-window diagnostics with 16 rows each')
    histogram = [0, 0, 0]
    for rows in calls:
        for queries in rows:
            if len(queries) != 2:
                raise ValueError('Expected the Q1 and Q4 helper slots')
            used = 0
            for available, done, arrival, begin, end, _, _, _, assigned, _, _, snapshot in queries:
                if available not in (0, 1) or assigned not in (0, 1) or done != assigned:
                    raise ValueError('Penultimate-window dispatch and completion disagree')
                if assigned and (not available or min(arrival, snapshot) <= 0 or not max(arrival, snapshot) <= begin <= end):
                    raise ValueError('Invalid penultimate-window helper timing')
                used += assigned
            histogram[used] += 1
    return histogram


def compare(root, group, owner, ref):
    if (Path((root/'reference.txt').read_text().strip()).resolve() != ref.resolve()
            or any((x/'main.txt').read_text().strip() != MAIN or (x/'kvaf.txt').read_text().strip() != KVAF
                   or (x/'physical_npu.txt').read_text().strip() != '4' for x in (root, ref))
            or (ref/'candidate.txt').exists()):
        raise ValueError('Expected the archived upstream parallel terminal and a B24 NPU4 LI candidate')
    revision = (root/'candidate.txt').read_text().strip()
    if revision == MAIN:
        raise ValueError('Candidate is the archived mainline itself')
    native, receipt = archived_oracle(owner, group)
    archive_folder = case_folder(ref, group, 'mainline')
    candidate_folder = case_folder(root, group, 'li-window')
    oracle_inputs = read(case_folder(owner, group, 'logical-ties').parent/'replay_input.json')
    old, _, old_thp, old_proof, old_cfg = measured_parallel(
        ref, group, 'mainline', MAIN, owner, native, receipt, oracle_inputs, verify_library=False)
    new, exact, new_thp, new_proof, new_cfg = measured_parallel(
        root, group, 'li-window', revision, owner, native, receipt, oracle_inputs, verify_library=True)
    if old_proof['library_sha256'] == new_proof['library_sha256']:
        raise ValueError('Candidate loaded the unchanged mainline operator binary')
    if Path(old_cfg['kvaf_root']).resolve() != Path(new_cfg['kvaf_root']).resolve():
        raise ValueError('Selected KVAF runtime differs from the baseline')
    if old_thp['bytes'] != new_thp['bytes']:
        raise ValueError('Registered DDR capacity differs from baseline')
    placement = pair_placement(old_thp, new_thp, local_baseline=True)
    runtime = old_proof['runtime'] == new_proof['runtime']
    old_li, old_fetches = details(archive_folder)
    new_li, new_fetches = details(candidate_folder)
    inputs = read(candidate_folder.parent / 'replay_input.json')
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 '
          f'exact_ID_sets={int(exact.sum())}/1280 KV_and_Index_state=PASS measured_rows=320/320 '
          f'ordered_Index=PASS separate_diagnostic_rows=320/320')
    print(f'G{group} mainline_archive', *(f'{k}_mean/P50_us={stats(v)}' for k, v in old.items()))
    print(f'G{group} LI_candidate', *(f'{k}_mean/P50_us={stats(v)}' for k, v in new.items()))
    print(f'G{group} LI_kernels_mean_us mainline=' + '/'.join(f'{v:.2f}' for v in old_li)
          + ' candidate=' + '/'.join(f'{v:.2f}' for v in new_li))
    if len(old_fetches) == len(new_fetches) == 20:
        old_reads = '/'.join(f'{mean(col):.2f}' for col in zip(*old_fetches))
        new_reads = '/'.join(f'{mean(col):.2f}' for col in zip(*new_fetches))
        print(f'G{group} prefetch/final_tokens_per_call_mean mainline={old_reads} candidate={new_reads}')
    else:
        print(f'G{group} prefetch/final_tokens_per_call_mean UNAVAILABLE')
    if placement != 'matched' or not runtime:
        raise ValueError(f'G{group}: comparison unverified: placement={placement} same_runtime={int(runtime)}')
    graph_gain = [100 * (a-b) / a for a, b in zip(old['graph'], new['graph'])]
    print(f'G{group} vs_mainline_archive', *(f'{k}_saved_mean/P50_us={saved(old[k], v)}' for k, v in new.items()),
          f'graph_gain_mean/P50_pct={graph_gain[0]:+.2f}/{graph_gain[1]:+.2f} placement={placement}')
    workers = critical_window_workers(candidate_folder)
    if workers:
        from statistics import median
        columns = ('arrival_after_snapshot','handoff','query_compute')
        print(f'G{group} last_finishing_row_and_Q diagnostic', *(
            f'{name}_mean/P50_us={mean(values):.2f}/{median(values):.2f}'
            for name, values in zip(columns, workers)))
    early = candidate_folder/'penultimate_workers.json'
    if early.is_file():
        none, one, two = penultimate_worker_counts(read(early)['calls'])
        print(f'G{group} penultimate_window diagnostic rows_with_0/1/2_helpers={none}/{one}/{two} '
              f'assigned_and_completed_queries={one+2*two}/640')
    from li_restore_report import serial_graph_line
    from strong_serial_g10 import comparison_line
    print(comparison_line(owner.parent, inputs, new['graph']) if group == 10 else serial_graph_line(group, inputs, new['graph']))
    atomic_json(root/f'g{group}/comparison.json', dict(group=group, production_revision=revision,
                archive=str(ref.resolve()), candidate=new, mainline=old))


def aggregate(rows):
    if len(rows) != len(GROUPS) or {r['group'] for r in rows} != set(GROUPS):
        raise ValueError('Expected one comparison for each of the five capture groups')
    previous = mean(r['mainline']['graph'][0] for r in rows)
    current = mean(r['candidate']['graph'][0] for r in rows)
    return previous, current, 100 * (previous-current)/previous


def summary(root):
    ref = Path((root/'reference.txt').read_text().strip()).resolve()
    revision = (root/'candidate.txt').read_text().strip()
    rows = [read(root/f'g{group}/comparison.json') for group in GROUPS]
    if any(r['production_revision'] != revision or Path(r['archive']) != ref for r in rows):
        raise ValueError('Cannot aggregate results from different candidates or references')
    previous, current, gain = aggregate(rows)
    cases_above_target = sum(row['candidate']['graph'][0] < 0.90*row['mainline']['graph'][0] for row in rows)
    print(f'FIVE_GROUP_EQUAL_WEIGHT graph_mean_us archive={previous:.3f} candidate={current:.3f} '
          f'saved_us={previous-current:+.3f} gain_pct={gain:+.3f} '
          f'groups_over_10pct={cases_above_target}/5 target_graph_mean_us={previous*0.90:.3f} target_reduction_over_10pct={"PASS" if current < previous*0.90 else "NOT_MET"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('group')
    sub.add_argument('root', type=Path)
    sub.add_argument('group', type=int, choices=GROUPS)
    sub.add_argument('owner', type=Path)
    sub.add_argument('reference', type=Path)
    sub = commands.add_parser('numa')
    sub.add_argument('thp', type=Path)
    sub = commands.add_parser('summary')
    sub.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.command == 'group':
        compare(args.root, args.group, args.owner, args.reference)
    elif args.command == 'numa':
        print(single_numa_node(read(args.thp)) or '')
    else:
        summary(args.root)
