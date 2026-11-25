# SPDX-License-Identifier: Apache-2.0
"""Compare the two incremental PR30 features on matched BS16/Q4 production replays."""
import argparse
import hashlib
from math import isclose
from pathlib import Path
from statistics import mean

import numpy as np

from cap_pair_report import SDK, SDK_TREE, oracle_for
from check_isolated_device import extract
from main_five_report import case_folder, measured_parallel, read
from peer_delta import pair_placement
from pr30_archive_compare import decomposition, marker
from pr30_terminal_phases import terminal_detail
from reporting import atomic_json
from terminal_batch_checks import diagnostic_pressure
from unified_main_device import EXPECTED_TEXT

GROUPS = (10, 4, 17, 7, 2)
FRESH_CALLS = 10
CALLS_PER_GRAPH = {'mainline': 20, 'dualcut': FRESH_CALLS, 'paced': FRESH_CALLS}
REVISIONS = {
    'mainline': 'd6cfd392f722dd3f96ef6e05175f503aac511dbb',
    'dualcut': 'bba4372065e549d0f2291bc08458d6250d4fa161',
    'paced': 'b261b4885edd79dabac822a2e6e3790c7f226ad6',
}
PAIRS = (('mainline', 'dualcut'), ('dualcut', 'paced'))
METRICS = ('span', 'LI', 'tail', 'kernel_gaps')
PER_CALL_MEANS = ('window10_per_call_us', 'event_mean_us', 'between_calls_per_call_us',
                  'event_edge_residual_per_call_us')
TERMINAL = 'parallel_terminal_kernel/1'
LI_TEXT = ('v9_kernel/0', 'v9_kernel/1')


def text_hashes(path):
    result = {}
    for name, sections in extract(Path(path)).items():
        for key, code in sections.items():
            tag = f'{name}/{key[1]}'
            if key[4] != '.text' or tag in result:
                raise ValueError(f'Unexpected executable section {key} in {name}')
            result[tag] = hashlib.sha256(code).hexdigest()
    if result.keys() != EXPECTED_TEXT.keys():
        raise ValueError(f'Incomplete executable sections in {path}')
    return result


def group_reference(root, group):
    group_marker = root/f'g{group}/reference.txt'
    value = group_marker.read_text().strip() if group_marker.exists() else marker(root, 'reference')
    if value == 'unavailable':
        return None
    if not value or not Path(value).is_absolute():
        raise ValueError(f'G{group}: expected an absolute archive path or unavailable')
    return Path(value).resolve()


def check_roots(root, archive):
    for arm, revision in REVISIONS.items():
        if marker(root, arm) != revision:
            raise ValueError(f'Wrong pinned build for {arm}')
    if archive is not None and marker(archive, 'main') != REVISIONS['mainline']:
        raise ValueError(f'Unexpected archived main revision in {archive}')
    for directory in ((root, archive) if archive is not None else (root,)):
        if (marker(directory, 'kvaf'), marker(directory, 'kvaf_tree'), marker(directory, 'physical_npu')) != (
                SDK, SDK_TREE, '4'):
            raise ValueError(f'Unexpected SDK, source tree or NPU in {directory}')
    if not marker(root, 'host'):
        raise ValueError('Missing replay hostname')
    if archive is None:
        return
    if marker(root, 'host') != marker(archive, 'host'):
        raise ValueError('Baseline belongs to a different host')
    if (marker(root, 'machine_id') and marker(archive, 'machine_id') and
            marker(root, 'machine_id') != marker(archive, 'machine_id')):
        raise ValueError('Baseline belongs to a different machine')


def receipt_summary(calls, paced):
    if len(calls) != FRESH_CALLS:
        raise ValueError('Incomplete measurement block')
    final = np.asarray([r['final'] for r in calls])
    prefetch = np.asarray([r['prefetch'] for r in calls])
    cqe = np.asarray([r['cqe'] for r in calls])
    if (prefetch.shape != (FRESH_CALLS, 16) or cqe.shape != (FRESH_CALLS, 16, 4) or final.shape != cqe.shape or
            any(x.dtype.kind not in 'iu' for x in (prefetch, cqe, final)) or
            np.any(prefetch < 0) or np.any(prefetch > 2048) or np.any(final < 0) or np.any(final > 2048) or
            np.any(cqe < 0) or np.any(cqe > 4)):
        raise ValueError('Malformed per-query fetch/CQ counts')
    final = final.astype(np.int64)
    # An unsplit, nonempty GET has exactly one completion; the paced kernel
    # requests another completion at each distinct intermediate prefix.
    wanted = (final > 0).astype(np.int64)
    eligible = (final.sum(axis=2, keepdims=True) >= 256) & (final > 1)
    if paced:
        previous = np.zeros_like(final)
        for numerator in (1, 6, 11):
            cut = (final * numerator + 15) // 16
            wanted += eligible & (cut > previous) & (cut < final)
            previous = cut
    if not np.array_equal(wanted, cqe):
        raise ValueError('Actual CQ receipts differ from the declared batching policy')
    split = cqe > 1
    return dict(prefetch=float(prefetch.sum())/len(calls), final=float(final.sum())/len(calls),
                split_q=int(split.sum()), split_rows=int(np.count_nonzero(split.any(axis=2))),
                eligible_q=int(eligible.sum()), cqe_hist=np.bincount(cqe.ravel().astype(np.int64), minlength=5).tolist())


def diagnostic(folder, policy, expected_calls, receipts=None):
    samples = read(folder/'diagnose/round-1/detail_samples.json')
    if [s['round'] for s in samples] != list(range(expected_calls)) or any(s['source_round'] != 1 for s in samples):
        raise ValueError(f'{folder}: incomplete diagnostic calls')
    detail = terminal_detail(folder, expected_calls=expected_calls)
    if detail['status'] != 'OK':
        raise ValueError(f'{folder}: incomplete terminal timeline')
    traffic = [s['algorithm']['totals'] for s in samples]
    pressure_keys = ('first_post_peak_4us_tokens', 'peak_pending_queries')
    if receipts is not None:
        if len(receipts) != len(samples):
            raise ValueError('Diagnostic receipt count differs from the measured timeline count')
        pressure = [{k: r[k] for k in pressure_keys} for r in receipts]
        for counts, observed, peaks in zip(traffic, receipts, pressure):
            if (counts['prefetch'] != sum(observed['prefetch']) or
                    counts['final'] != sum(sum(row) for row in observed['final']) or
                    not 0 <= peaks['first_post_peak_4us_tokens'] <= counts['final'] or
                    not 0 <= peaks['peak_pending_queries'] <= 64):
                raise ValueError('Diagnostic traffic disagrees with recorded queue receipts')
    if policy == 'single':
        upstream = [diagnostic_pressure(s, policy) for s in samples]
        if receipts is not None and pressure != upstream:
            raise ValueError('Live doorbells disagree with captured single-Post timestamps')
        pressure = upstream
    elif receipts is None:
        raise ValueError('Missing paced doorbell observations')
    return dict(
        **{k: mean(r[k] for r in traffic) for k in ('prefetch', 'final')},
        **{k: mean(r[k] for r in pressure) for k in pressure_keys},
        last_worker_mean_us={k: detail['phases_us'][k][0] for k in
                             ('WQE_and_publish', 'index_update_and_join', 'final_CQ_join', 'prefetch_CQ_join')},
    )


def feature_times(profile, checked):
    """Reconstruct timing from the measured four-kernel invocations, in microseconds."""
    samples = profile['samples']
    calls = len(samples)
    if calls not in (10, 20) or calls != profile['graph_calls']:
        raise ValueError('Expected one graph containing 10 or 20 complete calls')
    event = profile['graph_accounting']['event_us']
    event_mean = event / calls
    if not isclose(event_mean, checked['graph'][0], abs_tol=.025):
        raise ValueError('Event duration disagrees with the validated graph timing')
    parts = decomposition(profile, {'event_mean_us': event_mean}, checked)
    starts, finishes = [], []
    for sample in samples:
        accounting = sample['accounting']
        intervals = accounting['intervals_us']
        start = min(a for a, _ in intervals['li'])
        finish = intervals['topk_resolve_update'][0][1]
        if finishes and start < finishes[-1]:
            raise ValueError('Adjacent measured invocations overlap or are out of order')
        stages = accounting['stages_us']
        if not isclose(finish - start, stages['li'] + stages['topk_resolve_update']
                       + accounting['inter_kernel_gap_us'], abs_tol=.025):
            raise ValueError('Per-call span differs from LI, tail and gaps')
        starts.append(start)
        finishes.append(finish)
    between = sum(start - finish for start, finish in zip(starts[1:], finishes))
    window = 10
    averages = [(finishes[i + window - 1] - starts[i]) / window for i in range(0, calls, window)]
    # Only the device event is available at the graph boundaries. Its offset
    # from the profiler envelope cannot be assigned to individual calls.
    edge_residual = event - (finishes[-1] - starts[0])
    if not isclose(parts['event_outside_span_mean'], (between + edge_residual) / calls, abs_tol=.025):
        raise ValueError('Between-call gaps and event residual do not reconcile')
    return dict(span=parts['kernel_span'], LI=checked['LI1_to_LI3'], tail=checked['final_topk_through_tail'],
                kernel_gaps=parts['inter_kernel_gap'], window10_per_call_us=mean(averages),
                event_mean_us=event_mean, between_calls_per_call_us=between / calls,
                event_edge_residual_per_call_us=edge_residual / calls, event_total_us=event)


def collect_arm(root, group, arm, owner, oracle, host):
    folder = case_folder(root, group, arm)
    archived = arm == 'mainline'
    calls = CALLS_PER_GRAPH[arm]
    times, exact, thp, proof, cfg = measured_parallel(
        root, group, arm, REVISIONS[arm], owner, *oracle, verify_library=not archived, expected_calls=calls)
    timing = feature_times(read(folder/'diagnose/round-1/full_attribution.json'), times)
    if proof['runtime']['host'] != host or proof['runtime']['visible'] != '4':
        raise ValueError(f'G{group} {arm}: wrong measured host or device')
    if archived:
        text = proof['device_text_sha256']
        if text != EXPECTED_TEXT:
            raise ValueError(f'G{group}: upstream archive reported a different device binary')
    else:
        text = text_hashes(proof['loaded_library'])
    measurements = None
    policy = 'paced' if arm == 'paced' else 'single'
    detail_receipts = None
    if not archived:
        raw = read(folder/'ablation_receipts.json')
        if raw['policy'] != policy or raw['independent_status_allocations'] != calls:
            raise ValueError(f'G{group} {arm}: invalid policy or shared terminal CQ status')
        measurements = receipt_summary(raw['measured'], arm == 'paced')
        detail_receipts = raw['detail']
        receipt_summary(detail_receipts, arm == 'paced')
    diag = diagnostic(folder, policy, calls, detail_receipts)
    # measured_parallel verifies the exact iteration count of each arm.
    varying = {'production_root', 'production_revision', 'kvaf_server', 'output', 'trace_root', 'kvaf_root', 'iterations'}
    input_data = read(folder.parent/'replay_input.json')
    controls = dict(config={k: v for k, v in cfg.items() if k not in varying},
                    runtime=proof['runtime'], capture=proof['capture_operand_hashes'],
                    geometry=input_data['geometry'], identities=input_data['identities'],
                    pages=input_data['page_count'], registered_bytes=thp['bytes'])
    return timing, measurements, diag, text, controls, thp, int(exact.sum())


def saved(old, new):
    return {**{k: [a - b for a, b in zip(old[k], new[k])] for k in METRICS},
            **{k: old[k] - new[k] for k in PER_CALL_MEANS}}


def format_times(times, signed=False):
    spec = '+.2f' if signed else '.2f'
    fields = [f'{k}_mean/P50_us={format(times[k][0], spec)}/{format(times[k][1], spec)}' for k in METRICS]
    fields.extend(f'{k}={format(times[k], spec)}' for k in PER_CALL_MEANS)
    if 'event_total_us' in times:
        fields.append(f'event_total_us={times["event_total_us"]:.2f}')
    return ' '.join(fields)


def report(root, group, owner):
    archive = group_reference(root, group)
    check_roots(root, archive)
    oracle = oracle_for(owner, group)
    timing, measured, diagnostics, images, exact = {}, {}, {}, {}, {}
    arms = tuple(REVISIONS) if archive is not None else ('dualcut', 'paced')
    for arm in arms:
        source = archive if arm == 'mainline' else root
        timing[arm], receipts, diagnostics[arm], images[arm], controls, thp, exact[arm] = collect_arm(
            source, group, arm, owner, oracle, marker(root, 'host'))
        if arm == arms[0]:
            base_controls, base_thp = controls, thp
        else:
            if (controls != base_controls or pair_placement(base_thp, thp, local_baseline=True) != 'matched'):
                raise ValueError(f'G{group} {arm}: input, runtime or CPU transport placement differs')
        if arm != 'mainline':
            measured[arm] = receipts
    code_check = dict(main_tail_equals_dualcut=(
                          images['mainline'][TERMINAL] == images['dualcut'][TERMINAL] if archive else None),
                      dualcut_li_equals_paced=all(images['dualcut'][k] == images['paced'][k] for k in LI_TEXT))
    order = (root/f'g{group}/order.txt').read_text().split()
    if sorted(order) != ['dualcut', 'paced']:
        raise ValueError(f'G{group}: invalid arm order {order}')
    atomic_json(root/f'g{group}/feature_steps.json', dict(
        group=group, calls_per_graph={a: CALLS_PER_GRAPH[a] for a in arms},
        timing=timing, measured=measured, diagnostic=diagnostics,
        device_text=code_check, executable_sha256=images, native=REVISIONS,
        reference=str(archive) if archive is not None else None, order=order,
        runtime=base_controls['runtime'], exact_id_sets=exact))
    for arm in arms:
        calls = CALLS_PER_GRAPH[arm]
        rows, queries = calls * 16, calls * 16 * 4
        print(f'G{group} CHECK arm={arm} TopK_score_vs_strong_serial=PASS calls={calls}/{calls} '
              f'queries={queries}/{queries} KV_and_Index=PASS rows={rows}/{rows} '
              f'ordered_diagnostic_rows={rows}/{rows} placement=matched')
        print(f'G{group} {arm}', format_times(timing[arm]))
        if arm in measured:
            r = measured[arm]
            print(f'G{group} {arm} measured prefetch/final_tokens_per_call={r["prefetch"]:.2f}/{r["final"]:.2f} '
                  f'eligible_Q={r["eligible_q"]}/{queries} split_Q={r["split_q"]}/{queries} '
                  f'split_rows={r["split_rows"]}/{rows} '
                  f'Q_with_0/1/2/3/4_CQEs={"/".join(map(str,r["cqe_hist"]))}')
        d = diagnostics[arm]
        p = d['last_worker_mean_us']
        print(f'G{group} {arm} diagnostic prefetch/final_tokens_per_call={d["prefetch"]:.2f}/{d["final"]:.2f} '
              f'first_post_peak_4us_tokens={d["first_post_peak_4us_tokens"]:.2f} '
              f'peak_pending_Q={d["peak_pending_queries"]:.2f} '
              f'last_worker_WQE/Index/final_CQ/prefetch_CQ_mean_us=' +
              '/'.join(f'{v:.2f}' for v in p.values()))
    for (prior, arm), code_matched in zip(PAIRS, code_check.values()):
        if prior not in timing:
            print(f'G{group} {arm}_vs_{prior} UNAVAILABLE reason=no_matching_local_main_archive')
            continue
        print(f'G{group} {arm}_vs_{prior} saved', format_times(saved(timing[prior], timing[arm]), signed=True),
              'historical_main' if prior == 'mainline' else 'fresh_same_block',
              f'calls_per_graph={CALLS_PER_GRAPH[prior]}/{CALLS_PER_GRAPH[arm]}',
              f'unmodified_stage_binary={"MATCH" if code_matched else "DIFF"}')


def summary(root):
    rows = [read(root/f'g{g}/feature_steps.json') for g in GROUPS]
    for group, row in zip(GROUPS, rows):
        archive = group_reference(root, group)
        expected = CALLS_PER_GRAPH if archive is not None else {'dualcut': FRESH_CALLS, 'paced': FRESH_CALLS}
        if (row['group'] != group or row['calls_per_graph'] != expected or set(row['timing']) != set(expected)
                or row['native'] != REVISIONS or row['reference'] != (str(archive) if archive else None)
                or row['runtime'] != rows[0]['runtime']):
            raise ValueError('Mixed groups, baseline archives, revisions or runtimes')
    print('TIMING span=first_LI_start_to_last_tail_end span_mean=LI_mean+tail_mean+kernel_gaps_mean '
          'event_mean=span_mean+between_calls_per_call+event_edge_residual_per_call '
          'window10=first_LI_to_tenth_tail_divided_by_10 '
          'P50=median_of_individual_samples mainline_archive_graph_calls=20 candidate_graph_calls=10')
    print('DIAGNOSTIC_SEPARATE_REPLAY first_post_peak=first_doorbell_tokens_in_busiest_4us '
          'peak_pending=overlapping_submit_to_software_CQ_observation_intervals '
          'hardware_conflict_counter=unavailable')
    for prior, arm in PAIRS:
        missing = [str(row['group']) for row in rows if prior not in row['timing']]
        if missing:
            print(f'FIVE_GROUP_EQUAL_WEIGHT {arm}_vs_{prior} UNAVAILABLE missing_groups={",".join(missing)}')
            continue
        diffs = [saved(r['timing'][prior], r['timing'][arm]) for r in rows]
        stages = [f'{k}_saved_mean_us={mean(d[k][0] for d in diffs):+.3f}' for k in METRICS]
        stages.extend(f'{k}_saved={mean(d[k] for d in diffs):+.3f}' for k in PER_CALL_MEANS)
        print(f'FIVE_GROUP_EQUAL_WEIGHT {arm}_vs_{prior}', *stages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', nargs='?', type=int, choices=GROUPS)
    parser.add_argument('owner', nargs='?', type=Path)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summary(args.root)
    elif args.group is not None and args.owner:
        report(args.root, args.group, args.owner)
    else:
        parser.error('pass a group and an owner archive, or --summary')
