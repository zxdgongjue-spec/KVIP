# SPDX-License-Identifier: Apache-2.0
"""Compare the two incremental PR30 features on matched BS16/Q4 production replays."""
import argparse
import hashlib
from pathlib import Path
from statistics import mean

import numpy as np

from cap_pair_report import SDK, SDK_TREE, oracle_for
from check_isolated_device import extract
from main_five_report import case_folder, measured_parallel, read
from peer_delta import pair_placement
from pr30_archive_compare import marker
from pr30_terminal_phases import terminal_detail
from reporting import atomic_json
from terminal_batch_checks import diagnostic_pressure
from unified_main_device import EXPECTED_TEXT

GROUPS = (10, 4, 17, 7, 2)
REVISIONS = {
    'mainline': 'd6cfd392f722dd3f96ef6e05175f503aac511dbb',
    'dualcut': 'bba4372065e549d0f2291bc08458d6250d4fa161',
    'paced': 'b261b4885edd79dabac822a2e6e3790c7f226ad6',
}
PAIRS = (('mainline', 'dualcut'), ('dualcut', 'paced'))
METRICS = ('graph', 'LI', 'tail')
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


def check_roots(root, archive):
    for arm, revision in REVISIONS.items():
        if marker(root, arm) != revision:
            raise ValueError(f'Wrong pinned build for {arm}')
    if marker(archive, 'main') != REVISIONS['mainline']:
        raise ValueError(f'Unexpected archived main revision in {archive}')
    for directory in (root, archive):
        if (marker(directory, 'kvaf'), marker(directory, 'kvaf_tree'), marker(directory, 'physical_npu')) != (
                SDK, SDK_TREE, '4'):
            raise ValueError(f'Unexpected SDK, source tree or NPU in {directory}')
    if not marker(root, 'host') or marker(root, 'host') != marker(archive, 'host'):
        raise ValueError('Baseline belongs to a different host')
    if (marker(root, 'machine_id') and marker(archive, 'machine_id') and
            marker(root, 'machine_id') != marker(archive, 'machine_id')):
        raise ValueError('Baseline belongs to a different machine')


def receipt_summary(calls, paced):
    if len(calls) != 20:
        raise ValueError('Incomplete measurement block')
    final = np.asarray([r['final'] for r in calls])
    prefetch = np.asarray([r['prefetch'] for r in calls])
    cqe = np.asarray([r['cqe'] for r in calls])
    if (prefetch.shape != (20, 16) or cqe.shape != (20, 16, 4) or final.shape != cqe.shape or
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
    return dict(prefetch=float(prefetch.sum())/20, final=float(final.sum())/20,
                split_q=int(split.sum()), split_rows=int(np.count_nonzero(split.any(axis=2))),
                eligible_q=int(eligible.sum()), cqe_hist=np.bincount(cqe.ravel().astype(np.int64), minlength=5).tolist())


def diagnostic(folder, policy, receipts=None):
    samples = read(folder/'diagnose/round-1/detail_samples.json')
    if [s['round'] for s in samples] != list(range(20)) or any(s['source_round'] != 1 for s in samples):
        raise ValueError(f'{folder}: incomplete diagnostic calls')
    detail = terminal_detail(folder)
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


def collect_arm(root, group, arm, owner, oracle, host):
    folder = case_folder(root, group, arm)
    archived = arm == 'mainline'
    times, exact, thp, proof, cfg = measured_parallel(
        root, group, arm, REVISIONS[arm], owner, *oracle, verify_library=not archived)
    timing = dict(zip(METRICS, (times[k] for k in ('graph', 'LI1_to_LI3', 'final_topk_through_tail'))))
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
        if raw['policy'] != policy or raw['independent_status_allocations'] != 20:
            raise ValueError(f'G{group} {arm}: invalid policy or shared terminal CQ status')
        measurements = receipt_summary(raw['measured'], arm == 'paced')
        detail_receipts = raw['detail']
        receipt_summary(detail_receipts, arm == 'paced')
    diag = diagnostic(folder, policy, detail_receipts)
    varying = {'production_root', 'production_revision', 'kvaf_server', 'output', 'trace_root', 'kvaf_root'}
    input_data = read(folder.parent/'replay_input.json')
    controls = dict(config={k: v for k, v in cfg.items() if k not in varying},
                    runtime=proof['runtime'], capture=proof['capture_operand_hashes'],
                    geometry=input_data['geometry'], identities=input_data['identities'],
                    pages=input_data['page_count'], registered_bytes=thp['bytes'])
    return timing, measurements, diag, text, controls, thp, int(exact.sum())


def saved(old, new):
    return {k: [a - b for a, b in zip(old[k], new[k])] for k in METRICS}


def format_times(times, signed=False):
    spec = '+.2f' if signed else '.2f'
    return ' '.join(f'{k}_mean/P50_us={format(times[k][0], spec)}/{format(times[k][1], spec)}' for k in METRICS)


def report(root, group, owner):
    archive = Path(marker(root, 'reference')).resolve()
    check_roots(root, archive)
    oracle = oracle_for(owner, group)
    timing, measured, diagnostics, images, exact = {}, {}, {}, {}, {}
    for arm in REVISIONS:
        source = archive if arm == 'mainline' else root
        timing[arm], receipts, diagnostics[arm], images[arm], controls, thp, exact[arm] = collect_arm(
            source, group, arm, owner, oracle, marker(root, 'host'))
        if arm == 'mainline':
            base_controls, base_thp = controls, thp
        else:
            if (controls != base_controls or pair_placement(base_thp, thp, local_baseline=True) != 'matched'):
                raise ValueError(f'G{group} {arm}: input, runtime or CPU transport placement differs')
            measured[arm] = receipts
    code_check = dict(main_tail_equals_dualcut=images['mainline'][TERMINAL] == images['dualcut'][TERMINAL],
                      dualcut_li_equals_paced=all(images['dualcut'][k] == images['paced'][k] for k in LI_TEXT))
    order = (root/f'g{group}/order.txt').read_text().split()
    if sorted(order) != ['dualcut', 'paced']:
        raise ValueError(f'G{group}: invalid arm order {order}')
    atomic_json(root/f'g{group}/feature_steps.json', dict(
        group=group, timing=timing, measured=measured, diagnostic=diagnostics,
        device_text=code_check, executable_sha256=images, native=REVISIONS, reference=str(archive), order=order,
        runtime=base_controls['runtime'], exact_id_sets=exact))
    print(f'G{group} CHECK TopK_score_vs_strong_serial=PASS queries_per_arm=1280/1280 KV_and_Index=PASS '
          'rows_per_arm=320/320 ordered_diagnostic_rows_per_arm=320/320 placement=matched')
    for arm in REVISIONS:
        print(f'G{group} {arm}', format_times(timing[arm]))
        if arm in measured:
            r = measured[arm]
            print(f'G{group} {arm} measured prefetch/final_tokens_per_call={r["prefetch"]:.2f}/{r["final"]:.2f} '
                  f'eligible_Q={r["eligible_q"]}/1280 split_Q={r["split_q"]}/1280 split_rows={r["split_rows"]}/320 '
                  f'Q_with_0/1/2/3/4_CQEs={"/".join(map(str,r["cqe_hist"]))}')
        d = diagnostics[arm]
        p = d['last_worker_mean_us']
        print(f'G{group} {arm} diagnostic prefetch/final_tokens_per_call={d["prefetch"]:.2f}/{d["final"]:.2f} '
              f'first_post_peak_4us_tokens={d["first_post_peak_4us_tokens"]:.2f} '
              f'peak_pending_Q={d["peak_pending_queries"]:.2f} '
              f'last_worker_WQE/Index/final_CQ/prefetch_CQ_mean_us=' +
              '/'.join(f'{v:.2f}' for v in p.values()))
    for (prior, arm), code_matched in zip(PAIRS, code_check.values()):
        print(f'G{group} {arm}_vs_{prior} saved', format_times(saved(timing[prior], timing[arm]), signed=True),
              'historical_main' if prior == 'mainline' else 'fresh_same_block',
              f'unmodified_stage_binary={"MATCH" if code_matched else "DIFF"}')


def summary(root):
    rows = [read(root/f'g{g}/feature_steps.json') for g in GROUPS]
    if any(r['group'] != g or r['native'] != REVISIONS or r['reference'] != marker(root, 'reference') or
           r['runtime'] != rows[0]['runtime'] for g, r in zip(GROUPS, rows)):
        raise ValueError('Mixed groups, baseline archives, revisions or runtimes')
    print('TIMING graph_mean=graph_event_divided_by_20 graph_P50=median_first_LI_to_tail_end_span '
          'LI=three_LI_kernels tail=final_TopK_through_last_CQ_and_retire')
    print('DIAGNOSTIC_SEPARATE_REPLAY first_post_peak=first_doorbell_tokens_in_busiest_4us '
          'peak_pending=overlapping_submit_to_software_CQ_observation_intervals '
          'hardware_conflict_counter=unavailable')
    for prior, arm in PAIRS:
        diffs = [saved(r['timing'][prior], r['timing'][arm]) for r in rows]
        stages = ' '.join(f'{k}_saved_mean_us={mean(d[k][0] for d in diffs):+.3f}' for k in METRICS)
        print(f'FIVE_GROUP_EQUAL_WEIGHT {arm}_vs_{prior} {stages}')


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
