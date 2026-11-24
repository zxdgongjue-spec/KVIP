# SPDX-License-Identifier: Apache-2.0
"""Verify one completed production graph against the exact pre-512K graph."""
import sys
from pathlib import Path
from statistics import mean

from main_five_report import case_folder, read
from pr30_archive_compare import INDIVIDUAL, PACKED, find_arm, measure_arm, same_controls, same_work
from pr30_terminal_phases import print_terminal_detail, terminal_detail


def protocol_delta(candidate, group, archived, new):
    revision = (candidate/'main.txt').read_text().strip()
    if (archived/'main.txt').read_text().strip() != revision:
        raise ValueError('Single-call and packed candidate archives require identical native revisions')
    single = measure_arm(archived, 'cap512k', revision, group)
    if (single['protocol'] != INDIVIDUAL or
            any(single[k] != new[k] for k in ('binary', 'geometry', 'oracle', 'capture', 'cap_state')) or
            not same_controls(archived, candidate, single, new)):
        raise ValueError('The single-call archive differs in input, placement or binary')
    a, b = single['stats'], new['stats']
    print(f'G{group} same_candidate_packed_minus_single '
          f'event_mean_us={b["event_mean"]-a["event_mean"]:+.2f} '
          f'four_kernel_span_mean_us={b["kernel_span"][0]-a["kernel_span"][0]:+.2f} '
          f'tail_mean_us={b["final_topk_through_tail"][0]-a["final_topk_through_tail"][0]:+.2f}')


def report(candidate, group, baseline, repo, single_archive):
    baseline_arm = find_arm(baseline, 'exact', repo, group)
    if not baseline_arm:
        raise ValueError(f'G{group}: no matching pre-512K PR30 binary in {baseline}')
    ref = measure_arm(baseline, *baseline_arm, group)
    revision = (candidate/'main.txt').read_text().strip()
    new = measure_arm(candidate, 'cap512k', revision, group, verify_library=True)
    ref_folder = case_folder(baseline, group, baseline_arm[0])
    new_folder = case_folder(candidate, group, 'cap512k')
    proof = read(new_folder/'mainline_checks.json')
    if (new['protocol'] != PACKED or ref['protocol'] != PACKED or not same_work(ref, new) or
            not same_controls(baseline, candidate, ref, new) or
            proof['independent_buffers'] != dict(index=20, kv=20, winners=20, window=20) or
            proof['library_sha256'] == read(ref_folder/'mainline_checks.json')['library_sha256']):
        raise ValueError(f'G{group}: PR30 and the 512K candidate failed the production comparison checks')
    print(f'G{group} CHECK both_arms TopK_score_vs_strong_serial=PASS queries_per_arm=1280/1280 '
          f'KV=PASS ordered_Index=PASS rows_per_arm=320/320 independent_calls=20 measured_graphs_per_arm=1')
    metrics = {}
    for label, arm in [('PR30', ref), ('active_512k', new)]:
        s = arm['stats']
        values = (s['event_mean'], s['kernel_span'][1], *s['LI1_to_LI3'], *s['final_topk_through_tail'])
        metrics[label] = values
    metrics['candidate_minus_PR30'] = tuple(n-o for o,n in zip(metrics['PR30'], metrics['active_512k']))
    for label, values in metrics.items():
        fmt = '+.2f' if label == 'candidate_minus_PR30' else '.2f'
        pairs = [f'{k}_mean/P50_us={format(a, fmt)}/{format(b, fmt)}'
                 for k,a,b in zip(('graph', 'LI', 'tail'), values[::2], values[1::2])]
        print(f'G{group} {label}', *pairs)
    for label, folder in (('PR30', ref_folder), ('active_512k', new_folder)):
        requests = read(folder/'mainline_checks.json').get('history_cap', {}).get('measured_requests')
        if requests:
            print(f'G{group} {label} prefetch/final_tokens_per_call_mean='
                  f'{mean(requests["prefetch"]):.2f}/{mean(requests["final"]):.2f}')
    print_terminal_detail(group, dict(PR30=terminal_detail(ref_folder), active_512k=terminal_detail(new_folder)))
    try:
        protocol_delta(candidate, group, single_archive, new)
    except (OSError, KeyError, ValueError) as error:
        print(f'G{group} same_candidate_packed_minus_single UNAVAILABLE reason={error}')


if __name__ == '__main__':
    report(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
